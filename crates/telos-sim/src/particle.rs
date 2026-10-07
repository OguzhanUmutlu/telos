//! Pure-CPU particle simulation engine, emitters, and GPU quad data structures.
//!
//! Provides deterministic and physics-integrated particle simulation for:
//! - Block break debris (sampling real block texture sub-rectangles with ground collision)
//! - Block placement dust puffs
//! - Material-tinted footstep particles
//! - Ambient torch flame and rising smoke emitters
//! - Critical hit sparks, explosions, and hearts
//! - Networked particle event integration (`S2cParticleEvent`)

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use telos_protocol::messages::play::{ParticleEffectKind, S2cParticleEvent};

/// Visual category of simulated particle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParticleKind {
    /// Fragmented block debris quad with voxel ground collision and drag.
    Debris,
    /// Expanding, rising smoke puff with frame animation.
    Smoke,
    /// Flickering torch flame sprite with upward buoyancy.
    Flame,
    /// Low-lying dust puff from block placement or footstep.
    Dust,
    /// Critical strike spark spraying outwards.
    Spark,
    /// Floating entity status heart.
    Heart,
}

/// 48-byte GPU billboard particle representation matching std430/scalar buffer layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct ParticleGpu {
    /// World-space center coordinate (X, Y, Z).
    pub pos: [f32; 3],
    /// Billboard quad edge size in world units.
    pub size: f32,
    /// Texture coordinate sub-rectangle minimum [u0, v0].
    pub uv_min: [f32; 2],
    /// Texture coordinate sub-rectangle maximum [u1, v1].
    pub uv_max: [f32; 2],
    /// Packed RGBA8 color tint and alpha opacity (`r | (g << 8) | (b << 16) | (a << 24)`).
    pub color: u32,
    /// 2D texture array layer index.
    pub layer: u32,
    /// Texture source identifier (0 = Particle texture array, 1 = Terrain block texture array).
    pub tex_source: u32,
    /// Flags (e.g. unlit/emissive or rotation).
    pub flags: u32,
}

/// Single simulated CPU particle instance.
#[derive(Debug, Clone)]
pub struct Particle {
    /// World position (X, Y, Z).
    pub pos: Vec3,
    /// World velocity (blocks/sec).
    pub vel: Vec3,
    /// Tint color and alpha opacity [R, G, B, A].
    pub color: [u8; 4],
    /// Current rendered size in world units.
    pub size: f32,
    /// Base starting size.
    pub initial_size: f32,
    /// Current age in seconds.
    pub age: f32,
    /// Total lifetime in seconds before despawning.
    pub max_age: f32,
    /// Texture array layer index.
    pub layer: u32,
    /// Texture source (0 = particle array, 1 = terrain block array).
    pub tex_source: u32,
    /// Sub-texel UV min [u0, v0].
    pub uv_min: [f32; 2],
    /// Sub-texel UV max [u1, v1].
    pub uv_max: [f32; 2],
    /// Downward gravity acceleration (units/sec²).
    pub gravity: f32,
    /// Air drag friction factor (fraction/sec).
    pub drag: f32,
    /// Whether this particle tests voxel ground collision.
    pub collides_voxels: bool,
    /// Particle behavioral category.
    pub kind: ParticleKind,
}

/// High-performance CPU particle simulation system and spawner manager.
#[derive(Debug, Clone)]
pub struct ParticleSystem {
    particles: Vec<Particle>,
    max_particles: usize,
    rng_state: u64,
}

impl Default for ParticleSystem {
    fn default() -> Self {
        Self::new(4096)
    }
}

impl ParticleSystem {
    /// Creates a new particle system with the specified capacity limit.
    #[must_use]
    pub fn new(max_particles: usize) -> Self {
        Self {
            particles: Vec::with_capacity(max_particles.min(1024)),
            max_particles,
            rng_state: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// Returns the number of currently active simulated particles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// Returns `true` if there are no active particles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// Clears all currently active particles.
    pub fn clear(&mut self) {
        self.particles.clear();
    }

    /// Simple fast pseudo-random number generator (PCG / linear congruential) for zero-allocation particle variation.
    fn next_u32(&mut self) -> u32 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.rng_state >> 32) as u32
    }

    /// Returns a pseudo-random float in `[0.0, 1.0)`.
    #[allow(clippy::cast_precision_loss)]
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() & 0x00FF_FFFF) as f32 / 16_777_216.0
    }

    /// Returns a pseudo-random float in `[min, max]`.
    fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }

    /// Advances particle simulation by `dt` seconds, applying gravity, drag, voxel collision, and visual aging.
    pub fn tick<F>(&mut self, dt: f32, is_solid_block: F)
    where
        F: Fn(i32, i32, i32) -> bool,
    {
        if dt <= 0.0 || self.particles.is_empty() {
            return;
        }

        let clamped_dt = dt.min(5.0);
        let max_substep = 0.05;
        let mut remaining = clamped_dt;

        while remaining > 0.0 {
            let substep = remaining.min(max_substep);
            self.tick_substep(substep, &is_solid_block);
            remaining -= substep;
        }

        // Remove expired particles
        self.particles.retain(|p| p.age < p.max_age);
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn tick_substep<F>(&mut self, dt: f32, is_solid_block: &F)
    where
        F: Fn(i32, i32, i32) -> bool,
    {
        for p in &mut self.particles {
            p.age += dt;
            if p.age >= p.max_age {
                continue;
            }

            let t = (p.age / p.max_age).clamp(0.0, 1.0);

            // 1. Physics integration
            p.vel.y += p.gravity * dt;
            let drag_factor = (1.0 - p.drag * dt).max(0.0);
            p.vel *= drag_factor;

            let next_pos = p.pos + p.vel * dt;

            if p.collides_voxels {
                let bx = next_pos.x.floor() as i32;
                let by = next_pos.y.floor() as i32;
                let bz = next_pos.z.floor() as i32;

                if is_solid_block(bx, by, bz) {
                    // Halts downward velocity upon landing on a block surface
                    p.vel = Vec3::ZERO;
                    p.pos.y = by as f32 + 1.02;
                } else {
                    p.pos = next_pos;
                }
            } else {
                p.pos = next_pos;
            }

            // 2. Visual evolution by particle kind
            match p.kind {
                ParticleKind::Smoke => {
                    // Expands and animates through 8 puff frames (0..=7)
                    p.size = p.initial_size * (1.0 + t * 1.6);
                    let initial_a = 220.0;
                    p.color[3] = (initial_a * (1.0 - t).max(0.0)) as u8;
                    p.layer = ((t * 7.99) as u32).min(7);
                }
                ParticleKind::Flame => {
                    // Rapidly flickers, shrinks, and rises
                    p.size = p.initial_size * (1.0 - t * 0.4);
                    let initial_a = 255.0;
                    p.color[3] = (initial_a * (1.0 - t).max(0.0)) as u8;
                }
                ParticleKind::Debris => {
                    // Retains solid opacity until final 30% of lifetime
                    if t > 0.7 {
                        let fade = (1.0 - t) / 0.3;
                        p.color[3] = (255.0 * fade.max(0.0)) as u8;
                    }
                }
                ParticleKind::Dust => {
                    // Linear fade
                    p.color[3] = (200.0 * (1.0 - t).max(0.0)) as u8;
                }
                ParticleKind::Spark => {
                    p.size = p.initial_size * (1.0 - t * 0.5);
                    p.color[3] = (255.0 * (1.0 - t).max(0.0)) as u8;
                }
                ParticleKind::Heart => {
                    // Gentle vertical oscillation and fadeout
                    p.size = p.initial_size * (1.0 + 0.2 * (t * std::f32::consts::PI).sin());
                    if t > 0.6 {
                        p.color[3] = (255.0 * ((1.0 - t) / 0.4).max(0.0)) as u8;
                    }
                }
            }
        }
    }

    /// Spawns block breaking debris particles bursting from the center of a broken block.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn spawn_block_break(
        &mut self,
        center: Vec3,
        block_layer: u32,
        tint: [u8; 4],
        count: usize,
    ) {
        let spawn_count = count.min(self.max_particles.saturating_sub(self.particles.len()));
        for _ in 0..spawn_count {
            // Random offset within the 1x1x1 block cube
            let offset = Vec3::new(
                self.next_f32_range(-0.35, 0.35),
                self.next_f32_range(-0.35, 0.35),
                self.next_f32_range(-0.35, 0.35),
            );

            // Outward burst velocity with upward bias
            let dir_x = self.next_f32_range(-1.0, 1.0);
            let dir_y = self.next_f32_range(0.2, 1.5);
            let dir_z = self.next_f32_range(-1.0, 1.0);
            let speed = self.next_f32_range(1.5, 3.8);
            let vel = Vec3::new(dir_x, dir_y, dir_z).normalize() * speed;

            // Pick a random 4x4 sub-crop in the 16x16 block face texture
            let u_idx = (self.next_u32() % 4) as f32;
            let v_idx = (self.next_u32() % 4) as f32;
            let uv_min = [u_idx * 0.25, v_idx * 0.25];
            let uv_max = [uv_min[0] + 0.25, uv_min[1] + 0.25];

            let size = self.next_f32_range(0.08, 0.16);
            let max_age = self.next_f32_range(0.6, 1.2);

            self.particles.push(Particle {
                pos: center + offset,
                vel,
                color: tint,
                size,
                initial_size: size,
                age: 0.0,
                max_age,
                layer: block_layer,
                tex_source: 1, // Terrain block texture array
                uv_min,
                uv_max,
                gravity: -14.0,
                drag: 0.08,
                collides_voxels: true,
                kind: ParticleKind::Debris,
            });
        }
    }

    /// Spawns gentle dust puff particles around the outer faces of a placed block.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn spawn_block_place(
        &mut self,
        center: Vec3,
        block_layer: u32,
        tint: [u8; 4],
        count: usize,
    ) {
        let spawn_count = count.min(self.max_particles.saturating_sub(self.particles.len()));
        for _ in 0..spawn_count {
            let offset = Vec3::new(
                self.next_f32_range(-0.45, 0.45),
                self.next_f32_range(-0.45, 0.45),
                self.next_f32_range(-0.45, 0.45),
            );
            let vel = Vec3::new(
                self.next_f32_range(-0.4, 0.4),
                self.next_f32_range(0.2, 0.8),
                self.next_f32_range(-0.4, 0.4),
            );
            let size = self.next_f32_range(0.12, 0.22);
            let max_age = self.next_f32_range(0.3, 0.5);

            self.particles.push(Particle {
                pos: center + offset,
                vel,
                color: tint,
                size,
                initial_size: size,
                age: 0.0,
                max_age,
                layer: block_layer.min(7), // Particle generic puff or block layer
                tex_source: 0,             // Particle texture array (smoke puff)
                uv_min: [0.0, 0.0],
                uv_max: [1.0, 1.0],
                gravity: -2.0,
                drag: 0.2,
                collides_voxels: false,
                kind: ParticleKind::Dust,
            });
        }
    }

    /// Spawns dust particles beneath the player's feet when stepping on a surface.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn spawn_footstep(&mut self, pos: Vec3, tint: [u8; 4]) {
        let count = 4;
        let spawn_count = count.min(self.max_particles.saturating_sub(self.particles.len()));
        for _ in 0..spawn_count {
            let offset = Vec3::new(
                self.next_f32_range(-0.25, 0.25),
                0.05,
                self.next_f32_range(-0.25, 0.25),
            );
            let vel = Vec3::new(
                self.next_f32_range(-0.3, 0.3),
                self.next_f32_range(0.1, 0.4),
                self.next_f32_range(-0.3, 0.3),
            );
            let size = self.next_f32_range(0.14, 0.20);
            let max_age = self.next_f32_range(0.25, 0.4);

            self.particles.push(Particle {
                pos: pos + offset,
                vel,
                color: tint,
                size,
                initial_size: size,
                age: 0.0,
                max_age,
                layer: 0,      // generic_0 puff
                tex_source: 0, // Particle texture array
                uv_min: [0.0, 0.0],
                uv_max: [1.0, 1.0],
                gravity: 0.5,
                drag: 0.3,
                collides_voxels: false,
                kind: ParticleKind::Dust,
            });
        }
    }

    /// Spawns a single rising wisp of smoke above a torch wick.
    pub fn spawn_torch_smoke(&mut self, pos: Vec3) {
        if self.particles.len() >= self.max_particles {
            return;
        }

        let offset = Vec3::new(
            self.next_f32_range(-0.03, 0.03),
            0.42,
            self.next_f32_range(-0.03, 0.03),
        );
        let vel = Vec3::new(
            self.next_f32_range(-0.04, 0.04),
            self.next_f32_range(0.35, 0.55),
            self.next_f32_range(-0.04, 0.04),
        );
        let size = self.next_f32_range(0.12, 0.18);
        let max_age = self.next_f32_range(0.9, 1.4);

        self.particles.push(Particle {
            pos: pos + offset,
            vel,
            color: [220, 220, 220, 200],
            size,
            initial_size: size,
            age: 0.0,
            max_age,
            layer: 0,      // generic_0..7 animation
            tex_source: 0, // Particle texture array
            uv_min: [0.0, 0.0],
            uv_max: [1.0, 1.0],
            gravity: 0.15, // Buoyant upward float
            drag: 0.05,
            collides_voxels: false,
            kind: ParticleKind::Smoke,
        });
    }

    /// Spawns a tiny flickering flame sprite at the torch wick.
    pub fn spawn_torch_flame(&mut self, pos: Vec3) {
        if self.particles.len() >= self.max_particles {
            return;
        }

        let offset = Vec3::new(
            self.next_f32_range(-0.02, 0.02),
            0.36,
            self.next_f32_range(-0.02, 0.02),
        );
        let vel = Vec3::new(0.0, self.next_f32_range(0.1, 0.25), 0.0);
        let size = self.next_f32_range(0.10, 0.15);
        let max_age = self.next_f32_range(0.25, 0.45);

        self.particles.push(Particle {
            pos: pos + offset,
            vel,
            color: [255, 230, 180, 255],
            size,
            initial_size: size,
            age: 0.0,
            max_age,
            layer: 8,      // Layer 8 = flame.png
            tex_source: 0, // Particle texture array
            uv_min: [0.0, 0.0],
            uv_max: [1.0, 1.0],
            gravity: 0.2,
            drag: 0.1,
            collides_voxels: false,
            kind: ParticleKind::Flame,
        });
    }

    /// Spawns critical hit spark particles spraying outward from an impact position.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn spawn_crit(&mut self, pos: Vec3, count: usize) {
        let spawn_count = count.min(self.max_particles.saturating_sub(self.particles.len()));
        for _ in 0..spawn_count {
            let vel = Vec3::new(
                self.next_f32_range(-1.5, 1.5),
                self.next_f32_range(0.5, 2.5),
                self.next_f32_range(-1.5, 1.5),
            );
            let size = self.next_f32_range(0.18, 0.28);
            let max_age = self.next_f32_range(0.3, 0.6);

            self.particles.push(Particle {
                pos,
                vel,
                color: [255, 240, 150, 255],
                size,
                initial_size: size,
                age: 0.0,
                max_age,
                layer: 9,      // Layer 9 = spark_0.png
                tex_source: 0, // Particle texture array
                uv_min: [0.0, 0.0],
                uv_max: [1.0, 1.0],
                gravity: -8.0,
                drag: 0.15,
                collides_voxels: false,
                kind: ParticleKind::Spark,
            });
        }
    }

    /// Spawns a floating ambient status effect swirl particle around an entity.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn spawn_status_effect_swirl(&mut self, center: Vec3, color: [u8; 3]) {
        if self.particles.len() >= self.max_particles {
            return;
        }
        let angle = self.next_f32() * std::f32::consts::TAU;
        let radius = self.next_f32_range(0.3, 0.7);
        let offset = Vec3::new(
            angle.cos() * radius,
            self.next_f32_range(-0.5, 0.8),
            angle.sin() * radius,
        );
        let vel = Vec3::new(
            -angle.sin() * 0.2,
            self.next_f32_range(0.1, 0.3),
            angle.cos() * 0.2,
        );
        let size = self.next_f32_range(0.10, 0.16);
        let max_age = self.next_f32_range(0.6, 1.2);

        self.particles.push(Particle {
            pos: center + offset,
            vel,
            color: [color[0], color[1], color[2], 200],
            size,
            initial_size: size,
            age: 0.0,
            max_age,
            layer: 0,
            tex_source: 0,
            uv_min: [0.0, 0.0],
            uv_max: [1.0, 1.0],
            gravity: 0.05,
            drag: 0.05,
            collides_voxels: false,
            kind: ParticleKind::Dust,
        });
    }

    /// Spawns particles corresponding to a server-replicated `S2cParticleEvent`.
    pub fn spawn_from_event<F>(&mut self, event: &S2cParticleEvent, block_layer_lookup: F)
    where
        F: Fn(u32) -> (u32, [u8; 4]),
    {
        let center = Vec3::new(event.x, event.y, event.z);
        let count = usize::from(event.count.max(1));

        match event.effect {
            ParticleEffectKind::BlockBreak => {
                let (layer, tint) = block_layer_lookup(event.block_state_id);
                self.spawn_block_break(center, layer, tint, count);
            }
            ParticleEffectKind::BlockPlace => {
                let (layer, tint) = block_layer_lookup(event.block_state_id);
                self.spawn_block_place(center, layer, tint, count);
            }
            ParticleEffectKind::Footstep => {
                let (_layer, tint) = block_layer_lookup(event.block_state_id);
                self.spawn_footstep(center, tint);
            }
            ParticleEffectKind::Smoke => {
                for _ in 0..count {
                    self.spawn_torch_smoke(center);
                }
            }
            ParticleEffectKind::Flame => {
                for _ in 0..count {
                    self.spawn_torch_flame(center);
                }
            }
            ParticleEffectKind::Crit => {
                self.spawn_crit(center, count);
            }
            ParticleEffectKind::Explosion => {
                self.spawn_crit(center, count);
                for _ in 0..count {
                    self.spawn_torch_smoke(center);
                }
            }
            ParticleEffectKind::Heart => {
                for _ in 0..count {
                    if self.particles.len() >= self.max_particles {
                        break;
                    }
                    let offset = Vec3::new(
                        self.next_f32_range(-0.3, 0.3),
                        self.next_f32_range(0.0, 0.5),
                        self.next_f32_range(-0.3, 0.3),
                    );
                    let vel = Vec3::new(0.0, self.next_f32_range(0.3, 0.6), 0.0);
                    let size = self.next_f32_range(0.18, 0.25);
                    let max_age = self.next_f32_range(0.8, 1.4);

                    self.particles.push(Particle {
                        pos: center + offset,
                        vel,
                        color: [255, 100, 120, 255],
                        size,
                        initial_size: size,
                        age: 0.0,
                        max_age,
                        layer: 10,     // Layer 10 = heart.png
                        tex_source: 0, // Particle texture array
                        uv_min: [0.0, 0.0],
                        uv_max: [1.0, 1.0],
                        gravity: 0.1,
                        drag: 0.05,
                        collides_voxels: false,
                        kind: ParticleKind::Heart,
                    });
                }
            }
        }
    }

    /// Extracts active particles into contiguous 48-byte `ParticleGpu` buffer format.
    pub fn extract_gpu_particles(&self, out: &mut Vec<ParticleGpu>) {
        out.clear();
        out.reserve(self.particles.len());

        for p in &self.particles {
            let color = u32::from(p.color[0])
                | (u32::from(p.color[1]) << 8)
                | (u32::from(p.color[2]) << 16)
                | (u32::from(p.color[3]) << 24);

            out.push(ParticleGpu {
                pos: p.pos.to_array(),
                size: p.size,
                uv_min: p.uv_min,
                uv_max: p.uv_max,
                color,
                layer: p.layer,
                tex_source: p.tex_source,
                flags: 0,
            });
        }
    }
}
