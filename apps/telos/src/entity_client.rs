//! Client-side entity tracking, interpolation, hierarchical cuboid meshing, and combat raycasting.

use std::collections::VecDeque;

use glam::{DVec3, Quat, Vec2, Vec3};
use hashbrown::HashMap;
use telos_protocol::messages::{S2cEntityMove, S2cSpawnArrow, S2cSpawnEntity, S2cSpawnItem};
use telos_sim::EntityType;

/// GPU vertex format for instanced dynamic entity rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EntityVertexGpu {
    /// World-space position of vertex.
    pub pos: [f32; 3],
    /// Packed light: `(sky & 0xF) | ((block & 0xF) << 4) | ((normal_idx & 0x7) << 8)`.
    pub light: u32,
    /// Texture coordinates in normalized `[0.0, 1.0]`.
    pub uv: [f32; 2],
    /// Texture array layer index (0: Zombie, 1: Pig, 2: Cow).
    pub layer: u32,
    /// Hurt red tint flash factor in `[0.0, 1.0]`.
    pub hurt_tint: f32,
}

/// Push constants forwarded to `shaders/entity.vert`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EntityPushConstants {
    /// Combined view-projection matrix.
    pub view_proj: [f32; 16],
    /// 64-bit Buffer Device Address (BDA) pointing to `EntityVertexGpu` array.
    pub vertex_buffer_address: u64,
    /// Alignment padding to 80 bytes.
    pub pad: [u32; 2],
}

/// A replicated snapshot of a remote entity at a specific point in time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntitySnapshot {
    /// Local arrival timestamp in seconds.
    pub timestamp: f64,
    /// World position at snapshot.
    pub pos: DVec3,
    /// Body yaw in degrees.
    pub yaw: f32,
    /// Looking pitch in degrees.
    pub pitch: f32,
    /// Head yaw in degrees.
    pub head_yaw: f32,
}

/// Unwraps and linearly interpolates an angle in degrees across the [0, 360) boundary.
#[must_use]
pub fn lerp_angle(from: f32, to: f32, alpha: f32) -> f32 {
    let mut diff = (to - from) % 360.0;
    if diff < -180.0 {
        diff += 360.0;
    } else if diff > 180.0 {
        diff -= 360.0;
    }
    from + diff * alpha
}

/// Client representation of an active network entity.
#[derive(Debug, Clone)]
pub struct ClientEntity {
    /// Unique network entity identifier.
    pub net_id: u32,
    /// Entity archetype classification.
    pub entity_type: EntityType,
    /// Current interpolated world position.
    pub pos: DVec3,
    /// Target world position from server replication packet.
    pub target_pos: DVec3,
    /// Body orientation yaw angle in degrees.
    pub yaw: f32,
    /// Looking pitch angle in degrees.
    pub pitch: f32,
    /// Independent head yaw angle in degrees.
    pub head_yaw: f32,
    /// Target body yaw from server.
    pub target_yaw: f32,
    /// Continuous walk cycle accumulator for limb animation.
    pub walk_time: f32,
    /// Hurt flash timer in seconds (decays to 0.0).
    pub hurt_timer: f32,
    /// Attack swing animation timer in seconds (decays to 0.0).
    pub attack_swing_timer: f32,
    /// Death fall-over animation timer in seconds (decays to 0.0).
    pub death_timer: f32,
    /// Current health points.
    pub health: f32,
    /// Maximum health points.
    pub max_health: f32,
    /// Item ID if this entity is a dropped item (`EntityType::Item`).
    pub item_id: u32,
    /// Item stack count if this entity is a dropped item.
    pub item_count: u16,
    /// Ring buffer of historical entity snapshots for interpolation.
    pub snapshots: VecDeque<EntitySnapshot>,
}

impl ClientEntity {
    /// Creates a new `ClientEntity` initialized from an `S2cSpawnEntity` packet.
    #[must_use]
    pub fn from_spawn(msg: S2cSpawnEntity, now: f64) -> Self {
        let entity_type = EntityType::from_u8(msg.entity_type).unwrap_or(EntityType::Zombie);
        let pos = DVec3::new(msg.x, msg.y, msg.z);
        let snap = EntitySnapshot {
            timestamp: now,
            pos,
            yaw: msg.yaw,
            pitch: msg.pitch,
            head_yaw: msg.head_yaw,
        };
        let mut snapshots = VecDeque::with_capacity(16);
        snapshots.push_back(snap);
        Self {
            net_id: msg.net_id,
            entity_type,
            pos,
            target_pos: pos,
            yaw: msg.yaw,
            pitch: msg.pitch,
            head_yaw: msg.head_yaw,
            target_yaw: msg.yaw,
            walk_time: 0.0,
            hurt_timer: 0.0,
            attack_swing_timer: 0.0,
            death_timer: 0.0,
            health: msg.health,
            max_health: msg.max_health,
            item_id: 0,
            item_count: 0,
            snapshots,
        }
    }

    /// Creates a new `ClientEntity` representing a dropped item entity initialized from an `S2cSpawnItem` packet.
    #[must_use]
    pub fn from_spawn_item(msg: S2cSpawnItem, now: f64) -> Self {
        let pos = DVec3::new(msg.x, msg.y, msg.z);
        let snap = EntitySnapshot {
            timestamp: now,
            pos,
            yaw: 0.0,
            pitch: 0.0,
            head_yaw: 0.0,
        };
        let mut snapshots = VecDeque::with_capacity(16);
        snapshots.push_back(snap);
        Self {
            net_id: msg.net_id,
            entity_type: EntityType::Item,
            pos,
            target_pos: pos,
            yaw: 0.0,
            pitch: 0.0,
            head_yaw: 0.0,
            target_yaw: 0.0,
            walk_time: 0.0,
            hurt_timer: 0.0,
            attack_swing_timer: 0.0,
            death_timer: 0.0,
            health: 5.0,
            max_health: 5.0,
            item_id: msg.item_id,
            item_count: msg.count,
            snapshots,
        }
    }

    /// Creates a new `ClientEntity` representing a flying projectile arrow initialized from an `S2cSpawnArrow` packet.
    #[must_use]
    pub fn from_spawn_arrow(msg: S2cSpawnArrow, now: f64) -> Self {
        let pos = DVec3::new(msg.x, msg.y, msg.z);
        let snap = EntitySnapshot {
            timestamp: now,
            pos,
            yaw: msg.yaw,
            pitch: msg.pitch,
            head_yaw: msg.yaw,
        };
        let mut snapshots = VecDeque::with_capacity(16);
        snapshots.push_back(snap);
        Self {
            net_id: msg.net_id,
            entity_type: EntityType::Arrow,
            pos,
            target_pos: pos,
            yaw: msg.yaw,
            pitch: msg.pitch,
            head_yaw: msg.yaw,
            target_yaw: msg.yaw,
            walk_time: 0.0,
            hurt_timer: 0.0,
            attack_swing_timer: 0.0,
            death_timer: 0.0,
            health: 1.0,
            max_health: 1.0,
            item_id: 0,
            item_count: 0,
            snapshots,
        }
    }
}

/// Storage container for remote entities tracked on the client.
#[derive(Debug, Default)]
pub struct ClientEntityStore {
    entities: HashMap<u32, ClientEntity>,
    /// Accumulated client timeline in seconds.
    pub current_time: f64,
}

impl ClientEntityStore {
    /// Creates a new empty entity store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            current_time: 0.0,
        }
    }

    /// Number of active entities tracked.
    #[must_use]
    pub fn count(&self) -> usize {
        self.entities.len()
    }

    /// Looks up a client entity by network ID.
    #[must_use]
    pub fn get(&self, net_id: u32) -> Option<&ClientEntity> {
        self.entities.get(&net_id)
    }

    /// Returns an iterator over all tracked entities.
    pub fn iter(&self) -> impl Iterator<Item = &ClientEntity> {
        self.entities.values()
    }

    /// Handles an incoming `S2cSpawnEntity` packet.
    pub fn on_spawn(&mut self, msg: S2cSpawnEntity) {
        self.entities
            .insert(msg.net_id, ClientEntity::from_spawn(msg, self.current_time));
    }

    /// Handles an incoming `S2cSpawnItem` packet.
    pub fn on_spawn_item(&mut self, msg: S2cSpawnItem) {
        self.entities.insert(
            msg.net_id,
            ClientEntity::from_spawn_item(msg, self.current_time),
        );
    }

    /// Handles an incoming `S2cSpawnArrow` packet.
    pub fn on_spawn_arrow(&mut self, msg: S2cSpawnArrow) {
        self.entities.insert(
            msg.net_id,
            ClientEntity::from_spawn_arrow(msg, self.current_time),
        );
    }

    /// Handles an incoming `S2cDespawnEntity` packet.
    pub fn on_despawn(&mut self, net_ids: &[u32]) {
        for &id in net_ids {
            if let Some(entity) = self.entities.get(&id)
                && entity.death_timer <= 0.0
            {
                self.entities.remove(&id);
            }
        }
    }

    /// Handles an incoming `S2cEntityMove` packet.
    pub fn on_move(&mut self, msg: S2cEntityMove) {
        if let Some(entity) = self.entities.get_mut(&msg.net_id) {
            let snap = EntitySnapshot {
                timestamp: self.current_time,
                pos: DVec3::new(msg.x, msg.y, msg.z),
                yaw: msg.yaw,
                pitch: msg.pitch,
                head_yaw: msg.head_yaw,
            };
            entity.target_pos = snap.pos;
            entity.target_yaw = snap.yaw;
            entity.pitch = snap.pitch;
            entity.head_yaw = snap.head_yaw;
            entity.snapshots.push_back(snap);
            while entity.snapshots.len() > 16 {
                entity.snapshots.pop_front();
            }
        }
    }

    /// Handles an incoming `S2cEntityStatus` packet.
    pub fn on_status(&mut self, net_id: u32, status: u8) {
        if let Some(entity) = self.entities.get_mut(&net_id) {
            match status {
                2 => {
                    // Hurt red flash
                    entity.hurt_timer = 0.5;
                }
                3 => {
                    // Death: red flash and roll sideways over 1 second
                    entity.hurt_timer = 1.0;
                    entity.death_timer = 1.0;
                }
                4 => {
                    // Attack swing animation
                    entity.attack_swing_timer = 0.3;
                }
                _ => {}
            }
        }
    }

    /// Clears all tracked entities.
    pub fn clear(&mut self) {
        self.entities.clear();
    }

    /// Advances frame-time interpolation and animations.
    pub fn update(&mut self, dt: f32) {
        self.current_time += f64::from(dt);
        // Standard 100 ms interpolation delay (2 server ticks at 20 TPS)
        let render_time = self.current_time - 0.100;

        self.entities.retain(|_, entity| {
            // Decay hurt flash and attack swing timers
            entity.hurt_timer = (entity.hurt_timer - dt).max(0.0);
            entity.attack_swing_timer = (entity.attack_swing_timer - dt).max(0.0);
            if entity.death_timer > 0.0 {
                entity.death_timer = (entity.death_timer - dt).max(0.0);
                if entity.death_timer <= 0.0 {
                    return false;
                }
            }
            true
        });

        for entity in self.entities.values_mut() {
            let prev_pos = entity.pos;

            // Interpolate or extrapolate from snapshot history
            if entity.snapshots.len() >= 2 {
                let first_ts = entity.snapshots.front().unwrap().timestamp;
                let last_ts = entity.snapshots.back().unwrap().timestamp;

                if render_time <= first_ts {
                    // Underflow: clamp to oldest snapshot
                    let s = entity.snapshots.front().unwrap();
                    entity.pos = s.pos;
                    entity.yaw = s.yaw;
                    entity.pitch = s.pitch;
                    entity.head_yaw = s.head_yaw;
                } else if render_time >= last_ts {
                    // Starved / packet jitter: extrapolate up to 1 tick (50 ms)
                    let n = entity.snapshots.len();
                    let s_prev = &entity.snapshots[n - 2];
                    let s_last = &entity.snapshots[n - 1];
                    let delta_t = s_last.timestamp - s_prev.timestamp;
                    if delta_t > 0.001 {
                        let vel = (s_last.pos - s_prev.pos) / delta_t;
                        let extra = (render_time - last_ts).min(0.05);
                        entity.pos = s_last.pos + vel * extra;
                    } else {
                        entity.pos = s_last.pos;
                    }
                    entity.yaw = s_last.yaw;
                    entity.pitch = s_last.pitch;
                    entity.head_yaw = s_last.head_yaw;
                } else {
                    // Locate bounding snapshots S0 and S1
                    let mut s0 = &entity.snapshots[0];
                    let mut s1 = &entity.snapshots[1];
                    for i in 0..(entity.snapshots.len() - 1) {
                        if entity.snapshots[i].timestamp <= render_time
                            && entity.snapshots[i + 1].timestamp >= render_time
                        {
                            s0 = &entity.snapshots[i];
                            s1 = &entity.snapshots[i + 1];
                            break;
                        }
                    }

                    let dur = s1.timestamp - s0.timestamp;
                    #[allow(clippy::cast_possible_truncation)]
                    let alpha = if dur > 1e-5 {
                        ((render_time - s0.timestamp) / dur).clamp(0.0, 1.0) as f32
                    } else {
                        1.0
                    };

                    entity.pos = s0.pos + (s1.pos - s0.pos) * f64::from(alpha);
                    entity.yaw = lerp_angle(s0.yaw, s1.yaw, alpha);
                    entity.pitch = lerp_angle(s0.pitch, s1.pitch, alpha);
                    entity.head_yaw = lerp_angle(s0.head_yaw, s1.head_yaw, alpha);
                }
            } else if let Some(snap) = entity.snapshots.back() {
                entity.pos = snap.pos;
                entity.yaw = snap.yaw;
                entity.pitch = snap.pitch;
                entity.head_yaw = snap.head_yaw;
            }

            // Prune snapshots older than 500 ms before render_time (keep at least 2)
            let cutoff = render_time - 0.500;
            while entity.snapshots.len() > 2 && entity.snapshots.front().unwrap().timestamp < cutoff
            {
                entity.snapshots.pop_front();
            }

            // Accumulate walk animation when entity moves horizontally, or spin items
            if entity.entity_type == EntityType::Item {
                entity.walk_time += dt;
                entity.yaw = (entity.yaw + 90.0 * dt) % 360.0;
                entity.target_yaw = entity.yaw;
            } else {
                let delta = entity.pos - prev_pos;
                let horiz_dist_sq = delta.x * delta.x + delta.z * delta.z;
                if horiz_dist_sq > 1e-6 {
                    entity.walk_time += dt;
                } else {
                    entity.walk_time = (entity.walk_time - dt * 2.0).max(0.0);
                }
            }
        }
    }

    /// Performs slab-method ray intersection against entity bounding boxes.
    ///
    /// Returns `Some((hit_net_id, hit_distance))` if an entity was struck within `max_dist`.
    #[must_use]
    pub fn raycast(&self, ray_origin: DVec3, ray_dir: Vec3, max_dist: f32) -> Option<(u32, f32)> {
        let mut closest_hit: Option<(u32, f32)> = None;

        for entity in self.entities.values() {
            if entity.death_timer > 0.0
                || entity.entity_type == EntityType::Item
                || entity.entity_type == EntityType::Arrow
            {
                continue;
            }
            let aabb = entity.entity_type.default_aabb();
            if let Some(dist) = aabb.intersects_ray(entity.pos, ray_origin, ray_dir, max_dist) {
                if let Some((_, closest_d)) = closest_hit {
                    if dist < closest_d {
                        closest_hit = Some((entity.net_id, dist));
                    }
                } else {
                    closest_hit = Some((entity.net_id, dist));
                }
            }
        }

        closest_hit
    }

    /// Generates hierarchical cuboid geometry for all active entities.
    pub fn build_mesh(
        &self,
        get_light: impl Fn(DVec3) -> (u8, u8),
        out_vertices: &mut Vec<EntityVertexGpu>,
    ) {
        for entity in self.entities.values() {
            let (sky, block) = get_light(entity.pos);
            let hurt_tint = if entity.death_timer > 0.0 {
                (entity.death_timer / 1.0).clamp(0.0, 1.0)
            } else {
                (entity.hurt_timer / 0.5).clamp(0.0, 1.0)
            };
            let body_roll = if entity.death_timer > 0.0 {
                let progress = (1.0 - entity.death_timer / 1.0).clamp(0.0, 1.0);
                progress * 90.0
            } else {
                0.0
            };
            let layer = match entity.entity_type {
                EntityType::Zombie | EntityType::Player | EntityType::Item | EntityType::Arrow => 0,
                EntityType::Pig => 1,
                EntityType::Cow => 2,
            };

            match entity.entity_type {
                EntityType::Zombie | EntityType::Player => {
                    build_humanoid_mesh(
                        entity,
                        sky,
                        block,
                        layer,
                        hurt_tint,
                        body_roll,
                        out_vertices,
                    );
                }
                EntityType::Pig => {
                    build_pig_mesh(
                        entity,
                        sky,
                        block,
                        layer,
                        hurt_tint,
                        body_roll,
                        out_vertices,
                    );
                }
                EntityType::Cow => {
                    build_cow_mesh(
                        entity,
                        sky,
                        block,
                        layer,
                        hurt_tint,
                        body_roll,
                        out_vertices,
                    );
                }
                EntityType::Item => {
                    build_item_mesh(entity, sky, block, hurt_tint, out_vertices);
                }
                EntityType::Arrow => {
                    build_arrow_mesh(entity, sky, block, hurt_tint, out_vertices);
                }
            }
        }
    }
}

fn build_arrow_mesh(
    entity: &ClientEntity,
    sky: u8,
    block: u8,
    hurt_tint: f32,
    out: &mut Vec<EntityVertexGpu>,
) {
    let body_quat = Quat::from_rotation_y((-entity.yaw).to_radians());
    emit_cuboid(
        entity.pos,
        body_quat,
        0.0,
        Vec3::ZERO,
        Vec3::new(entity.pitch, 0.0, 0.0),
        Vec3::new(-0.025, -0.025, -0.3),
        Vec3::new(0.025, 0.025, 0.3),
        (0.0, 0.0, 1.0, 1.0, 8.0),
        sky,
        block,
        0,
        hurt_tint,
        out,
    );
}

fn build_item_mesh(
    entity: &ClientEntity,
    sky: u8,
    block: u8,
    hurt_tint: f32,
    out: &mut Vec<EntityVertexGpu>,
) {
    let bob_y = (entity.walk_time * 3.0).sin() * 0.08 + 0.12;
    let item_pos = entity.pos + DVec3::new(0.0, f64::from(bob_y), 0.0);
    let body_quat = Quat::from_rotation_y((-entity.yaw).to_radians());

    // Small 4x4x4 pixel (0.25m) floating item box centered horizontally
    emit_cuboid(
        item_pos,
        body_quat,
        0.0,
        Vec3::new(0.0, 2.0 * S, 0.0),
        Vec3::ZERO,
        Vec3::new(-2.0 * S, 0.0, -2.0 * S),
        Vec3::new(2.0 * S, 4.0 * S, 2.0 * S),
        (0.0, 0.0, 4.0, 4.0, 4.0),
        sky,
        block,
        0,
        hurt_tint,
        out,
    );
}

// ----------------------------------------------------------------------------
// Model geometry builders
// ----------------------------------------------------------------------------

const S: f32 = 1.0 / 16.0;

#[allow(clippy::too_many_lines)]
fn build_humanoid_mesh(
    entity: &ClientEntity,
    sky: u8,
    block: u8,
    layer: u32,
    hurt_tint: f32,
    body_roll: f32,
    out: &mut Vec<EntityVertexGpu>,
) {
    let body_quat = Quat::from_rotation_y((-entity.yaw).to_radians())
        * Quat::from_rotation_z(body_roll.to_radians());
    let death_lift = (body_roll / 90.0) * 0.25;
    let walk_anim = (entity.walk_time * 6.0).sin();

    // 1. Head: 8x8x8 centered at y=24..32, pivot at (0, 24, 0)
    let head_rot_yaw = entity.head_yaw - entity.yaw;
    let head_pitch = entity.pitch;
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(0.0, 24.0 * S, 0.0),
        Vec3::new(head_pitch, head_rot_yaw, 0.0),
        Vec3::new(-4.0 * S, 24.0 * S, -4.0 * S),
        Vec3::new(4.0 * S, 32.0 * S, 4.0 * S),
        (0.0, 0.0, 8.0, 8.0, 8.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );

    // 2. Body: 8x12x4 at y=12..24, pivot at (0, 24, 0)
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(0.0, 24.0 * S, 0.0),
        Vec3::ZERO,
        Vec3::new(-4.0 * S, 12.0 * S, -2.0 * S),
        Vec3::new(4.0 * S, 24.0 * S, 2.0 * S),
        (16.0, 16.0, 8.0, 12.0, 4.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );

    let swing_offset = if entity.attack_swing_timer > 0.0 {
        let progress = (1.0 - entity.attack_swing_timer / 0.3).clamp(0.0, 1.0);
        (progress * std::f32::consts::PI).sin() * 40.0
    } else {
        0.0
    };

    // 3. Right Arm: 4x12x4, pivot at (-5, 22, 0), outstretched forward (-90 deg pitch)
    let r_arm_pitch = -90.0 + walk_anim * 8.0 - swing_offset;
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(-5.0 * S, 22.0 * S, 0.0),
        Vec3::new(r_arm_pitch, 0.0, 0.0),
        Vec3::new(-7.0 * S, 10.0 * S, -2.0 * S),
        Vec3::new(-3.0 * S, 22.0 * S, 2.0 * S),
        (40.0, 16.0, 4.0, 12.0, 4.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );

    // 4. Left Arm: 4x12x4, pivot at (5, 22, 0), outstretched forward (-90 deg pitch)
    let l_arm_pitch = -90.0 - walk_anim * 8.0 - swing_offset;
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(5.0 * S, 22.0 * S, 0.0),
        Vec3::new(l_arm_pitch, 0.0, 0.0),
        Vec3::new(3.0 * S, 10.0 * S, -2.0 * S),
        Vec3::new(7.0 * S, 22.0 * S, 2.0 * S),
        (40.0, 16.0, 4.0, 12.0, 4.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );

    // 5. Right Leg: 4x12x4, pivot at (-2, 12, 0)
    let r_leg_pitch = walk_anim * 28.0;
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(-1.9 * S, 12.0 * S, 0.0),
        Vec3::new(r_leg_pitch, 0.0, 0.0),
        Vec3::new(-3.9 * S, 0.0, -2.0 * S),
        Vec3::new(-0.1 * S, 12.0 * S, 2.0 * S),
        (0.0, 16.0, 4.0, 12.0, 4.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );

    // 6. Left Leg: 4x12x4, pivot at (2, 12, 0)
    let l_leg_pitch = -walk_anim * 28.0;
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(1.9 * S, 12.0 * S, 0.0),
        Vec3::new(l_leg_pitch, 0.0, 0.0),
        Vec3::new(0.1 * S, 0.0, -2.0 * S),
        Vec3::new(3.9 * S, 12.0 * S, 2.0 * S),
        (0.0, 16.0, 4.0, 12.0, 4.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );
}

#[allow(clippy::too_many_lines)]
fn build_pig_mesh(
    entity: &ClientEntity,
    sky: u8,
    block: u8,
    layer: u32,
    hurt_tint: f32,
    body_roll: f32,
    out: &mut Vec<EntityVertexGpu>,
) {
    let body_quat = Quat::from_rotation_y((-entity.yaw).to_radians())
        * Quat::from_rotation_z(body_roll.to_radians());
    let death_lift = (body_roll / 90.0) * 0.25;
    let walk_anim = (entity.walk_time * 6.0).sin();

    // 1. Horizontal Body: 10x8x16 at y=6..14, z=-8..8
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(0.0, 10.0 * S, 0.0),
        Vec3::ZERO,
        Vec3::new(-5.0 * S, 6.0 * S, -8.0 * S),
        Vec3::new(5.0 * S, 14.0 * S, 8.0 * S),
        (28.0, 8.0, 10.0, 16.0, 8.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );

    // 2. Head: 8x8x8 at y=8..16, z=-14..-6, pivot at (0, 12, -6)
    let head_rot_yaw = entity.head_yaw - entity.yaw;
    let head_pitch = entity.pitch;
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(0.0, 12.0 * S, -6.0 * S),
        Vec3::new(head_pitch, head_rot_yaw, 0.0),
        Vec3::new(-4.0 * S, 8.0 * S, -14.0 * S),
        Vec3::new(4.0 * S, 16.0 * S, -6.0 * S),
        (0.0, 0.0, 8.0, 8.0, 8.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );

    // 3. Snout: 4x3x1 at y=9..12, z=-15..-14
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(0.0, 12.0 * S, -6.0 * S),
        Vec3::new(head_pitch, head_rot_yaw, 0.0),
        Vec3::new(-2.0 * S, 9.0 * S, -15.0 * S),
        Vec3::new(2.0 * S, 12.0 * S, -14.0 * S),
        (16.0, 16.0, 4.0, 3.0, 1.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );

    // 4 Legs: 4x6x4 at y=0..6
    let leg_pitch = walk_anim * 25.0;
    // Front Right
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(-3.0 * S, 6.0 * S, -5.0 * S),
        Vec3::new(leg_pitch, 0.0, 0.0),
        Vec3::new(-5.0 * S, 0.0, -7.0 * S),
        Vec3::new(-S, 6.0 * S, -3.0 * S),
        (0.0, 16.0, 4.0, 6.0, 4.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );
    // Front Left
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(3.0 * S, 6.0 * S, -5.0 * S),
        Vec3::new(-leg_pitch, 0.0, 0.0),
        Vec3::new(1.0 * S, 0.0, -7.0 * S),
        Vec3::new(5.0 * S, 6.0 * S, -3.0 * S),
        (0.0, 16.0, 4.0, 6.0, 4.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );
    // Back Right
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(-3.0 * S, 6.0 * S, 5.0 * S),
        Vec3::new(-leg_pitch, 0.0, 0.0),
        Vec3::new(-5.0 * S, 0.0, 3.0 * S),
        Vec3::new(-S, 6.0 * S, 7.0 * S),
        (0.0, 16.0, 4.0, 6.0, 4.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );
    // Back Left
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(3.0 * S, 6.0 * S, 5.0 * S),
        Vec3::new(leg_pitch, 0.0, 0.0),
        Vec3::new(1.0 * S, 0.0, 3.0 * S),
        Vec3::new(5.0 * S, 6.0 * S, 7.0 * S),
        (0.0, 16.0, 4.0, 6.0, 4.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );
}

#[allow(clippy::too_many_lines)]
fn build_cow_mesh(
    entity: &ClientEntity,
    sky: u8,
    block: u8,
    layer: u32,
    hurt_tint: f32,
    body_roll: f32,
    out: &mut Vec<EntityVertexGpu>,
) {
    let body_quat = Quat::from_rotation_y((-entity.yaw).to_radians())
        * Quat::from_rotation_z(body_roll.to_radians());
    let death_lift = (body_roll / 90.0) * 0.25;
    let walk_anim = (entity.walk_time * 6.0).sin();

    // 1. Horizontal Body: 12x10x18 at y=12..22, z=-9..9
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(0.0, 17.0 * S, 0.0),
        Vec3::ZERO,
        Vec3::new(-6.0 * S, 12.0 * S, -9.0 * S),
        Vec3::new(6.0 * S, 22.0 * S, 9.0 * S),
        (18.0, 4.0, 12.0, 18.0, 10.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );

    // 2. Head: 8x8x6 at y=16..24, z=-14..-8, pivot at (0, 20, -8)
    let head_rot_yaw = entity.head_yaw - entity.yaw;
    let head_pitch = entity.pitch;
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(0.0, 20.0 * S, -8.0 * S),
        Vec3::new(head_pitch, head_rot_yaw, 0.0),
        Vec3::new(-4.0 * S, 16.0 * S, -14.0 * S),
        Vec3::new(4.0 * S, 24.0 * S, -8.0 * S),
        (0.0, 0.0, 8.0, 8.0, 6.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );

    // 3. Horns: 1x3x1 on each side of head
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(0.0, 20.0 * S, -8.0 * S),
        Vec3::new(head_pitch, head_rot_yaw, 0.0),
        Vec3::new(-5.0 * S, 23.0 * S, -12.0 * S),
        Vec3::new(-4.0 * S, 26.0 * S, -11.0 * S),
        (22.0, 0.0, 1.0, 3.0, 1.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(0.0, 20.0 * S, -8.0 * S),
        Vec3::new(head_pitch, head_rot_yaw, 0.0),
        Vec3::new(4.0 * S, 23.0 * S, -12.0 * S),
        Vec3::new(5.0 * S, 26.0 * S, -11.0 * S),
        (22.0, 0.0, 1.0, 3.0, 1.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );

    // 4 Legs: 4x12x4 at y=0..12
    let leg_pitch = walk_anim * 25.0;
    // Front Right
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(-4.0 * S, 12.0 * S, -6.0 * S),
        Vec3::new(leg_pitch, 0.0, 0.0),
        Vec3::new(-6.0 * S, 0.0, -8.0 * S),
        Vec3::new(-2.0 * S, 12.0 * S, -4.0 * S),
        (0.0, 16.0, 4.0, 12.0, 4.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );
    // Front Left
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(4.0 * S, 12.0 * S, -6.0 * S),
        Vec3::new(-leg_pitch, 0.0, 0.0),
        Vec3::new(2.0 * S, 0.0, -8.0 * S),
        Vec3::new(6.0 * S, 12.0 * S, -4.0 * S),
        (0.0, 16.0, 4.0, 12.0, 4.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );
    // Back Right
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(-4.0 * S, 12.0 * S, 6.0 * S),
        Vec3::new(-leg_pitch, 0.0, 0.0),
        Vec3::new(-6.0 * S, 0.0, 4.0 * S),
        Vec3::new(-2.0 * S, 12.0 * S, 8.0 * S),
        (0.0, 16.0, 4.0, 12.0, 4.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );
    // Back Left
    emit_cuboid(
        entity.pos,
        body_quat,
        death_lift,
        Vec3::new(4.0 * S, 12.0 * S, 6.0 * S),
        Vec3::new(leg_pitch, 0.0, 0.0),
        Vec3::new(2.0 * S, 0.0, 4.0 * S),
        Vec3::new(6.0 * S, 12.0 * S, 8.0 * S),
        (0.0, 16.0, 4.0, 12.0, 4.0),
        sky,
        block,
        layer,
        hurt_tint,
        out,
    );
}

// ----------------------------------------------------------------------------
// Low-level cuboid face emitter
// ----------------------------------------------------------------------------

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn emit_cuboid(
    entity_pos: DVec3,
    body_quat: Quat,
    death_lift: f32,
    pivot: Vec3,
    rotation_deg: Vec3,
    min_pt: Vec3,
    max_pt: Vec3,
    uv_spec: (f32, f32, f32, f32, f32), // (u, v, w, h, d)
    sky: u8,
    block: u8,
    layer: u32,
    hurt_tint: f32,
    out: &mut Vec<EntityVertexGpu>,
) {
    let (u0, v0, w, h, d) = uv_spec;
    let inv = 1.0 / 64.0;

    // UV regions on 64x64 skin
    let uv_right = [
        (u0) * inv,
        (v0 + d) * inv,
        (u0 + d) * inv,
        (v0 + d + h) * inv,
    ];
    let uv_front = [
        (u0 + d) * inv,
        (v0 + d) * inv,
        (u0 + d + w) * inv,
        (v0 + d + h) * inv,
    ];
    let uv_left = [
        (u0 + d + w) * inv,
        (v0 + d) * inv,
        (u0 + 2.0 * d + w) * inv,
        (v0 + d + h) * inv,
    ];
    let uv_back = [
        (u0 + 2.0 * d + w) * inv,
        (v0 + d) * inv,
        (u0 + 2.0 * d + 2.0 * w) * inv,
        (v0 + d + h) * inv,
    ];
    let uv_top = [
        (u0 + d) * inv,
        (v0) * inv,
        (u0 + d + w) * inv,
        (v0 + d) * inv,
    ];
    let uv_bottom = [
        (u0 + d + w) * inv,
        (v0) * inv,
        (u0 + d + 2.0 * w) * inv,
        (v0 + d) * inv,
    ];

    let limb_quat = Quat::from_euler(
        glam::EulerRot::YXZ,
        rotation_deg.y.to_radians(),
        rotation_deg.x.to_radians(),
        rotation_deg.z.to_radians(),
    );

    let transform_pt = |p: Vec3| -> [f32; 3] {
        let rotated_limb = pivot + limb_quat * (p - pivot);
        let rotated_body = body_quat * rotated_limb;
        let world = entity_pos.as_vec3() + rotated_body + Vec3::new(0.0, death_lift, 0.0);
        [world.x, world.y, world.z]
    };

    // 8 box corners in local space
    let p000 = min_pt;
    let p100 = Vec3::new(max_pt.x, min_pt.y, min_pt.z);
    let p010 = Vec3::new(min_pt.x, max_pt.y, min_pt.z);
    let p110 = Vec3::new(max_pt.x, max_pt.y, min_pt.z);
    let p001 = Vec3::new(min_pt.x, min_pt.y, max_pt.z);
    let p101 = Vec3::new(max_pt.x, min_pt.y, max_pt.z);
    let p011 = Vec3::new(min_pt.x, max_pt.y, max_pt.z);
    let p111 = max_pt;

    // Helper to emit a quad (2 triangles = 6 vertices)
    let mut emit_quad = |pts: [Vec3; 4], uvs: [Vec2; 4], norm_idx: u32| {
        let light = u32::from(sky & 0xF) | (u32::from(block & 0xF) << 4) | ((norm_idx & 0x7) << 8);
        let indices = [0, 1, 2, 2, 1, 3];
        for &idx in &indices {
            out.push(EntityVertexGpu {
                pos: transform_pt(pts[idx]),
                light,
                uv: [uvs[idx].x, uvs[idx].y],
                layer,
                hurt_tint,
            });
        }
    };

    // 1. Top face (+Y, norm 1)
    emit_quad(
        [p011, p111, p010, p110],
        [
            Vec2::new(uv_top[0], uv_top[3]),
            Vec2::new(uv_top[2], uv_top[3]),
            Vec2::new(uv_top[0], uv_top[1]),
            Vec2::new(uv_top[2], uv_top[1]),
        ],
        1,
    );

    // 2. Bottom face (-Y, norm 0)
    emit_quad(
        [p000, p100, p001, p101],
        [
            Vec2::new(uv_bottom[0], uv_bottom[1]),
            Vec2::new(uv_bottom[2], uv_bottom[1]),
            Vec2::new(uv_bottom[0], uv_bottom[3]),
            Vec2::new(uv_bottom[2], uv_bottom[3]),
        ],
        0,
    );

    // 3. Front face (+Z, norm 3)
    emit_quad(
        [p001, p101, p011, p111],
        [
            Vec2::new(uv_front[0], uv_front[3]),
            Vec2::new(uv_front[2], uv_front[3]),
            Vec2::new(uv_front[0], uv_front[1]),
            Vec2::new(uv_front[2], uv_front[1]),
        ],
        3,
    );

    // 4. Back face (-Z, norm 2)
    emit_quad(
        [p100, p000, p110, p010],
        [
            Vec2::new(uv_back[0], uv_back[3]),
            Vec2::new(uv_back[2], uv_back[3]),
            Vec2::new(uv_back[0], uv_back[1]),
            Vec2::new(uv_back[2], uv_back[1]),
        ],
        2,
    );

    // 5. Left face (+X, norm 5)
    emit_quad(
        [p101, p100, p111, p110],
        [
            Vec2::new(uv_left[0], uv_left[3]),
            Vec2::new(uv_left[2], uv_left[3]),
            Vec2::new(uv_left[0], uv_left[1]),
            Vec2::new(uv_left[2], uv_left[1]),
        ],
        5,
    );

    // 6. Right face (-X, norm 4)
    emit_quad(
        [p000, p001, p010, p011],
        [
            Vec2::new(uv_right[0], uv_right[3]),
            Vec2::new(uv_right[2], uv_right[3]),
            Vec2::new(uv_right[0], uv_right[1]),
            Vec2::new(uv_right[2], uv_right[1]),
        ],
        4,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lerp_angle_unwrapping() {
        // Simple interpolation
        assert!((lerp_angle(10.0, 50.0, 0.5) - 30.0).abs() < 1e-4);

        // Crossing 0 / 360 clockwise (350 -> 10 = +20 diff)
        assert!((lerp_angle(350.0, 10.0, 0.5) - 360.0).abs() < 1e-4);

        // Crossing 0 / 360 counter-clockwise (10 -> 350 = -20 diff)
        assert!((lerp_angle(10.0, 350.0, 0.5) - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_entity_snapshot_interpolation() {
        let mut store = ClientEntityStore::new();

        let spawn_msg = S2cSpawnEntity {
            net_id: 1,
            entity_type: 0,
            x: 0.0,
            y: 64.0,
            z: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            head_yaw: 0.0,
            health: 20.0,
            max_health: 20.0,
        };
        store.on_spawn(spawn_msg);

        // Move 100 ms later to (10.0, 64.0, 0.0)
        store.current_time = 0.100;
        store.on_move(S2cEntityMove {
            net_id: 1,
            x: 10.0,
            y: 64.0,
            z: 0.0,
            yaw: 90.0,
            pitch: 0.0,
            head_yaw: 90.0,
            on_ground: true,
        });

        // Advance to 150 ms (render_time = 150 - 100 = 50 ms, halfway between 0 and 100 ms)
        store.update(0.050);

        let entity = store.get(1).unwrap();
        assert!((entity.pos.x - 5.0).abs() < 0.1);
        assert!((entity.yaw - 45.0).abs() < 0.5);
    }

    #[test]
    fn test_entity_extrapolation_bounded() {
        let mut store = ClientEntityStore::new();

        store.on_spawn(S2cSpawnEntity {
            net_id: 2,
            entity_type: 0,
            x: 0.0,
            y: 64.0,
            z: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            head_yaw: 0.0,
            health: 20.0,
            max_health: 20.0,
        });

        // 50 ms later
        store.current_time = 0.050;
        store.on_move(S2cEntityMove {
            net_id: 2,
            x: 5.0,
            y: 64.0,
            z: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            head_yaw: 0.0,
            on_ground: true,
        });

        // Advance to 300 ms (render_time = 200 ms, starved by 150 ms)
        store.update(0.250);

        let entity = store.get(2).unwrap();
        // Speed = 5.0 / 0.050 = 100.0 blocks/sec.
        // Bounded extra = 50 ms -> max extra pos = 5.0 + 100.0 * 0.05 = 10.0
        assert!(entity.pos.x <= 10.01);
    }

    #[test]
    fn test_entity_status_attack_and_death_animations() {
        let mut store = ClientEntityStore::new();

        store.on_spawn(S2cSpawnEntity {
            net_id: 10,
            entity_type: 0, // Zombie
            x: 0.0,
            y: 64.0,
            z: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            head_yaw: 0.0,
            health: 20.0,
            max_health: 20.0,
        });

        // 1. Attack swing animation
        store.on_status(10, 4);
        let entity = store.get(10).unwrap();
        assert!((entity.attack_swing_timer - 0.3).abs() < 1e-4);

        // Raycast can hit living entity
        let hit = store.raycast(DVec3::new(0.0, 64.0, -3.0), Vec3::new(0.0, 0.0, 1.0), 5.0);
        assert_eq!(hit.map(|(id, _)| id), Some(10));

        // 2. Death status
        store.on_status(10, 3);
        let entity = store.get(10).unwrap();
        assert!((entity.death_timer - 1.0).abs() < 1e-4);
        assert!((entity.hurt_timer - 1.0).abs() < 1e-4);

        // Dying entity is excluded from combat raycasting
        let hit_after_death =
            store.raycast(DVec3::new(0.0, 64.0, -3.0), Vec3::new(0.0, 0.0, 1.0), 5.0);
        assert!(hit_after_death.is_none());

        // 3. Despawn packet arrives while dying - entity retained until animation finishes
        store.on_despawn(&[10]);
        assert!(store.get(10).is_some());

        // 4. Update decays timers
        store.update(0.5);
        let entity = store.get(10).unwrap();
        assert!((entity.death_timer - 0.5).abs() < 1e-4);
        assert!(entity.attack_swing_timer.abs() < 1e-4);

        // 5. Update past 1.0s total -> entity cleanly removed from store
        store.update(0.6);
        assert!(store.get(10).is_none());
    }

    #[test]
    fn test_item_entity_spawn_bob_and_mesh() {
        let mut store = ClientEntityStore::new();

        store.on_spawn_item(S2cSpawnItem {
            net_id: 42,
            item_id: 1, // stone
            count: 64,
            x: 10.0,
            y: 65.0,
            z: 20.0,
            vel_x: 0.0,
            vel_y: 0.0,
            vel_z: 0.0,
        });

        assert_eq!(store.count(), 1);
        let item = store.get(42).unwrap();
        assert_eq!(item.entity_type, EntityType::Item);
        assert_eq!(item.item_id, 1);
        assert_eq!(item.item_count, 64);

        // Raycasting should NOT hit item entities
        let hit = store.raycast(DVec3::new(10.0, 65.0, 15.0), Vec3::new(0.0, 0.0, 1.0), 10.0);
        assert!(hit.is_none());

        // Update 0.5s: yaw advances by 45 degrees (90 deg/s)
        store.update(0.5);
        let item_updated = store.get(42).unwrap();
        assert!((item_updated.yaw - 45.0).abs() < 1e-3);
        assert!((item_updated.walk_time - 0.5).abs() < 1e-4);

        // Build mesh: 1 item cuboid emits 36 vertices (6 faces * 6 vertices)
        let mut vertices = Vec::new();
        store.build_mesh(|_| (15, 0), &mut vertices);
        assert_eq!(vertices.len(), 36);
    }

    #[test]
    fn test_arrow_entity_spawn_and_mesh() {
        let mut store = ClientEntityStore::new();

        store.on_spawn_arrow(S2cSpawnArrow {
            net_id: 99,
            x: 5.0,
            y: 64.0,
            z: 5.0,
            vel_x: 0.0,
            vel_y: 0.0,
            vel_z: 1.0,
            yaw: 180.0,
            pitch: -10.0,
        });

        assert_eq!(store.count(), 1);
        let arrow = store.get(99).unwrap();
        assert_eq!(arrow.entity_type, EntityType::Arrow);
        assert!((arrow.yaw - 180.0).abs() < 1e-4);
        assert!((arrow.pitch - (-10.0)).abs() < 1e-4);

        // Raycasting should NOT hit arrow entities
        let hit = store.raycast(DVec3::new(5.0, 64.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 10.0);
        assert!(hit.is_none());

        // Build mesh: 1 arrow cuboid emits 36 vertices
        let mut vertices = Vec::new();
        store.build_mesh(|_| (15, 0), &mut vertices);
        assert_eq!(vertices.len(), 36);
    }
}
