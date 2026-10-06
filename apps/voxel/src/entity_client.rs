//! Client-side entity tracking, interpolation, hierarchical cuboid meshing, and combat raycasting.

use glam::{DVec3, Quat, Vec2, Vec3};
use hashbrown::HashMap;
use vx_protocol::messages::{S2cEntityMove, S2cSpawnEntity};
use vx_sim::EntityType;

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
    /// Current health points.
    pub health: f32,
    /// Maximum health points.
    pub max_health: f32,
}

impl ClientEntity {
    /// Creates a new `ClientEntity` initialized from an `S2cSpawnEntity` packet.
    #[must_use]
    pub fn from_spawn(msg: S2cSpawnEntity) -> Self {
        let entity_type = EntityType::from_u8(msg.entity_type).unwrap_or(EntityType::Zombie);
        let pos = DVec3::new(msg.x, msg.y, msg.z);
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
            health: msg.health,
            max_health: msg.max_health,
        }
    }
}

/// Storage container for remote entities tracked on the client.
#[derive(Debug, Default)]
pub struct ClientEntityStore {
    entities: HashMap<u32, ClientEntity>,
}

impl ClientEntityStore {
    /// Creates a new empty entity store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
        }
    }

    /// Number of active entities tracked.
    #[must_use]
    pub fn count(&self) -> usize {
        self.entities.len()
    }

    /// Returns an iterator over all tracked entities.
    pub fn iter(&self) -> impl Iterator<Item = &ClientEntity> {
        self.entities.values()
    }

    /// Handles an incoming `S2cSpawnEntity` packet.
    pub fn on_spawn(&mut self, msg: S2cSpawnEntity) {
        self.entities
            .insert(msg.net_id, ClientEntity::from_spawn(msg));
    }

    /// Handles an incoming `S2cDespawnEntity` packet.
    pub fn on_despawn(&mut self, net_ids: &[u32]) {
        for &id in net_ids {
            self.entities.remove(&id);
        }
    }

    /// Handles an incoming `S2cEntityMove` packet.
    pub fn on_move(&mut self, msg: S2cEntityMove) {
        if let Some(entity) = self.entities.get_mut(&msg.net_id) {
            entity.target_pos = DVec3::new(msg.x, msg.y, msg.z);
            entity.target_yaw = msg.yaw;
            entity.pitch = msg.pitch;
            entity.head_yaw = msg.head_yaw;
        }
    }

    /// Handles an incoming `S2cEntityStatus` packet.
    pub fn on_status(&mut self, net_id: u32, status: u8) {
        if let Some(entity) = self.entities.get_mut(&net_id) {
            if status == 2 {
                // Hurt red flash
                entity.hurt_timer = 0.5;
            } else if status == 3 {
                // Death
                entity.hurt_timer = 1.0;
            }
        }
    }

    /// Clears all tracked entities.
    pub fn clear(&mut self) {
        self.entities.clear();
    }

    /// Advances frame-time interpolation and animations.
    pub fn update(&mut self, dt: f32) {
        for entity in self.entities.values_mut() {
            // Decay hurt flash timer
            entity.hurt_timer = (entity.hurt_timer - dt).max(0.0);

            // Interpolate position toward replicated target
            let t = (dt * 15.0).min(1.0);
            let delta = entity.target_pos - entity.pos;
            entity.pos += delta * f64::from(t);

            // Accumulate walk animation when moving horizontally
            let horiz_dist_sq = delta.x * delta.x + delta.z * delta.z;
            if horiz_dist_sq > 0.0001 {
                entity.walk_time += dt;
            } else {
                entity.walk_time = (entity.walk_time - dt * 2.0).max(0.0);
            }

            // Smooth body yaw
            let diff = entity.target_yaw - entity.yaw;
            entity.yaw += diff * t;
        }
    }

    /// Performs slab-method ray intersection against entity bounding boxes.
    ///
    /// Returns `Some((hit_net_id, hit_distance))` if an entity was struck within `max_dist`.
    #[must_use]
    pub fn raycast(&self, ray_origin: DVec3, ray_dir: Vec3, max_dist: f32) -> Option<(u32, f32)> {
        let mut closest_hit: Option<(u32, f32)> = None;

        for entity in self.entities.values() {
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
            let hurt_tint = (entity.hurt_timer / 0.5).clamp(0.0, 1.0);
            let layer = match entity.entity_type {
                EntityType::Zombie | EntityType::Player => 0,
                EntityType::Pig => 1,
                EntityType::Cow => 2,
            };

            match entity.entity_type {
                EntityType::Zombie | EntityType::Player => {
                    build_humanoid_mesh(entity, sky, block, layer, hurt_tint, out_vertices);
                }
                EntityType::Pig => {
                    build_pig_mesh(entity, sky, block, layer, hurt_tint, out_vertices);
                }
                EntityType::Cow => {
                    build_cow_mesh(entity, sky, block, layer, hurt_tint, out_vertices);
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Model geometry builders
// ----------------------------------------------------------------------------

const S: f32 = 1.0 / 16.0;

fn build_humanoid_mesh(
    entity: &ClientEntity,
    sky: u8,
    block: u8,
    layer: u32,
    hurt_tint: f32,
    out: &mut Vec<EntityVertexGpu>,
) {
    let body_yaw = entity.yaw;
    let walk_anim = (entity.walk_time * 6.0).sin();

    // 1. Head: 8x8x8 centered at y=24..32, pivot at (0, 24, 0)
    let head_rot_yaw = entity.head_yaw - entity.yaw;
    let head_pitch = entity.pitch;
    emit_cuboid(
        entity.pos,
        body_yaw,
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
        body_yaw,
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

    // 3. Right Arm: 4x12x4, pivot at (-5, 22, 0), outstretched forward (-90 deg pitch)
    let r_arm_pitch = -90.0 + walk_anim * 8.0;
    emit_cuboid(
        entity.pos,
        body_yaw,
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
    let l_arm_pitch = -90.0 - walk_anim * 8.0;
    emit_cuboid(
        entity.pos,
        body_yaw,
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
        body_yaw,
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
        body_yaw,
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
    out: &mut Vec<EntityVertexGpu>,
) {
    let body_yaw = entity.yaw;
    let walk_anim = (entity.walk_time * 6.0).sin();

    // 1. Horizontal Body: 10x8x16 at y=6..14, z=-8..8
    emit_cuboid(
        entity.pos,
        body_yaw,
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
        body_yaw,
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
        body_yaw,
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
        body_yaw,
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
        body_yaw,
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
        body_yaw,
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
        body_yaw,
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
    out: &mut Vec<EntityVertexGpu>,
) {
    let body_yaw = entity.yaw;
    let walk_anim = (entity.walk_time * 6.0).sin();

    // 1. Horizontal Body: 12x10x18 at y=12..22, z=-9..9
    emit_cuboid(
        entity.pos,
        body_yaw,
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
        body_yaw,
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
        body_yaw,
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
        body_yaw,
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
        body_yaw,
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
        body_yaw,
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
        body_yaw,
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
        body_yaw,
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
    body_yaw: f32,
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
    let body_quat = Quat::from_rotation_y((-body_yaw).to_radians());

    let transform_pt = |p: Vec3| -> [f32; 3] {
        let rotated_limb = pivot + limb_quat * (p - pivot);
        let rotated_body = body_quat * rotated_limb;
        let world = entity_pos.as_vec3() + rotated_body;
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
