//! Entity archetypes, components, mob AI, and movement simulation systems.

use bevy_ecs::bundle::Bundle;
use bevy_ecs::component::Component;
use bevy_ecs::query::Without;
use bevy_ecs::system::{Query, Res, Resource};
use glam::{DVec3, Vec3};

use crate::attributes::{CombatTracker, Health};

/// Marker component indicating an entity is outside active simulation distance and frozen in place.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SimulationFrozen;

/// High-level classification of an in-game entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum EntityType {
    /// Human player.
    #[default]
    Player = 0,
    /// Hostile undead humanoid mob.
    Zombie = 1,
    /// Passive quadruped farm animal.
    Pig = 2,
    /// Passive bovine farm animal.
    Cow = 3,
}

impl EntityType {
    /// Converts a wire byte code into an `EntityType`.
    #[must_use]
    pub const fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::Player),
            1 => Some(Self::Zombie),
            2 => Some(Self::Pig),
            3 => Some(Self::Cow),
            _ => None,
        }
    }

    /// Converts this `EntityType` to its wire byte code.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Returns human-readable name of the entity type.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Player => "Player",
            Self::Zombie => "Zombie",
            Self::Pig => "Pig",
            Self::Cow => "Cow",
        }
    }

    /// Default bounding box half extents and Y center offset for this entity type.
    #[must_use]
    pub fn default_aabb(self) -> EntityAabb {
        match self {
            Self::Player => EntityAabb {
                half_size: Vec3::new(0.3, 0.9, 0.3),
                y_offset: 0.9,
            },
            Self::Zombie => EntityAabb {
                half_size: Vec3::new(0.3, 0.975, 0.3),
                y_offset: 0.975,
            },
            Self::Pig => EntityAabb {
                half_size: Vec3::new(0.45, 0.45, 0.45),
                y_offset: 0.45,
            },
            Self::Cow => EntityAabb {
                half_size: Vec3::new(0.45, 0.7, 0.45),
                y_offset: 0.7,
            },
        }
    }
}

/// Network identifier and entity type marker component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Component)]
pub struct NetEntity {
    /// Network ID assigned by the authoritative server.
    pub net_id: u32,
    /// Type category of the entity.
    pub entity_type: EntityType,
}

/// Double-precision world position of an entity.
#[derive(Debug, Clone, Copy, PartialEq, Component, Default)]
pub struct Position(pub DVec3);

impl Position {
    /// Creates a new `Position` from double coordinates.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self(DVec3::new(x, y, z))
    }
}

/// Euler rotation angles of an entity.
#[derive(Debug, Clone, Copy, PartialEq, Component, Default)]
pub struct Rotation {
    /// Body yaw in degrees.
    pub yaw: f32,
    /// Pitch angle in degrees (looking up/down).
    pub pitch: f32,
    /// Head yaw in degrees (can look around independently of body).
    pub head_yaw: f32,
}

impl Rotation {
    /// Creates a new `Rotation` with identical body and head yaw.
    #[must_use]
    pub const fn new(yaw: f32, pitch: f32) -> Self {
        Self {
            yaw,
            pitch,
            head_yaw: yaw,
        }
    }
}

/// Linear velocity vector of an entity in blocks per tick.
#[derive(Debug, Clone, Copy, PartialEq, Component, Default)]
pub struct Velocity(pub Vec3);

/// Axis-aligned bounding box component centered relative to the base position.
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct EntityAabb {
    /// Half-dimensions of the box along each axis `(half_x, half_y, half_z)`.
    pub half_size: Vec3,
    /// Vertical offset from entity base position to box center.
    pub y_offset: f32,
}

impl EntityAabb {
    /// Computes the world-space minimum and maximum corners given base position.
    #[must_use]
    pub fn bounds(&self, pos: DVec3) -> (DVec3, DVec3) {
        let center = pos + DVec3::new(0.0, f64::from(self.y_offset), 0.0);
        let min = center
            - DVec3::new(
                f64::from(self.half_size.x),
                f64::from(self.half_size.y),
                f64::from(self.half_size.z),
            );
        let max = center
            + DVec3::new(
                f64::from(self.half_size.x),
                f64::from(self.half_size.y),
                f64::from(self.half_size.z),
            );
        (min, max)
    }

    /// Performs slab-method ray intersection test against this bounding box in world space.
    ///
    /// Returns distance along ray `t` if an intersection occurred within `max_dist`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn intersects_ray(
        &self,
        pos: DVec3,
        ray_origin: DVec3,
        ray_dir: Vec3,
        max_dist: f32,
    ) -> Option<f32> {
        let (min, max) = self.bounds(pos);

        let mut t_min = 0.0_f64;
        let mut t_max = f64::from(max_dist);

        for i in 0..3 {
            let (orig_comp, min_comp, max_comp, dir_comp) = match i {
                0 => (ray_origin.x, min.x, max.x, f64::from(ray_dir.x)),
                1 => (ray_origin.y, min.y, max.y, f64::from(ray_dir.y)),
                _ => (ray_origin.z, min.z, max.z, f64::from(ray_dir.z)),
            };

            if dir_comp.abs() < 1e-6 {
                // Ray is parallel to slab. No hit if origin is outside slab.
                if orig_comp < min_comp || orig_comp > max_comp {
                    return None;
                }
            } else {
                let inv_d = 1.0 / dir_comp;
                let mut t1 = (min_comp - orig_comp) * inv_d;
                let mut t2 = (max_comp - orig_comp) * inv_d;
                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }
                t_min = t_min.max(t1);
                t_max = t_max.min(t2);
                if t_min > t_max {
                    return None;
                }
            }
        }

        if t_max < 0.0 || t_min > f64::from(max_dist) {
            None
        } else {
            Some(t_min.max(0.0) as f32)
        }
    }
}

/// Behavioral nature of a mob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MobKind {
    /// Hostile mob that targets and attacks players.
    Hostile,
    /// Passive mob that flees when harmed and wanders peacefully.
    Passive,
}

/// Current artificial intelligence decision state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AiState {
    /// Stationary idle state with remaining ticks countdown.
    Idle {
        /// Ticks remaining before selecting next action.
        timer: u32,
    },
    /// Moving toward a designated target location.
    Wandering {
        /// Target world position.
        target: DVec3,
        /// Maximum ticks to spend attempting to reach target.
        timer: u32,
    },
    /// Pursuing a player or enemy entity.
    Chasing {
        /// Network ID of target entity.
        target_net_id: u32,
    },
    /// Fleeing away from a danger source.
    Fleeing {
        /// Source location fleeing away from.
        away_from: DVec3,
        /// Ticks remaining in panic flee mode.
        timer: u32,
    },
}

/// Artificial intelligence controller component for mobs.
#[derive(Debug, Clone, Component)]
pub struct Mob {
    /// Category of mob (hostile vs passive).
    pub kind: MobKind,
    /// Current AI decision state.
    pub ai_state: AiState,
    /// Base horizontal movement speed in blocks per tick.
    pub base_speed: f32,
    /// Pseudo-random step seed for wander target selection.
    pub rng_seed: u64,
}

impl Mob {
    /// Creates a hostile mob controller (e.g. Zombie).
    #[must_use]
    pub fn new_hostile(seed: u64) -> Self {
        Self {
            kind: MobKind::Hostile,
            ai_state: AiState::Idle { timer: 20 },
            base_speed: 0.12,
            rng_seed: seed,
        }
    }

    /// Creates a passive mob controller (e.g. Pig, Cow).
    #[must_use]
    pub fn new_passive(seed: u64) -> Self {
        Self {
            kind: MobKind::Passive,
            ai_state: AiState::Idle { timer: 40 },
            base_speed: 0.06,
            rng_seed: seed,
        }
    }
}

/// Hurt animation and invulnerability cooldown timer (counts down each tick).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Component, Default)]
pub struct HurtTime(pub u8);

/// Complete component bundle representing a spawned mob in the ECS world.
#[derive(Debug, Clone, Bundle)]
pub struct MobBundle {
    /// Network identification.
    pub net: NetEntity,
    /// World position.
    pub pos: Position,
    /// Orientation angles.
    pub rot: Rotation,
    /// Linear velocity.
    pub vel: Velocity,
    /// Bounding box.
    pub aabb: EntityAabb,
    /// Living health points.
    pub health: Health,
    /// Combat cooldown tracker.
    pub combat: CombatTracker,
    /// Hurt animation timer.
    pub hurt_time: HurtTime,
    /// AI decision controller.
    pub mob: Mob,
}

impl MobBundle {
    /// Creates a new Zombie mob bundle.
    #[must_use]
    pub fn new_zombie(net_id: u32, pos: DVec3, seed: u64) -> Self {
        Self {
            net: NetEntity {
                net_id,
                entity_type: EntityType::Zombie,
            },
            pos: Position(pos),
            rot: Rotation::default(),
            vel: Velocity::default(),
            aabb: EntityType::Zombie.default_aabb(),
            health: Health::new(20.0),
            combat: CombatTracker::default(),
            hurt_time: HurtTime::default(),
            mob: Mob::new_hostile(seed),
        }
    }

    /// Creates a new Pig mob bundle.
    #[must_use]
    pub fn new_pig(net_id: u32, pos: DVec3, seed: u64) -> Self {
        Self {
            net: NetEntity {
                net_id,
                entity_type: EntityType::Pig,
            },
            pos: Position(pos),
            rot: Rotation::default(),
            vel: Velocity::default(),
            aabb: EntityType::Pig.default_aabb(),
            health: Health::new(10.0),
            combat: CombatTracker::default(),
            hurt_time: HurtTime::default(),
            mob: Mob::new_passive(seed),
        }
    }

    /// Creates a new Cow mob bundle.
    #[must_use]
    pub fn new_cow(net_id: u32, pos: DVec3, seed: u64) -> Self {
        Self {
            net: NetEntity {
                net_id,
                entity_type: EntityType::Cow,
            },
            pos: Position(pos),
            rot: Rotation::default(),
            vel: Velocity::default(),
            aabb: EntityType::Cow.default_aabb(),
            health: Health::new(10.0),
            combat: CombatTracker::default(),
            hurt_time: HurtTime::default(),
            mob: Mob::new_passive(seed),
        }
    }
}

/// Resource holding current active player positions for AI spatial awareness.
#[derive(Debug, Clone, Resource, Default)]
pub struct PlayerPositions(pub Vec<(u32, DVec3)>);

/// System that executes artificial intelligence state machines for all active mobs.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::needless_pass_by_value
)]
pub fn mob_ai_system(
    mut mobs: Query<(&mut Mob, &mut Rotation, &mut Velocity, &Position), Without<SimulationFrozen>>,
    players: Option<Res<PlayerPositions>>,
) {
    let player_list = players.as_ref().map_or(&[][..], |p| p.0.as_slice());

    for (mut mob, mut rot, mut vel, pos) in &mut mobs {
        match mob.kind {
            MobKind::Hostile => {
                // Find nearest player within 24 blocks detection radius
                let mut nearest_player = None;
                let mut min_dist_sq = 24.0 * 24.0;

                for &(player_id, player_pos) in player_list {
                    let dsq = pos.0.distance_squared(player_pos);
                    if dsq < min_dist_sq {
                        min_dist_sq = dsq;
                        nearest_player = Some((player_id, player_pos));
                    }
                }

                if let Some((target_id, target_pos)) = nearest_player {
                    mob.ai_state = AiState::Chasing {
                        target_net_id: target_id,
                    };

                    let dx = target_pos.x - pos.0.x;
                    let dz = target_pos.z - pos.0.z;
                    let horiz_dist = (dx * dx + dz * dz).sqrt();

                    if horiz_dist > 0.05 {
                        let nx = dx / horiz_dist;
                        let nz = dz / horiz_dist;
                        vel.0.x = (nx * f64::from(mob.base_speed)) as f32;
                        vel.0.z = (nz * f64::from(mob.base_speed)) as f32;

                        let angle_rad = dz.atan2(dx);
                        let yaw_deg = angle_rad.to_degrees() as f32 - 90.0;
                        rot.yaw = yaw_deg;
                        rot.head_yaw = yaw_deg;
                    }
                } else {
                    // No players nearby -> wander peacefully or idle
                    match mob.ai_state {
                        AiState::Idle { mut timer } => {
                            if timer == 0 {
                                // Pick new wander target using deterministic PRNG
                                mob.rng_seed = mob
                                    .rng_seed
                                    .wrapping_mul(6_364_136_223_846_793_005)
                                    .wrapping_add(1);
                                let rx = ((mob.rng_seed & 0xFFFF) as f64 / 65535.0) * 16.0 - 8.0;
                                let rz =
                                    (((mob.rng_seed >> 16) & 0xFFFF) as f64 / 65535.0) * 16.0 - 8.0;
                                let target = DVec3::new(pos.0.x + rx, pos.0.y, pos.0.z + rz);
                                mob.ai_state = AiState::Wandering { target, timer: 100 };
                            } else {
                                timer -= 1;
                                mob.ai_state = AiState::Idle { timer };
                                vel.0.x = 0.0;
                                vel.0.z = 0.0;
                            }
                        }
                        AiState::Wandering { target, mut timer } => {
                            let dx = target.x - pos.0.x;
                            let dz = target.z - pos.0.z;
                            let dist_sq = dx * dx + dz * dz;

                            if dist_sq < 0.25 || timer == 0 {
                                mob.ai_state = AiState::Idle { timer: 60 };
                                vel.0.x = 0.0;
                                vel.0.z = 0.0;
                            } else {
                                timer -= 1;
                                mob.ai_state = AiState::Wandering { target, timer };
                                let horiz_dist = dist_sq.sqrt();
                                let nx = dx / horiz_dist;
                                let nz = dz / horiz_dist;
                                let speed = mob.base_speed * 0.6;
                                vel.0.x = (nx * f64::from(speed)) as f32;
                                vel.0.z = (nz * f64::from(speed)) as f32;

                                let angle_rad = dz.atan2(dx);
                                let yaw_deg = angle_rad.to_degrees() as f32 - 90.0;
                                rot.yaw = yaw_deg;
                                rot.head_yaw = yaw_deg;
                            }
                        }
                        _ => {
                            mob.ai_state = AiState::Idle { timer: 40 };
                        }
                    }
                }
            }
            MobKind::Passive => match mob.ai_state {
                AiState::Fleeing {
                    away_from,
                    mut timer,
                } => {
                    if timer == 0 {
                        mob.ai_state = AiState::Idle { timer: 60 };
                    } else {
                        timer -= 1;
                        mob.ai_state = AiState::Fleeing { away_from, timer };
                        let dx = pos.0.x - away_from.x;
                        let dz = pos.0.z - away_from.z;
                        let dist = (dx * dx + dz * dz).sqrt().max(0.1);
                        let nx = dx / dist;
                        let nz = dz / dist;
                        let speed = mob.base_speed * 2.2;
                        vel.0.x = (nx * f64::from(speed)) as f32;
                        vel.0.z = (nz * f64::from(speed)) as f32;

                        let angle_rad = dz.atan2(dx);
                        let yaw_deg = angle_rad.to_degrees() as f32 - 90.0;
                        rot.yaw = yaw_deg;
                        rot.head_yaw = yaw_deg;
                    }
                }
                AiState::Idle { mut timer } => {
                    if timer == 0 {
                        mob.rng_seed = mob
                            .rng_seed
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1);
                        let rx = ((mob.rng_seed & 0xFFFF) as f64 / 65535.0) * 12.0 - 6.0;
                        let rz = (((mob.rng_seed >> 16) & 0xFFFF) as f64 / 65535.0) * 12.0 - 6.0;
                        let target = DVec3::new(pos.0.x + rx, pos.0.y, pos.0.z + rz);
                        mob.ai_state = AiState::Wandering { target, timer: 80 };
                    } else {
                        timer -= 1;
                        mob.ai_state = AiState::Idle { timer };
                        vel.0.x = 0.0;
                        vel.0.z = 0.0;
                    }
                }
                AiState::Wandering { target, mut timer } => {
                    let dx = target.x - pos.0.x;
                    let dz = target.z - pos.0.z;
                    let dist_sq = dx * dx + dz * dz;

                    if dist_sq < 0.25 || timer == 0 {
                        mob.ai_state = AiState::Idle { timer: 80 };
                        vel.0.x = 0.0;
                        vel.0.z = 0.0;
                    } else {
                        timer -= 1;
                        mob.ai_state = AiState::Wandering { target, timer };
                        let horiz_dist = dist_sq.sqrt();
                        let nx = dx / horiz_dist;
                        let nz = dz / horiz_dist;
                        vel.0.x = (nx * f64::from(mob.base_speed)) as f32;
                        vel.0.z = (nz * f64::from(mob.base_speed)) as f32;

                        let angle_rad = dz.atan2(dx);
                        let yaw_deg = angle_rad.to_degrees() as f32 - 90.0;
                        rot.yaw = yaw_deg;
                        rot.head_yaw = yaw_deg;
                    }
                }
                AiState::Chasing { .. } => {
                    mob.ai_state = AiState::Idle { timer: 40 };
                }
            },
        }
    }
}

/// System that applies physics velocity integration, gravity, and drag to entities.
pub fn mob_movement_system(
    mut query: Query<(&mut Position, &mut Velocity), Without<SimulationFrozen>>,
) {
    for (mut pos, mut vel) in &mut query {
        // Integrate horizontal and vertical velocity
        pos.0.x += f64::from(vel.0.x);
        pos.0.y += f64::from(vel.0.y);
        pos.0.z += f64::from(vel.0.z);

        // Apply gravity
        vel.0.y = (vel.0.y - 0.08).max(-2.0);

        // Apply horizontal drag/friction
        vel.0.x *= 0.6;
        vel.0.z *= 0.6;
    }
}

/// System that counts down entity hurt animation and triggers panic flee on passive mobs.
pub fn mob_hurt_decay_system(
    mut query: Query<(&mut HurtTime, &mut Mob, &Position), Without<SimulationFrozen>>,
) {
    for (mut hurt, mut mob, pos) in &mut query {
        if hurt.0 > 0 {
            hurt.0 -= 1;
            if mob.kind == MobKind::Passive {
                if let AiState::Fleeing { .. } = mob.ai_state {
                    // Already fleeing
                } else {
                    // Panic flee away from current position
                    mob.ai_state = AiState::Fleeing {
                        away_from: pos.0,
                        timer: 80,
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_conversions() {
        assert_eq!(EntityType::from_u8(0), Some(EntityType::Player));
        assert_eq!(EntityType::from_u8(1), Some(EntityType::Zombie));
        assert_eq!(EntityType::from_u8(2), Some(EntityType::Pig));
        assert_eq!(EntityType::from_u8(3), Some(EntityType::Cow));
        assert_eq!(EntityType::from_u8(4), None);
        assert_eq!(EntityType::Zombie.to_u8(), 1);
        assert_eq!(EntityType::Pig.name(), "Pig");
    }

    #[test]
    fn test_aabb_ray_intersection() {
        let aabb = EntityAabb {
            half_size: Vec3::new(0.5, 1.0, 0.5),
            y_offset: 1.0,
        };
        let pos = DVec3::new(10.0, 64.0, 10.0);

        // Ray directly aiming at box center from 5 blocks away
        let ray_origin = DVec3::new(10.0, 65.0, 5.0);
        let ray_dir = Vec3::new(0.0, 0.0, 1.0);
        let hit = aabb.intersects_ray(pos, ray_origin, ray_dir, 10.0);
        assert!(hit.is_some());
        let dist = hit.unwrap();
        assert!((dist - 4.5).abs() < 1e-4);

        // Ray aiming away
        let ray_miss = Vec3::new(1.0, 0.0, 0.0);
        assert!(
            aabb.intersects_ray(pos, ray_origin, ray_miss, 10.0)
                .is_none()
        );

        // Ray out of reach
        assert!(aabb.intersects_ray(pos, ray_origin, ray_dir, 3.0).is_none());
    }

    #[test]
    fn test_mob_ai_chase_player() {
        let mut world = bevy_ecs::world::World::new();
        let zombie = MobBundle::new_zombie(10, DVec3::new(0.0, 64.0, 0.0), 12345);
        let e = world.spawn(zombie).id();

        world.insert_resource(PlayerPositions(vec![(1, DVec3::new(10.0, 64.0, 0.0))]));

        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(mob_ai_system);

        schedule.run(&mut world);

        let mob = world.get::<Mob>(e).unwrap();
        assert_eq!(mob.ai_state, AiState::Chasing { target_net_id: 1 });

        let vel = world.get::<Velocity>(e).unwrap();
        assert!(vel.0.x > 0.05); // Moving positive X towards player
    }

    #[test]
    fn test_mob_hurt_decay_triggers_flee() {
        let mut world = bevy_ecs::world::World::new();
        let mut pig = MobBundle::new_pig(20, DVec3::new(5.0, 64.0, 5.0), 999);
        pig.hurt_time.0 = 5;
        let e = world.spawn(pig).id();

        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(mob_hurt_decay_system);

        schedule.run(&mut world);

        let hurt = world.get::<HurtTime>(e).unwrap();
        assert_eq!(hurt.0, 4);

        let mob = world.get::<Mob>(e).unwrap();
        match mob.ai_state {
            AiState::Fleeing { timer, .. } => assert_eq!(timer, 80),
            _ => panic!("Expected pig to flee when hurt"),
        }
    }
}
