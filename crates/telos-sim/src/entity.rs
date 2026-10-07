//! Entity archetypes, components, mob AI, and movement simulation systems.

use bevy_ecs::bundle::Bundle;
use bevy_ecs::component::Component;
use bevy_ecs::query::Without;
use bevy_ecs::system::{Query, Res, Resource};
use glam::{DVec3, Vec3};
use telos_core::coords::{BlockPos, Face};

use crate::attributes::{CombatTracker, Health};
use crate::inventory::ItemStack;

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
    /// Dropped floating item stack.
    Item = 4,
    /// Airborne or embedded projectile arrow.
    Arrow = 5,
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
            4 => Some(Self::Item),
            5 => Some(Self::Arrow),
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
            Self::Item => "Item",
            Self::Arrow => "Arrow",
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
            Self::Item => EntityAabb {
                half_size: Vec3::new(0.125, 0.125, 0.125),
                y_offset: 0.125,
            },
            Self::Arrow => EntityAabb {
                half_size: Vec3::new(0.15, 0.15, 0.15),
                y_offset: 0.15,
            },
        }
    }
}

/// Lifetime before a dropped item entity despawns (6000 ticks = 5 minutes at 20 TPS).
pub const ITEM_DESPAWN_TICKS: u16 = 6000;
/// Default pickup delay when dropped by a player (10 ticks = 0.5s).
pub const PLAYER_DROP_PICKUP_DELAY: u16 = 10;
/// Proximity radius for merging matching item entities (1.5 blocks).
pub const ITEM_MERGE_RADIUS: f64 = 1.5;
/// Proximity radius for player inventory pickup (1.5 blocks).
pub const ITEM_PICKUP_RADIUS: f64 = 1.5;

/// Dropped item entity component tracking its inventory stack, pickup delay, and age.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Component)]
pub struct ItemEntity {
    /// Item stack represented by this entity.
    pub stack: ItemStack,
    /// Delay in ticks before a survival player can pick up this item (e.g. 10 on drop).
    pub pickup_delay: u16,
    /// Lifetime in ticks since creation. Despawns when reaching `ITEM_DESPAWN_TICKS`.
    pub age: u16,
}

impl ItemEntity {
    /// Creates a new `ItemEntity` with given stack and pickup delay.
    #[must_use]
    pub const fn new(stack: ItemStack, pickup_delay: u16) -> Self {
        Self {
            stack,
            pickup_delay,
            age: 0,
        }
    }
}

/// Performs a single-tick physics simulation step for a dropped item entity against terrain collision.
#[allow(clippy::cast_possible_truncation)]
pub fn tick_item_physics_step(
    pos: &mut DVec3,
    vel: &mut Vec3,
    mut is_solid: impl FnMut(i32, i32, i32) -> bool,
) {
    // 1. Gravity acceleration (0.04 blocks/tick^2 downward)
    vel.y -= 0.04;

    // 2. Air resistance / damping
    vel.x *= 0.98;
    vel.y *= 0.98;
    vel.z *= 0.98;

    // 3. Candidate target position
    let next_x = pos.x + f64::from(vel.x);
    let next_y = pos.y + f64::from(vel.y);
    let next_z = pos.z + f64::from(vel.z);

    // Floor voxel coordinate check
    let block_x = next_x.floor() as i32;
    let block_y = next_y.floor() as i32;
    let block_z = next_z.floor() as i32;

    if is_solid(block_x, block_y, block_z) {
        // Floor contact: resting on top of the solid block
        pos.y = f64::from(block_y + 1);
        vel.y = 0.0;
        // Ground friction
        vel.x *= 0.6;
        vel.z *= 0.6;

        // Advance horizontal coordinates if passing through air
        let check_above_x = is_solid(next_x.floor() as i32, block_y + 1, pos.z.floor() as i32);
        if check_above_x {
            vel.x = 0.0;
        } else {
            pos.x = next_x;
        }
        let check_above_z = is_solid(pos.x.floor() as i32, block_y + 1, next_z.floor() as i32);
        if check_above_z {
            vel.z = 0.0;
        } else {
            pos.z = next_z;
        }
    } else {
        // In air: free movement
        pos.x = next_x;
        pos.y = next_y;
        pos.z = next_z;
    }
}

/// Attempts to merge `source` into `target`. Returns true if items were transferred.
pub fn merge_item_stacks(target: &mut ItemStack, source: &mut ItemStack) -> bool {
    if source.is_empty() || target.item != source.item || target.count >= 64 {
        return false;
    }
    let space = 64 - target.count;
    let transfer = source.count.min(space);
    target.count += transfer;
    source.count -= transfer;
    source.normalize();
    transfer > 0
}

/// Minimum charging ticks before a bow can release an arrow (3 ticks = 0.15s).
pub const BOW_MIN_CHARGE_TICKS: u16 = 3;
/// Full charge duration for a bow (20 ticks = 1.0s).
pub const BOW_FULL_CHARGE_TICKS: u16 = 20;
/// Minimum release speed of an arrow in meters/second (10.0 m/s = 0.5 blocks/tick).
pub const BOW_MIN_RELEASE_SPEED: f32 = 10.0;
/// Maximum release speed of an arrow in meters/second (60.0 m/s = 3.0 blocks/tick).
pub const BOW_MAX_RELEASE_SPEED: f32 = 60.0;
/// Downward gravitational acceleration for arrows per tick (0.05 blocks/tick).
pub const ARROW_GRAVITY: f64 = 0.05;
/// Air drag multiplier applied to arrow velocity per tick.
pub const ARROW_AIR_DRAG: f64 = 0.99;
/// Lifetime in ticks before a stuck arrow despawns (1200 ticks = 60s).
pub const ARROW_DESPAWN_STUCK_TICKS: u16 = 1200;
/// Lifetime in ticks before a flying arrow despawns (600 ticks = 30s).
pub const ARROW_DESPAWN_FLYING_TICKS: u16 = 600;
/// Pickup proximity radius for stuck arrows (1.5 blocks).
pub const ARROW_PICKUP_RADIUS: f64 = 1.5;

/// Projectile arrow entity component tracking shooter, embedded state, age, and damage.
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct ArrowEntity {
    /// Entity network ID of the shooter who fired this arrow, if known.
    pub shooter_id: Option<u32>,
    /// Whether the arrow is embedded into a solid block face.
    pub in_ground: bool,
    /// Struck block position if embedded.
    pub stuck_block: Option<BlockPos>,
    /// Lifetime in ticks since creation.
    pub age: u16,
    /// Base impact damage scaled by release velocity.
    pub damage: f32,
    /// Whether a survival player can pick up this arrow when stuck.
    pub pickup_allowed: bool,
}

impl ArrowEntity {
    /// Creates a new `ArrowEntity` with given shooter, base damage, and pickup flag.
    #[must_use]
    pub const fn new(shooter_id: Option<u32>, damage: f32, pickup_allowed: bool) -> Self {
        Self {
            shooter_id,
            in_ground: false,
            stuck_block: None,
            age: 0,
            damage,
            pickup_allowed,
        }
    }
}

/// Result of advancing an arrow's physics for one tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArrowStepOutcome {
    /// Arrow continues flying through the air.
    Flying,
    /// Arrow struck a solid voxel and embedded into its face.
    HitBlock {
        /// Coordinates of the struck block.
        hit_block: BlockPos,
        /// Face entered by the ray.
        face: Face,
        /// Exact intersection point in world space.
        hit_pos: DVec3,
    },
    /// Arrow struck a living entity.
    HitEntity {
        /// Network ID of the struck entity.
        target_id: u32,
        /// Distance along the trajectory ray to the intersection point.
        distance: f32,
    },
}

/// Performs a single-tick physics simulation step for a projectile arrow.
#[allow(clippy::cast_possible_truncation)]
pub fn tick_arrow_physics_step<F, E>(
    pos: &mut DVec3,
    vel: &mut DVec3,
    yaw: &mut f32,
    pitch: &mut f32,
    mut is_solid: F,
    mut check_entities: E,
) -> ArrowStepOutcome
where
    F: FnMut(BlockPos) -> bool,
    E: FnMut(DVec3, Vec3, f32) -> Option<(u32, f32)>,
{
    // Apply gravity & drag to velocity
    vel.y -= ARROW_GRAVITY;
    *vel *= ARROW_AIR_DRAG;

    let speed = vel.length();
    if speed < 1e-4 {
        return ArrowStepOutcome::Flying;
    }

    let dir_v3 = (*vel / speed).as_vec3();
    let start_pos = *pos;
    let max_dist = speed as f32;

    // 1. Check entity collision along ray [0, max_dist]
    let entity_hit = check_entities(start_pos, dir_v3, max_dist);

    // 2. Check voxel collision along ray [0, max_dist]
    let voxel_hit =
        telos_core::raycast::raycast_voxels(start_pos.as_vec3(), dir_v3, max_dist, &mut is_solid);

    // Determine which hit was closer
    match (entity_hit, voxel_hit) {
        (Some((entity_id, ent_dist)), Some(v_hit)) if ent_dist <= v_hit.distance => {
            ArrowStepOutcome::HitEntity {
                target_id: entity_id,
                distance: ent_dist,
            }
        }
        (Some((entity_id, ent_dist)), None) => ArrowStepOutcome::HitEntity {
            target_id: entity_id,
            distance: ent_dist,
        },
        (_, Some(v_hit)) => {
            *pos = v_hit.hit_point.as_dvec3();
            *vel = DVec3::ZERO;
            ArrowStepOutcome::HitBlock {
                hit_block: v_hit.pos,
                face: v_hit.face,
                hit_pos: *pos,
            }
        }
        (None, None) => {
            *pos += *vel;
            // Update yaw and pitch aligned with new velocity vector
            *yaw = (-vel.x).atan2(vel.z).to_degrees() as f32;
            *pitch = (-vel.y)
                .atan2((vel.x * vel.x + vel.z * vel.z).sqrt())
                .to_degrees() as f32;
            ArrowStepOutcome::Flying
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

/// Navigation follower component managing waypoint progression, jump impulses, and steering.
#[derive(Debug, Clone, Component, Default)]
pub struct PathFollower {
    /// Active planned navigation path.
    pub path: Option<crate::nav::NavPath>,
    /// Target entity network ID currently tracked.
    pub target_entity: Option<u32>,
    /// Last target position when path was computed.
    pub last_target_pos: Option<telos_core::coords::BlockPos>,
    /// Consecutive ticks mob position remained stuck near the same coordinates.
    pub stuck_ticks: u32,
    /// Last recorded position for stuck detection.
    pub last_pos: DVec3,
    /// Cooldown ticks remaining before next repath query is permitted.
    pub repath_cooldown: u32,
}

impl PathFollower {
    /// Creates a new default `PathFollower`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Hurt animation and invulnerability cooldown timer (counts down each tick).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Component, Default)]
pub struct HurtTime(pub u8);

/// Combat attack cooldown component controlling attack damage, reach, and cadence.
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct AttackCooldown {
    /// Ticks remaining before the entity can execute another attack.
    pub current: u32,
    /// Cooldown reset interval in ticks.
    pub interval: u32,
    /// Base melee damage inflicted per hit.
    pub damage: f32,
    /// Maximum reach distance in blocks to land an attack.
    pub reach: f32,
}

impl AttackCooldown {
    /// Creates a new `AttackCooldown` configuration.
    #[must_use]
    pub const fn new(interval: u32, damage: f32, reach: f32) -> Self {
        Self {
            current: 0,
            interval,
            damage,
            reach,
        }
    }

    /// Advances the cooldown timer by one tick, clamping at 0.
    pub fn tick(&mut self) {
        if self.current > 0 {
            self.current -= 1;
        }
    }

    /// Returns `true` if the cooldown has elapsed and an attack can be performed.
    #[must_use]
    pub const fn can_attack(&self) -> bool {
        self.current == 0
    }

    /// Resets the cooldown timer to the full interval.
    pub fn reset(&mut self) {
        self.current = self.interval;
    }
}

impl Default for AttackCooldown {
    fn default() -> Self {
        Self {
            current: 0,
            interval: 20,
            damage: 3.0,
            reach: 1.8,
        }
    }
}

/// Tests whether an unobstructed direct line-of-sight exists between `from` and `to`.
///
/// Raycasts voxels between eye positions and returns `true` if no solid obstacle intervenes.
#[must_use]
pub fn has_line_of_sight<F>(from: Vec3, to: Vec3, is_solid: F) -> bool
where
    F: FnMut(telos_core::coords::BlockPos) -> bool,
{
    let diff = to - from;
    let dist = diff.length();
    if dist < 1e-4 {
        return true;
    }
    let hit = telos_core::raycast::raycast_voxels(from, diff, dist, is_solid);
    match hit {
        None => true,
        Some(h) => h.distance >= dist - 0.05,
    }
}

/// Player targeting information for mob sensory perception and combat AI.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetablePlayer {
    /// Network ID of the player.
    pub net_id: u32,
    /// Double-precision world position of the player feet.
    pub pos: DVec3,
    /// Whether the player can be targeted by hostile mobs (e.g. survival mode, not flying).
    pub targetable: bool,
}

impl TargetablePlayer {
    /// Creates a new `TargetablePlayer`.
    #[must_use]
    pub const fn new(net_id: u32, pos: DVec3, targetable: bool) -> Self {
        Self {
            net_id,
            pos,
            targetable,
        }
    }
}

/// Resource holding current active player positions and targetability for AI spatial awareness.
#[derive(Debug, Clone, Resource, Default)]
pub struct PlayerPositions(pub Vec<TargetablePlayer>);

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
    /// 3D navigation path follower.
    pub path_follower: PathFollower,
    /// Attack cooldown and melee damage properties.
    pub attack_cooldown: AttackCooldown,
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
            path_follower: PathFollower::default(),
            attack_cooldown: AttackCooldown::new(20, 3.0, 1.8),
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
            path_follower: PathFollower::default(),
            attack_cooldown: AttackCooldown::new(u32::MAX, 0.0, 0.0),
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
            path_follower: PathFollower::default(),
            attack_cooldown: AttackCooldown::new(u32::MAX, 0.0, 0.0),
        }
    }
}

/// System that executes artificial intelligence state machines and waypoint steering for all active mobs.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    clippy::type_complexity
)]
pub fn mob_ai_system(
    mut mobs: Query<
        (
            &mut Mob,
            &mut PathFollower,
            &mut Rotation,
            &mut Velocity,
            &Position,
            Option<&mut AttackCooldown>,
        ),
        Without<SimulationFrozen>,
    >,
    players: Option<Res<PlayerPositions>>,
) {
    let player_list = players.as_ref().map_or(&[][..], |p| p.0.as_slice());

    for (mut mob, mut follower, mut rot, mut vel, pos, mut attack_cooldown) in &mut mobs {
        if let Some(ref mut cd) = attack_cooldown {
            cd.tick();
        }

        // Stuck detection
        if pos.0.distance_squared(follower.last_pos) < 0.0025 {
            follower.stuck_ticks += 1;
            if follower.stuck_ticks >= 20 {
                follower.path = None;
                follower.stuck_ticks = 0;
                follower.repath_cooldown = 0;
            }
        } else {
            follower.stuck_ticks = 0;
            follower.last_pos = pos.0;
        }

        match mob.kind {
            MobKind::Hostile => {
                // Determine target player
                let mut target_player = None;

                // Check existing target if chasing
                if let AiState::Chasing { target_net_id } = mob.ai_state
                    && let Some(tp) = player_list
                        .iter()
                        .find(|p| p.net_id == target_net_id && p.targetable)
                {
                    let dsq = pos.0.distance_squared(tp.pos);
                    if dsq <= 24.0 * 24.0 {
                        target_player = Some((tp.net_id, tp.pos));
                    }
                }

                // If no current valid target, acquire nearest targetable player within 16 blocks
                if target_player.is_none() {
                    let mut min_dist_sq = 16.0 * 16.0;
                    for p in player_list {
                        if !p.targetable {
                            continue;
                        }
                        let dsq = pos.0.distance_squared(p.pos);
                        if dsq < min_dist_sq {
                            min_dist_sq = dsq;
                            target_player = Some((p.net_id, p.pos));
                        }
                    }
                }

                if let Some((target_id, target_pos)) = target_player {
                    mob.ai_state = AiState::Chasing {
                        target_net_id: target_id,
                    };
                    follower.target_entity = Some(target_id);

                    let speed = mob.base_speed;
                    let dx = target_pos.x - pos.0.x;
                    let dz = target_pos.z - pos.0.z;
                    let dy = (target_pos.y - pos.0.y).abs();
                    let horiz_dist = (dx * dx + dz * dz).sqrt();

                    let reach = attack_cooldown.as_ref().map_or(1.8, |cd| cd.reach);

                    // Always face the target when chasing
                    if horiz_dist > 0.01 {
                        let angle_rad = dz.atan2(dx);
                        let yaw_deg = angle_rad.to_degrees() as f32 - 90.0;
                        rot.yaw = yaw_deg;
                        rot.head_yaw = yaw_deg;
                    }

                    // If within melee reach, halt horizontal movement so mob doesn't overshoot
                    if horiz_dist <= f64::from(reach) && dy <= 1.5 {
                        vel.0.x = 0.0;
                        vel.0.z = 0.0;
                    } else {
                        let mut followed_path = false;

                        // Follow planned navigation waypoints if available
                        if let Some(ref mut path) = follower.path {
                            if let Some(wp) = path.current_waypoint() {
                                let wp_center = DVec3::new(
                                    f64::from(wp.x()) + 0.5,
                                    f64::from(wp.y()),
                                    f64::from(wp.z()) + 0.5,
                                );
                                let wdx = wp_center.x - pos.0.x;
                                let wdz = wp_center.z - pos.0.z;
                                let wh_dist = (wdx * wdx + wdz * wdz).sqrt();

                                if wh_dist < 0.45 && (pos.0.y - wp_center.y).abs() < 1.25 {
                                    path.advance();
                                }

                                if let Some(active_wp) = path.current_waypoint() {
                                    let active_center = DVec3::new(
                                        f64::from(active_wp.x()) + 0.5,
                                        f64::from(active_wp.y()),
                                        f64::from(active_wp.z()) + 0.5,
                                    );
                                    let adx = active_center.x - pos.0.x;
                                    let adz = active_center.z - pos.0.z;
                                    let adist = (adx * adx + adz * adz).sqrt();

                                    if adist > 0.05 {
                                        let nx = adx / adist;
                                        let nz = adz / adist;
                                        vel.0.x = (nx * f64::from(speed)) as f32;
                                        vel.0.z = (nz * f64::from(speed)) as f32;

                                        let cur_y = pos.0.y.floor() as i32;
                                        if active_wp.y() > cur_y && vel.0.y.abs() < 0.1 {
                                            vel.0.y = 0.42;
                                        }

                                        followed_path = true;
                                    }
                                }
                            }
                            if path.is_finished() {
                                follower.path = None;
                            }
                        }

                        // Direct steering fallback if no waypoint path
                        if !followed_path && horiz_dist > 0.05 {
                            let nx = dx / horiz_dist;
                            let nz = dz / horiz_dist;
                            vel.0.x = (nx * f64::from(speed)) as f32;
                            vel.0.z = (nz * f64::from(speed)) as f32;
                        }
                    }
                } else {
                    // No players nearby -> wander peacefully or idle
                    follower.target_entity = None;
                    match mob.ai_state {
                        AiState::Idle { mut timer } => {
                            if timer == 0 {
                                mob.rng_seed = mob
                                    .rng_seed
                                    .wrapping_mul(6_364_136_223_846_793_005)
                                    .wrapping_add(1);
                                let rx = ((mob.rng_seed & 0xFFFF) as f64 / 65535.0) * 16.0 - 8.0;
                                let rz =
                                    (((mob.rng_seed >> 16) & 0xFFFF) as f64 / 65535.0) * 16.0 - 8.0;
                                let target = DVec3::new(pos.0.x + rx, pos.0.y, pos.0.z + rz);
                                mob.ai_state = AiState::Wandering { target, timer: 100 };
                                follower.path = None;
                            } else {
                                timer -= 1;
                                mob.ai_state = AiState::Idle { timer };
                                vel.0.x = 0.0;
                                vel.0.z = 0.0;
                            }
                        }
                        AiState::Wandering { target, mut timer } => {
                            let speed = mob.base_speed * 0.6;
                            let mut followed_path = false;

                            if let Some(ref mut path) = follower.path {
                                if let Some(wp) = path.current_waypoint() {
                                    let wp_center = DVec3::new(
                                        f64::from(wp.x()) + 0.5,
                                        f64::from(wp.y()),
                                        f64::from(wp.z()) + 0.5,
                                    );
                                    let dx = wp_center.x - pos.0.x;
                                    let dz = wp_center.z - pos.0.z;
                                    let horiz_dist = (dx * dx + dz * dz).sqrt();

                                    if horiz_dist < 0.45 && (pos.0.y - wp_center.y).abs() < 1.25 {
                                        path.advance();
                                    }

                                    if let Some(active_wp) = path.current_waypoint() {
                                        let active_center = DVec3::new(
                                            f64::from(active_wp.x()) + 0.5,
                                            f64::from(active_wp.y()),
                                            f64::from(active_wp.z()) + 0.5,
                                        );
                                        let adx = active_center.x - pos.0.x;
                                        let adz = active_center.z - pos.0.z;
                                        let adist = (adx * adx + adz * adz).sqrt();

                                        if adist > 0.05 {
                                            let nx = adx / adist;
                                            let nz = adz / adist;
                                            vel.0.x = (nx * f64::from(speed)) as f32;
                                            vel.0.z = (nz * f64::from(speed)) as f32;

                                            let cur_y = pos.0.y.floor() as i32;
                                            if active_wp.y() > cur_y && vel.0.y.abs() < 0.1 {
                                                vel.0.y = 0.42;
                                            }

                                            let angle_rad = adz.atan2(adx);
                                            let yaw_deg = angle_rad.to_degrees() as f32 - 90.0;
                                            rot.yaw = yaw_deg;
                                            rot.head_yaw = yaw_deg;
                                            followed_path = true;
                                        }
                                    }
                                }
                                if path.is_finished() {
                                    follower.path = None;
                                }
                            }

                            if !followed_path {
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
                                    vel.0.x = (nx * f64::from(speed)) as f32;
                                    vel.0.z = (nz * f64::from(speed)) as f32;

                                    let angle_rad = dz.atan2(dx);
                                    let yaw_deg = angle_rad.to_degrees() as f32 - 90.0;
                                    rot.yaw = yaw_deg;
                                    rot.head_yaw = yaw_deg;
                                }
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
                        follower.path = None;
                    } else {
                        timer -= 1;
                        mob.ai_state = AiState::Idle { timer };
                        vel.0.x = 0.0;
                        vel.0.z = 0.0;
                    }
                }
                AiState::Wandering { target, mut timer } => {
                    let speed = mob.base_speed;
                    let mut followed_path = false;

                    if let Some(ref mut path) = follower.path {
                        if let Some(wp) = path.current_waypoint() {
                            let wp_center = DVec3::new(
                                f64::from(wp.x()) + 0.5,
                                f64::from(wp.y()),
                                f64::from(wp.z()) + 0.5,
                            );
                            let dx = wp_center.x - pos.0.x;
                            let dz = wp_center.z - pos.0.z;
                            let horiz_dist = (dx * dx + dz * dz).sqrt();

                            if horiz_dist < 0.45 && (pos.0.y - wp_center.y).abs() < 1.25 {
                                path.advance();
                            }

                            if let Some(active_wp) = path.current_waypoint() {
                                let active_center = DVec3::new(
                                    f64::from(active_wp.x()) + 0.5,
                                    f64::from(active_wp.y()),
                                    f64::from(active_wp.z()) + 0.5,
                                );
                                let adx = active_center.x - pos.0.x;
                                let adz = active_center.z - pos.0.z;
                                let adist = (adx * adx + adz * adz).sqrt();

                                if adist > 0.05 {
                                    let nx = adx / adist;
                                    let nz = adz / adist;
                                    vel.0.x = (nx * f64::from(speed)) as f32;
                                    vel.0.z = (nz * f64::from(speed)) as f32;

                                    let cur_y = pos.0.y.floor() as i32;
                                    if active_wp.y() > cur_y && vel.0.y.abs() < 0.1 {
                                        vel.0.y = 0.42;
                                    }

                                    let angle_rad = adz.atan2(adx);
                                    let yaw_deg = angle_rad.to_degrees() as f32 - 90.0;
                                    rot.yaw = yaw_deg;
                                    rot.head_yaw = yaw_deg;
                                    followed_path = true;
                                }
                            }
                        }
                        if path.is_finished() {
                            follower.path = None;
                        }
                    }

                    if !followed_path {
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
                            vel.0.x = (nx * f64::from(speed)) as f32;
                            vel.0.z = (nz * f64::from(speed)) as f32;

                            let angle_rad = dz.atan2(dx);
                            let yaw_deg = angle_rad.to_degrees() as f32 - 90.0;
                            rot.yaw = yaw_deg;
                            rot.head_yaw = yaw_deg;
                        }
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
        assert_eq!(EntityType::from_u8(4), Some(EntityType::Item));
        assert_eq!(EntityType::from_u8(5), Some(EntityType::Arrow));
        assert_eq!(EntityType::from_u8(6), None);
        assert_eq!(EntityType::Zombie.to_u8(), 1);
        assert_eq!(EntityType::Item.to_u8(), 4);
        assert_eq!(EntityType::Arrow.to_u8(), 5);
        assert_eq!(EntityType::Pig.name(), "Pig");
        assert_eq!(EntityType::Item.name(), "Item");
        assert_eq!(EntityType::Arrow.name(), "Arrow");
    }

    #[test]
    fn test_arrow_physics_step() {
        let mut pos = DVec3::new(0.0, 10.0, 0.0);
        let mut vel = DVec3::new(1.0, 0.0, 0.0);
        let mut yaw = 0.0_f32;
        let mut pitch = 0.0_f32;

        let outcome = tick_arrow_physics_step(
            &mut pos,
            &mut vel,
            &mut yaw,
            &mut pitch,
            |_| false,
            |_, _, _| None,
        );
        assert_eq!(outcome, ArrowStepOutcome::Flying);
        assert!(pos.x > 0.0);
        assert!(pos.y < 10.0);
        assert!(vel.y < 0.0);

        let mut pos2 = DVec3::new(0.0, 10.0, 0.0);
        let mut vel2 = DVec3::new(2.0, 0.0, 0.0);
        let outcome2 = tick_arrow_physics_step(
            &mut pos2,
            &mut vel2,
            &mut yaw,
            &mut pitch,
            |b| b.x() == 1,
            |_, _, _| None,
        );
        match outcome2 {
            ArrowStepOutcome::HitBlock { hit_block, .. } => {
                assert_eq!(hit_block.x(), 1);
                assert_eq!(vel2, DVec3::ZERO);
            }
            _ => panic!("Expected block hit"),
        }

        let mut pos3 = DVec3::new(0.0, 10.0, 0.0);
        let mut vel3 = DVec3::new(2.0, 0.0, 0.0);
        let outcome3 = tick_arrow_physics_step(
            &mut pos3,
            &mut vel3,
            &mut yaw,
            &mut pitch,
            |_| false,
            |_, _, _| Some((42, 1.2)),
        );
        match outcome3 {
            ArrowStepOutcome::HitEntity {
                target_id,
                distance,
            } => {
                assert_eq!(target_id, 42);
                assert!((distance - 1.2).abs() < 1e-4);
            }
            _ => panic!("Expected entity hit"),
        }
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

        world.insert_resource(PlayerPositions(vec![TargetablePlayer::new(
            1,
            DVec3::new(10.0, 64.0, 0.0),
            true,
        )]));

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
