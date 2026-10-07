//! Player physics simulation: swept AABB collision, gravity, jump, walk/sprint/sneak, and flight.

use glam::{DVec3, FloatExt, Vec3};

/// Player physical collision width and depth in blocks.
pub const PLAYER_WIDTH: f64 = 0.6;
/// Player physical collision height in blocks.
pub const PLAYER_HEIGHT: f64 = 1.8;
/// Player eye level offset from feet in blocks.
pub const EYE_HEIGHT: f64 = 1.62;

/// Downward gravitational acceleration in blocks/s^2.
pub const GRAVITY: f32 = 32.0;
/// Instantaneous upward vertical impulse on jump in blocks/s.
pub const JUMP_VELOCITY: f32 = 8.4;
/// Walking speed in blocks/s.
pub const WALK_SPEED: f32 = 4.317;
/// Sprinting speed in blocks/s.
pub const SPRINT_SPEED: f32 = 5.612;
/// Sneaking speed in blocks/s.
pub const SNEAK_SPEED: f32 = 1.3;
/// Base creative flight speed in blocks/s.
pub const FLY_SPEED: f32 = 14.0;
/// Sprinting creative flight speed in blocks/s.
pub const FLY_SPRINT_SPEED: f32 = 35.0;
/// Maximum step-up height for walking over slabs and stairs without jumping.
pub const STEP_HEIGHT: f64 = 0.5;

/// Player game mode controlling physics rules and flight capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GameMode {
    /// Survival mode: strict swept AABB collision, gravity, fall damage, and no flight.
    Survival,
    /// Creative mode: god mode, flight toggle, instant block breaking, and creative flight.
    #[default]
    Creative,
}

impl GameMode {
    /// Returns the user-facing display name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Survival => "Survival",
            Self::Creative => "Creative",
        }
    }
}

/// Axis-aligned bounding box in world space with double precision coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    /// Minimum corner [`min_x`, `min_y`, `min_z`].
    pub min: DVec3,
    /// Maximum corner [`max_x`, `max_y`, `max_z`].
    pub max: DVec3,
}

impl Aabb {
    /// Creates a new AABB from minimum and maximum coordinates.
    #[must_use]
    pub const fn new(min: DVec3, max: DVec3) -> Self {
        Self { min, max }
    }

    /// Creates a full 1x1x1 unit voxel AABB at integer block coordinates.
    #[must_use]
    pub fn from_block(x: i32, y: i32, z: i32) -> Self {
        Self {
            min: DVec3::new(f64::from(x), f64::from(y), f64::from(z)),
            max: DVec3::new(f64::from(x + 1), f64::from(y + 1), f64::from(z + 1)),
        }
    }

    /// Creates an axis-aligned sub-box within an integer block (e.g. 0..16 coordinates).
    #[must_use]
    pub fn from_sub_box(x: i32, y: i32, z: i32, min_sub: [u8; 3], max_sub: [u8; 3]) -> Self {
        let bx = f64::from(x);
        let by = f64::from(y);
        let bz = f64::from(z);
        Self {
            min: DVec3::new(
                bx + f64::from(min_sub[0]) / 16.0,
                by + f64::from(min_sub[1]) / 16.0,
                bz + f64::from(min_sub[2]) / 16.0,
            ),
            max: DVec3::new(
                bx + f64::from(max_sub[0]) / 16.0,
                by + f64::from(max_sub[1]) / 16.0,
                bz + f64::from(max_sub[2]) / 16.0,
            ),
        }
    }

    /// Checks if this AABB intersects `other`.
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x < other.max.x
            && self.max.x > other.min.x
            && self.min.y < other.max.y
            && self.max.y > other.min.y
            && self.min.z < other.max.z
            && self.max.z > other.min.z
    }

    /// Displaces this AABB by an offset vector.
    #[must_use]
    pub fn offset(&self, d: DVec3) -> Self {
        Self {
            min: self.min + d,
            max: self.max + d,
        }
    }
}

/// Instantaneous player movement input buttons.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PlayerInputState {
    /// Forward movement key (e.g. W).
    pub forward: bool,
    /// Backward movement key (e.g. S).
    pub backward: bool,
    /// Left strafe key (e.g. A).
    pub left: bool,
    /// Right strafe key (e.g. D).
    pub right: bool,
    /// Jump key (Space).
    pub jump: bool,
    /// Sneak / descend key (Shift).
    pub sneak: bool,
    /// Sprint key (Ctrl).
    pub sprint: bool,
}

/// Authoritative client-side player physics controller.
#[derive(Debug, Clone)]
pub struct PlayerPhysicsController {
    /// World position of player's feet.
    pub pos: DVec3,
    /// Current velocity vector in blocks per second.
    pub vel: Vec3,
    /// Active game mode.
    pub game_mode: GameMode,
    /// Whether player is currently in flying mode.
    pub flying: bool,
    /// Whether player has firm contact with solid ground.
    pub on_ground: bool,
    /// Double-tap Space timer for flight toggling (in seconds).
    pub space_tap_timer: f32,
    /// Space key state in previous frame.
    pub space_was_down: bool,
    /// Bounding box width and depth.
    pub width: f64,
    /// Bounding box height.
    pub height: f64,
    /// Eye height offset above feet position.
    pub eye_height: f64,
    /// Movement speed multiplier from active status effects.
    pub speed_multiplier: f32,
    /// Jump velocity bonus from active status effects.
    pub jump_boost: f32,
}

impl Default for PlayerPhysicsController {
    fn default() -> Self {
        Self::new(DVec3::new(0.0, 64.0, 0.0), GameMode::Creative)
    }
}

impl PlayerPhysicsController {
    /// Creates a new physics controller at `pos` with the given `game_mode`.
    #[must_use]
    pub fn new(pos: DVec3, game_mode: GameMode) -> Self {
        let flying = game_mode == GameMode::Creative;
        Self {
            pos,
            vel: Vec3::ZERO,
            game_mode,
            flying,
            on_ground: false,
            space_tap_timer: 0.0,
            space_was_down: false,
            width: PLAYER_WIDTH,
            height: PLAYER_HEIGHT,
            eye_height: EYE_HEIGHT,
            speed_multiplier: 1.0,
            jump_boost: 0.0,
        }
    }

    /// Returns the current player bounding box at feet position `pos`.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let half_w = self.width * 0.5;
        Aabb {
            min: DVec3::new(self.pos.x - half_w, self.pos.y, self.pos.z - half_w),
            max: DVec3::new(
                self.pos.x + half_w,
                self.pos.y + self.height,
                self.pos.z + half_w,
            ),
        }
    }

    /// Computes eye camera position in world space.
    #[must_use]
    pub fn eye_pos(&self) -> Vec3 {
        Vec3::new(
            self.pos.x as f32,
            (self.pos.y + self.eye_height) as f32,
            self.pos.z as f32,
        )
    }

    /// Sets feet position from an eye camera position.
    pub fn set_pos_from_eye(&mut self, eye: Vec3) {
        self.pos = DVec3::new(
            f64::from(eye.x),
            f64::from(eye.y) - self.eye_height,
            f64::from(eye.z),
        );
    }

    /// Toggles between Survival and Creative game modes.
    pub fn toggle_game_mode(&mut self) -> GameMode {
        match self.game_mode {
            GameMode::Survival => {
                self.game_mode = GameMode::Creative;
            }
            GameMode::Creative => {
                self.game_mode = GameMode::Survival;
                self.flying = false;
            }
        }
        self.game_mode
    }

    /// Advances the physics simulation by `dt` seconds using swept AABB voxel collision.
    ///
    /// `get_colliders` is invoked with `(x, y, z, &mut Vec<Aabb>)` to query solid block colliders.
    #[allow(clippy::too_many_lines)]
    pub fn update<F>(
        &mut self,
        dt: f32,
        yaw_deg: f32,
        input: &PlayerInputState,
        mut get_colliders: F,
    ) where
        F: FnMut(i32, i32, i32, &mut Vec<Aabb>),
    {
        let dt = dt.clamp(0.001, 0.05);

        // 1. Double-tap Space detection for flight toggle in Creative mode
        if input.jump && !self.space_was_down {
            if self.game_mode == GameMode::Creative && self.space_tap_timer > 0.0 {
                self.flying = !self.flying;
                self.space_tap_timer = 0.0;
                if self.flying {
                    self.vel.y = 0.0;
                }
            } else {
                self.space_tap_timer = 0.35;
            }
        }
        self.space_was_down = input.jump;
        self.space_tap_timer = (self.space_tap_timer - dt).max(0.0);

        // Survival mode never permits flight
        if self.game_mode == GameMode::Survival {
            self.flying = false;
        }

        let yaw_rad = yaw_deg.to_radians();
        let forward_h = Vec3::new(yaw_rad.cos(), 0.0, yaw_rad.sin()).normalize_or_zero();
        let right_h = Vec3::new(-yaw_rad.sin(), 0.0, yaw_rad.cos()).normalize_or_zero();

        let mut wish_dir = Vec3::ZERO;
        if input.forward {
            wish_dir += forward_h;
        }
        if input.backward {
            wish_dir -= forward_h;
        }
        if input.right {
            wish_dir += right_h;
        }
        if input.left {
            wish_dir -= right_h;
        }
        if wish_dir != Vec3::ZERO {
            wish_dir = wish_dir.normalize();
        }

        if self.flying {
            // Creative flight dynamics
            let fly_speed = if input.sprint {
                FLY_SPRINT_SPEED
            } else {
                FLY_SPEED
            };
            let mut target_vel = wish_dir * fly_speed;

            if input.jump {
                target_vel.y += fly_speed;
            }
            if input.sneak {
                target_vel.y -= fly_speed;
            }

            let accel = 18.0 * dt;
            if target_vel == Vec3::ZERO {
                let drag = 0.01f32.powf(dt);
                self.vel *= drag;
                if self.vel.length_squared() < 1e-4 {
                    self.vel = Vec3::ZERO;
                }
            } else {
                self.vel = self.vel.lerp(target_vel, accel.min(1.0));
            }

            self.pos += self.vel.as_dvec3() * f64::from(dt);
            self.on_ground = false;
        } else {
            // Walking / survival dynamics
            let base_speed = if input.sprint {
                SPRINT_SPEED
            } else if input.sneak {
                SNEAK_SPEED
            } else {
                WALK_SPEED
            };
            let move_speed = base_speed * self.speed_multiplier.max(0.1);

            let target_h = wish_dir * move_speed;
            let accel = if self.on_ground { 15.0 * dt } else { 3.5 * dt };

            if wish_dir == Vec3::ZERO {
                let friction = if self.on_ground {
                    0.0001f32.powf(dt)
                } else {
                    0.92f32.powf(dt)
                };
                self.vel.x *= friction;
                self.vel.z *= friction;
                if self.vel.x.abs() < 1e-3 {
                    self.vel.x = 0.0;
                }
                if self.vel.z.abs() < 1e-3 {
                    self.vel.z = 0.0;
                }
            } else {
                self.vel.x = self.vel.x.lerp(target_h.x, accel.min(1.0));
                self.vel.z = self.vel.z.lerp(target_h.z, accel.min(1.0));
            }

            // Jump
            if input.jump && self.on_ground {
                self.vel.y = JUMP_VELOCITY + self.jump_boost;
                self.on_ground = false;
            }

            // Gravity
            self.vel.y = (self.vel.y - GRAVITY * dt).max(-60.0);

            let wish_disp = self.vel.as_dvec3() * f64::from(dt);

            // Execute swept collision resolution
            self.move_and_slide(wish_disp, input.sneak, &mut get_colliders);
        }
    }

    /// Performs swept AABB movement with axis separation, sneak edge guard, and step-up support.
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    fn move_and_slide<F>(&mut self, wish_disp: DVec3, sneak: bool, get_colliders: &mut F)
    where
        F: FnMut(i32, i32, i32, &mut Vec<Aabb>),
    {
        let mut colliders = Vec::with_capacity(32);

        // Query broadphase bounding box encompassing starting and ending positions
        let broad_box = self.aabb().offset(wish_disp);
        let min_x = (self.pos.x - self.width * 0.5).min(broad_box.min.x).floor() as i32 - 1;
        let max_x = (self.pos.x + self.width * 0.5).max(broad_box.max.x).floor() as i32 + 1;
        let min_y = (self.pos.y).min(broad_box.min.y).floor() as i32 - 1;
        let max_y = (self.pos.y + self.height).max(broad_box.max.y).floor() as i32 + 1;
        let min_z = (self.pos.z - self.width * 0.5).min(broad_box.min.z).floor() as i32 - 1;
        let max_z = (self.pos.z + self.width * 0.5).max(broad_box.max.z).floor() as i32 + 1;

        for bx in min_x..=max_x {
            for by in min_y..=max_y {
                for bz in min_z..=max_z {
                    get_colliders(bx, by, bz, &mut colliders);
                }
            }
        }

        let mut player_box = self.aabb();

        // 1. Resolve Y axis first (vertical gravity & jump)
        let mut actual_dy = wish_disp.y;
        for c in &colliders {
            if actual_dy < 0.0 {
                // Falling down: test landing on top of collider
                if player_box.min.y >= c.max.y - 1e-4
                    && player_box.min.y + actual_dy < c.max.y
                    && player_box.max.x > c.min.x + 1e-4
                    && player_box.min.x < c.max.x - 1e-4
                    && player_box.max.z > c.min.z + 1e-4
                    && player_box.min.z < c.max.z - 1e-4
                {
                    actual_dy = actual_dy.max(c.max.y - player_box.min.y);
                }
            } else if actual_dy > 0.0 {
                // Jumping up: test hitting ceiling
                if player_box.max.y <= c.min.y + 1e-4
                    && player_box.max.y + actual_dy > c.min.y
                    && player_box.max.x > c.min.x + 1e-4
                    && player_box.min.x < c.max.x - 1e-4
                    && player_box.max.z > c.min.z + 1e-4
                    && player_box.min.z < c.max.z - 1e-4
                {
                    actual_dy = actual_dy.min(c.min.y - player_box.max.y);
                }
            }
        }

        player_box.min.y += actual_dy;
        player_box.max.y += actual_dy;

        if wish_disp.y < 0.0 && actual_dy > wish_disp.y + 1e-6 {
            self.on_ground = true;
            self.vel.y = 0.0;
        } else if wish_disp.y > 0.0 && actual_dy < wish_disp.y - 1e-6 {
            self.vel.y = 0.0;
        } else if actual_dy < -1e-4 {
            self.on_ground = false;
        }

        // 2. Sneak edge-guard: prevent walking off drops when sneaking on ground
        let (mut wish_dx, mut wish_dz) = (wish_disp.x, wish_disp.z);
        if self.on_ground && sneak {
            // Check if offset in X maintains solid ground support under player base center
            let test_x_box = Aabb::new(
                DVec3::new(
                    self.pos.x + wish_dx - 0.05,
                    self.pos.y - 0.5,
                    self.pos.z - 0.05,
                ),
                DVec3::new(
                    self.pos.x + wish_dx + 0.05,
                    self.pos.y + 0.05,
                    self.pos.z + 0.05,
                ),
            );
            let has_support_x = colliders.iter().any(|c| c.intersects(&test_x_box));
            if !has_support_x {
                wish_dx = 0.0;
            }

            // Check if offset in Z maintains solid ground support under player base center
            let test_z_box = Aabb::new(
                DVec3::new(
                    self.pos.x - 0.05,
                    self.pos.y - 0.5,
                    self.pos.z + wish_dz - 0.05,
                ),
                DVec3::new(
                    self.pos.x + 0.05,
                    self.pos.y + 0.05,
                    self.pos.z + wish_dz + 0.05,
                ),
            );
            let has_support_z = colliders.iter().any(|c| c.intersects(&test_z_box));
            if !has_support_z {
                wish_dz = 0.0;
            }
        }

        // 3. Resolve X and Z with optional step-up
        let box_pre_horizontal = player_box;
        let (normal_box, normal_dx, normal_dz) =
            Self::resolve_horizontal(player_box, wish_dx, wish_dz, &colliders);

        if self.on_ground
            && (normal_dx.abs() < wish_dx.abs() - 1e-4 || normal_dz.abs() < wish_dz.abs() - 1e-4)
        {
            // Collision occurred with a step or obstacle: attempt step-up
            let mut stepped_box = box_pre_horizontal;
            let mut can_step_up = true;

            // Check vertical headroom for step
            for c in &colliders {
                if stepped_box.max.y <= c.min.y + 1e-4
                    && stepped_box.max.y + STEP_HEIGHT > c.min.y
                    && stepped_box.max.x > c.min.x + 1e-4
                    && stepped_box.min.x < c.max.x - 1e-4
                    && stepped_box.max.z > c.min.z + 1e-4
                    && stepped_box.min.z < c.max.z - 1e-4
                {
                    can_step_up = false;
                    break;
                }
            }

            if can_step_up {
                stepped_box.min.y += STEP_HEIGHT;
                stepped_box.max.y += STEP_HEIGHT;

                let (mut elevated_box, stepped_dx, stepped_dz) =
                    Self::resolve_horizontal(stepped_box, wish_dx, wish_dz, &colliders);

                // Step back down to ground
                let mut step_down_dy = -STEP_HEIGHT;
                for c in &colliders {
                    if elevated_box.min.y >= c.max.y - 1e-4
                        && elevated_box.min.y + step_down_dy < c.max.y
                        && elevated_box.max.x > c.min.x + 1e-4
                        && elevated_box.min.x < c.max.x - 1e-4
                        && elevated_box.max.z > c.min.z + 1e-4
                        && elevated_box.min.z < c.max.z - 1e-4
                    {
                        step_down_dy = step_down_dy.max(c.max.y - elevated_box.min.y);
                    }
                }
                elevated_box.min.y += step_down_dy;
                elevated_box.max.y += step_down_dy;

                let normal_dist_sq = normal_dx * normal_dx + normal_dz * normal_dz;
                let step_dist_sq = stepped_dx * stepped_dx + stepped_dz * stepped_dz;

                if step_dist_sq > normal_dist_sq + 1e-4 {
                    player_box = elevated_box;
                    self.on_ground = true;
                } else {
                    player_box = normal_box;
                }
            } else {
                player_box = normal_box;
            }
        } else {
            player_box = normal_box;
        }

        // Commit final position to self.pos
        self.pos.x = f64::midpoint(player_box.min.x, player_box.max.x);
        self.pos.y = player_box.min.y;
        self.pos.z = f64::midpoint(player_box.min.z, player_box.max.z);
    }

    /// Resolves horizontal movement along X then Z against solid colliders.
    #[must_use]
    fn resolve_horizontal(
        mut b: Aabb,
        mut dx: f64,
        mut dz: f64,
        colliders: &[Aabb],
    ) -> (Aabb, f64, f64) {
        // Resolve X
        for c in colliders {
            if dx < 0.0 {
                if b.min.x >= c.max.x - 1e-4
                    && b.min.x + dx < c.max.x
                    && b.max.y > c.min.y + 1e-4
                    && b.min.y < c.max.y - 1e-4
                    && b.max.z > c.min.z + 1e-4
                    && b.min.z < c.max.z - 1e-4
                {
                    dx = dx.max(c.max.x - b.min.x);
                }
            } else if dx > 0.0
                && b.max.x <= c.min.x + 1e-4
                && b.max.x + dx > c.min.x
                && b.max.y > c.min.y + 1e-4
                && b.min.y < c.max.y - 1e-4
                && b.max.z > c.min.z + 1e-4
                && b.min.z < c.max.z - 1e-4
            {
                dx = dx.min(c.min.x - b.max.x);
            }
        }
        b.min.x += dx;
        b.max.x += dx;

        // Resolve Z
        for c in colliders {
            if dz < 0.0 {
                if b.min.z >= c.max.z - 1e-4
                    && b.min.z + dz < c.max.z
                    && b.max.x > c.min.x + 1e-4
                    && b.min.x < c.max.x - 1e-4
                    && b.max.y > c.min.y + 1e-4
                    && b.min.y < c.max.y - 1e-4
                {
                    dz = dz.max(c.max.z - b.min.z);
                }
            } else if dz > 0.0
                && b.max.z <= c.min.z + 1e-4
                && b.max.z + dz > c.min.z
                && b.max.x > c.min.x + 1e-4
                && b.min.x < c.max.x - 1e-4
                && b.max.y > c.min.y + 1e-4
                && b.min.y < c.max.y - 1e-4
            {
                dz = dz.min(c.min.z - b.max.z);
            }
        }
        b.min.z += dz;
        b.max.z += dz;

        (b, dx, dz)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_intersection() {
        let a = Aabb::new(DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 1.0, 1.0));
        let b = Aabb::new(DVec3::new(0.5, 0.5, 0.5), DVec3::new(1.5, 1.5, 1.5));
        let c = Aabb::new(DVec3::new(2.0, 0.0, 0.0), DVec3::new(3.0, 1.0, 1.0));

        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_gravity_and_ground_snap() {
        let mut controller =
            PlayerPhysicsController::new(DVec3::new(0.0, 65.0, 0.0), GameMode::Survival);
        let input = PlayerInputState::default();

        // Floor at Y = 64
        let colliders = |bx: i32, by: i32, bz: i32, out: &mut Vec<Aabb>| {
            if by == 63 && bx.abs() <= 2 && bz.abs() <= 2 {
                out.push(Aabb::from_block(bx, by, bz));
            }
        };

        // Fall under gravity over several frames
        for _ in 0..40 {
            controller.update(0.05, 0.0, &input, colliders);
        }

        assert!(controller.on_ground);
        assert!((controller.pos.y - 64.0).abs() < 1e-3);
        assert_eq!(controller.vel.y, 0.0);
    }

    #[test]
    fn test_jump_impulse() {
        let mut controller =
            PlayerPhysicsController::new(DVec3::new(0.0, 64.0, 0.0), GameMode::Survival);
        controller.on_ground = true;

        let input = PlayerInputState {
            jump: true,
            ..Default::default()
        };

        let colliders = |bx: i32, by: i32, bz: i32, out: &mut Vec<Aabb>| {
            if by == 63 && bx.abs() <= 2 && bz.abs() <= 2 {
                out.push(Aabb::from_block(bx, by, bz));
            }
        };

        controller.update(0.05, 0.0, &input, colliders);

        assert!(!controller.on_ground);
        assert!(controller.vel.y > 0.0);
        assert!(controller.pos.y > 64.0);
    }

    #[test]
    fn test_step_up_over_slab() {
        let mut controller =
            PlayerPhysicsController::new(DVec3::new(0.0, 64.0, 0.0), GameMode::Survival);
        controller.on_ground = true;

        let input = PlayerInputState {
            forward: true,
            ..Default::default()
        };

        // Ground at Y = 63, and half-slab at X = 1, Y = 64 with height 0.5
        let colliders = |bx: i32, by: i32, bz: i32, out: &mut Vec<Aabb>| {
            if by == 63 && bx.abs() <= 4 && bz.abs() <= 2 {
                out.push(Aabb::from_block(bx, by, bz));
            } else if (1..=4).contains(&bx) && by == 64 && bz.abs() <= 2 {
                out.push(Aabb::from_sub_box(bx, by, bz, [0, 0, 0], [16, 8, 16]));
            }
        };

        // Walk forward in +X direction (yaw = 0 -> cos(0) = 1 in X)
        for _ in 0..20 {
            controller.update(0.05, 0.0, &input, colliders);
        }

        // Player should have stepped up onto the 0.5-high slab
        assert!(controller.pos.x > 0.8);
        assert!(controller.pos.y >= 64.5 - 1e-3);
    }

    #[test]
    fn test_sneak_edge_guard() {
        let mut controller =
            PlayerPhysicsController::new(DVec3::new(0.5, 64.0, 0.5), GameMode::Survival);
        controller.on_ground = true;

        let input = PlayerInputState {
            forward: true,
            sneak: true,
            ..Default::default()
        };

        // Platform ends at X = 0 (only block 0, 63, 0 exists: X in [0.0, 1.0])
        let colliders = |bx: i32, by: i32, bz: i32, out: &mut Vec<Aabb>| {
            if bx == 0 && by == 63 && bz == 0 {
                out.push(Aabb::from_block(bx, by, bz));
            }
        };

        for _ in 0..30 {
            controller.update(0.05, 0.0, &input, colliders);
        }

        // Player should not have fallen off the edge
        assert!(controller.on_ground);
        assert!(controller.pos.x <= 1.0);
        assert_eq!(controller.pos.y, 64.0);
    }
}
