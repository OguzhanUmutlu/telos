//! Deterministic player movement physics, coordinate quantization, and step simulation.

use glam::{DVec3, FloatExt, Vec3};
use vx_protocol::messages::{InputFrame, input_buttons};

/// Maximum allowable player speed in blocks per second before triggering server anti-cheat clamp.
pub const MAX_LEGAL_SPEED: f32 = 120.0;

/// Movement physics mode for player simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MoveMode {
    /// Frictionless free-flight spectator / creative camera without block collisions.
    #[default]
    NoClipFly,
    /// Ground walking and jumping physics with gravity and drag.
    Walk,
}

/// Authoritative movement state representing position, velocity, and orientation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveState {
    /// World position (double precision).
    pub pos: DVec3,
    /// World velocity in blocks per second.
    pub vel: Vec3,
    /// Camera look yaw in degrees `[0.0, 360.0)`.
    pub yaw: f32,
    /// Camera look pitch in degrees `[-90.0, 90.0]`.
    pub pitch: f32,
    /// Whether player has ground contact.
    pub on_ground: bool,
    /// Whether player is currently in flying mode.
    pub flying: bool,
}

impl Default for MoveState {
    fn default() -> Self {
        Self {
            pos: DVec3::ZERO,
            vel: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            on_ground: false,
            flying: true,
        }
    }
}

impl MoveState {
    /// Creates a new `MoveState` with given position, orientation, and flight flag.
    #[must_use]
    pub const fn new(pos: DVec3, yaw: f32, pitch: f32, flying: bool) -> Self {
        Self {
            pos,
            vel: Vec3::ZERO,
            yaw,
            pitch,
            on_ground: false,
            flying,
        }
    }

    /// Checks if this state is approximately equal to `other` within strict tolerances.
    #[must_use]
    pub fn is_approx_equal(&self, other: &Self, pos_tol: f64, vel_tol: f32) -> bool {
        self.pos.distance_squared(other.pos) <= pos_tol * pos_tol
            && self.vel.distance_squared(other.vel) <= vel_tol * vel_tol
            && self.flying == other.flying
            && self.on_ground == other.on_ground
    }
}

/// Quantizes an angle in degrees into a 16-bit integer (`65536 = 360` degrees).
#[must_use]
pub fn quantize_yaw(degrees: f32) -> u16 {
    let norm = degrees.rem_euclid(360.0) / 360.0;
    (norm * 65536.0) as u16
}

/// Dequantizes a 16-bit yaw integer into degrees `[0.0, 360.0)`.
#[must_use]
pub fn dequantize_yaw(quant: u16) -> f32 {
    (f32::from(quant) / 65536.0) * 360.0
}

/// Quantizes a pitch angle in `[-90.0, 90.0]` into a signed 16-bit integer (`32767 = 90` degrees).
#[must_use]
pub fn quantize_pitch(degrees: f32) -> i16 {
    let clamped = degrees.clamp(-90.0, 90.0) / 90.0;
    (clamped * 32767.0) as i16
}

/// Dequantizes a signed 16-bit pitch integer into degrees in `[-90.0, 90.0]`.
#[must_use]
pub fn dequantize_pitch(quant: i16) -> f32 {
    (f32::from(quant) / 32767.0) * 90.0
}

/// Computes a 3D unit direction vector from yaw and pitch angles in degrees.
#[must_use]
pub fn angles_to_direction(yaw_deg: f32, pitch_deg: f32) -> Vec3 {
    let yaw_rad = yaw_deg.to_radians();
    let pitch_rad = pitch_deg.to_radians();
    Vec3::new(
        yaw_rad.cos() * pitch_rad.cos(),
        pitch_rad.sin(),
        yaw_rad.sin() * pitch_rad.cos(),
    )
    .normalize_or_zero()
}

/// Simulates a single deterministic movement step for the given input frame.
///
/// This function is executed identically on the client during local prediction
/// and on the authoritative server during tick processing, ensuring exact alignment.
#[allow(clippy::too_many_lines, clippy::if_not_else)]
pub fn simulate_movement_step(state: &mut MoveState, input: &InputFrame, mode: MoveMode, dt: f32) {
    // 1. Update orientation directly from quantized input angles
    state.yaw = dequantize_yaw(input.yaw);
    state.pitch = dequantize_pitch(input.pitch);

    let buttons = input.buttons;
    let is_fwd = (buttons & input_buttons::FORWARD) != 0;
    let is_back = (buttons & input_buttons::BACK) != 0;
    let is_left = (buttons & input_buttons::LEFT) != 0;
    let is_right = (buttons & input_buttons::RIGHT) != 0;
    let is_jump = (buttons & input_buttons::JUMP) != 0;
    let is_sneak = (buttons & input_buttons::SNEAK) != 0;
    let is_sprint = (buttons & input_buttons::SPRINT) != 0;
    let is_fly_up = (buttons & input_buttons::FLY_UP) != 0;
    let is_fly_down = (buttons & input_buttons::FLY_DOWN) != 0;

    match mode {
        MoveMode::NoClipFly => {
            state.flying = true;
            state.on_ground = false;

            let yaw_rad = state.yaw.to_radians();
            let pitch_rad = state.pitch.to_radians();

            let forward = Vec3::new(
                yaw_rad.cos() * pitch_rad.cos(),
                pitch_rad.sin(),
                yaw_rad.sin() * pitch_rad.cos(),
            )
            .normalize_or_zero();

            let right = Vec3::new(-yaw_rad.sin(), 0.0, yaw_rad.cos()).normalize_or_zero();
            let world_up = Vec3::Y;

            let mut wish = Vec3::ZERO;
            if is_fwd {
                wish += forward;
            }
            if is_back {
                wish -= forward;
            }
            if is_right {
                wish += right;
            }
            if is_left {
                wish -= right;
            }
            if is_fly_up || is_jump {
                wish += world_up;
            }
            if is_fly_down || is_sneak {
                wish -= world_up;
            }

            let base_speed = if is_sprint { 35.0 } else { 14.0 };
            let accel = 20.0 * dt;

            if wish != Vec3::ZERO {
                let target_vel = wish.normalize() * base_speed;
                state.vel = state.vel.lerp(target_vel, accel.min(1.0));
            } else {
                let drag = 0.005f32.powf(dt);
                state.vel *= drag;
                if state.vel.length_squared() < 1e-4 {
                    state.vel = Vec3::ZERO;
                }
            }

            // Clamp velocity to legal anti-cheat limit
            if state.vel.length() > MAX_LEGAL_SPEED {
                state.vel = state.vel.normalize() * MAX_LEGAL_SPEED;
            }

            state.pos += state.vel.as_dvec3() * f64::from(dt);
        }
        MoveMode::Walk => {
            state.flying = false;
            let yaw_rad = state.yaw.to_radians();
            let forward_h = Vec3::new(yaw_rad.cos(), 0.0, yaw_rad.sin()).normalize_or_zero();
            let right_h = Vec3::new(-yaw_rad.sin(), 0.0, yaw_rad.cos()).normalize_or_zero();

            let mut wish_h = Vec3::ZERO;
            if is_fwd {
                wish_h += forward_h;
            }
            if is_back {
                wish_h -= forward_h;
            }
            if is_right {
                wish_h += right_h;
            }
            if is_left {
                wish_h -= right_h;
            }

            let speed = if is_sprint { 5.612 } else { 4.317 };
            if wish_h != Vec3::ZERO {
                let target_h = wish_h.normalize() * speed;
                state.vel.x = state.vel.x.lerp(target_h.x, (15.0 * dt).min(1.0));
                state.vel.z = state.vel.z.lerp(target_h.z, (15.0 * dt).min(1.0));
            } else {
                let friction = 0.01f32.powf(dt);
                state.vel.x *= friction;
                state.vel.z *= friction;
            }

            // Jump
            if is_jump && state.on_ground {
                state.vel.y = 8.4;
                state.on_ground = false;
            }

            // Gravity & air drag
            if !state.on_ground {
                state.vel.y = (state.vel.y - 32.0 * dt).max(-60.0);
            }

            // Clamp velocity
            if state.vel.length() > MAX_LEGAL_SPEED {
                state.vel = state.vel.normalize() * MAX_LEGAL_SPEED;
            }

            state.pos += state.vel.as_dvec3() * f64::from(dt);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_round_trip() {
        for deg in [0.0, 45.0, 90.0, 180.0, 270.0, 359.9] {
            let q = quantize_yaw(deg);
            let d = dequantize_yaw(q);
            assert!((d - deg).abs() < 0.01);
        }

        for pitch in [-90.0, -45.0, 0.0, 30.0, 89.9] {
            let q = quantize_pitch(pitch);
            let d = dequantize_pitch(q);
            assert!((d - pitch).abs() < 0.01);
        }
    }

    #[test]
    fn test_deterministic_movement() {
        let mut state_a = MoveState::new(DVec3::new(10.0, 64.0, 10.0), 90.0, 0.0, true);
        let mut state_b = state_a;

        let input = InputFrame {
            tick: 1,
            buttons: input_buttons::FORWARD | input_buttons::SPRINT,
            yaw: quantize_yaw(90.0),
            pitch: quantize_pitch(0.0),
            hotbar: 0,
        };

        for _ in 0..10 {
            simulate_movement_step(&mut state_a, &input, MoveMode::NoClipFly, 0.05);
            simulate_movement_step(&mut state_b, &input, MoveMode::NoClipFly, 0.05);
        }

        assert_eq!(state_a, state_b);
        assert!(state_a.pos.z > 10.0);
    }
}
