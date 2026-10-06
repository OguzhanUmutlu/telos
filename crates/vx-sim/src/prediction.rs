//! Client movement prediction buffer, authoritative server reconciliation, and visual error smoothing.

use std::collections::VecDeque;

use glam::{DVec3, Vec3};
use vx_protocol::messages::{InputFrame, S2cPlayerMovementAck};

use crate::movement::{MoveMode, MoveState, simulate_movement_step};

/// Capacity of the circular prediction buffer (128 ticks = ~6.4 seconds at 20 TPS).
pub const PREDICTION_BUFFER_CAPACITY: usize = 128;

/// A recorded prediction step tying an input frame to its resulting simulated state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictionEntry {
    /// Client tick index when the input was sampled and applied.
    pub tick: u32,
    /// Input frame that was simulated.
    pub input: InputFrame,
    /// Predicted state resulting from this step.
    pub state: MoveState,
}

/// Resulting metrics when reconciliation causes a rollback and replay.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReconciliationResult {
    /// Number of subsequent input frames replayed during reconciliation.
    pub replayed_ticks: usize,
    /// 3D positional deviation vector between the previous prediction and replayed outcome.
    pub error: Vec3,
    /// Whether this correction was triggered by a server-mandated teleport.
    pub is_teleport: bool,
}

/// Circular history buffer storing input frames and predicted movement states.
#[derive(Debug, Clone)]
pub struct PredictionBuffer {
    history: VecDeque<PredictionEntry>,
    current_state: MoveState,
}

impl Default for PredictionBuffer {
    fn default() -> Self {
        Self::new(MoveState::default())
    }
}

impl PredictionBuffer {
    /// Creates a new `PredictionBuffer` starting from the given initial `MoveState`.
    #[must_use]
    pub fn new(initial_state: MoveState) -> Self {
        Self {
            history: VecDeque::with_capacity(PREDICTION_BUFFER_CAPACITY),
            current_state: initial_state,
        }
    }

    /// Returns the current latest predicted state.
    #[must_use]
    pub const fn current_state(&self) -> &MoveState {
        &self.current_state
    }

    /// Directly overrides the current state (e.g. initial spawn or respawn).
    pub fn set_state(&mut self, state: MoveState) {
        self.current_state = state;
        self.history.clear();
    }

    /// Resets the prediction buffer to an authoritative state and clears history.
    pub fn reset_to(&mut self, state: MoveState) {
        self.set_state(state);
    }

    /// Returns up to 16 unacknowledged input frames strictly newer than `last_acked_tick`.
    #[must_use]
    pub fn unacked_frames(&self, last_acked_tick: u32) -> Vec<InputFrame> {
        let unacked: Vec<InputFrame> = self
            .history
            .iter()
            .filter(|e| e.tick > last_acked_tick)
            .map(|e| e.input)
            .collect();
        if unacked.len() > 16 {
            unacked[unacked.len() - 16..].to_vec()
        } else {
            unacked
        }
    }

    /// Advances local prediction by simulating one input frame and recording it in the buffer.
    pub fn push_and_predict(&mut self, input: InputFrame, mode: MoveMode, dt: f32) -> MoveState {
        let mut state = self.current_state;
        simulate_movement_step(&mut state, &input, mode, dt);

        if self.history.len() >= PREDICTION_BUFFER_CAPACITY {
            self.history.pop_front();
        }

        self.history.push_back(PredictionEntry {
            tick: input.tick,
            input,
            state,
        });

        self.current_state = state;
        state
    }

    /// Reconciles local prediction against an authoritative server movement acknowledgment.
    ///
    /// If the predicted state matches the server outcome within tolerance and no teleport occurred,
    /// history up to `ack.client_tick_ack` is cleanly discarded with zero replay cost.
    ///
    /// If a misprediction or teleport is detected, the state is rolled back to the server's
    /// authoritative state and all subsequent pending inputs are re-simulated.
    pub fn reconcile(
        &mut self,
        ack: &S2cPlayerMovementAck,
        mode: MoveMode,
        dt: f32,
    ) -> Option<ReconciliationResult> {
        let authoritative = MoveState {
            pos: DVec3::new(ack.x, ack.y, ack.z),
            vel: Vec3::new(ack.vx, ack.vy, ack.vz),
            yaw: ack.yaw,
            pitch: ack.pitch,
            on_ground: ack.on_ground,
            flying: ack.flying,
        };

        // Forced teleport or hard desync outside buffer range
        if ack.teleport_id > 0 {
            let old_pos = self.current_state.pos;
            self.current_state = authoritative;
            self.history.clear();
            return Some(ReconciliationResult {
                replayed_ticks: 0,
                error: (old_pos - authoritative.pos).as_vec3(),
                is_teleport: true,
            });
        }

        // Search for matching client tick in prediction history
        let match_idx = self
            .history
            .iter()
            .position(|e| e.tick == ack.client_tick_ack);

        let Some(idx) = match_idx else {
            // Tick is older than buffer history window; prune everything if newer
            if let Some(front) = self.history.front()
                && ack.client_tick_ack < front.tick
            {
                // Stale ack older than history window, ignore safely
                return None;
            }
            return None;
        };

        let predicted_at_tick = self.history[idx].state;

        // Check if client prediction was accurate within tolerances (10 mm pos, 0.05 m/s vel)
        let is_accurate = predicted_at_tick.is_approx_equal(&authoritative, 0.01, 0.05);

        if is_accurate {
            // Perfect prediction! Prune acknowledged entries from the front of the queue
            self.history.drain(..=idx);
            return None;
        }

        // Misprediction detected: Rollback & Replay
        let old_latest_pos = self.current_state.pos;
        let mut running_state = authoritative;

        // Prune history strictly prior to the acknowledged tick
        self.history.drain(..idx);

        // Update the acked entry itself
        if let Some(acked_entry) = self.history.front_mut() {
            acked_entry.state = running_state;
        }

        let replayed_count = self.history.len().saturating_sub(1);

        // Replay all subsequent inputs using updated authoritative baseline
        for entry in self.history.iter_mut().skip(1) {
            simulate_movement_step(&mut running_state, &entry.input, mode, dt);
            entry.state = running_state;
        }

        self.current_state = running_state;
        let error = (old_latest_pos - running_state.pos).as_vec3();

        Some(ReconciliationResult {
            replayed_ticks: replayed_count,
            error,
            is_teleport: false,
        })
    }
}

/// Smooths visual render position errors across frames during reconciliation.
#[derive(Debug, Clone, Copy, Default)]
pub struct VisualSmoothing {
    /// Current decaying visual offset applied to rendering camera.
    pub render_offset: Vec3,
}

impl VisualSmoothing {
    /// Creates a new visual smoothing tracker with zero offset.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            render_offset: Vec3::ZERO,
        }
    }

    /// Decays the visual offset towards zero using exponential half-life (~60 ms).
    pub fn update(&mut self, dt: f32) {
        if self.render_offset.length_squared() < 1e-6 {
            self.render_offset = Vec3::ZERO;
            return;
        }

        // Half-life of 60 ms
        let factor = 0.5f32.powf(dt / 0.06);
        self.render_offset *= factor;
    }

    /// Adds reconciliation error to the smoothing filter.
    ///
    /// Large corrections (> 2.0 blocks) or teleports snap immediately with zero smoothing.
    pub fn add_error(&mut self, error: Vec3, is_teleport: bool) {
        if is_teleport || error.length() > 2.0 {
            self.render_offset = Vec3::ZERO;
        } else {
            self.render_offset += error;
        }
    }

    /// Resets the visual smoothing offset immediately to zero.
    pub fn reset(&mut self) {
        self.render_offset = Vec3::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vx_protocol::messages::input_buttons;

    #[test]
    fn test_prediction_buffer_accurate_ack_no_replay() {
        let initial = MoveState::new(DVec3::new(0.0, 64.0, 0.0), 0.0, 0.0, true);
        let mut buffer = PredictionBuffer::new(initial);

        let input = InputFrame {
            tick: 1,
            buttons: input_buttons::FORWARD,
            yaw: 0,
            pitch: 0,
            hotbar: 0,
        };

        let predicted = buffer.push_and_predict(input, MoveMode::NoClipFly, 0.05);

        let ack = S2cPlayerMovementAck {
            client_tick_ack: 1,
            server_tick: 10,
            x: predicted.pos.x,
            y: predicted.pos.y,
            z: predicted.pos.z,
            vx: predicted.vel.x,
            vy: predicted.vel.y,
            vz: predicted.vel.z,
            yaw: predicted.yaw,
            pitch: predicted.pitch,
            on_ground: predicted.on_ground,
            flying: predicted.flying,
            teleport_id: 0,
        };

        let res = buffer.reconcile(&ack, MoveMode::NoClipFly, 0.05);
        assert_eq!(res, None);
        assert_eq!(buffer.history.len(), 0);
    }

    #[test]
    fn test_prediction_buffer_misprediction_replay() {
        let initial = MoveState::new(DVec3::new(0.0, 64.0, 0.0), 0.0, 0.0, true);
        let mut buffer = PredictionBuffer::new(initial);

        // Push 3 inputs
        for tick in 1..=3 {
            let input = InputFrame {
                tick,
                buttons: input_buttons::FORWARD,
                yaw: 0,
                pitch: 0,
                hotbar: 0,
            };
            buffer.push_and_predict(input, MoveMode::NoClipFly, 0.05);
        }

        assert_eq!(buffer.history.len(), 3);

        // Server reports tick 1 had a disturbance (e.g. knockback/external impulse)
        let ack = S2cPlayerMovementAck {
            client_tick_ack: 1,
            server_tick: 5,
            x: 0.0,
            y: 65.0, // Server moved player up by 1 block
            z: 0.0,
            vx: 0.0,
            vy: 2.0,
            vz: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            on_ground: false,
            flying: true,
            teleport_id: 0,
        };

        let res = buffer.reconcile(&ack, MoveMode::NoClipFly, 0.05);
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res.replayed_ticks, 2);
        assert!(!res.is_teleport);
        // Current state position Y should reflect the replayed upward perturbation
        assert!(buffer.current_state().pos.y > 64.5);
    }

    #[test]
    fn test_visual_smoothing_decay() {
        let mut smoothing = VisualSmoothing::new();
        smoothing.add_error(Vec3::new(1.0, 0.0, 0.0), false);
        assert!((smoothing.render_offset.x - 1.0).abs() < 1e-4);

        smoothing.update(0.06); // 1 half-life
        assert!((smoothing.render_offset.x - 0.5).abs() < 0.05);

        smoothing.update(0.30); // multiple half-lives
        assert!(smoothing.render_offset.length() < 0.05);
    }
}
