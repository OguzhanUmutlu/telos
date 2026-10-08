//! Adaptive low-latency jitter buffer for real-time voice streaming.
//!
//! Buffers 2–3 frames (40–60 ms) to smooth out network jitter, reorders packets,
//! and emits `None` gaps to trigger Packet Loss Concealment (PLC).

use std::collections::BTreeMap;

/// Bounded jitter buffer holding sequenced voice packets.
#[derive(Debug, Clone)]
pub struct VoiceJitterBuffer {
    /// Sequenced packets: sequence -> raw compressed frame.
    packets: BTreeMap<u64, Vec<u8>>,
    /// Expected sequence number of next frame to decode.
    expected_seq: Option<u64>,
    /// Minimum initial frames required before starting playback (default: 2 frames = 40 ms).
    target_depth: usize,
    /// Maximum buffered frames before dropping oldest to prevent latency buildup (default: 6 frames = 120 ms).
    max_capacity: usize,
    /// Whether the buffer has finished prebuffering.
    is_playing: bool,
}

impl VoiceJitterBuffer {
    /// Creates a new jitter buffer with default 2-frame target depth and 6-frame cap.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(2, 6)
    }

    /// Creates a jitter buffer with custom target depth and maximum capacity.
    #[must_use]
    pub fn with_config(target_depth: usize, max_capacity: usize) -> Self {
        Self {
            packets: BTreeMap::new(),
            expected_seq: None,
            target_depth: target_depth.max(1),
            max_capacity: max_capacity.max(target_depth + 1),
            is_playing: false,
        }
    }

    /// Pushes a new incoming network voice packet into the jitter buffer.
    pub fn push(&mut self, sequence: u64, data: Vec<u8>) {
        if let Some(expected) = self.expected_seq {
            // Discard packets that arrived too late to be played
            if sequence < expected {
                return;
            }
        }

        self.packets.insert(sequence, data);

        // Cap buffer to avoid unbounded latency creep
        while self.packets.len() > self.max_capacity {
            self.packets.pop_first();
        }

        // Start playing once target depth is reached
        if !self.is_playing && self.packets.len() >= self.target_depth {
            self.is_playing = true;
            if self.expected_seq.is_none() {
                self.expected_seq = self.packets.keys().next().copied();
            }
        }
    }

    /// Pops the next frame to be decoded.
    ///
    /// - Returns `Some(Some(bytes))` for an available in-order packet.
    /// - Returns `Some(None)` when a packet in the sequence was dropped (triggers PLC).
    /// - Returns `None` when buffer is empty or prebuffering.
    pub fn pop(&mut self) -> Option<Option<Vec<u8>>> {
        if !self.is_playing {
            return None;
        }

        let expected = match self.expected_seq {
            Some(seq) => seq,
            None => {
                if let Some(&first_seq) = self.packets.keys().next() {
                    self.expected_seq = Some(first_seq);
                    first_seq
                } else {
                    self.is_playing = false;
                    return None;
                }
            }
        };

        if let Some(data) = self.packets.remove(&expected) {
            self.expected_seq = Some(expected + 1);
            Some(Some(data))
        } else if let Some(&next_available) = self.packets.keys().next() {
            if next_available > expected {
                // Lost packet detected: bump sequence and request PLC
                self.expected_seq = Some(expected + 1);
                Some(None)
            } else {
                // Out-of-order stale packet in map, discard and re-pop
                self.packets.remove(&next_available);
                self.pop()
            }
        } else {
            // Underrun: no packets left, reset playback flag until prebuffered again
            self.is_playing = false;
            self.expected_seq = None;
            None
        }
    }

    /// Clears the jitter buffer and resets sequence tracking.
    pub fn reset(&mut self) {
        self.packets.clear();
        self.expected_seq = None;
        self.is_playing = false;
    }

    /// Returns the number of currently buffered packets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.packets.len()
    }

    /// Returns `true` if the jitter buffer contains no packets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.packets.is_empty()
    }
}

impl Default for VoiceJitterBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jitter_buffer_in_order() {
        let mut jb = VoiceJitterBuffer::with_config(2, 6);
        jb.push(100, vec![1]);
        assert_eq!(jb.pop(), None, "should buffer until target depth (2)");

        jb.push(101, vec![2]);
        // Now has 2 frames, should start playback
        assert_eq!(jb.pop(), Some(Some(vec![1])));
        assert_eq!(jb.pop(), Some(Some(vec![2])));
        assert_eq!(jb.pop(), None, "buffer exhausted");
    }

    #[test]
    fn test_jitter_buffer_packet_loss_plc() {
        let mut jb = VoiceJitterBuffer::with_config(2, 6);
        jb.push(10, vec![10]);
        jb.push(12, vec![12]); // Packet 11 is lost

        assert_eq!(jb.pop(), Some(Some(vec![10])));
        // Next expected is 11, but only 12 is available -> emits Some(None) for PLC
        assert_eq!(
            jb.pop(),
            Some(None),
            "should trigger PLC for missing seq 11"
        );
        // Next is 12
        assert_eq!(jb.pop(), Some(Some(vec![12])));
    }

    #[test]
    fn test_jitter_buffer_reordering() {
        let mut jb = VoiceJitterBuffer::with_config(2, 6);
        // Packet 20 arrives before 19
        jb.push(20, vec![20]);
        jb.push(19, vec![19]);

        assert_eq!(jb.pop(), Some(Some(vec![19])));
        assert_eq!(jb.pop(), Some(Some(vec![20])));
    }
}
