//! Coordinate argument parsing for absolute, relative (`~`), and local (`^`) vectors.

use crate::command::reader::{CommandSyntaxError, StringReader};
use glam::{Vec2, Vec3};

/// A single axis coordinate value which can be absolute, relative (`~`), or local (`^`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoordinateArg {
    /// Absolute coordinate in world space.
    Absolute(f32),
    /// Relative offset from the executor's position (`~<offset>`).
    Relative(f32),
    /// Local offset aligned with the executor's view rotation (`^<offset>`).
    Local(f32),
}

impl CoordinateArg {
    /// Resolves an absolute or relative coordinate given a base coordinate value.
    ///
    /// # Panics
    /// Panics if called on a `CoordinateArg::Local`. Local coordinates must be resolved
    /// as a unified 3D vector via [`Vec3Arg::resolve`].
    #[must_use]
    pub fn resolve(&self, base: f32) -> f32 {
        match *self {
            Self::Absolute(val) => val,
            Self::Relative(offset) => base + offset,
            Self::Local(_) => panic!("Local coordinates cannot be resolved along a single axis"),
        }
    }

    /// Returns true if this coordinate is a local (`^`) coordinate.
    #[must_use]
    pub const fn is_local(&self) -> bool {
        matches!(self, Self::Local(_))
    }

    /// Parses a single coordinate component from the string reader.
    pub fn parse(reader: &mut StringReader) -> Result<Self, CommandSyntaxError> {
        reader.skip_whitespace();
        let start = reader.cursor();

        if let Some(c) = reader.peek() {
            if c == '~' {
                reader.read_char();
                // Check if an offset follows immediately without whitespace
                if matches!(reader.peek(), Some(next) if next.is_ascii_digit() || next == '-' || next == '+')
                {
                    let offset = reader.read_f32()?;
                    return Ok(Self::Relative(offset));
                }
                return Ok(Self::Relative(0.0));
            } else if c == '^' {
                reader.read_char();
                if matches!(reader.peek(), Some(next) if next.is_ascii_digit() || next == '-' || next == '+')
                {
                    let offset = reader.read_f32()?;
                    return Ok(Self::Local(offset));
                }
                return Ok(Self::Local(0.0));
            }
        }

        let val = reader.read_f32().map_err(|_| CommandSyntaxError::Custom {
            cursor: start,
            message: "Expected coordinate number, '~', or '^'".to_string(),
        })?;

        Ok(Self::Absolute(val))
    }
}

/// A 3D coordinate vector argument composed of three coordinate axes (X, Y, Z).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3Arg {
    /// X axis coordinate.
    pub x: CoordinateArg,
    /// Y axis coordinate.
    pub y: CoordinateArg,
    /// Z axis coordinate.
    pub z: CoordinateArg,
}

impl Vec3Arg {
    /// Creates a new `Vec3Arg` from 3 components.
    #[must_use]
    pub const fn new(x: CoordinateArg, y: CoordinateArg, z: CoordinateArg) -> Self {
        Self { x, y, z }
    }

    /// Parses a 3D coordinate vector (`X Y Z`) from the reader.
    ///
    /// Validates that local (`^`) and relative/absolute coordinates are not mixed.
    pub fn parse(reader: &mut StringReader) -> Result<Self, CommandSyntaxError> {
        let x = CoordinateArg::parse(reader)?;

        reader.skip_whitespace();
        let y = CoordinateArg::parse(reader)?;

        reader.skip_whitespace();
        let z = CoordinateArg::parse(reader)?;

        let locals = u8::from(x.is_local()) + u8::from(y.is_local()) + u8::from(z.is_local());
        if locals > 0 && locals < 3 {
            return Err(CommandSyntaxError::Custom {
                cursor: reader.cursor(),
                message: "Cannot mix local coordinates (^) with absolute/relative coordinates"
                    .to_string(),
            });
        }

        Ok(Self { x, y, z })
    }

    /// Resolves the coordinate vector into world-space coordinates.
    ///
    /// - For absolute/relative: resolves each axis independently against `origin`.
    /// - For local (`^left ^up ^forward`): resolves rotated by `yaw_pitch` (in degrees).
    #[must_use]
    pub fn resolve(&self, origin: Vec3, yaw_pitch: Vec2) -> Vec3 {
        if self.x.is_local() {
            let left_val = match self.x {
                CoordinateArg::Local(v) => v,
                _ => 0.0,
            };
            let up_val = match self.y {
                CoordinateArg::Local(v) => v,
                _ => 0.0,
            };
            let forward_val = match self.z {
                CoordinateArg::Local(v) => v,
                _ => 0.0,
            };

            let yaw_rad = yaw_pitch.x.to_radians();
            let pitch_rad = yaw_pitch.y.to_radians();

            // Forward vector in Classic Voxel coordinate system (Y is up, -Z is forward when yaw=0)
            let fwd = Vec3::new(
                -yaw_rad.sin() * pitch_rad.cos(),
                -pitch_rad.sin(),
                yaw_rad.cos() * pitch_rad.cos(),
            )
            .normalize_or_zero();

            // Right vector (perpendicular to forward on horizontal plane)
            let right = Vec3::new(yaw_rad.cos(), 0.0, yaw_rad.sin()).normalize_or_zero();
            // Local up vector
            let up = right.cross(fwd).normalize_or_zero();
            let left = -right;

            origin + (left * left_val) + (up * up_val) + (fwd * forward_val)
        } else {
            Vec3::new(
                self.x.resolve(origin.x),
                self.y.resolve(origin.y),
                self.z.resolve(origin.z),
            )
        }
    }
}
