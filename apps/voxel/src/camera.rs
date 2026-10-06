//! Camera, frustum culling, and 3D fly controller.

use glam::{Mat4, Vec3, Vec4};

/// 6 planes of a viewing frustum for AABB visibility testing.
#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    /// Planes stored as `(normal.x, normal.y, normal.z, d)` where `n.dot(p) + d = 0`.
    planes: [Vec4; 6],
}

impl Frustum {
    /// Extracts the 6 frustum clipping planes from a combined View-Projection matrix.
    ///
    /// Assumes Vulkan clip volume: $X, Y \in [-W, W]$ and $Z \in [0, W]$.
    #[must_use]
    pub fn from_view_proj(vp: &Mat4) -> Self {
        let r0 = vp.row(0);
        let r1 = vp.row(1);
        let r2 = vp.row(2);
        let r3 = vp.row(3);

        let mut planes = [
            r3 + r0, // Left:   w + x >= 0
            r3 - r0, // Right:  w - x >= 0
            r3 + r1, // Bottom: w + y >= 0
            r3 - r1, // Top:    w - y >= 0
            r3 - r2, // Near:   w - z >= 0 (in reversed-Z, z = w at near plane)
            r2,      // Far:    z >= 0     (in reversed-Z, z = 0 at far plane)
        ];

        // Normalize plane equations so distance calculations are metric
        for plane in &mut planes {
            let normal_len = Vec3::new(plane.x, plane.y, plane.z).length();
            if normal_len > 0.0 {
                *plane /= normal_len;
            }
        }

        Self { planes }
    }

    /// Tests if an axis-aligned bounding box (AABB) intersects or is inside the frustum.
    ///
    /// Returns `true` if the box is potentially visible, `false` if it is completely culled.
    #[must_use]
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            let normal = Vec3::new(plane.x, plane.y, plane.z);
            let d = plane.w;

            // Positive vertex: corner furthest along the plane normal
            let p = Vec3::new(
                if normal.x >= 0.0 { max.x } else { min.x },
                if normal.y >= 0.0 { max.y } else { min.y },
                if normal.z >= 0.0 { max.z } else { min.z },
            );

            if normal.dot(p) + d < 0.0 {
                return false;
            }
        }
        true
    }
}

/// 3D perspective camera with yaw and pitch orientation.
#[derive(Debug, Clone)]
pub struct Camera {
    /// World position of the camera eye.
    pub position: Vec3,
    /// Yaw angle in radians (horizontal look direction).
    pub yaw: f32,
    /// Pitch angle in radians (vertical look direction, clamped to ±89°).
    pub pitch: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near clipping plane distance.
    pub z_near: f32,
    /// Far clipping plane distance.
    pub z_far: f32,
}

impl Camera {
    /// Creates a new camera at the specified position.
    #[must_use]
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            yaw: -std::f32::consts::FRAC_PI_2, // Looking along -Z by default
            pitch: 0.0,
            fov_y: 70.0f32.to_radians(),
            z_near: 0.1,
            z_far: 1000.0,
        }
    }

    /// Unit direction vector the camera is facing.
    #[must_use]
    pub fn forward(&self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        Vec3::new(
            self.yaw.cos() * cos_pitch,
            self.pitch.sin(),
            self.yaw.sin() * cos_pitch,
        )
        .normalize()
    }

    /// Unit direction vector pointing to the camera's right.
    #[must_use]
    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    /// View matrix for the camera transform.
    #[must_use]
    pub fn view_matrix(&self) -> Mat4 {
        let dir = self.forward();
        Mat4::look_to_rh(self.position, dir, Vec3::Y)
    }

    /// Perspective projection matrix for the given aspect ratio ($W/H$) with reversed-Z (ADR-02).
    ///
    /// Maps $z_{\text{near}} \to 1.0$ and $z_{\text{far}} \to 0.0$ for optimal floating-point depth precision.
    #[must_use]
    pub fn projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        let f = 1.0 / (self.fov_y * 0.5).tan();
        Mat4::from_cols(
            Vec4::new(f / aspect_ratio, 0.0, 0.0, 0.0),
            Vec4::new(0.0, f, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, -1.0),
            Vec4::new(0.0, 0.0, self.z_near, 0.0),
        )
    }

    /// Combined View-Projection matrix.
    #[must_use]
    pub fn view_proj_matrix(&self, aspect_ratio: f32) -> Mat4 {
        self.projection_matrix(aspect_ratio) * self.view_matrix()
    }
}

/// First-person fly camera controller handling WASD movement and mouse look.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug)]
pub struct FlyController {
    /// Move forward state (W).
    pub forward: bool,
    /// Move backward state (S).
    pub backward: bool,
    /// Strafe left state (A).
    pub left: bool,
    /// Strafe right state (D).
    pub right: bool,
    /// Fly up state (Space).
    pub up: bool,
    /// Fly down state (Shift).
    pub down: bool,
    /// Sprint modifier state (Ctrl).
    pub sprint: bool,

    /// Base movement speed in blocks per second.
    pub base_speed: f32,
    /// Sprint speed multiplier.
    pub sprint_multiplier: f32,
    /// Mouse sensitivity radians per raw delta unit.
    pub mouse_sensitivity: f32,
    /// Whether pointer lock / mouse capture is active.
    pub mouse_captured: bool,
}

impl Default for FlyController {
    fn default() -> Self {
        Self {
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            sprint: false,
            base_speed: 25.0,
            sprint_multiplier: 3.0,
            mouse_sensitivity: 0.0025,
            mouse_captured: false,
        }
    }
}

impl FlyController {
    /// Processes mouse motion delta when captured.
    pub fn on_mouse_move(&mut self, camera: &mut Camera, delta_x: f64, delta_y: f64) {
        if !self.mouse_captured {
            return;
        }

        #[allow(clippy::cast_possible_truncation)]
        let dx = delta_x as f32 * self.mouse_sensitivity;
        #[allow(clippy::cast_possible_truncation)]
        let dy = delta_y as f32 * self.mouse_sensitivity;

        camera.yaw += dx;
        camera.pitch -= dy;

        // Clamp pitch to prevent flipping upside down
        let max_pitch = 89.0f32.to_radians();
        camera.pitch = camera.pitch.clamp(-max_pitch, max_pitch);
    }

    /// Integrates keyboard movement over delta time.
    pub fn update(&self, camera: &mut Camera, dt: f32) {
        let speed = if self.sprint {
            self.base_speed * self.sprint_multiplier
        } else {
            self.base_speed
        };

        let forward = camera.forward();
        let right = camera.right();

        let mut move_dir = Vec3::ZERO;

        if self.forward {
            move_dir += forward;
        }
        if self.backward {
            move_dir -= forward;
        }
        if self.right {
            move_dir += right;
        }
        if self.left {
            move_dir -= right;
        }
        if self.up {
            move_dir += Vec3::Y;
        }
        if self.down {
            move_dir -= Vec3::Y;
        }

        if move_dir.length_squared() > 0.0 {
            camera.position += move_dir.normalize() * speed * dt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_direction_vectors() {
        let mut camera = Camera::new(Vec3::ZERO);
        camera.yaw = -std::f32::consts::FRAC_PI_2; // -90 deg -> look along -Z
        camera.pitch = 0.0;

        let fwd = camera.forward();
        let right = camera.right();

        assert!((fwd.x).abs() < 1e-4);
        assert!((fwd.y).abs() < 1e-4);
        assert!((fwd.z - (-1.0)).abs() < 1e-4);

        assert!((right.x - 1.0).abs() < 1e-4);
        assert!((right.y).abs() < 1e-4);
        assert!((right.z).abs() < 1e-4);

        // Perpendicularity
        assert!(fwd.dot(right).abs() < 1e-5);
    }

    #[test]
    fn test_frustum_culling_front_and_behind() {
        let mut camera = Camera::new(Vec3::new(0.0, 0.0, 0.0));
        camera.yaw = -std::f32::consts::FRAC_PI_2; // Looking along -Z
        camera.pitch = 0.0;

        let vp = camera.view_proj_matrix(16.0 / 9.0);
        let frustum = Frustum::from_view_proj(&vp);

        // Target box in front of camera
        let visible_box_min = Vec3::new(-5.0, -5.0, -25.0);
        let visible_box_max = Vec3::new(5.0, 5.0, -15.0);
        assert!(frustum.intersects_aabb(visible_box_min, visible_box_max));

        // Target box behind camera
        let behind_box_min = Vec3::new(-5.0, -5.0, 15.0);
        let behind_box_max = Vec3::new(5.0, 5.0, 25.0);
        assert!(!frustum.intersects_aabb(behind_box_min, behind_box_max));

        // Target box far off to the side
        let side_box_min = Vec3::new(500.0, -5.0, -25.0);
        let side_box_max = Vec3::new(510.0, 5.0, -15.0);
        assert!(!frustum.intersects_aabb(side_box_min, side_box_max));
    }
}
