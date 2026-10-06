//! Comprehensive unit tests for CPU particle simulation and GPU extraction in telos-sim.

use glam::Vec3;
use telos_protocol::messages::play::{ParticleEffectKind, S2cParticleEvent};
use telos_sim::particle::{ParticleGpu, ParticleSystem};

#[test]
fn test_particle_system_lifecycle_and_aging() {
    let mut sys = ParticleSystem::new(1024);
    assert_eq!(sys.len(), 0);
    assert!(sys.is_empty());

    sys.spawn_torch_smoke(Vec3::new(10.0, 64.0, 10.0));
    assert_eq!(sys.len(), 1);

    // Initial state check
    let mut gpu_particles = Vec::new();
    sys.extract_gpu_particles(&mut gpu_particles);
    assert_eq!(gpu_particles.len(), 1);
    assert_eq!(gpu_particles[0].tex_source, 0); // Particle texture array

    // Advance 0.5 seconds
    sys.tick(0.5, |_x, _y, _z| false);
    assert_eq!(sys.len(), 1);

    sys.extract_gpu_particles(&mut gpu_particles);
    // Size should have expanded and position should have risen
    assert!(gpu_particles[0].size > 0.12);
    assert!(gpu_particles[0].pos[1] > 64.4);

    // Advance past max_age (1.5s total)
    sys.tick(1.5, |_x, _y, _z| false);
    assert_eq!(sys.len(), 0);
    assert!(sys.is_empty());
}

#[test]
fn test_block_break_debris_and_ground_collision() {
    let mut sys = ParticleSystem::new(1024);
    let center = Vec3::new(5.0, 65.0, 5.0);
    sys.spawn_block_break(center, 2, [100, 150, 50, 255], 20);
    assert_eq!(sys.len(), 20);

    // Solid floor at y <= 64
    let is_solid = |_x: i32, y: i32, _z: i32| y <= 64;

    // Simulate for 0.4 seconds: gravity pulls downward, particles land at surface
    for _ in 0..10 {
        sys.tick(0.04, is_solid);
    }

    let mut gpu_particles = Vec::new();
    sys.extract_gpu_particles(&mut gpu_particles);
    assert_eq!(gpu_particles.len(), 20);

    // All debris must be above or at the floor surface (y >= 64.0)
    for p in &gpu_particles {
        assert!(
            p.pos[1] >= 64.0,
            "Particle penetrated floor: pos.y = {}",
            p.pos[1]
        );
        assert_eq!(p.tex_source, 1); // Sample from block texture array
        assert_eq!(p.layer, 2);
    }
}

#[test]
fn test_gpu_particle_buffer_packing_and_alignment() {
    assert_eq!(
        std::mem::size_of::<ParticleGpu>(),
        48,
        "ParticleGpu must be exactly 48 bytes for std430 alignment"
    );

    let mut sys = ParticleSystem::new(1024);
    sys.spawn_block_place(Vec3::new(0.0, 10.0, 0.0), 3, [200, 200, 200, 255], 8);

    let mut gpu_buf = Vec::new();
    sys.extract_gpu_particles(&mut gpu_buf);
    assert_eq!(gpu_buf.len(), 8);

    // Verify Pod bytes conversion succeeds
    let bytes = bytemuck::cast_slice::<ParticleGpu, u8>(&gpu_buf);
    assert_eq!(bytes.len(), 8 * 48);
}

#[test]
fn test_network_particle_event_spawn() {
    let mut sys = ParticleSystem::new(1024);

    let event = S2cParticleEvent {
        effect: ParticleEffectKind::BlockBreak,
        x: 10.0,
        y: 20.0,
        z: 30.0,
        count: 16,
        speed: 1.0,
        block_state_id: 1, // Stone
    };

    sys.spawn_from_event(&event, |id| {
        if id == 1 {
            (0, [128, 128, 128, 255])
        } else {
            (1, [255, 255, 255, 255])
        }
    });

    assert_eq!(sys.len(), 16);

    let mut gpu_particles = Vec::new();
    sys.extract_gpu_particles(&mut gpu_particles);
    assert_eq!(gpu_particles.len(), 16);
    assert_eq!(gpu_particles[0].layer, 0);
    assert_eq!(gpu_particles[0].tex_source, 1);
}

#[test]
fn test_capacity_cap_enforcement() {
    let mut sys = ParticleSystem::new(50);
    // Attempt to spawn 200 particles
    sys.spawn_block_break(Vec3::ZERO, 0, [255, 255, 255, 255], 200);
    assert_eq!(sys.len(), 50);

    // Spawning more should not exceed capacity
    sys.spawn_torch_smoke(Vec3::ZERO);
    assert_eq!(sys.len(), 50);
}
