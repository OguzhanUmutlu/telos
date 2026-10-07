//! Verification tests for combat AI, line-of-sight raycasting, target aggro, and attack cooldowns.

use glam::{DVec3, Vec3};
use telos_core::coords::BlockPos;
use telos_sim::entity::{
    AiState, AttackCooldown, Mob, MobBundle, PlayerPositions, TargetablePlayer, Velocity,
    has_line_of_sight, mob_ai_system,
};

#[test]
fn test_los_unobstructed() {
    let mob_eye = Vec3::new(0.5, 65.62, 0.5);
    let player_eye = Vec3::new(10.5, 65.62, 0.5);

    // All blocks between are air (not solid)
    let is_solid = |_pos: BlockPos| false;
    assert!(has_line_of_sight(mob_eye, player_eye, is_solid));
}

#[test]
fn test_los_obstructed_by_solid_wall() {
    let mob_eye = Vec3::new(0.5, 65.62, 0.5);
    let player_eye = Vec3::new(10.5, 65.62, 0.5);

    // Solid stone wall at x = 5
    let is_solid = |pos: BlockPos| pos.x() == 5;
    assert!(!has_line_of_sight(mob_eye, player_eye, is_solid));
}

#[test]
fn test_aggro_ignores_creative_players() {
    let mut world = bevy_ecs::world::World::new();
    let zombie = MobBundle::new_zombie(10, DVec3::new(0.0, 64.0, 0.0), 12345);
    let e = world.spawn(zombie).id();

    // Player 1 is within 8 blocks, but NOT targetable (creative / flying)
    world.insert_resource(PlayerPositions(vec![TargetablePlayer::new(
        1,
        DVec3::new(8.0, 64.0, 0.0),
        false,
    )]));

    let mut schedule = bevy_ecs::schedule::Schedule::default();
    schedule.add_systems(mob_ai_system);

    schedule.run(&mut world);

    let mob = world.get::<Mob>(e).unwrap();
    // Mob should NOT be chasing the untargetable player
    assert!(!matches!(mob.ai_state, AiState::Chasing { .. }));

    // Now make player targetable (survival)
    world.insert_resource(PlayerPositions(vec![TargetablePlayer::new(
        1,
        DVec3::new(8.0, 64.0, 0.0),
        true,
    )]));

    schedule.run(&mut world);

    let mob = world.get::<Mob>(e).unwrap();
    assert_eq!(mob.ai_state, AiState::Chasing { target_net_id: 1 });

    let vel = world.get::<Velocity>(e).unwrap();
    assert!(vel.0.x > 0.05);
}

#[test]
fn test_attack_cooldown_ticks_down() {
    let mut cd = AttackCooldown::new(20, 3.0, 1.8);
    assert!(cd.can_attack());
    assert_eq!(cd.current, 0);

    // Strike resets cooldown to 20
    cd.reset();
    assert!(!cd.can_attack());
    assert_eq!(cd.current, 20);

    // Tick down 19 ticks
    for _ in 0..19 {
        cd.tick();
        assert!(!cd.can_attack());
    }
    assert_eq!(cd.current, 1);

    // Final tick restores readiness
    cd.tick();
    assert_eq!(cd.current, 0);
    assert!(cd.can_attack());
}
