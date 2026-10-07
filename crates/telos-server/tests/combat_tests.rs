//! Integration tests for server-authoritative mob combat AI, melee attacks, armor mitigation, and loot drops.

use glam::DVec3;
use telos_core::coords::BlockPos;
use telos_core::ident::Identifier;
use telos_net::{Connection, Lane, MemoryConnection, Payload};
use telos_protocol::bounded::BoundedString;
use telos_protocol::messages::{
    AuthMode, C2sClientSettings, C2sConfigAck, C2sHello, C2sInteractEntity, C2sLoginStart,
    C2sMessage, C2sPlayerPosition, S2cMessage,
};
use telos_server::{Server, ServerConfig};
use telos_sim::entity::{EntityType, HurtTime};
use telos_sim::inventory::{
    ARMOR_BOOTS_SLOT, ARMOR_CHESTPLATE_SLOT, ARMOR_HELMET_SLOT, ARMOR_LEGGINGS_SLOT,
};
use telos_sim::{Experience, Health, Inventory, ItemStack};
use telos_voxel::state::BlockStateId;

fn setup_server_and_player() -> (Server, MemoryConnection<C2sMessage, S2cMessage>, u64, f64) {
    let config = ServerConfig {
        tps: 20,
        view_distance: 3,
        vertical_view_distance: 2,
        chunks_per_tick_per_player: 10,
        ..Default::default()
    };
    let mut server = Server::new(12345, config);

    let (server_conn, client_conn) = MemoryConnection::pair_default();
    let session_id = server.add_connection(Box::new(server_conn));

    // Handshake: Hello
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::Hello(C2sHello {
                protocol: 1,
                build: BoundedString::new("0.1.0").unwrap(),
                features: 0,
            })),
        )
        .expect("send Hello");
    server.tick();

    // LoginStart
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::LoginStart(C2sLoginStart {
                username: BoundedString::new("CombatPlayer").unwrap(),
                mode: AuthMode::Offline,
            })),
        )
        .expect("send LoginStart");
    server.tick();

    // Drain login messages
    while let Ok(Some(_)) = client_conn.try_recv() {}

    // ClientSettings + ConfigAck
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ClientSettings(C2sClientSettings {
                view_distance: 3,
                simulation_distance: 3,
                locale: BoundedString::new("en_US").unwrap(),
            })),
        )
        .unwrap();
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ConfigAck(C2sConfigAck)),
        )
        .unwrap();
    server.tick();

    // Drain initial join and chunk messages
    while let Ok(Some(_)) = client_conn.try_recv() {}

    let surface_y = f64::from(server.world().get_surface_y(128, 160));
    (server, client_conn, session_id, surface_y)
}

#[test]
fn test_zombie_attacks_survival_player() {
    let (mut server, client_conn, session_id, surface_y) = setup_server_and_player();
    let sy = surface_y as i32;

    // Clear flat area
    for x in 126..=130 {
        for z in 158..=162 {
            let _ = server
                .world_mut()
                .set_block(BlockPos::new(x, sy, z), BlockStateId::new(1));
            for y in 1..=3 {
                let _ = server
                    .world_mut()
                    .set_block(BlockPos::new(x, sy + y, z), BlockStateId::AIR);
            }
        }
    }

    let player_y = surface_y + 1.0;
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerPosition(C2sPlayerPosition {
                x: 128.0,
                y: player_y,
                z: 160.0,
                yaw: 0.0,
                pitch: 0.0,
                on_ground: true,
            })),
        )
        .unwrap();
    server.tick();

    // Spawn Zombie immediately adjacent at distance 1.2 blocks (within 1.8 reach)
    let _zombie_id = server.spawn_mob(EntityType::Zombie, DVec3::new(128.0, player_y, 161.2));

    // Initial player health is 20.0
    let player_ecs = server
        .get_session(session_id)
        .and_then(|s| s.ecs_entity)
        .unwrap();
    let initial_health = server.ecs_world().get::<Health>(player_ecs).unwrap().cur;
    assert!((initial_health - 20.0).abs() < 1e-4);

    // Tick server: Zombie perceives player, initiates attack, and inflicts damage
    server.tick();

    let health_after_attack = server.ecs_world().get::<Health>(player_ecs).unwrap().cur;
    // Zombie base attack damage is 3.0 HP (no armor on player -> 17.0 HP)
    assert!(
        health_after_attack < 20.0,
        "Player health should decrease after zombie melee attack: {health_after_attack}"
    );
    assert!(
        (health_after_attack - 17.0).abs() < 0.01,
        "Player health should be ~17.0 after unmitigated 3.0 hit: {health_after_attack}"
    );

    let hurt_time = server.ecs_world().get::<HurtTime>(player_ecs).unwrap().0;
    assert_eq!(hurt_time, 10, "Player HurtTime should be set to 10");
}

#[test]
fn test_player_armor_mitigates_zombie_damage() {
    let (mut server, client_conn, session_id, surface_y) = setup_server_and_player();
    let sy = surface_y as i32;

    for x in 126..=130 {
        for z in 158..=162 {
            let _ = server
                .world_mut()
                .set_block(BlockPos::new(x, sy, z), BlockStateId::new(1));
            for y in 1..=3 {
                let _ = server
                    .world_mut()
                    .set_block(BlockPos::new(x, sy + y, z), BlockStateId::AIR);
            }
        }
    }

    let player_y = surface_y + 1.0;
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerPosition(C2sPlayerPosition {
                x: 128.0,
                y: player_y,
                z: 160.0,
                yaw: 0.0,
                pitch: 0.0,
                on_ground: true,
            })),
        )
        .unwrap();
    server.tick();

    let player_ecs = server
        .get_session(session_id)
        .and_then(|s| s.ecs_entity)
        .unwrap();

    // Equip full iron armor (helmet, chestplate, leggings, boots)
    let item_reg = server.registries().item_registry();
    let helmet_id = item_reg
        .get_by_ident(&Identifier::new("telos", "iron_helmet").unwrap())
        .expect("iron_helmet must exist");
    let chest_id = item_reg
        .get_by_ident(&Identifier::new("telos", "iron_chestplate").unwrap())
        .expect("iron_chestplate must exist");
    let legs_id = item_reg
        .get_by_ident(&Identifier::new("telos", "iron_leggings").unwrap())
        .expect("iron_leggings must exist");
    let boots_id = item_reg
        .get_by_ident(&Identifier::new("telos", "iron_boots").unwrap())
        .expect("iron_boots must exist");

    {
        let mut inv = server
            .ecs_world_mut()
            .get_mut::<Inventory>(player_ecs)
            .unwrap();
        inv.slots[ARMOR_HELMET_SLOT] = ItemStack::new(helmet_id, 1);
        inv.slots[ARMOR_CHESTPLATE_SLOT] = ItemStack::new(chest_id, 1);
        inv.slots[ARMOR_LEGGINGS_SLOT] = ItemStack::new(legs_id, 1);
        inv.slots[ARMOR_BOOTS_SLOT] = ItemStack::new(boots_id, 1);
    }

    // Spawn Zombie adjacent
    let _zombie_id = server.spawn_mob(EntityType::Zombie, DVec3::new(128.0, player_y, 161.2));

    server.tick();

    let health_after_attack = server.ecs_world().get::<Health>(player_ecs).unwrap().cur;
    // With 15 armor defense, damage is ~1.38 HP instead of 3.0 HP
    let damage_taken = 20.0 - health_after_attack;
    assert!(
        damage_taken < 2.0,
        "Armored player should take under 2.0 damage, took: {damage_taken}"
    );
    assert!(
        damage_taken > 1.0,
        "Armored player should still take positive damage, took: {damage_taken}"
    );
}

#[test]
fn test_zombie_death_drops_rotten_flesh() {
    let (mut server, client_conn, session_id, surface_y) = setup_server_and_player();
    let player_y = surface_y + 1.0;
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerPosition(C2sPlayerPosition {
                x: 128.0,
                y: player_y,
                z: 160.0,
                yaw: 0.0,
                pitch: 0.0,
                on_ground: true,
            })),
        )
        .unwrap();
    server.tick();

    // Spawn Zombie within reach (1.5 blocks)
    let zombie_id = server.spawn_mob(EntityType::Zombie, DVec3::new(128.0, player_y, 161.5));

    let player_ecs = server
        .get_session(session_id)
        .and_then(|s| s.ecs_entity)
        .unwrap();

    let initial_xp = server
        .ecs_world()
        .get::<Experience>(player_ecs)
        .unwrap()
        .total_xp;

    // Attack zombie 5 times to deplete 20.0 HP (4.0 dmg per hit)
    for _ in 0..5 {
        client_conn
            .send(
                Lane::Control,
                Payload::Msg(C2sMessage::InteractEntity(C2sInteractEntity {
                    target_net_id: zombie_id,
                    action: 0, // Attack
                })),
            )
            .unwrap();
        // Elapse 11 ticks for 10-tick damage invulnerability window
        for _ in 0..11 {
            server.tick();
        }
    }

    // Verify zombie is despawned
    assert!(
        server.tracked_mobs.get(&zombie_id).is_none(),
        "Zombie should be despawned after death"
    );

    // Verify XP increased by 5
    let new_xp = server
        .ecs_world()
        .get::<Experience>(player_ecs)
        .unwrap()
        .total_xp;
    assert_eq!(new_xp, initial_xp + 5, "Killing zombie should award 5 XP");

    // Verify rotten_flesh is in player's inventory
    let item_reg = server.registries().item_registry();
    let rf_id = item_reg
        .get_by_ident(&Identifier::new("telos", "rotten_flesh").unwrap())
        .expect("rotten_flesh must exist");

    let inv = server.ecs_world().get::<Inventory>(player_ecs).unwrap();
    let has_flesh = inv.slots.iter().any(|s| s.item == rf_id && s.count > 0);
    assert!(
        has_flesh,
        "Player inventory should receive rotten_flesh loot drop"
    );
}

#[test]
fn test_pig_death_drops_porkchop() {
    let (mut server, client_conn, session_id, surface_y) = setup_server_and_player();
    let player_y = surface_y + 1.0;
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerPosition(C2sPlayerPosition {
                x: 128.0,
                y: player_y,
                z: 160.0,
                yaw: 0.0,
                pitch: 0.0,
                on_ground: true,
            })),
        )
        .unwrap();
    server.tick();

    let pig_id = server.spawn_mob(EntityType::Pig, DVec3::new(128.0, player_y, 161.5));

    let player_ecs = server
        .get_session(session_id)
        .and_then(|s| s.ecs_entity)
        .unwrap();

    let initial_xp = server
        .ecs_world()
        .get::<Experience>(player_ecs)
        .unwrap()
        .total_xp;

    // Attack pig 3 times to deplete 10.0 HP (4.0 dmg per hit)
    for _ in 0..3 {
        client_conn
            .send(
                Lane::Control,
                Payload::Msg(C2sMessage::InteractEntity(C2sInteractEntity {
                    target_net_id: pig_id,
                    action: 0,
                })),
            )
            .unwrap();
        // Elapse 11 ticks for 10-tick damage invulnerability window
        for _ in 0..11 {
            server.tick();
        }
    }

    assert!(
        server.tracked_mobs.get(&pig_id).is_none(),
        "Pig should be despawned after death"
    );

    let new_xp = server
        .ecs_world()
        .get::<Experience>(player_ecs)
        .unwrap()
        .total_xp;
    assert_eq!(new_xp, initial_xp + 2, "Killing pig should award 2 XP");

    let item_reg = server.registries().item_registry();
    let pc_id = item_reg
        .get_by_ident(&Identifier::new("telos", "porkchop").unwrap())
        .expect("porkchop must exist");

    let inv = server.ecs_world().get::<Inventory>(player_ecs).unwrap();
    let has_pork = inv.slots.iter().any(|s| s.item == pc_id && s.count > 0);
    assert!(
        has_pork,
        "Player inventory should receive porkchop loot drop"
    );
}
