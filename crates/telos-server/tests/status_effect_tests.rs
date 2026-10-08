//! Integration tests for server-authoritative status effect lifecycle, potion modifiers,
//! delta wire synchronization, and damage mitigation.

#![allow(clippy::collapsible_if, clippy::float_cmp)]

use glam::DVec3;
use telos_core::coords::BlockPos;
use telos_net::{Connection, Lane, MemoryConnection, Payload};
use telos_protocol::bounded::BoundedString;
use telos_protocol::messages::{
    AuthMode, C2sChatMessage, C2sClientSettings, C2sConfigAck, C2sHello, C2sLoginStart, C2sMessage,
    C2sPlayerCommand, PlayerCommandKind, S2cMessage,
};
use telos_server::{Server, ServerConfig};
use telos_sim::effect::StatusEffectKind;
use telos_sim::inventory::{Inventory, ItemStack};
use telos_sim::potion::PotionType;
use telos_voxel::state::BlockStateId;

fn setup_server_with_player() -> (Server, MemoryConnection<C2sMessage, S2cMessage>, u64) {
    let config = ServerConfig {
        tps: 20,
        view_distance: 2,
        vertical_view_distance: 2,
        ..Default::default()
    };
    let mut server = Server::new(12345, config);
    let (server_conn, client_conn) = MemoryConnection::pair_default();
    let session_id = server.add_connection(Box::new(server_conn));

    // 1. Hello
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::Hello(C2sHello {
                protocol: 1,
                build: BoundedString::new("0.1.0").unwrap(),
                features: 0,
            })),
        )
        .unwrap();
    server.tick();
    let _ = client_conn.try_recv().unwrap();

    // 2. Login
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::LoginStart(C2sLoginStart {
                username: BoundedString::new("Tester").unwrap(),
                mode: AuthMode::Offline,
            })),
        )
        .unwrap();
    server.tick();
    let _ = client_conn.try_recv().unwrap();

    // 3. Settings & ConfigAck
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ClientSettings(C2sClientSettings {
                view_distance: 2,
                simulation_distance: 2,
                locale: BoundedString::new("en_US").unwrap(),
            })),
        )
        .unwrap();
    server.tick();
    while let Ok(Some(_)) = client_conn.try_recv() {}

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ConfigAck(C2sConfigAck)),
        )
        .unwrap();
    server.tick();

    // Drain initial login packets
    while let Ok(Some(_)) = client_conn.try_recv() {}

    (server, client_conn, session_id)
}

#[test]
fn test_status_effect_command_and_delta_packets() {
    let (mut server, client_conn, _session_id) = setup_server_with_player();

    // 1. Give Speed II for 5 seconds (100 ticks) via chat command
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ChatMessage(C2sChatMessage {
                message: BoundedString::new("/effect give @s speed 5 1").unwrap(),
            })),
        )
        .unwrap();
    server.tick();

    // Verify S2cEntityEffect packet is received
    let mut received_speed = false;
    while let Ok(Some(payload)) = client_conn.try_recv() {
        if let Some(S2cMessage::EntityEffect(eff)) = payload.into_msg() {
            if eff.effect_id == StatusEffectKind::Speed.id() {
                assert_eq!(eff.amplifier, 1);
                assert!(eff.duration_ticks == 99 || eff.duration_ticks == 100);
                received_speed = true;
            }
        }
    }
    assert!(received_speed, "Expected S2cEntityEffect for Speed II");

    // 2. Clear speed effect
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ChatMessage(C2sChatMessage {
                message: BoundedString::new("/effect clear @s speed").unwrap(),
            })),
        )
        .unwrap();
    server.tick();

    // Verify S2cRemoveEntityEffect packet is received
    let mut received_remove = false;
    while let Ok(Some(payload)) = client_conn.try_recv() {
        if let Some(S2cMessage::RemoveEntityEffect(rem)) = payload.into_msg() {
            if rem.effect_id == StatusEffectKind::Speed.id() {
                received_remove = true;
            }
        }
    }
    assert!(received_remove, "Expected S2cRemoveEntityEffect for Speed");
}

#[test]
fn test_potion_consumption_and_bottle_refund() {
    let (mut server, client_conn, session_id) = setup_server_with_player();

    // Equip player with a Potion item (36) in selected slot 0
    let entity = server.get_session(session_id).unwrap().ecs_entity.unwrap();
    {
        let mut inv = server.ecs_world_mut().get_mut::<Inventory>(entity).unwrap();
        inv.slots[0] = ItemStack::new(36, 1); // 1 Potion
    }

    // Drink potion of Night Vision (id 19)
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::PlayerCommand(C2sPlayerCommand {
                command: PlayerCommandKind::DrinkPotion {
                    potion_type: PotionType::NightVision.id(),
                },
            })),
        )
        .unwrap();
    server.tick();

    // Verify entity received NightVision effect
    let mut received_nv = false;
    while let Ok(Some(payload)) = client_conn.try_recv() {
        if let Some(S2cMessage::EntityEffect(eff)) = payload.into_msg() {
            if eff.effect_id == StatusEffectKind::NightVision.id() {
                assert!(eff.duration_ticks == 3599 || eff.duration_ticks == 3600);
                received_nv = true;
            }
        }
    }
    assert!(received_nv, "Expected S2cEntityEffect for Night Vision");

    // Verify inventory slot was replaced with Glass Bottle (38)
    let inv = server.ecs_world().get::<Inventory>(entity).unwrap();
    assert_eq!(inv.slots[0].item, 38);
    assert_eq!(inv.slots[0].count, 1);
}

#[test]
fn test_effect_duration_expiration() {
    let (mut server, client_conn, _session_id) = setup_server_with_player();

    // Give short effect (1 second = 20 ticks)
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ChatMessage(C2sChatMessage {
                message: BoundedString::new("/effect give @s blindness 1 0").unwrap(),
            })),
        )
        .unwrap();
    server.tick();

    let mut got_effect = false;
    while let Ok(Some(payload)) = client_conn.try_recv() {
        if let Some(S2cMessage::EntityEffect(eff)) = payload.into_msg() {
            if eff.effect_id == StatusEffectKind::Blindness.id() {
                got_effect = true;
            }
        }
    }
    assert!(got_effect, "Expected Blindness effect packet");

    // Advance 21 ticks
    for _ in 0..21 {
        server.tick();
    }

    let mut got_remove = false;
    while let Ok(Some(payload)) = client_conn.try_recv() {
        if let Some(S2cMessage::RemoveEntityEffect(rem)) = payload.into_msg() {
            if rem.effect_id == StatusEffectKind::Blindness.id() {
                got_remove = true;
            }
        }
    }
    assert!(got_remove, "Expected RemoveEntityEffect upon expiration");
}

#[test]
fn test_fire_resistance_hazard_immunity() {
    let (mut server, client_conn, session_id) = setup_server_with_player();

    // Place lava (31) at player's location
    let world_name = server.get_session(session_id).unwrap().world_name.clone();
    let pos = BlockPos::new(0, 64, 0);
    server
        .worlds_mut()
        .get_or_default_mut(&world_name)
        .set_block(pos, BlockStateId(31));

    // Teleport player into lava
    server
        .get_session_mut(session_id)
        .unwrap()
        .teleport(DVec3::new(0.5, 64.0, 0.5));
    server.tick();

    // Initially without Fire Resistance: takes lava damage within 10 ticks
    for _ in 0..10 {
        server.tick();
    }
    let entity = server.get_session(session_id).unwrap().ecs_entity.unwrap();
    let health_before = server
        .ecs_world()
        .get::<telos_sim::Health>(entity)
        .unwrap()
        .cur;
    assert!(
        health_before < 20.0,
        "Player in lava should take damage without Fire Resistance"
    );

    // Heal player and grant Fire Resistance
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ChatMessage(C2sChatMessage {
                message: BoundedString::new("/effect give @s fire_resistance 60 0").unwrap(),
            })),
        )
        .unwrap();
    server.tick();
    {
        let mut h = server
            .ecs_world_mut()
            .get_mut::<telos_sim::Health>(entity)
            .unwrap();
        h.cur = 20.0;
    }

    // Advance 20 ticks in lava with Fire Resistance
    for _ in 0..20 {
        server.tick();
    }
    let health_after = server
        .ecs_world()
        .get::<telos_sim::Health>(entity)
        .unwrap()
        .cur;
    assert_eq!(
        health_after, 20.0,
        "Player with Fire Resistance should take 0 damage from lava"
    );
}
