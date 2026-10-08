//! Integration tests for server-authoritative 3D spatial voice chat routing,
//! distance falloff cutoff, dimension isolation, and datagram flood rate limiting.

#![allow(clippy::float_cmp)]

use glam::DVec3;
use telos_net::{Connection, Lane, MemoryConnection, Payload};
use telos_protocol::bounded::{BoundedString, BoundedVec};
use telos_protocol::messages::{
    AuthMode, C2sClientSettings, C2sConfigAck, C2sHello, C2sLoginStart, C2sMessage, C2sVoiceData,
    ConnectionPhase, S2cMessage,
};
use telos_server::{Server, ServerConfig};

fn setup_server() -> Server {
    let config = ServerConfig {
        tps: 20,
        view_distance: 2,
        vertical_view_distance: 2,
        ..Default::default()
    };
    Server::new(12345, config)
}

fn add_player(
    server: &mut Server,
    username: &str,
    pos: DVec3,
) -> (u64, MemoryConnection<C2sMessage, S2cMessage>) {
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
    while let Ok(Some(_)) = client_conn.try_recv() {}

    // 2. Login
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::LoginStart(C2sLoginStart {
                username: BoundedString::new(username).unwrap(),
                mode: AuthMode::Offline,
            })),
        )
        .unwrap();
    server.tick();
    while let Ok(Some(_)) = client_conn.try_recv() {}

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

    // Drain initial login & chunk spawn packets
    while let Ok(Some(_)) = client_conn.try_recv() {}

    // Position player authoritatively
    if let Some(session) = server.get_session_mut(session_id) {
        assert_eq!(session.phase, ConnectionPhase::Play);
        session.position = pos;
    }

    (session_id, client_conn)
}

#[test]
fn test_voice_proximity_routing_in_range() {
    let mut server = setup_server();

    // Alice at (0, 64, 0), Bob at (10, 64, 0) - distance 10.0 blocks (audible)
    let (alice_id, alice_conn) = add_player(&mut server, "Alice", DVec3::new(0.0, 64.0, 0.0));
    let (_bob_id, bob_conn) = add_player(&mut server, "Bob", DVec3::new(10.0, 64.0, 0.0));

    // Drain any join messages
    while let Ok(Some(_)) = alice_conn.try_recv() {}
    while let Ok(Some(_)) = bob_conn.try_recv() {}

    let voice_packet = C2sVoiceData {
        sequence: 101,
        opus_frame: BoundedVec::new(vec![0xFC, 0xAA, 0xBB, 0xCC]).unwrap(),
    };

    alice_conn
        .send(
            Lane::Unreliable,
            Payload::Msg(C2sMessage::VoiceData(voice_packet)),
        )
        .unwrap();

    server.tick();

    // Alice should NOT receive her own voice datagram back
    while let Ok(Some(payload)) = alice_conn.try_recv() {
        if let Some(S2cMessage::VoiceData(_)) = payload.into_msg() {
            panic!("Speaker should not receive echo of their own voice");
        }
    }

    // Bob SHOULD receive Alice's voice packet with authoritative speaker position
    let mut received_voice = None;
    while let Ok(Some(payload)) = bob_conn.try_recv() {
        if let Some(S2cMessage::VoiceData(v)) = payload.into_msg() {
            received_voice = Some(v);
            break;
        }
    }

    let bob_voice = received_voice.expect("Bob should receive Alice's voice packet");
    assert_eq!(bob_voice.sequence, 101);
    assert_eq!(bob_voice.opus_frame.as_slice(), &[0xFC, 0xAA, 0xBB, 0xCC]);
    // Speaker position should reflect Alice's mouth position (y + 1.62)
    assert_eq!(bob_voice.position[0], 0.0);
    assert!((bob_voice.position[1] - 65.62).abs() < 1e-4);
    assert_eq!(bob_voice.position[2], 0.0);

    let alice_uuid = server.get_session(alice_id).unwrap().player_uuid;
    assert_eq!(bob_voice.speaker_uuid, alice_uuid);
}

#[test]
fn test_voice_proximity_cutoff_out_of_range() {
    let mut server = setup_server();

    // Alice at (0, 64, 0), Charlie at (40, 64, 0) - distance 40.0 blocks (> 32 max radius)
    let (_alice_id, alice_conn) = add_player(&mut server, "Alice", DVec3::new(0.0, 64.0, 0.0));
    let (_charlie_id, charlie_conn) =
        add_player(&mut server, "Charlie", DVec3::new(40.0, 64.0, 0.0));

    while let Ok(Some(_)) = alice_conn.try_recv() {}
    while let Ok(Some(_)) = charlie_conn.try_recv() {}

    let voice_packet = C2sVoiceData {
        sequence: 202,
        opus_frame: BoundedVec::new(vec![0xFC, 0x11, 0x22]).unwrap(),
    };

    alice_conn
        .send(
            Lane::Unreliable,
            Payload::Msg(C2sMessage::VoiceData(voice_packet)),
        )
        .unwrap();

    server.tick();

    // Charlie should NOT receive the voice packet because distance > 32 blocks
    while let Ok(Some(payload)) = charlie_conn.try_recv() {
        if let Some(S2cMessage::VoiceData(_)) = payload.into_msg() {
            panic!("Player outside 32-block radius received voice datagram");
        }
    }
}

#[test]
fn test_voice_dimension_isolation() {
    let mut server = setup_server();

    // Alice in overworld at (0, 64, 0), David in nether at (0, 64, 0)
    let (_alice_id, alice_conn) = add_player(&mut server, "Alice", DVec3::new(0.0, 64.0, 0.0));
    let (david_id, david_conn) = add_player(&mut server, "David", DVec3::new(0.0, 64.0, 0.0));

    // Place David in the Nether dimension
    if let Some(david_session) = server.get_session_mut(david_id) {
        david_session.world_name = "nether".to_string();
    }

    while let Ok(Some(_)) = alice_conn.try_recv() {}
    while let Ok(Some(_)) = david_conn.try_recv() {}

    let voice_packet = C2sVoiceData {
        sequence: 303,
        opus_frame: BoundedVec::new(vec![0xFC, 0x33, 0x44]).unwrap(),
    };

    alice_conn
        .send(
            Lane::Unreliable,
            Payload::Msg(C2sMessage::VoiceData(voice_packet)),
        )
        .unwrap();

    server.tick();

    // David should NOT receive overworld voice packets even at 0 distance
    while let Ok(Some(payload)) = david_conn.try_recv() {
        if let Some(S2cMessage::VoiceData(_)) = payload.into_msg() {
            panic!("Player in different dimension received voice datagram");
        }
    }
}

#[test]
fn test_voice_rate_limit_flood_protection() {
    let mut server = setup_server();

    let (_alice_id, alice_conn) = add_player(&mut server, "Alice", DVec3::new(0.0, 64.0, 0.0));
    let (_bob_id, bob_conn) = add_player(&mut server, "Bob", DVec3::new(5.0, 64.0, 0.0));

    while let Ok(Some(_)) = alice_conn.try_recv() {}
    while let Ok(Some(_)) = bob_conn.try_recv() {}

    // Alice attempts to send 75 voice frames in a single tick burst
    for seq in 0..75 {
        let voice_packet = C2sVoiceData {
            sequence: seq,
            opus_frame: BoundedVec::new(vec![0xFC, seq as u8]).unwrap(),
        };
        alice_conn
            .send(
                Lane::Unreliable,
                Payload::Msg(C2sMessage::VoiceData(voice_packet)),
            )
            .unwrap();
    }

    server.tick();

    let mut delivered_count = 0;
    while let Ok(Some(payload)) = bob_conn.try_recv() {
        if let Some(S2cMessage::VoiceData(_)) = payload.into_msg() {
            delivered_count += 1;
        }
    }

    // Rate limit must cap at 60 frames per second
    assert_eq!(
        delivered_count, 60,
        "Server should deliver exactly 60 packets and discard burst overflow"
    );
}
