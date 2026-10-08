//! Integration and property tests for telos-protocol codecs and wire types.

use proptest::prelude::*;
use std::sync::Arc;
use telos_core::coords::ChunkPos;
use telos_core::form::{
    ActionForm, CustomForm, FormCancelReason, FormResponseData, FormValue, ModalForm, ModalFormData,
};
use telos_protocol::bounded::{BoundedString, BoundedVec};
use telos_protocol::codec::{
    MAX_FRAME_SIZE, PacketHeader, decode_c2s, decode_s2c, encode_c2s, encode_s2c, peek_frame,
};
use telos_protocol::error::ProtocolError;
use telos_protocol::messages::config::{
    CustomFuelWire, CustomShapedRecipeWire, CustomShapelessRecipeWire, CustomSmeltingRecipeWire,
};
use telos_protocol::messages::{
    AdvancementProgressWire, AuthMode, BlockActionKind, C2sBlockAction, C2sChatMessage,
    C2sClientSettings, C2sCloseContainer, C2sCommandSuggest, C2sConfigAck, C2sHello, C2sKeepAlive,
    C2sKnownRegistries, C2sLoginStart, C2sMessage, C2sModalFormResponse, C2sPlayerCommand,
    C2sPlayerPosition, ChunkPayload, ConnectionPhase, CustomBlockDefWire, CustomItemDefWire,
    Disconnect, DisconnectReason, LodPayload, ParticleEffectKind, PlayerCommandKind,
    S2cAdvancementToast, S2cAdvancementUpdate, S2cBlockActionAck, S2cBlockEvent, S2cBlockUpdate,
    S2cChatMessage, S2cChunkData, S2cChunkUnload, S2cCloseContainer, S2cCommandSuggestions,
    S2cConfigDone, S2cContainerProperty, S2cContentManifest, S2cEntityEffect, S2cGameMode,
    S2cHelloReply, S2cJoinGame, S2cKeepAlive, S2cLodNodeData, S2cLodNodeUnload, S2cLoginSuccess,
    S2cMessage, S2cModalFormRequest, S2cOpenContainer, S2cParticleEvent, S2cRecipeManifest,
    S2cRegistryData, S2cRemoveEntityEffect, S2cSpawnArrow, S2cSpawnItem, S2cUniformChunk, SlotData,
};
use telos_protocol::varint::{
    decode_varint, decode_varint_zigzag, decode_varlong, encode_varint, encode_varint_zigzag,
    encode_varlong, varint_size,
};
use telos_voxel::{chunk::ChunkSnapshot, light::ChunkLight, state::BlockStateId};

#[test]
fn test_varint_boundary_roundtrips() {
    let test_values = [
        0u32,
        1,
        127,
        128,
        255,
        256,
        16383,
        16384,
        2_097_151,
        2_097_152,
        268_435_455,
        268_435_456,
        u32::MAX,
    ];

    for &val in &test_values {
        let mut buf = Vec::new();
        encode_varint(val, &mut buf);
        assert_eq!(buf.len(), varint_size(val));

        let mut cursor = &buf[..];
        let decoded = decode_varint(&mut cursor).expect("failed to decode varint");
        assert_eq!(decoded, val);
        assert!(cursor.is_empty());
    }
}

#[test]
fn test_zigzag_roundtrips() {
    let test_values = [0i32, -1, 1, -128, 127, i32::MIN, i32::MAX];

    for &val in &test_values {
        let mut buf = Vec::new();
        encode_varint_zigzag(val, &mut buf);

        let mut cursor = &buf[..];
        let decoded = decode_varint_zigzag(&mut cursor).expect("failed to decode zigzag");
        assert_eq!(decoded, val);
        assert!(cursor.is_empty());
    }
}

#[test]
fn test_varlong_boundary_roundtrips() {
    let test_values = [0u64, 1, 127, 128, u64::from(u32::MAX), u64::MAX];

    for &val in &test_values {
        let mut buf = Vec::new();
        encode_varlong(val, &mut buf);

        let mut cursor = &buf[..];
        let decoded = decode_varlong(&mut cursor).expect("failed to decode varlong");
        assert_eq!(decoded, val);
        assert!(cursor.is_empty());
    }
}

#[test]
fn test_bounded_types_enforce_limits() {
    // String within limit
    let valid_str = BoundedString::<10>::new("1234567890").unwrap();
    assert_eq!(valid_str.len(), 10);

    // String exceeding limit
    let invalid_str = BoundedString::<10>::new("12345678901");
    assert!(matches!(
        invalid_str,
        Err(ProtocolError::StringTooLong {
            actual: 11,
            limit: 10
        })
    ));

    // Vector within limit
    let mut vec = BoundedVec::<u32, 2>::empty();
    assert!(vec.push(1).is_ok());
    assert!(vec.push(2).is_ok());
    // Exceeding capacity
    assert!(matches!(
        vec.push(3),
        Err(ProtocolError::VecTooLong {
            actual: 3,
            limit: 2
        })
    ));
}

#[test]
fn test_c2s_messages_roundtrip() {
    let messages = vec![
        C2sMessage::Hello(C2sHello {
            protocol: 1,
            build: BoundedString::new("test-client").unwrap(),
            features: 0xCAFE_BABE,
        }),
        C2sMessage::LoginStart(C2sLoginStart {
            username: BoundedString::new("Steve").unwrap(),
            mode: AuthMode::Offline,
        }),
        C2sMessage::KnownRegistries(C2sKnownRegistries {
            known_hashes: BoundedVec::new(vec![[7u8; 16], [8u8; 16]]).unwrap(),
        }),
        C2sMessage::ClientSettings(C2sClientSettings {
            view_distance: 32,
            simulation_distance: 12,
            locale: BoundedString::new("en_US").unwrap(),
        }),
        C2sMessage::ConfigAck(C2sConfigAck),
        C2sMessage::KeepAlive(C2sKeepAlive { id: 42_000_123 }),
        C2sMessage::ChatMessage(C2sChatMessage {
            message: BoundedString::new("Hello server!").unwrap(),
        }),
        C2sMessage::PlayerPosition(C2sPlayerPosition {
            x: 100.5,
            y: 64.0,
            z: -250.75,
            yaw: 180.0,
            pitch: -15.5,
            on_ground: true,
        }),
        C2sMessage::BlockAction(C2sBlockAction {
            sequence: 42,
            action: BlockActionKind::Break,
            x: -15,
            y: 64,
            z: 200,
            input_tick: 105,
        }),
        C2sMessage::BlockAction(C2sBlockAction {
            sequence: 43,
            action: BlockActionKind::Place {
                state_id: BlockStateId::new(4),
                hit_face: 1,
            },
            x: -15,
            y: 65,
            z: 200,
            input_tick: 106,
        }),
        C2sMessage::PlayerCommand(C2sPlayerCommand {
            command: PlayerCommandKind::ShootBow { charge_ticks: 20 },
        }),
        C2sMessage::Disconnect(Disconnect {
            reason: DisconnectReason::Normal,
            message: BoundedString::new("Quitting game").unwrap(),
        }),
    ];

    for msg in messages {
        let phase = msg.phase().unwrap_or(ConnectionPhase::Hello);
        let mut buf = Vec::new();
        encode_c2s(&msg, &mut buf);

        let mut cursor = &buf[..];
        let decoded = decode_c2s(phase, &mut cursor).expect("failed to decode C2S message");
        assert_eq!(decoded, msg);
        assert!(cursor.is_empty());
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_s2c_messages_roundtrip() {
    let messages = vec![
        S2cMessage::HelloReply(S2cHelloReply {
            protocol: 1,
            features: 0x1234_5678,
            server_id: [1u8; 16],
        }),
        S2cMessage::LoginSuccess(S2cLoginSuccess {
            player_uuid: [2u8; 16],
            username: BoundedString::new("Alex").unwrap(),
        }),
        S2cMessage::RegistryData(S2cRegistryData {
            registry_id: BoundedString::new("telos:item").unwrap(),
            content_hash: [0xaa; 32],
            entries: BoundedVec::new(vec![
                BoundedString::new("telos:stone").unwrap(),
                BoundedString::new("telos:dirt").unwrap(),
            ])
            .unwrap(),
        }),
        S2cMessage::ConfigDone(S2cConfigDone),
        S2cMessage::KeepAlive(S2cKeepAlive { id: 99_999 }),
        S2cMessage::ChatMessage(S2cChatMessage {
            sender: BoundedString::new("Server").unwrap(),
            message: BoundedString::new("Welcome!").unwrap(),
            timestamp: 123_456_789,
        }),
        S2cMessage::JoinGame(S2cJoinGame {
            entity_id: 1,
            spawn_x: 0.0,
            spawn_y: 70.0,
            spawn_z: 0.0,
            view_distance: 12,
        }),
        S2cMessage::UniformChunk(S2cUniformChunk {
            chunk_x: 3,
            chunk_y: 4,
            chunk_z: 5,
            version: 1,
            block_state: BlockStateId(1),
            sky_light: 15,
            block_light: 0,
        }),
        S2cMessage::ChunkUnload(S2cChunkUnload {
            chunk_x: 10,
            chunk_y: -2,
            chunk_z: 30,
        }),
        S2cMessage::LodNodeData(S2cLodNodeData {
            level: 2,
            node_x: -5,
            node_y: 1,
            node_z: 12,
            version: 7,
            quad_count: 14,
            palette_count: 3,
            payload: LodPayload::Wire(BoundedVec::new(vec![0x44, 0x33, 0x22, 0x11]).unwrap()),
        }),
        S2cMessage::LodNodeUnload(S2cLodNodeUnload {
            level: 2,
            node_x: -5,
            node_y: 1,
            node_z: 12,
        }),
        S2cMessage::BlockUpdate(S2cBlockUpdate {
            x: -15,
            y: 65,
            z: 200,
            state_id: BlockStateId::new(4),
            version: 12,
        }),
        S2cMessage::BlockActionAck(S2cBlockActionAck { sequence: 43 }),
        S2cMessage::ParticleEvent(S2cParticleEvent {
            effect: ParticleEffectKind::Smoke,
            x: 15.0,
            y: 70.5,
            z: -3.5,
            count: 12,
            speed: 0.25,
            block_state_id: 15,
        }),
        S2cMessage::SpawnItem(S2cSpawnItem {
            net_id: 88,
            item_id: 5,
            count: 3,
            x: 10.0,
            y: 65.0,
            z: -12.0,
            vel_x: 0.05,
            vel_y: 0.1,
            vel_z: -0.05,
        }),
        S2cMessage::SpawnArrow(S2cSpawnArrow {
            net_id: 99,
            x: 12.5,
            y: 66.0,
            z: -14.25,
            vel_x: 1.5,
            vel_y: 0.25,
            vel_z: -0.5,
            yaw: 45.0,
            pitch: -10.0,
        }),
        S2cMessage::AdvancementUpdate(S2cAdvancementUpdate {
            reset_all: true,
            advancements: BoundedVec::new(vec![AdvancementProgressWire::new(
                BoundedString::new("telos:story/root").unwrap(),
                true,
                100,
            )])
            .unwrap(),
        }),
        S2cMessage::AdvancementToast(S2cAdvancementToast {
            id: BoundedString::new("telos:story/root").unwrap(),
            title: BoundedString::new("Telos").unwrap(),
            icon_item: 1,
            frame: 0,
        }),
        S2cMessage::Disconnect(Disconnect {
            reason: DisconnectReason::ServerFull,
            message: BoundedString::new("Server is full").unwrap(),
        }),
    ];

    for msg in messages {
        let phase = msg.phase().unwrap_or(ConnectionPhase::Hello);
        let mut buf = Vec::new();
        encode_s2c(&msg, &mut buf);

        let mut cursor = &buf[..];
        let decoded = decode_s2c(phase, &mut cursor).expect("failed to decode S2C message");
        assert_eq!(decoded, msg);
        assert!(cursor.is_empty());
    }
}

#[test]
fn test_lod_node_data_memory_and_wire_roundtrip() {
    let words = vec![0x1122_3344, 0x5566_7788];
    let msg = S2cMessage::LodNodeData(S2cLodNodeData {
        level: 2,
        node_x: -10,
        node_y: 4,
        node_z: 25,
        version: 3,
        quad_count: 1,
        palette_count: 2,
        payload: LodPayload::Memory(Arc::new(words.clone())),
    });

    let mut buf = Vec::new();
    encode_s2c(&msg, &mut buf);

    let mut cursor = &buf[..];
    let decoded = decode_s2c(ConnectionPhase::Play, &mut cursor).unwrap();
    match decoded {
        S2cMessage::LodNodeData(data) => {
            assert_eq!(data.level, 2);
            assert_eq!(data.node_x, -10);
            assert_eq!(data.node_y, 4);
            assert_eq!(data.node_z, 25);
            assert_eq!(data.version, 3);
            assert_eq!(data.quad_count, 1);
            assert_eq!(data.palette_count, 2);
            assert_eq!(data.payload.to_words(), words);
        }
        other => panic!("Unexpected decoded message: {other:?}"),
    }
}

#[test]
fn test_chunk_data_snapshot_and_wire_roundtrip() {
    let snap = ChunkSnapshot::new_uniform(
        ChunkPos::new(10, 20, 30),
        BlockStateId(2),
        false,
        Some(ChunkLight::default()),
    );
    let msg = S2cMessage::ChunkData(S2cChunkData {
        chunk_x: 10,
        chunk_y: 20,
        chunk_z: 30,
        version: 1,
        epoch: 42,
        payload: ChunkPayload::Snapshot(Arc::new(snap.clone())),
    });

    let mut buf = Vec::new();
    encode_s2c(&msg, &mut buf);

    let mut cursor = &buf[..];
    let decoded =
        decode_s2c(ConnectionPhase::Play, &mut cursor).expect("failed to decode ChunkData");
    assert!(cursor.is_empty());

    if let S2cMessage::ChunkData(decoded_chunk) = decoded {
        assert_eq!(decoded_chunk.chunk_x, 10);
        assert_eq!(decoded_chunk.chunk_y, 20);
        assert_eq!(decoded_chunk.chunk_z, 30);
        assert_eq!(decoded_chunk.version, 1);
        assert_eq!(decoded_chunk.epoch, 42);

        let decoded_snap = decoded_chunk
            .payload
            .to_snapshot(decoded_chunk.pos())
            .unwrap();
        assert_eq!(decoded_snap.position(), snap.position());
        assert_eq!(decoded_snap.blocks(), snap.blocks());
        assert_eq!(decoded_snap.light(), snap.light());
    } else {
        panic!("Decoded unexpected message variant");
    }
}

#[test]
fn test_phase_isolation_enforcement() {
    // Encode a Play chat message
    let chat = C2sMessage::ChatMessage(C2sChatMessage {
        message: BoundedString::new("early message").unwrap(),
    });
    let mut buf = Vec::new();
    encode_c2s(&chat, &mut buf);

    // Attempting to decode during Hello phase must fail early with PhaseMismatch
    let mut cursor = &buf[..];
    let res = decode_c2s(ConnectionPhase::Hello, &mut cursor);
    assert!(res.is_err());
    assert!(matches!(
        res,
        Err(ProtocolError::PhaseMismatch {
            expected: ConnectionPhase::Hello,
            actual: 3
        })
    ));

    // Also test spoofed header phase (header claims Hello, but payload is msg_id 1 which does not exist in Hello)
    let payload = vec![1, 0]; // msg_id 1, followed by empty body
    let header = PacketHeader {
        magic: telos_protocol::PROTOCOL_MAGIC,
        version: telos_protocol::PROTOCOL_VERSION,
        flags: telos_protocol::PacketFlags::NONE,
        phase: telos_protocol::PHASE_HELLO,
        payload_len: payload.len() as u32,
        checksum: crc32fast::hash(&payload),
    };
    let mut spoofed = Vec::new();
    header.encode(&mut spoofed);
    spoofed.extend_from_slice(&payload);

    let mut cursor = &spoofed[..];
    let spoof_res = decode_c2s(ConnectionPhase::Hello, &mut cursor);
    assert!(matches!(
        spoof_res,
        Err(ProtocolError::UnknownMessageId {
            phase: "Hello",
            id: 1
        })
    ));
}

#[test]
fn test_frame_bounds_and_peek() {
    let msg = C2sMessage::KeepAlive(C2sKeepAlive { id: 1234 });
    let mut buf = Vec::new();
    encode_c2s(&msg, &mut buf);

    // Peek when buffer is incomplete (less than 16 bytes)
    assert_eq!(peek_frame(&buf[..1]).unwrap(), None);
    assert_eq!(peek_frame(&buf[..15]).unwrap(), None);

    // Peek when buffer has header but incomplete payload
    assert_eq!(peek_frame(&buf[..17]).unwrap(), None);

    // Peek when buffer is complete
    let peeked = peek_frame(&buf).unwrap().expect("should have full frame");
    assert_eq!(peeked.0, telos_protocol::PACKET_HEADER_SIZE);
    assert_eq!(peeked.0 + peeked.1, buf.len());

    // Artificial oversized frame with valid header
    let header = PacketHeader {
        magic: telos_protocol::PROTOCOL_MAGIC,
        version: telos_protocol::PROTOCOL_VERSION,
        flags: telos_protocol::PacketFlags::NONE,
        phase: telos_protocol::PHASE_PLAY,
        payload_len: (MAX_FRAME_SIZE + 1) as u32,
        checksum: 0,
    };
    let mut oversized = Vec::new();
    header.encode(&mut oversized);
    oversized.extend(vec![0u8; 10]);
    assert!(matches!(
        peek_frame(&oversized),
        Err(ProtocolError::FrameTooLarge { .. })
    ));
}

#[test]
fn test_checksum_corruption_detection() {
    let msg = C2sMessage::ChatMessage(C2sChatMessage {
        message: BoundedString::new("Important text").unwrap(),
    });
    let mut buf = Vec::new();
    encode_c2s(&msg, &mut buf);

    // Corrupt one bit in the payload (after 16-byte header)
    buf[16] ^= 0x01;

    let mut cursor = &buf[..];
    let res = decode_c2s(ConnectionPhase::Play, &mut cursor);
    assert!(matches!(res, Err(ProtocolError::ChecksumMismatch { .. })));
}

#[test]
fn test_invalid_magic_rejected() {
    let msg = C2sMessage::KeepAlive(C2sKeepAlive { id: 99 });
    let mut buf = Vec::new();
    encode_c2s(&msg, &mut buf);

    // Corrupt magic bytes
    buf[0] = b'B';
    buf[1] = b'A';
    buf[2] = b'D';
    buf[3] = b'!';

    let mut cursor = &buf[..];
    let res = decode_c2s(ConnectionPhase::Play, &mut cursor);
    assert!(matches!(
        res,
        Err(ProtocolError::InvalidMagic([b'B', b'A', b'D', b'!']))
    ));
}

#[test]
fn test_protocol_version_mismatch_rejected() {
    let msg = C2sMessage::KeepAlive(C2sKeepAlive { id: 99 });
    let mut buf = Vec::new();
    encode_c2s(&msg, &mut buf);

    // Corrupt version field (offset 4..6 in header)
    let bad_version: u16 = 999;
    buf[4..6].copy_from_slice(&bad_version.to_le_bytes());

    // Update checksum since header changed? Checksum is for payload, not header
    let mut cursor = &buf[..];
    let res = decode_c2s(ConnectionPhase::Play, &mut cursor);
    assert!(matches!(
        res,
        Err(ProtocolError::ProtocolVersionMismatch {
            expected: 1,
            actual: 999,
        })
    ));
}

#[test]
fn test_trailing_bytes_rejected() {
    let msg = C2sMessage::KeepAlive(C2sKeepAlive { id: 100 });
    let mut buf = Vec::new();
    encode_c2s(&msg, &mut buf);

    // Decode directly into payload, append 1 trailing byte and re-wrap in header
    let mut payload = Vec::new();
    encode_varint(msg.message_id(), &mut payload);
    msg.encode_body(&mut payload);
    payload.push(0xEE); // rogue trailing byte

    let header = PacketHeader {
        magic: telos_protocol::PROTOCOL_MAGIC,
        version: telos_protocol::PROTOCOL_VERSION,
        flags: telos_protocol::PacketFlags::NONE,
        phase: telos_protocol::PHASE_PLAY,
        payload_len: payload.len() as u32,
        checksum: crc32fast::hash(&payload),
    };
    let mut corrupted = Vec::new();
    header.encode(&mut corrupted);
    corrupted.extend_from_slice(&payload);

    let mut cursor = &corrupted[..];
    let res = decode_c2s(ConnectionPhase::Play, &mut cursor);
    assert!(matches!(
        res,
        Err(ProtocolError::TrailingBytes { count: 1 })
    ));
}

#[test]
fn test_rigorous_semantic_domain_validation() {
    // 1. NaN coordinate rejection
    let pos_nan = C2sMessage::PlayerPosition(C2sPlayerPosition {
        x: f64::NAN,
        y: 64.0,
        z: 0.0,
        yaw: 0.0,
        pitch: 0.0,
        on_ground: true,
    });
    let mut buf = Vec::new();
    encode_c2s(&pos_nan, &mut buf);
    let mut cursor = &buf[..];
    let res = decode_c2s(ConnectionPhase::Play, &mut cursor);
    assert!(matches!(
        res,
        Err(ProtocolError::InvalidValue {
            field: "position",
            ..
        })
    ));

    // 2. Out-of-bounds coordinate rejection
    let pos_oob = C2sMessage::PlayerPosition(C2sPlayerPosition {
        x: 50_000_000.0,
        y: 64.0,
        z: 0.0,
        yaw: 0.0,
        pitch: 0.0,
        on_ground: true,
    });
    let mut buf = Vec::new();
    encode_c2s(&pos_oob, &mut buf);
    let mut cursor = &buf[..];
    let res = decode_c2s(ConnectionPhase::Play, &mut cursor);
    assert!(matches!(
        res,
        Err(ProtocolError::InvalidValue {
            field: "position",
            ..
        })
    ));

    // 3. Invalid hit face rejection (> 5)
    let bad_action = C2sMessage::BlockAction(C2sBlockAction {
        sequence: 1,
        action: BlockActionKind::Place {
            state_id: BlockStateId::new(1),
            hit_face: 6, // invalid face
        },
        x: 0,
        y: 64,
        z: 0,
        input_tick: 1,
    });
    let mut buf = Vec::new();
    encode_c2s(&bad_action, &mut buf);
    let mut cursor = &buf[..];
    let res = decode_c2s(ConnectionPhase::Play, &mut cursor);
    assert!(matches!(
        res,
        Err(ProtocolError::InvalidValue {
            field: "hit_face",
            ..
        })
    ));

    // 4. Invalid username rejection
    let bad_login = C2sMessage::LoginStart(C2sLoginStart {
        username: BoundedString::new("bad user!").unwrap(), // contains spaces and !
        mode: AuthMode::Offline,
    });
    let mut buf = Vec::new();
    encode_c2s(&bad_login, &mut buf);
    let mut cursor = &buf[..];
    let res = decode_c2s(ConnectionPhase::Login, &mut cursor);
    assert!(matches!(
        res,
        Err(ProtocolError::InvalidValue {
            field: "username",
            ..
        })
    ));
}

#[test]
fn test_transparent_lz4_compression_on_large_payload() {
    // Construct a large registry payload (> 512 bytes)
    let mut entries = Vec::new();
    for i in 0..100 {
        entries.push(BoundedString::new(format!("telos:custom_block_variant_{i}")).unwrap());
    }
    let msg = S2cMessage::RegistryData(S2cRegistryData {
        registry_id: BoundedString::new("telos:blocks").unwrap(),
        content_hash: [0xbb; 32],
        entries: BoundedVec::new(entries).unwrap(),
    });

    let mut buf = Vec::new();
    encode_s2c(&msg, &mut buf);

    // Verify header indicates compression
    let mut cursor = &buf[..];
    let header = PacketHeader::decode(&mut cursor).unwrap();
    assert!(
        header
            .flags
            .contains(telos_protocol::PacketFlags::COMPRESSED)
    );

    // Verify roundtrip decoding transparently decompresses and validates checksum
    let mut decode_cursor = &buf[..];
    let decoded = decode_s2c(ConnectionPhase::Config, &mut decode_cursor).unwrap();
    assert_eq!(decoded, msg);
    assert!(decode_cursor.is_empty());
}

#[test]
fn test_command_suggestion_packets_roundtrip() {
    // 1. C2sCommandSuggest
    let req = C2sMessage::CommandSuggest(C2sCommandSuggest {
        id: 42,
        command: BoundedString::new("/tp @p ~10 ~ ").unwrap(),
        cursor: 13,
    });

    let mut buf = Vec::new();
    encode_c2s(&req, &mut buf);

    let mut cursor = &buf[..];
    let decoded =
        decode_c2s(ConnectionPhase::Play, &mut cursor).expect("failed to decode C2sCommandSuggest");
    assert_eq!(decoded, req);
    assert!(cursor.is_empty());

    // 2. S2cCommandSuggestions
    let matches = vec![
        BoundedString::new("~").unwrap(),
        BoundedString::new("~10").unwrap(),
    ];
    let tooltips = vec![
        BoundedString::new("current y").unwrap(),
        BoundedString::new("offset y").unwrap(),
    ];
    let resp = S2cMessage::CommandSuggestions(S2cCommandSuggestions {
        id: 42,
        start: 13,
        length: 0,
        matches: BoundedVec::new(matches).unwrap(),
        tooltips: BoundedVec::new(tooltips).unwrap(),
    });

    let mut buf_s2c = Vec::new();
    encode_s2c(&resp, &mut buf_s2c);

    let mut cursor_s2c = &buf_s2c[..];
    let decoded_s2c = decode_s2c(ConnectionPhase::Play, &mut cursor_s2c)
        .expect("failed to decode S2cCommandSuggestions");
    assert_eq!(decoded_s2c, resp);
    assert!(cursor_s2c.is_empty());
}

#[test]
fn test_container_packets_roundtrip() {
    // 1. S2cOpenContainer
    let mut slots = Vec::new();
    slots.push(SlotData { item: 1, count: 64 });
    for _ in 1..27 {
        slots.push(SlotData { item: 0, count: 0 });
    }
    let open_msg = S2cMessage::OpenContainer(S2cOpenContainer {
        window_id: 1,
        container_kind: 0,
        title: BoundedString::new("Chest").unwrap(),
        slots: BoundedVec::new(slots).unwrap(),
        x: 10,
        y: 64,
        z: -20,
    });
    let mut buf = Vec::new();
    encode_s2c(&open_msg, &mut buf);
    let mut cursor = &buf[..];
    let decoded_open =
        decode_s2c(ConnectionPhase::Play, &mut cursor).expect("decode S2cOpenContainer");
    assert_eq!(decoded_open, open_msg);
    assert!(cursor.is_empty());

    // 2. C2sCloseContainer
    let close_c2s = C2sMessage::CloseContainer(C2sCloseContainer { window_id: 1 });
    let mut buf_c2s = Vec::new();
    encode_c2s(&close_c2s, &mut buf_c2s);
    let mut cursor_c2s = &buf_c2s[..];
    let decoded_close_c2s =
        decode_c2s(ConnectionPhase::Play, &mut cursor_c2s).expect("decode C2sCloseContainer");
    assert_eq!(decoded_close_c2s, close_c2s);
    assert!(cursor_c2s.is_empty());

    // 3. S2cCloseContainer
    let close_s2c = S2cMessage::CloseContainer(S2cCloseContainer { window_id: 1 });
    let mut buf_s2c = Vec::new();
    encode_s2c(&close_s2c, &mut buf_s2c);
    let mut cursor_s2c = &buf_s2c[..];
    let decoded_close_s2c =
        decode_s2c(ConnectionPhase::Play, &mut cursor_s2c).expect("decode S2cCloseContainer");
    assert_eq!(decoded_close_s2c, close_s2c);
    assert!(cursor_s2c.is_empty());

    // 4. S2cBlockEvent
    let block_event = S2cMessage::BlockEvent(S2cBlockEvent {
        x: 10,
        y: 64,
        z: -20,
        action: 1,
        param: 1,
    });
    let mut buf_event = Vec::new();
    encode_s2c(&block_event, &mut buf_event);
    let mut cursor_event = &buf_event[..];
    let decoded_event =
        decode_s2c(ConnectionPhase::Play, &mut cursor_event).expect("decode S2cBlockEvent");
    assert_eq!(decoded_event, block_event);
    assert!(cursor_event.is_empty());

    // 5. S2cContainerProperty
    let prop_msg = S2cMessage::ContainerProperty(S2cContainerProperty {
        window_id: 1,
        property_id: 2, // cook_progress
        value: 145,
    });
    let mut buf_prop = Vec::new();
    encode_s2c(&prop_msg, &mut buf_prop);
    let mut cursor_prop = &buf_prop[..];
    let decoded_prop =
        decode_s2c(ConnectionPhase::Play, &mut cursor_prop).expect("decode S2cContainerProperty");
    assert_eq!(decoded_prop, prop_msg);
    assert!(cursor_prop.is_empty());
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_modal_form_wire_roundtrips() {
    // 1. S2cModalFormRequest ActionForm
    let form = ModalFormData::Action(
        ActionForm::new("Help Menu", "Choose an option:")
            .button("Commands")
            .button("Rules"),
    );
    let req = S2cModalFormRequest::new(101, &form).expect("create form request");
    let req_msg = S2cMessage::ModalFormRequest(req);
    let mut req_buf = Vec::new();
    encode_s2c(&req_msg, &mut req_buf);
    let mut cursor = &req_buf[..];
    let decoded_s2c =
        decode_s2c(ConnectionPhase::Play, &mut cursor).expect("decode S2cModalFormRequest");
    assert_eq!(decoded_s2c, req_msg);
    assert!(cursor.is_empty());

    if let S2cMessage::ModalFormRequest(r) = decoded_s2c {
        assert_eq!(r.form_id, 101);
        let parsed_form = r.parse_form().expect("parse form data");
        assert_eq!(parsed_form, form);
    } else {
        panic!("expected ModalFormRequest");
    }

    // 2. C2sModalFormResponse Action success
    let action_resp = FormResponseData::Action { button_index: 1 };
    let c2s_resp = C2sModalFormResponse::success(101, &action_resp).expect("create c2s response");
    let c2s_msg = C2sMessage::ModalFormResponse(c2s_resp);
    let mut c2s_buf = Vec::new();
    encode_c2s(&c2s_msg, &mut c2s_buf);
    let mut cursor = &c2s_buf[..];
    let decoded_c2s =
        decode_c2s(ConnectionPhase::Play, &mut cursor).expect("decode C2sModalFormResponse");
    assert_eq!(decoded_c2s, c2s_msg);
    assert!(cursor.is_empty());

    if let C2sMessage::ModalFormResponse(r) = decoded_c2s {
        assert_eq!(r.form_id, 101);
        assert!(r.has_response);
        let parsed_resp = r.parse_action_response().expect("parse action response");
        assert_eq!(parsed_resp, action_resp);
    } else {
        panic!("expected ModalFormResponse");
    }

    // 3. C2sModalFormResponse Cancelled
    let cancel_resp = C2sModalFormResponse::cancelled(101, FormCancelReason::UserClosed);
    let cancel_msg = C2sMessage::ModalFormResponse(cancel_resp);
    let mut cancel_buf = Vec::new();
    encode_c2s(&cancel_msg, &mut cancel_buf);
    let mut cursor = &cancel_buf[..];
    let decoded_cancel = decode_c2s(ConnectionPhase::Play, &mut cursor).expect("decode cancel");
    assert_eq!(decoded_cancel, cancel_msg);
    if let C2sMessage::ModalFormResponse(r) = decoded_cancel {
        assert!(!r.has_response);
        assert_eq!(r.cancel_reason, Some(0));
    } else {
        panic!("expected cancelled response");
    }

    // 4. ModalForm roundtrip
    let modal = ModalFormData::Modal(ModalForm::new("Confirm", "Are you sure?", "Yes", "No"));
    let modal_req = S2cModalFormRequest::new(102, &modal).expect("create modal req");
    let mut modal_buf = Vec::new();
    encode_s2c(&S2cMessage::ModalFormRequest(modal_req), &mut modal_buf);
    let mut cursor_modal = &modal_buf[..];
    let decoded_modal = decode_s2c(ConnectionPhase::Play, &mut cursor_modal).expect("decode modal");
    if let S2cMessage::ModalFormRequest(r) = decoded_modal {
        assert_eq!(r.form_id, 102);
        assert_eq!(r.parse_form().unwrap(), modal);
    } else {
        panic!("expected modal req");
    }

    let modal_resp = FormResponseData::Modal { confirmed: true };
    let modal_c2s = C2sModalFormResponse::success(102, &modal_resp).unwrap();
    let mut modal_c2s_buf = Vec::new();
    encode_c2s(
        &C2sMessage::ModalFormResponse(modal_c2s),
        &mut modal_c2s_buf,
    );
    let mut cursor_modal_c2s = &modal_c2s_buf[..];
    let decoded_modal_c2s =
        decode_c2s(ConnectionPhase::Play, &mut cursor_modal_c2s).expect("decode modal resp");
    if let C2sMessage::ModalFormResponse(r) = decoded_modal_c2s {
        assert_eq!(r.parse_modal_response().unwrap(), modal_resp);
    } else {
        panic!("expected modal resp");
    }

    // 5. CustomForm roundtrip
    let custom = ModalFormData::Custom(
        CustomForm::new("Config")
            .label("Info")
            .toggle("Feature", true),
    );
    let custom_req = S2cModalFormRequest::new(103, &custom).expect("create custom req");
    let mut custom_buf = Vec::new();
    encode_s2c(&S2cMessage::ModalFormRequest(custom_req), &mut custom_buf);
    let mut cursor_custom = &custom_buf[..];
    let decoded_custom =
        decode_s2c(ConnectionPhase::Play, &mut cursor_custom).expect("decode custom");
    if let S2cMessage::ModalFormRequest(r) = decoded_custom {
        assert_eq!(r.form_id, 103);
        assert_eq!(r.parse_form().unwrap(), custom);
    } else {
        panic!("expected custom req");
    }

    let custom_resp = FormResponseData::Custom {
        values: vec![FormValue::Null, FormValue::Bool(true)],
    };
    let custom_c2s = C2sModalFormResponse::success(103, &custom_resp).unwrap();
    let mut custom_c2s_buf = Vec::new();
    encode_c2s(
        &C2sMessage::ModalFormResponse(custom_c2s),
        &mut custom_c2s_buf,
    );
    let mut cursor_custom_c2s = &custom_c2s_buf[..];
    let decoded_custom_c2s =
        decode_c2s(ConnectionPhase::Play, &mut cursor_custom_c2s).expect("decode custom resp");
    if let C2sMessage::ModalFormResponse(r) = decoded_custom_c2s {
        assert_eq!(r.parse_custom_response().unwrap(), custom_resp);
    } else {
        panic!("expected custom resp");
    }
}

#[test]
fn test_game_mode_wire_roundtrip() {
    for mode in 0..=3 {
        let msg = S2cMessage::GameMode(S2cGameMode::new(mode, 0b0011_1111, 20.0, 4.5, 6.0));
        let mut buf = Vec::new();
        encode_s2c(&msg, &mut buf);
        let mut cursor = &buf[..];
        let decoded = decode_s2c(ConnectionPhase::Play, &mut cursor).expect("decode S2cGameMode");
        assert_eq!(msg, decoded);
        assert!(cursor.is_empty());

        if let S2cMessage::GameMode(g) = decoded {
            assert_eq!(g.game_mode, mode);
            assert_eq!(g.flags, 0b0011_1111);
            assert!((g.fly_speed - 20.0).abs() < 1e-5);
            assert!((g.walk_speed - 4.5).abs() < 1e-5);
            assert!((g.reach_distance - 6.0).abs() < 1e-5);
        } else {
            panic!("expected GameMode");
        }
    }
}

#[test]
fn test_content_manifest_wire_roundtrip() {
    let custom_blocks = vec![
        CustomBlockDefWire {
            identifier: BoundedString::new("custom:ruby_block").unwrap(),
            state_id: 85,
            flags: 0x0005,
            shape_kind: 0,
            light_emission: 12,
            hardness: 3.0,
            blast_resistance: 9.0,
            texture_name: BoundedString::new("ruby_block").unwrap(),
            base_color: [220, 20, 60, 255],
        },
        CustomBlockDefWire {
            identifier: BoundedString::new("custom:glowing_crystal").unwrap(),
            state_id: 86,
            flags: 0x0025,
            shape_kind: 3,
            light_emission: 15,
            hardness: 1.0,
            blast_resistance: 3.0,
            texture_name: BoundedString::new("glowing_crystal").unwrap(),
            base_color: [0, 255, 255, 255],
        },
    ];

    let custom_items = vec![
        CustomItemDefWire {
            identifier: BoundedString::new("custom:ruby").unwrap(),
            item_id: 91,
            name: BoundedString::new("Ruby Gem").unwrap(),
            max_stack_size: 64,
            item_type_kind: 0,
        },
        CustomItemDefWire {
            identifier: BoundedString::new("custom:ruby_pickaxe").unwrap(),
            item_id: 92,
            name: BoundedString::new("Ruby Pickaxe").unwrap(),
            max_stack_size: 1,
            item_type_kind: 2,
        },
    ];

    let manifest = S2cContentManifest {
        custom_blocks: BoundedVec::new(custom_blocks).unwrap(),
        custom_items: BoundedVec::new(custom_items).unwrap(),
    };

    let msg = S2cMessage::ContentManifest(manifest.clone());
    let mut buf = Vec::new();
    encode_s2c(&msg, &mut buf);

    let mut cursor = &buf[..];
    let decoded =
        decode_s2c(ConnectionPhase::Config, &mut cursor).expect("decode S2cContentManifest");
    assert_eq!(msg, decoded);
    assert!(cursor.is_empty());

    if let S2cMessage::ContentManifest(m) = decoded {
        assert_eq!(m.custom_blocks.len(), 2);
        assert_eq!(m.custom_blocks[0].identifier.as_str(), "custom:ruby_block");
        assert_eq!(m.custom_blocks[0].light_emission, 12);
        assert_eq!(m.custom_blocks[0].base_color, [220, 20, 60, 255]);
        assert_eq!(m.custom_items.len(), 2);
        assert_eq!(m.custom_items[1].name.as_str(), "Ruby Pickaxe");
        assert_eq!(m.custom_items[1].max_stack_size, 1);
    } else {
        panic!("expected ContentManifest");
    }
}

#[test]
fn test_s2c_recipe_manifest_roundtrip() {
    let shaped = vec![CustomShapedRecipeWire {
        width: 3,
        height: 3,
        pattern: BoundedVec::new(vec![101, 101, 101, 0, 5, 0, 0, 5, 0]).unwrap(),
        result_item: 102,
        result_count: 1,
        mirrored: true,
        remainder_item: 0,
    }];

    let shapeless = vec![CustomShapelessRecipeWire {
        ingredients: BoundedVec::new(vec![101, 201]).unwrap(),
        result_item: 202,
        result_count: 4,
        remainder_item: 0,
    }];

    let smelting = vec![CustomSmeltingRecipeWire {
        input_item: 101,
        output_item: 103,
        output_count: 1,
        cook_duration: 160,
        experience: 0.7,
    }];

    let fuels = vec![CustomFuelWire {
        item_id: 301,
        burn_duration_ticks: 2400,
    }];

    let manifest = S2cRecipeManifest {
        shaped_recipes: BoundedVec::new(shaped).unwrap(),
        shapeless_recipes: BoundedVec::new(shapeless).unwrap(),
        smelting_recipes: BoundedVec::new(smelting).unwrap(),
        fuels: BoundedVec::new(fuels).unwrap(),
    };

    let msg = S2cMessage::RecipeManifest(manifest.clone());
    let mut buf = Vec::new();
    encode_s2c(&msg, &mut buf);

    let mut cursor = &buf[..];
    let decoded =
        decode_s2c(ConnectionPhase::Config, &mut cursor).expect("decode S2cRecipeManifest");
    assert_eq!(msg, decoded);
    assert!(cursor.is_empty());

    if let S2cMessage::RecipeManifest(m) = decoded {
        assert_eq!(m.shaped_recipes.len(), 1);
        assert_eq!(m.shaped_recipes[0].width, 3);
        assert_eq!(m.shaped_recipes[0].result_item, 102);
        assert!(m.shaped_recipes[0].mirrored);
        assert_eq!(m.shapeless_recipes.len(), 1);
        assert_eq!(m.shapeless_recipes[0].result_count, 4);
        assert_eq!(m.smelting_recipes.len(), 1);
        assert_eq!(m.smelting_recipes[0].cook_duration, 160);
        assert!((m.smelting_recipes[0].experience - 0.7).abs() < 1e-6);
        assert_eq!(m.fuels.len(), 1);
        assert_eq!(m.fuels[0].burn_duration_ticks, 2400);
    } else {
        panic!("expected RecipeManifest");
    }
}

#[test]
fn test_s2c_entity_effect_and_remove_envelope_roundtrip() {
    let effect_msg = S2cMessage::EntityEffect(S2cEntityEffect {
        entity_id: 100,
        effect_id: 18, // Blindness
        amplifier: 2,
        duration_ticks: 600,
        ambient: false,
        show_particles: true,
        show_icon: true,
    });

    let mut buf = Vec::new();
    encode_s2c(&effect_msg, &mut buf);

    let mut cursor = &buf[..];
    let decoded =
        decode_s2c(ConnectionPhase::Play, &mut cursor).expect("decode S2cMessage::EntityEffect");
    assert_eq!(effect_msg, decoded);
    assert!(cursor.is_empty());

    let remove_msg = S2cMessage::RemoveEntityEffect(S2cRemoveEntityEffect {
        entity_id: 100,
        effect_id: 18,
    });

    let mut buf_remove = Vec::new();
    encode_s2c(&remove_msg, &mut buf_remove);

    let mut cursor_remove = &buf_remove[..];
    let decoded_remove = decode_s2c(ConnectionPhase::Play, &mut cursor_remove)
        .expect("decode S2cMessage::RemoveEntityEffect");
    assert_eq!(remove_msg, decoded_remove);
    assert!(cursor_remove.is_empty());
}

proptest! {
    #[test]
    fn prop_varint_roundtrip(val: u32) {
        let mut buf = Vec::new();
        encode_varint(val, &mut buf);
        let mut cursor = &buf[..];
        let decoded = decode_varint(&mut cursor).unwrap();
        prop_assert_eq!(val, decoded);
        prop_assert!(cursor.is_empty());
    }

    #[test]
    fn prop_zigzag_roundtrip(val: i32) {
        let mut buf = Vec::new();
        encode_varint_zigzag(val, &mut buf);
        let mut cursor = &buf[..];
        let decoded = decode_varint_zigzag(&mut cursor).unwrap();
        prop_assert_eq!(val, decoded);
        prop_assert!(cursor.is_empty());
    }

    #[test]
    fn prop_varlong_roundtrip(val: u64) {
        let mut buf = Vec::new();
        encode_varlong(val, &mut buf);
        let mut cursor = &buf[..];
        let decoded = decode_varlong(&mut cursor).unwrap();
        prop_assert_eq!(val, decoded);
        prop_assert!(cursor.is_empty());
    }
}
