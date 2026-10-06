//! Integration and property tests for vx-protocol codecs and wire types.

use proptest::prelude::*;
use vx_protocol::bounded::{BoundedString, BoundedVec};
use vx_protocol::codec::{
    MAX_FRAME_SIZE, decode_c2s, decode_s2c, encode_c2s, encode_s2c, peek_frame,
};
use vx_protocol::error::ProtocolError;
use vx_protocol::messages::{
    AuthMode, C2sChatMessage, C2sClientSettings, C2sConfigAck, C2sHello, C2sKeepAlive,
    C2sKnownRegistries, C2sLoginStart, C2sMessage, ConnectionPhase, Disconnect, DisconnectReason,
    S2cChatMessage, S2cConfigDone, S2cHelloReply, S2cKeepAlive, S2cLoginSuccess, S2cMessage,
    S2cRegistryData,
};
use vx_protocol::varint::{
    decode_varint, decode_varint_zigzag, decode_varlong, encode_varint, encode_varint_zigzag,
    encode_varlong, varint_size,
};

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
            registry_id: BoundedString::new("voxel:item").unwrap(),
            entries: BoundedVec::new(vec![
                BoundedString::new("voxel:stone").unwrap(),
                BoundedString::new("voxel:dirt").unwrap(),
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
fn test_phase_isolation_enforcement() {
    // Encode a Play chat message
    let chat = C2sMessage::ChatMessage(C2sChatMessage {
        message: BoundedString::new("early message").unwrap(),
    });
    let mut buf = Vec::new();
    encode_c2s(&chat, &mut buf);

    // Attempting to decode during Hello phase must fail with UnknownMessageId or error
    let mut cursor = &buf[..];
    let res = decode_c2s(ConnectionPhase::Hello, &mut cursor);
    assert!(res.is_err());
    assert!(matches!(
        res,
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

    // Peek when buffer is incomplete
    assert_eq!(peek_frame(&buf[..1]).unwrap(), None);

    // Peek when buffer is complete
    let peeked = peek_frame(&buf).unwrap().expect("should have full frame");
    assert_eq!(peeked.0 + peeked.1, buf.len());

    // Artificial oversized frame
    let mut oversized = Vec::new();
    encode_varint((MAX_FRAME_SIZE + 1) as u32, &mut oversized);
    oversized.extend(vec![0u8; 10]);
    assert!(matches!(
        peek_frame(&oversized),
        Err(ProtocolError::FrameTooLarge { .. })
    ));
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
