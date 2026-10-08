//! Integration tests for cryptographic player authentication, TOFU key pinning, and Account Authority certificates.

use telos_core::profile::{AccountAuthority, IdentityKeypair, PlayerUuid};
use telos_net::{Connection, Lane, MemoryConnection, Payload};
use telos_protocol::bounded::BoundedString;
use telos_protocol::messages::{
    AuthMode, C2sHello, C2sLoginProof, C2sLoginStart, C2sMessage, DisconnectReason, S2cMessage,
};
use telos_server::{Server, ServerConfig};

fn perform_hello(server: &mut Server, client_conn: &mut dyn Connection<C2sMessage, S2cMessage>) {
    let hello = C2sMessage::Hello(C2sHello {
        protocol: 1,
        features: 0,
        build: BoundedString::new("0.1.0").unwrap(),
    });
    client_conn
        .send(Lane::Control, Payload::Msg(hello))
        .expect("send hello");

    server.tick();

    let reply = client_conn.try_recv().expect("recv").expect("reply");
    let S2cMessage::HelloReply(_) = reply.into_msg().expect("msg") else {
        panic!("expected HelloReply");
    };
}

#[test]
fn test_offline_login_deterministic_uuid() {
    let config = ServerConfig {
        online_mode: false,
        ..Default::default()
    };
    let mut server = Server::new(1337, config);

    let (server_conn, mut client_conn) = MemoryConnection::pair_default();
    server.add_connection(Box::new(server_conn));

    perform_hello(&mut server, &mut client_conn);

    let login = C2sMessage::LoginStart(C2sLoginStart {
        username: BoundedString::new("Alex").unwrap(),
        mode: AuthMode::Offline,
    });
    client_conn
        .send(Lane::Control, Payload::Msg(login))
        .expect("send login");

    server.tick();

    let succ = client_conn
        .try_recv()
        .expect("recv")
        .expect("login success");
    let S2cMessage::LoginSuccess(succ) = succ.into_msg().expect("msg") else {
        panic!("expected LoginSuccess");
    };

    let expected_uuid = PlayerUuid::from_offline_name("Alex");
    assert_eq!(succ.player_uuid, expected_uuid.to_bytes());
    assert_eq!(succ.username.as_str(), "Alex");
}

#[test]
fn test_online_mode_rejects_offline_client() {
    let config = ServerConfig {
        online_mode: true,
        ..Default::default()
    };
    let mut server = Server::new(1337, config);

    let (server_conn, mut client_conn) = MemoryConnection::pair_default();
    server.add_connection(Box::new(server_conn));

    perform_hello(&mut server, &mut client_conn);

    let login = C2sMessage::LoginStart(C2sLoginStart {
        username: BoundedString::new("Steve").unwrap(),
        mode: AuthMode::Offline,
    });
    client_conn
        .send(Lane::Control, Payload::Msg(login))
        .expect("send login");

    server.tick();

    let msg = client_conn.try_recv().expect("recv").expect("disconnect");
    let S2cMessage::Disconnect(disc) = msg.into_msg().expect("msg") else {
        panic!("expected Disconnect");
    };
    assert_eq!(disc.reason, DisconnectReason::AuthFailed);
    assert!(disc.message.as_str().contains("online authentication"));
}

#[test]
fn test_keyed_cryptographic_challenge_response_handshake() {
    let config = ServerConfig {
        online_mode: true,
        ..Default::default()
    };
    let mut server = Server::new(1337, config);

    let (server_conn, mut client_conn) = MemoryConnection::pair_default();
    server.add_connection(Box::new(server_conn));

    perform_hello(&mut server, &mut client_conn);

    let client_kp = IdentityKeypair::generate().expect("keypair");
    let username = "Alice";

    // 1. Client initiates Keyed login
    let login = C2sMessage::LoginStart(C2sLoginStart {
        username: BoundedString::new(username).unwrap(),
        mode: AuthMode::Keyed,
    });
    client_conn
        .send(Lane::Control, Payload::Msg(login))
        .expect("send login start");

    server.tick();

    // 2. Server responds with cryptographic challenge nonce
    let challenge_msg = client_conn.try_recv().expect("recv").expect("challenge");
    let S2cMessage::LoginChallenge(challenge) = challenge_msg.into_msg().expect("msg") else {
        panic!("expected LoginChallenge");
    };
    assert_ne!(challenge.challenge_nonce, [0u8; 32]);

    // 3. Client constructs canonical signature payload and signs
    let mut payload = Vec::with_capacity(15 + 32 + username.len());
    payload.extend_from_slice(b"telos-login-v1:");
    payload.extend_from_slice(&challenge.challenge_nonce);
    payload.extend_from_slice(username.as_bytes());

    let signature = client_kp.sign(&payload);

    let proof = C2sMessage::LoginProof(C2sLoginProof {
        public_key: client_kp.public_key(),
        signature: signature.to_bytes(),
        certificate_data: None,
    });
    client_conn
        .send(Lane::Control, Payload::Msg(proof))
        .expect("send proof");

    server.tick();

    // 4. Server verifies signature and accepts login
    let succ_msg = client_conn
        .try_recv()
        .expect("recv")
        .expect("login success");
    let S2cMessage::LoginSuccess(succ) = succ_msg.into_msg().expect("msg") else {
        panic!("expected LoginSuccess");
    };

    let expected_uuid = PlayerUuid::from_public_key(&client_kp.public_key());
    assert_eq!(succ.player_uuid, expected_uuid.to_bytes());
    assert_eq!(succ.username.as_str(), "Alice");
}

#[test]
fn test_keyed_login_tampered_signature_rejected() {
    let config = ServerConfig {
        online_mode: false,
        ..Default::default()
    };
    let mut server = Server::new(1337, config);

    let (server_conn, mut client_conn) = MemoryConnection::pair_default();
    server.add_connection(Box::new(server_conn));

    perform_hello(&mut server, &mut client_conn);

    let client_kp = IdentityKeypair::generate().expect("keypair");
    let username = "Mallory";

    let login = C2sMessage::LoginStart(C2sLoginStart {
        username: BoundedString::new(username).unwrap(),
        mode: AuthMode::Keyed,
    });
    client_conn
        .send(Lane::Control, Payload::Msg(login))
        .expect("send login start");

    server.tick();

    let challenge_msg = client_conn.try_recv().expect("recv").expect("challenge");
    let S2cMessage::LoginChallenge(_) = challenge_msg.into_msg().expect("msg") else {
        panic!("expected LoginChallenge");
    };

    // Mallory sends corrupted signature
    let corrupted_sig = [0xEEu8; 64];
    let proof = C2sMessage::LoginProof(C2sLoginProof {
        public_key: client_kp.public_key(),
        signature: corrupted_sig,
        certificate_data: None,
    });
    client_conn
        .send(Lane::Control, Payload::Msg(proof))
        .expect("send bad proof");

    server.tick();

    let msg = client_conn.try_recv().expect("recv").expect("disconnect");
    let S2cMessage::Disconnect(disc) = msg.into_msg().expect("msg") else {
        panic!("expected Disconnect");
    };
    assert_eq!(disc.reason, DisconnectReason::AuthFailed);
    assert!(
        disc.message
            .as_str()
            .contains("Invalid cryptographic login signature")
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_tofu_hijacking_prevention() {
    let config = ServerConfig {
        enforce_tofu: true,
        ..Default::default()
    };
    let mut server = Server::new(1337, config);

    let kp_alice = IdentityKeypair::generate().expect("alice kp");
    let kp_bob = IdentityKeypair::generate().expect("bob kp");
    let username = "Alice";

    // --- 1. Alice connects first and pins her key ---
    {
        let (server_conn, mut client_conn) = MemoryConnection::pair_default();
        server.add_connection(Box::new(server_conn));
        perform_hello(&mut server, &mut client_conn);

        let login = C2sMessage::LoginStart(C2sLoginStart {
            username: BoundedString::new(username).unwrap(),
            mode: AuthMode::Keyed,
        });
        client_conn
            .send(Lane::Control, Payload::Msg(login))
            .unwrap();
        server.tick();

        let S2cMessage::LoginChallenge(challenge) =
            client_conn.try_recv().unwrap().unwrap().into_msg().unwrap()
        else {
            panic!("expected challenge");
        };

        let mut payload = Vec::with_capacity(15 + 32 + username.len());
        payload.extend_from_slice(b"telos-login-v1:");
        payload.extend_from_slice(&challenge.challenge_nonce);
        payload.extend_from_slice(username.as_bytes());

        let sig = kp_alice.sign(&payload);
        client_conn
            .send(
                Lane::Control,
                Payload::Msg(C2sMessage::LoginProof(C2sLoginProof {
                    public_key: kp_alice.public_key(),
                    signature: sig.to_bytes(),
                    certificate_data: None,
                })),
            )
            .unwrap();
        server.tick();

        let S2cMessage::LoginSuccess(_) =
            client_conn.try_recv().unwrap().unwrap().into_msg().unwrap()
        else {
            panic!("expected success");
        };
    }

    // --- 2. Bob attempts to log in as "Alice" with Bob's key ---
    {
        let (server_conn, mut client_conn) = MemoryConnection::pair_default();
        server.add_connection(Box::new(server_conn));
        perform_hello(&mut server, &mut client_conn);

        let login = C2sMessage::LoginStart(C2sLoginStart {
            username: BoundedString::new(username).unwrap(),
            mode: AuthMode::Keyed,
        });
        client_conn
            .send(Lane::Control, Payload::Msg(login))
            .unwrap();
        server.tick();

        let S2cMessage::LoginChallenge(challenge) =
            client_conn.try_recv().unwrap().unwrap().into_msg().unwrap()
        else {
            panic!("expected challenge");
        };

        let mut payload = Vec::with_capacity(15 + 32 + username.len());
        payload.extend_from_slice(b"telos-login-v1:");
        payload.extend_from_slice(&challenge.challenge_nonce);
        payload.extend_from_slice(username.as_bytes());

        let sig = kp_bob.sign(&payload);
        client_conn
            .send(
                Lane::Control,
                Payload::Msg(C2sMessage::LoginProof(C2sLoginProof {
                    public_key: kp_bob.public_key(),
                    signature: sig.to_bytes(),
                    certificate_data: None,
                })),
            )
            .unwrap();
        server.tick();

        let msg = client_conn.try_recv().unwrap().unwrap().into_msg().unwrap();
        let S2cMessage::Disconnect(disc) = msg else {
            panic!("expected Disconnect");
        };
        assert_eq!(disc.reason, DisconnectReason::AuthFailed);
        assert!(
            disc.message
                .as_str()
                .contains("bound to a different public key")
        );
    }

    // --- 3. Offline client attempts to log in as "Alice" ---
    {
        let (server_conn, mut client_conn) = MemoryConnection::pair_default();
        server.add_connection(Box::new(server_conn));
        perform_hello(&mut server, &mut client_conn);

        let login = C2sMessage::LoginStart(C2sLoginStart {
            username: BoundedString::new(username).unwrap(),
            mode: AuthMode::Offline,
        });
        client_conn
            .send(Lane::Control, Payload::Msg(login))
            .unwrap();
        server.tick();

        let msg = client_conn.try_recv().unwrap().unwrap().into_msg().unwrap();
        let S2cMessage::Disconnect(disc) = msg else {
            panic!("expected Disconnect");
        };
        assert_eq!(disc.reason, DisconnectReason::AuthFailed);
        assert!(
            disc.message
                .as_str()
                .contains("reserved for authenticated cryptographic players")
        );
    }
}

#[test]
fn test_trusted_account_authority_certificate() {
    let mut authority = AccountAuthority::new("AuthNet").expect("authority");
    let authority_pubkey = authority.public_key();

    let config = ServerConfig {
        trusted_authority_keys: vec![authority_pubkey],
        ..Default::default()
    };
    let mut server = Server::new(1337, config);

    let player_kp = IdentityKeypair::generate().expect("player kp");
    let username = "CertifiedPlayer";

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let cert = authority
        .issue_player_cert(username, &player_kp.public_key(), 3600, now)
        .expect("issue cert");
    let cert_bytes = serde_json::to_vec(&cert).expect("serialize cert");

    let (server_conn, mut client_conn) = MemoryConnection::pair_default();
    server.add_connection(Box::new(server_conn));
    perform_hello(&mut server, &mut client_conn);

    let login = C2sMessage::LoginStart(C2sLoginStart {
        username: BoundedString::new(username).unwrap(),
        mode: AuthMode::Keyed,
    });
    client_conn
        .send(Lane::Control, Payload::Msg(login))
        .unwrap();
    server.tick();

    let S2cMessage::LoginChallenge(challenge) =
        client_conn.try_recv().unwrap().unwrap().into_msg().unwrap()
    else {
        panic!("expected challenge");
    };

    let mut payload = Vec::with_capacity(15 + 32 + username.len());
    payload.extend_from_slice(b"telos-login-v1:");
    payload.extend_from_slice(&challenge.challenge_nonce);
    payload.extend_from_slice(username.as_bytes());

    let sig = player_kp.sign(&payload);
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::LoginProof(C2sLoginProof {
                public_key: player_kp.public_key(),
                signature: sig.to_bytes(),
                certificate_data: Some(cert_bytes),
            })),
        )
        .unwrap();
    server.tick();

    let S2cMessage::LoginSuccess(succ) =
        client_conn.try_recv().unwrap().unwrap().into_msg().unwrap()
    else {
        panic!("expected success");
    };
    assert_eq!(succ.username.as_str(), username);
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_usercache_file_persistence_across_restart() {
    let temp = tempfile::tempdir().expect("tempdir");
    let save_dir = temp.path().to_path_buf();

    let kp_alice = IdentityKeypair::generate().expect("alice kp");
    let kp_bob = IdentityKeypair::generate().expect("bob kp");
    let username = "PersistentUser";

    // 1. Run server, login Alice, save and drop server
    {
        let config = ServerConfig {
            save_directory: Some(save_dir.clone()),
            ..Default::default()
        };
        let mut server = Server::new(1337, config);

        let (server_conn, mut client_conn) = MemoryConnection::pair_default();
        server.add_connection(Box::new(server_conn));
        perform_hello(&mut server, &mut client_conn);

        let login = C2sMessage::LoginStart(C2sLoginStart {
            username: BoundedString::new(username).unwrap(),
            mode: AuthMode::Keyed,
        });
        client_conn
            .send(Lane::Control, Payload::Msg(login))
            .unwrap();
        server.tick();

        let S2cMessage::LoginChallenge(challenge) =
            client_conn.try_recv().unwrap().unwrap().into_msg().unwrap()
        else {
            panic!("expected challenge");
        };

        let mut payload = Vec::with_capacity(15 + 32 + username.len());
        payload.extend_from_slice(b"telos-login-v1:");
        payload.extend_from_slice(&challenge.challenge_nonce);
        payload.extend_from_slice(username.as_bytes());

        let sig = kp_alice.sign(&payload);
        client_conn
            .send(
                Lane::Control,
                Payload::Msg(C2sMessage::LoginProof(C2sLoginProof {
                    public_key: kp_alice.public_key(),
                    signature: sig.to_bytes(),
                    certificate_data: None,
                })),
            )
            .unwrap();
        server.tick();

        let S2cMessage::LoginSuccess(_) =
            client_conn.try_recv().unwrap().unwrap().into_msg().unwrap()
        else {
            panic!("expected success");
        };

        server.save_and_flush().expect("flush");
    }

    // Verify usercache.json exists on disk
    let usercache_file = save_dir.join("usercache.json");
    assert!(usercache_file.exists());

    // 2. Restart new server instance using same save_directory
    {
        let config = ServerConfig {
            save_directory: Some(save_dir),
            ..Default::default()
        };
        let mut server = Server::new(1337, config);

        let (server_conn, mut client_conn) = MemoryConnection::pair_default();
        server.add_connection(Box::new(server_conn));
        perform_hello(&mut server, &mut client_conn);

        // Bob tries to steal Alice's username
        let login = C2sMessage::LoginStart(C2sLoginStart {
            username: BoundedString::new(username).unwrap(),
            mode: AuthMode::Keyed,
        });
        client_conn
            .send(Lane::Control, Payload::Msg(login))
            .unwrap();
        server.tick();

        let S2cMessage::LoginChallenge(challenge) =
            client_conn.try_recv().unwrap().unwrap().into_msg().unwrap()
        else {
            panic!("expected challenge");
        };

        let mut payload = Vec::with_capacity(15 + 32 + username.len());
        payload.extend_from_slice(b"telos-login-v1:");
        payload.extend_from_slice(&challenge.challenge_nonce);
        payload.extend_from_slice(username.as_bytes());

        let sig = kp_bob.sign(&payload);
        client_conn
            .send(
                Lane::Control,
                Payload::Msg(C2sMessage::LoginProof(C2sLoginProof {
                    public_key: kp_bob.public_key(),
                    signature: sig.to_bytes(),
                    certificate_data: None,
                })),
            )
            .unwrap();
        server.tick();

        let msg = client_conn.try_recv().unwrap().unwrap().into_msg().unwrap();
        let S2cMessage::Disconnect(disc) = msg else {
            panic!("expected Disconnect");
        };
        assert_eq!(disc.reason, DisconnectReason::AuthFailed);
        assert!(
            disc.message
                .as_str()
                .contains("bound to a different public key")
        );
    }
}
