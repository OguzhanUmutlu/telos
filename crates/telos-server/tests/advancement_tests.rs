//! Integration tests for server-authoritative advancement evaluation, toasts, and persistence.

use std::fs;
use telos_net::{Connection, Lane, MemoryConnection, Payload};
use telos_protocol::bounded::BoundedString;
use telos_protocol::messages::{
    AuthMode, C2sClientSettings, C2sConfigAck, C2sHello, C2sLoginStart, C2sMessage, S2cMessage,
};
use telos_server::{Server, ServerConfig};

fn complete_handshake(
    server: &mut Server,
    client_conn: &MemoryConnection<C2sMessage, S2cMessage>,
    username: &str,
) -> (u32, Vec<S2cMessage>) {
    // 1. Client sends Hello
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

    // Drain HelloReply
    let _ = client_conn.try_recv().unwrap();

    // 2. Client sends LoginStart
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::LoginStart(C2sLoginStart {
                username: BoundedString::new(username).unwrap(),
                mode: AuthMode::Offline,
            })),
        )
        .expect("send LoginStart");
    server.tick();

    // Drain LoginSuccess, RegistryData (blocks), RegistryData (items), ConfigDone
    while let Ok(Some(_)) = client_conn.try_recv() {}

    // 3. Client sends ClientSettings and ConfigAck
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ClientSettings(C2sClientSettings {
                view_distance: 2,
                simulation_distance: 2,
                locale: BoundedString::new("en_US").unwrap(),
            })),
        )
        .expect("send ClientSettings");
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ConfigAck(C2sConfigAck)),
        )
        .expect("send ConfigAck");
    server.tick();

    // Collect all join messages
    let mut msgs = Vec::new();
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(msg) = incoming.into_msg() {
            msgs.push(msg);
        }
    }
    (1, msgs)
}

#[test]
fn test_advancements_root_auto_grant_on_join() {
    let config = ServerConfig {
        tps: 20,
        view_distance: 2,
        vertical_view_distance: 2,
        ..Default::default()
    };
    let mut server = Server::new(20001, config);
    let (server_conn, client_conn) = MemoryConnection::pair_default();
    let _session_id = server.add_connection(Box::new(server_conn));

    let (_, join_msgs) = complete_handshake(&mut server, &client_conn, "AdvancementTester");

    // Verify initial AdvancementUpdate reset_all message is sent on join containing root advancements
    let mut found_initial_update = false;
    for msg in &join_msgs {
        if let S2cMessage::AdvancementUpdate(update) = msg
            && update.reset_all
        {
            let ids: Vec<&str> = update.advancements.iter().map(|a| a.id.as_str()).collect();
            assert!(
                ids.contains(&"telos:story/root"),
                "Expected story root in initial advancement update"
            );
            assert!(
                ids.contains(&"telos:adventure/root"),
                "Expected adventure root in initial advancement update"
            );
            found_initial_update = true;
        }
    }
    assert!(
        found_initial_update,
        "Expected initial S2cAdvancementUpdate message with reset_all"
    );
}

#[test]
fn test_advancements_persistence_across_sessions() {
    let tmp_dir = tempfile::tempdir().expect("create tempdir");
    let save_dir = tmp_dir.path().to_path_buf();

    let config = ServerConfig {
        tps: 20,
        view_distance: 2,
        vertical_view_distance: 2,
        save_directory: Some(save_dir.clone()),
        ..Default::default()
    };
    let mut server = Server::new(20002, config.clone());

    // Connect player and complete handshake
    let (server_conn, client_conn) = MemoryConnection::pair_default();
    let session_id = server.add_connection(Box::new(server_conn));
    let (_, join_msgs) = complete_handshake(&mut server, &client_conn, "PersistentPlayer");

    // Starter inventory contains Oak Log (5), which triggers Getting Wood toast on join
    let mut received_toast = false;
    for msg in &join_msgs {
        if let S2cMessage::AdvancementToast(toast) = msg
            && toast.id.as_str() == "telos:story/mine_wood"
        {
            received_toast = true;
            assert_eq!(toast.title.as_str(), "Getting Wood");
        }
    }
    // If not in join messages, trigger it explicitly
    if !received_toast {
        server.evaluate_player_advancements(
            session_id,
            &telos_sim::CriterionTrigger::Inventory {
                item: 5, // Oak Log
                count: 1,
            },
        );
        while let Ok(Some(incoming)) = client_conn.try_recv() {
            if let Some(S2cMessage::AdvancementToast(toast)) = incoming.into_msg()
                && toast.id.as_str() == "telos:story/mine_wood"
            {
                received_toast = true;
            }
        }
    }
    assert!(received_toast, "Expected Getting Wood advancement toast");

    // Force server save and flush
    let _ = server.save_and_flush();

    // Verify advancement JSON file was persisted to disk
    let adv_file = save_dir.join("advancements").join("PersistentPlayer.json");
    assert!(
        adv_file.exists(),
        "Expected PersistentPlayer.json to exist on disk"
    );
    let content = fs::read_to_string(&adv_file).expect("read advancement json");
    assert!(
        content.contains("telos:story/mine_wood"),
        "JSON should contain unlocked advancement"
    );

    // Drop first server
    drop(server);

    // Create a new server loading the same save directory
    let mut server2 = Server::new(20003, config);
    let (server_conn2, client_conn2) = MemoryConnection::pair_default();
    let _ = server2.add_connection(Box::new(server_conn2));
    let (_, join_msgs2) = complete_handshake(&mut server2, &client_conn2, "PersistentPlayer");

    // Verify the reconnected player has Getting Wood pre-completed in the initial reset_all update
    let mut reloaded_unlocked = false;
    for msg in join_msgs2 {
        if let S2cMessage::AdvancementUpdate(update) = msg
            && update.reset_all
            && update
                .advancements
                .iter()
                .any(|a| a.id.as_str() == "telos:story/mine_wood")
        {
            reloaded_unlocked = true;
        }
    }
    assert!(
        reloaded_unlocked,
        "Expected previously unlocked advancement to be synchronized upon reload"
    );
}
