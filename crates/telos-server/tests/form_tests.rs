//! Integration tests for server-authoritative modal forms, response routing,
//! paginated `/help` form navigation, and pending form lifecycle.

use telos_core::form::{FormCancelReason, FormResponseData, ModalFormData};
use telos_net::{Connection, Lane, MemoryConnection, Payload};
use telos_protocol::bounded::BoundedString;
use telos_protocol::messages::{
    AuthMode, C2sChatMessage, C2sClientSettings, C2sConfigAck, C2sHello, C2sLoginStart, C2sMessage,
    C2sModalFormResponse, S2cMessage,
};
use telos_server::{Server, ServerConfig};

fn setup_server_and_player() -> (Server, MemoryConnection<C2sMessage, S2cMessage>, u64) {
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

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::LoginStart(C2sLoginStart {
                username: BoundedString::new("FormTester").unwrap(),
                mode: AuthMode::Offline,
            })),
        )
        .expect("send LoginStart");
    server.tick();

    while let Ok(Some(_)) = client_conn.try_recv() {}

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ClientSettings(C2sClientSettings {
                view_distance: 3,
                simulation_distance: 3,
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

    while let Ok(Some(_)) = client_conn.try_recv() {}

    (server, client_conn, session_id)
}

#[test]
fn test_help_command_triggers_paginated_action_form() {
    let (mut server, client_conn, session_id) = setup_server_and_player();

    // Player executes "/help" command in chat
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ChatMessage(C2sChatMessage {
                message: BoundedString::new("/help").unwrap(),
            })),
        )
        .expect("send chat /help");

    server.tick();

    // Check that server responded with S2cModalFormRequest
    let mut form_req = None;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::ModalFormRequest(req)) = incoming.into_msg() {
            form_req = Some(req);
        }
    }

    let form_req = form_req.expect("Expected S2cModalFormRequest from server");
    assert!(
        server
            .pending_forms
            .contains_key(&(session_id, form_req.form_id))
    );

    let form_data = form_req.parse_form().expect("Valid form data JSON");
    match form_data {
        ModalFormData::Action(action_form) => {
            assert!(action_form.title.contains("Page 1/"));
            // First page has 6 command buttons + Next Page + Close
            assert!(action_form.buttons.len() >= 8);
            assert!(action_form.buttons[0].text.contains("/time"));
        }
        _ => panic!("Expected ActionForm for /help"),
    }
}

#[test]
fn test_help_form_next_page_and_detail_navigation() {
    let (mut server, client_conn, session_id) = setup_server_and_player();

    // Direct dispatch of help form
    let form_id = server
        .send_help_form(session_id, 0)
        .expect("Form dispatched");
    server.tick();

    // Drain initial form request
    let mut form_req = None;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::ModalFormRequest(req)) = incoming.into_msg() {
            form_req = Some(req);
        }
    }
    assert_eq!(form_req.unwrap().form_id, form_id);

    // Click button 6: [ Next Page > ]
    let next_page_resp =
        C2sModalFormResponse::success(form_id, &FormResponseData::Action { button_index: 6 })
            .expect("Response created");

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ModalFormResponse(next_page_resp)),
        )
        .expect("send response");

    server.tick();

    // Receive page 2 form request
    let mut page2_req = None;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::ModalFormRequest(req)) = incoming.into_msg() {
            page2_req = Some(req);
        }
    }
    let page2_req = page2_req.expect("Expected page 2 form request");
    let page2_data = page2_req.parse_form().expect("Valid JSON");
    if let ModalFormData::Action(action) = page2_data {
        assert!(action.title.contains("Page 2/"));
    } else {
        panic!("Expected ActionForm on page 2");
    }

    // On page 2, click button 0 (/enchant) to inspect details
    let detail_resp = C2sModalFormResponse::success(
        page2_req.form_id,
        &FormResponseData::Action { button_index: 0 },
    )
    .expect("Response created");

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ModalFormResponse(detail_resp)),
        )
        .expect("send response");

    server.tick();

    // Receive detail form request
    let mut detail_req = None;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::ModalFormRequest(req)) = incoming.into_msg() {
            detail_req = Some(req);
        }
    }
    let detail_req = detail_req.expect("Expected detail form request");
    let detail_data = detail_req.parse_form().expect("Valid JSON");
    if let ModalFormData::Action(action) = detail_data {
        assert!(action.title.contains("/enchant"));
        assert!(action.content.contains("Syntax:"));
        assert_eq!(action.buttons[0].text, "[ < Back to Command Index ]");
    } else {
        panic!("Expected ActionForm for details");
    }

    // Click Back to return to page 2
    let back_resp = C2sModalFormResponse::success(
        detail_req.form_id,
        &FormResponseData::Action { button_index: 0 },
    )
    .expect("Response created");

    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ModalFormResponse(back_resp)),
        )
        .expect("send response");

    server.tick();

    // Receive returned page 2 request
    let mut return_req = None;
    while let Ok(Some(incoming)) = client_conn.try_recv() {
        if let Some(S2cMessage::ModalFormRequest(req)) = incoming.into_msg() {
            return_req = Some(req);
        }
    }
    let return_req = return_req.expect("Expected return form request");
    if let ModalFormData::Action(action) = return_req.parse_form().unwrap() {
        assert!(action.title.contains("Page 2/"));
    } else {
        panic!("Expected ActionForm return");
    }
}

#[test]
fn test_form_cancellation_and_cleanup() {
    let (mut server, client_conn, session_id) = setup_server_and_player();

    let form_id = server
        .send_help_form(session_id, 0)
        .expect("Form dispatched");
    assert!(server.pending_forms.contains_key(&(session_id, form_id)));

    // Client dismisses the form
    let cancel = C2sModalFormResponse::cancelled(form_id, FormCancelReason::UserClosed);
    client_conn
        .send(
            Lane::Control,
            Payload::Msg(C2sMessage::ModalFormResponse(cancel)),
        )
        .expect("send cancel");

    server.tick();

    // Pending form must be cleaned up
    assert!(!server.pending_forms.contains_key(&(session_id, form_id)));
}
