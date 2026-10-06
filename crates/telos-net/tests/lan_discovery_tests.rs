//! Integration tests for LAN discovery beacon emission and listening.

use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};
use std::time::Duration;
use telos_net::lan::{LanBeacon, LanDiscoveryListener};

#[test]
fn test_lan_discovery_direct_emission_and_listener_polling() {
    // Bind listener on an ephemeral loopback port
    let listener_sock = UdpSocket::bind("127.0.0.1:0").expect("bind listener socket");
    let listener_port = listener_sock.local_addr().unwrap().port();
    let mut listener = LanDiscoveryListener::from_socket(listener_sock);

    // Prepare sender socket and beacon
    let sender_sock = UdpSocket::bind("127.0.0.1:0").expect("bind sender socket");
    let beacon = LanBeacon {
        flags: 0,
        protocol: 1,
        quic_port: 45678,
        cert_spki_sha256: [0xAA; 32],
        current_players: 2,
        max_players: 10,
        game_mode: 1, // Creative
        motd: "Creative Build Server".into(),
    };

    let encoded = beacon.encode();
    let target = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, listener_port));

    // Send beacon to listener
    sender_sock.send_to(&encoded, target).expect("send beacon");

    // Give kernel a tiny moment to process the local packet
    std::thread::sleep(Duration::from_millis(10));

    // Poll listener
    let servers = listener.active_servers();
    assert_eq!(servers.len(), 1);
    let s = &servers[0];
    assert_eq!(s.beacon.motd, "Creative Build Server");
    assert_eq!(s.beacon.quic_port, 45678);
    assert_eq!(s.beacon.current_players, 2);
    assert_eq!(s.beacon.max_players, 10);
    assert_eq!(s.beacon.game_mode, 1);
    assert_eq!(s.connect_addr().port(), 45678);
}
