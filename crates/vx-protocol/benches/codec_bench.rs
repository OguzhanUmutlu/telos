//! Criterion benchmarks for protocol serialization and deserialization.

#![allow(missing_docs)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use vx_protocol::bounded::{BoundedString, BoundedVec};
use vx_protocol::codec::{decode_c2s, decode_s2c, encode_c2s, encode_s2c};
use vx_protocol::messages::{
    C2sHello, C2sMessage, ConnectionPhase, S2cChatMessage, S2cMessage, S2cRegistryData,
};

fn bench_hello_codec(c: &mut Criterion) {
    let hello = C2sMessage::Hello(C2sHello {
        protocol: 1,
        build: BoundedString::new("voxel-0.1.0-dev").unwrap(),
        features: 0b1011_0101,
    });

    let mut buf = Vec::with_capacity(64);
    encode_c2s(&hello, &mut buf);

    c.bench_function("encode_c2s_hello", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(64);
            encode_c2s(black_box(&hello), &mut out);
            black_box(out);
        });
    });

    c.bench_function("decode_c2s_hello", |b| {
        b.iter(|| {
            let mut cursor = black_box(&buf[..]);
            let decoded = decode_c2s(ConnectionPhase::Hello, &mut cursor).unwrap();
            black_box(decoded);
        });
    });
}

fn bench_chat_codec(c: &mut Criterion) {
    let chat = S2cMessage::ChatMessage(S2cChatMessage {
        sender: BoundedString::new("ServerAdmin").unwrap(),
        message: BoundedString::new("Welcome to the voxel world! High performance networking.")
            .unwrap(),
        timestamp: 1_700_000_000_000,
    });

    let mut buf = Vec::with_capacity(128);
    encode_s2c(&chat, &mut buf);

    c.bench_function("encode_s2c_chat", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(128);
            encode_s2c(black_box(&chat), &mut out);
            black_box(out);
        });
    });

    c.bench_function("decode_s2c_chat", |b| {
        b.iter(|| {
            let mut cursor = black_box(&buf[..]);
            let decoded = decode_s2c(ConnectionPhase::Play, &mut cursor).unwrap();
            black_box(decoded);
        });
    });
}

fn bench_registry_codec(c: &mut Criterion) {
    let mut entries = Vec::with_capacity(128);
    for i in 0..100 {
        entries.push(BoundedString::new(format!("voxel:block_type_{i}")).unwrap());
    }
    let reg_msg = S2cMessage::RegistryData(S2cRegistryData {
        registry_id: BoundedString::new("voxel:block").unwrap(),
        content_hash: [0x55; 32],
        entries: BoundedVec::new(entries).unwrap(),
    });

    let mut buf = Vec::with_capacity(4096);
    encode_s2c(&reg_msg, &mut buf);

    c.bench_function("encode_s2c_registry_100_entries", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(4096);
            encode_s2c(black_box(&reg_msg), &mut out);
            black_box(out);
        });
    });

    c.bench_function("decode_s2c_registry_100_entries", |b| {
        b.iter(|| {
            let mut cursor = black_box(&buf[..]);
            let decoded = decode_s2c(ConnectionPhase::Config, &mut cursor).unwrap();
            black_box(decoded);
        });
    });
}

criterion_group!(
    benches,
    bench_hello_codec,
    bench_chat_codec,
    bench_registry_codec
);
criterion_main!(benches);
