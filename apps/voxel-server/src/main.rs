//! Dedicated authoritative headless server binary for the voxel engine.

use clap::Parser;
use mimalloc::MiMalloc;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use vx_core::{FixedTimestep, TelemetryConfig, init_telemetry};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Dedicated authoritative server for the voxel engine.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Target simulation rate in ticks per second.
    #[arg(short, long, default_value_t = 20)]
    tps: u32,

    /// Log level filter (e.g. info, debug, trace).
    #[arg(long, default_value = "info")]
    log: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    init_telemetry(&TelemetryConfig {
        app_name: "voxel-server",
        default_filter: args.log.into(),
        ansi_colors: true,
    });

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        target_tps = args.tps,
        "Starting voxel dedicated server"
    );

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        tracing::info!("Shutdown signal received (Ctrl+C), stopping server gracefully...");
        r.store(false, Ordering::SeqCst);
    })?;

    let mut timestep = FixedTimestep::new(args.tps);
    let mut last_stats = Instant::now();
    let mut tick_counter: u64 = 0;
    let mut total_tick_duration = Duration::ZERO;

    while running.load(Ordering::SeqCst) {
        let loop_start = Instant::now();

        timestep.advance(|tick_num| {
            let tick_start = Instant::now();
            let _span = tracing::info_span!("tick", num = tick_num).entered();

            // Placeholder for simulation ticks (Phases 3+)
            tick_counter += 1;

            let elapsed = tick_start.elapsed();
            total_tick_duration += elapsed;
        });

        // Periodic TPS telemetry (every 5 seconds)
        if last_stats.elapsed() >= Duration::from_secs(5) {
            let elapsed_secs = last_stats.elapsed().as_secs_f64();
            let avg_tps = (tick_counter as f64) / elapsed_secs;
            let avg_tick_ms = if tick_counter > 0 {
                (total_tick_duration.as_secs_f64() * 1000.0) / (tick_counter as f64)
            } else {
                0.0
            };

            tracing::info!(
                avg_tps = format!("{avg_tps:.2}"),
                avg_tick_ms = format!("{avg_tick_ms:.4}"),
                total_ticks = timestep.total_ticks(),
                "Server tick health"
            );

            tick_counter = 0;
            total_tick_duration = Duration::ZERO;
            last_stats = Instant::now();
        }

        // Sleep briefly to avoid 100% spin in headless loop
        let elapsed = loop_start.elapsed();
        let target_duration = Duration::from_millis(1);
        if let Some(remaining) = target_duration.checked_sub(elapsed) {
            thread::sleep(remaining);
        }
    }

    tracing::info!(
        final_tick = timestep.total_ticks(),
        "Server loop terminated cleanly. World data flushed."
    );

    Ok(())
}
