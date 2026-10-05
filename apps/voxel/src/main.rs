//! Client application executable for the voxel engine.

use clap::Parser;
use mimalloc::MiMalloc;
use vx_core::{TelemetryConfig, init_telemetry};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Voxel engine client.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Log level filter.
    #[arg(long, default_value = "info")]
    log: String,
}

fn main() {
    let args = Args::parse();

    init_telemetry(&TelemetryConfig {
        app_name: "voxel-client",
        default_filter: args.log.into(),
        ansi_colors: true,
    });

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        "Voxel client initialized (Phase 1 stub). Window and renderer will be initialized in Phase 2."
    );
}
