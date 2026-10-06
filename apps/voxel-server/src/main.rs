//! Dedicated authoritative headless server binary for the Telos voxel engine.

use anyhow::Context;
use clap::Parser;
use mimalloc::MiMalloc;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use vx_core::{AppDirs, FixedTimestep, TelemetryConfig, init_telemetry};
use vx_server::{Server, ServerConfig};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const BANNER: &str = r"
  _____     _           
 |_   _|___| |___ ___   
   | | | -_| | . |_ -|  
   |_| |___|_|___|___|  
  Telos Voxel Server
";

/// Dedicated authoritative server for the Telos voxel engine.
#[derive(Parser, Debug)]
#[command(author, version, about = "Telos dedicated authoritative voxel server", long_about = None)]
struct Args {
    /// Path to server configuration TOML file.
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Generate a default template `server.toml` file in current directory and exit.
    #[arg(long)]
    init_config: bool,

    /// Override network bind address (e.g. "0.0.0.0:25565").
    #[arg(short, long)]
    bind: Option<String>,

    /// Override network port (applies to bind address).
    #[arg(short, long)]
    port: Option<u16>,

    /// Procedural world generation seed override.
    #[arg(short, long)]
    seed: Option<u64>,

    /// Target simulation rate in ticks per second.
    #[arg(long)]
    tps: Option<u32>,

    /// Default view distance in chunks.
    #[arg(short, long)]
    view_distance: Option<u32>,

    /// World persistence save directory path.
    #[arg(long)]
    save_dir: Option<PathBuf>,

    /// Run in portable mode with all data, configuration, and logs isolated in current directory.
    #[arg(long)]
    portable: bool,

    /// Custom root application data directory override.
    #[arg(long)]
    data_dir: Option<PathBuf>,

    /// Custom configuration directory override.
    #[arg(long)]
    config_dir: Option<PathBuf>,

    /// Message of the day shown in connection listings.
    #[arg(long)]
    motd: Option<String>,

    /// Log level filter (e.g. info, debug, trace).
    #[arg(long, default_value = "info")]
    log: String,
}

#[allow(clippy::too_many_lines)]
fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // 1. Resolve application directories adhering to platform standards / portable mode
    let app_dirs = if args.portable {
        AppDirs::portable(".")
    } else if let Some(ref data) = args.data_dir {
        let mut dirs = AppDirs::from_data_dir(data);
        if let Some(ref cfg) = args.config_dir {
            dirs = dirs.with_config_dir(cfg);
        }
        dirs
    } else {
        let mut dirs = AppDirs::standard_with_local_fallback();
        if let Some(ref cfg) = args.config_dir {
            dirs = dirs.with_config_dir(cfg);
        }
        dirs
    };

    if args.init_config {
        let path = args
            .config
            .unwrap_or_else(|| app_dirs.config_file("server.toml"));
        if path.exists() {
            anyhow::bail!(
                "A configuration file already exists at '{}'.",
                path.display()
            );
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, ServerConfig::default_toml_template()).with_context(|| {
            format!(
                "Failed to write server.toml template to '{}'",
                path.display()
            )
        })?;
        println!("Generated 'server.toml' template at '{}'.", path.display());
        return Ok(());
    }

    init_telemetry(&TelemetryConfig {
        app_name: "voxel-server",
        default_filter: args.log.into(),
        ansi_colors: true,
    });

    println!("{BANNER}");
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        portable = app_dirs.is_portable(),
        data_dir = %app_dirs.data_dir().display(),
        config_dir = %app_dirs.config_dir().display(),
        "Starting Telos dedicated voxel server..."
    );

    // 2. Ensure runtime directories exist
    let _ = app_dirs.ensure_dirs_exist();

    // 3. Resolve configuration from file (if provided or default exists)
    let config_path = args.config.or_else(|| {
        let local_file = PathBuf::from("server.toml");
        if local_file.exists() {
            Some(local_file)
        } else {
            let app_file = app_dirs.config_file("server.toml");
            if app_file.exists() {
                Some(app_file)
            } else {
                None
            }
        }
    });

    let mut config = if let Some(ref path) = config_path {
        tracing::info!(path = %path.display(), "Loading server configuration from file");
        ServerConfig::load_from_file(path)
            .with_context(|| format!("Failed to load configuration from {}", path.display()))?
    } else {
        tracing::info!("No server.toml found; using default server configuration");
        ServerConfig::default()
    };

    // 4. Apply CLI overrides
    if let Some(bind_override) = args.bind {
        config.bind_address = bind_override;
    }
    if let Some(port_override) = args.port {
        let ip = config.bind_address.rsplit(':').nth(1).unwrap_or("0.0.0.0");
        config.bind_address = format!("{ip}:{port_override}");
    }
    if let Some(tps_override) = args.tps {
        config.tps = tps_override;
    }
    if let Some(vd_override) = args.view_distance {
        config.view_distance = vd_override;
    }
    if let Some(save_dir_override) = args.save_dir {
        config.save_directory = Some(save_dir_override);
    } else if config.save_directory.is_none() {
        config.save_directory = Some(app_dirs.world_save_dir("world"));
    }
    if let Some(motd_override) = args.motd {
        config.motd = motd_override;
    }

    let seed = args.seed.unwrap_or(1337);

    // 3. Initialize server instance
    let mut server = Server::new(seed, config.clone());

    // 4. Bind QUIC listener
    let bind_addr: SocketAddr = config
        .bind_address
        .parse()
        .with_context(|| format!("Invalid bind address '{}'", config.bind_address))?;
    server
        .listen_addr(bind_addr)
        .with_context(|| format!("Failed to bind QUIC listener on {bind_addr}"))?;

    tracing::info!(
        bind_address = %config.bind_address,
        target_tps = config.tps,
        view_distance = config.view_distance,
        seed = seed,
        worlds = server.worlds().len(),
        motd = %config.motd,
        "Server initialized and accepting connections"
    );

    for name in server.worlds().world_names() {
        let gen_kind = server
            .world_named(&name)
            .map(vx_server::ServerWorld::generator_kind)
            .unwrap_or_default();
        tracing::info!(world = %name, generator = ?gen_kind, "Dimension active");
    }

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        tracing::info!("Shutdown signal received (Ctrl+C), stopping server gracefully...");
        r.store(false, Ordering::SeqCst);
    })?;

    let mut timestep = FixedTimestep::new(config.tps);
    let mut last_stats = Instant::now();
    let mut tick_counter: u64 = 0;
    let mut total_tick_duration = Duration::ZERO;

    while running.load(Ordering::SeqCst) && !server.is_shutdown_requested() {
        let loop_start = Instant::now();

        timestep.advance(|tick_num| {
            let tick_start = Instant::now();
            let _span = tracing::info_span!("tick", num = tick_num).entered();

            server.tick();
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
                active_sessions = server.session_count(),
                total_ticks = timestep.total_ticks(),
                "Server tick health"
            );

            tick_counter = 0;
            total_tick_duration = Duration::ZERO;
            last_stats = Instant::now();
        }

        let elapsed = loop_start.elapsed();
        let target_duration = Duration::from_millis(1);
        if let Some(remaining) = target_duration.checked_sub(elapsed) {
            thread::sleep(remaining);
        }
    }

    tracing::info!("Flushing dirty world data across all dimensions...");
    let saved = server.save_and_flush()?;
    tracing::info!(
        saved_chunks = saved,
        final_tick = timestep.total_ticks(),
        "Server loop terminated cleanly. All world data saved to disk."
    );

    Ok(())
}
