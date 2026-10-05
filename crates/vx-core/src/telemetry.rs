//! Structured logging and telemetry initialization.

use std::{
    borrow::Cow,
    sync::atomic::{AtomicBool, Ordering},
};
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

static TELEMETRY_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Configuration options for engine telemetry.
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Service or app name for tracing spans.
    pub app_name: &'static str,
    /// Default log filter directive if `VOXEL_LOG` / `RUST_LOG` is not set.
    pub default_filter: Cow<'static, str>,
    /// Whether to enable ANSI colors in console output.
    pub ansi_colors: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            app_name: "voxel",
            default_filter: Cow::Borrowed("info"),
            ansi_colors: true,
        }
    }
}

/// Initializes global structured logging and telemetry.
///
/// Can be safely called multiple times; subsequent calls are no-ops.
///
/// Filters are configured via the `VOXEL_LOG` environment variable, falling back
/// to `RUST_LOG`, and finally to `config.default_filter`.
pub fn init_telemetry(config: &TelemetryConfig) {
    if TELEMETRY_INITIALIZED.swap(true, Ordering::SeqCst) {
        return;
    }

    let filter = EnvFilter::try_from_env("VOXEL_LOG")
        .or_else(|_| EnvFilter::try_from_env("RUST_LOG"))
        .unwrap_or_else(|_| EnvFilter::new(&config.default_filter));

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_ansi(config.ansi_colors)
        .with_target(true)
        .with_thread_names(true);

    let registry = tracing_subscriber::registry().with(filter).with(fmt_layer);

    #[cfg(feature = "tracy")]
    {
        let tracy_layer = tracing_tracy::TracyLayer::default();
        registry.with(tracy_layer).init();
        tracing::info!(
            app = config.app_name,
            "Telemetry initialized with Tracy profiling enabled"
        );
    }

    #[cfg(not(feature = "tracy"))]
    {
        registry.init();
        tracing::info!(app = config.app_name, "Telemetry initialized");
    }
}
