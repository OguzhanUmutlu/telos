//! Platform-native application data layout and directory resolution.
//!
//! Provides standards-compliant path discovery across Linux (XDG Base Directory specification),
//! Windows (%APPDATA%, %LOCALAPPDATA%), and macOS (~/Library/Application Support, ~/Library/Caches).
//!
//! Supports portable mode (`--portable`), custom data directory overrides (`--data-dir`),
//! and transparent fallback to local workspace directories (`./worlds`, `./server.toml`)
//! or legacy `~/.voxel`.

use std::path::{Path, PathBuf};

/// Application name used for subdirectories in platform-native data roots.
pub const APP_NAME: &str = "telos";

/// Resolved application directories for settings, persistent world saves,
/// transient caches, and rolling logs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppDirs {
    data_dir: PathBuf,
    config_dir: PathBuf,
    cache_dir: PathBuf,
    logs_dir: PathBuf,
    is_portable: bool,
}

impl AppDirs {
    /// Resolves standard platform directories with default fallbacks.
    #[must_use]
    pub fn standard() -> Self {
        resolve_standard_dirs()
    }

    /// Resolves directories checking for local development working directory fallback first.
    ///
    /// If `./worlds` or `./server.toml` exists in the current working directory,
    /// returns an instance rooted at the current directory. Otherwise resolves
    /// standard platform directories.
    #[must_use]
    pub fn standard_with_local_fallback() -> Self {
        if let Ok(cwd) = std::env::current_dir()
            && (cwd.join("worlds").is_dir() || cwd.join("server.toml").is_file())
        {
            return Self {
                data_dir: cwd.clone(),
                config_dir: cwd.clone(),
                cache_dir: cwd.join(".cache"),
                logs_dir: cwd.join("logs"),
                is_portable: false,
            };
        }
        Self::standard()
    }

    /// Resolves directories in portable mode inside the specified base path.
    ///
    /// All configuration, world data, caches, and logs are isolated within subdirectories
    /// of `base_dir`.
    pub fn portable(base_dir: impl Into<PathBuf>) -> Self {
        let base = base_dir.into();
        Self {
            data_dir: base.join("data"),
            config_dir: base.join("config"),
            cache_dir: base.join("cache"),
            logs_dir: base.join("logs"),
            is_portable: true,
        }
    }

    /// Resolves directories rooted at a custom data directory override.
    ///
    /// The `config_dir`, `cache_dir`, and `logs_dir` are derived from subdirectories of `data_dir`.
    pub fn from_data_dir(data_dir: impl Into<PathBuf>) -> Self {
        let data = data_dir.into();
        Self {
            config_dir: data.join("config"),
            cache_dir: data.join("cache"),
            logs_dir: data.join("logs"),
            data_dir: data,
            is_portable: false,
        }
    }

    /// Overrides the persistent data directory.
    #[must_use]
    pub fn with_data_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.data_dir = path.into();
        self
    }

    /// Overrides the configuration directory.
    #[must_use]
    pub fn with_config_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.config_dir = path.into();
        self
    }

    /// Overrides the transient cache directory.
    #[must_use]
    pub fn with_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = path.into();
        self
    }

    /// Overrides the logging directory.
    #[must_use]
    pub fn with_logs_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.logs_dir = path.into();
        self
    }

    /// Returns the root data directory (containing saves, assets, and mods).
    #[must_use]
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Returns the configuration directory (containing settings files).
    #[must_use]
    pub fn config_dir(&self) -> &Path {
        &self.config_dir
    }

    /// Returns the transient disk cache directory (containing LOD clipmaps, shaders).
    #[must_use]
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Returns the rolling log output directory.
    #[must_use]
    pub fn logs_dir(&self) -> &Path {
        &self.logs_dir
    }

    /// Returns true if running in portable mode.
    #[must_use]
    pub fn is_portable(&self) -> bool {
        self.is_portable
    }

    /// Returns the directory where persistent world saves live: `<data_dir>/worlds`.
    #[must_use]
    pub fn saves_dir(&self) -> PathBuf {
        self.data_dir.join("worlds")
    }

    /// Returns the specific world save directory: `<data_dir>/worlds/<world_name>`.
    #[must_use]
    pub fn world_save_dir(&self, world_name: &str) -> PathBuf {
        self.saves_dir().join(world_name)
    }

    /// Returns the path to a specific configuration file: `<config_dir>/<filename>`.
    #[must_use]
    pub fn config_file(&self, filename: &str) -> PathBuf {
        self.config_dir.join(filename)
    }

    /// Returns the transient LOD disk cache directory for a given server:
    /// `<cache_dir>/lod/<server_id>`.
    #[must_use]
    pub fn lod_cache_dir(&self, server_id: &str) -> PathBuf {
        self.cache_dir.join("lod").join(server_id)
    }

    /// Returns the transient compiled shader cache directory:
    /// `<cache_dir>/shaders`.
    #[must_use]
    pub fn shader_cache_dir(&self) -> PathBuf {
        self.cache_dir.join("shaders")
    }

    /// Returns the custom asset pack storage directory:
    /// `<data_dir>/assets`.
    #[must_use]
    pub fn assets_dir(&self) -> PathBuf {
        self.data_dir.join("assets")
    }

    /// Returns the mod storage directory:
    /// `<data_dir>/mods`.
    #[must_use]
    pub fn mods_dir(&self) -> PathBuf {
        self.data_dir.join("mods")
    }

    /// Ensures that the primary directories (`data_dir`, `config_dir`, `cache_dir`, `logs_dir`, `saves_dir`)
    /// exist on disk, creating them if necessary.
    ///
    /// # Errors
    /// Returns an `std::io::Error` if directory creation fails.
    pub fn ensure_dirs_exist(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.data_dir)?;
        std::fs::create_dir_all(&self.config_dir)?;
        std::fs::create_dir_all(&self.cache_dir)?;
        std::fs::create_dir_all(&self.logs_dir)?;
        std::fs::create_dir_all(self.saves_dir())?;
        Ok(())
    }
}

/// Resolves standard platform directories according to OS conventions.
fn resolve_standard_dirs() -> AppDirs {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));

    // Legacy fallback check: if ~/.telos exists as a directory and ~/.local/share/telos does not
    let legacy_home = home.join(".telos");
    if legacy_home.is_dir() {
        let is_primary_missing = dirs::data_dir().is_none_or(|p| !p.join(APP_NAME).exists());
        if is_primary_missing {
            return AppDirs {
                data_dir: legacy_home.clone(),
                config_dir: legacy_home.join("config"),
                cache_dir: legacy_home.join("cache"),
                logs_dir: legacy_home.join("logs"),
                is_portable: false,
            };
        }
    }

    #[cfg(target_os = "windows")]
    {
        let data_dir = dirs::data_dir()
            .map(|p| p.join("Telos"))
            .unwrap_or_else(|| home.join("AppData").join("Roaming").join("Telos"));
        let config_dir = data_dir.join("config");
        let cache_dir = dirs::cache_dir()
            .map(|p| p.join("Telos").join("cache"))
            .unwrap_or_else(|| {
                home.join("AppData")
                    .join("Local")
                    .join("Telos")
                    .join("cache")
            });
        let logs_dir = dirs::data_local_dir()
            .map(|p| p.join("Telos").join("logs"))
            .unwrap_or_else(|| {
                home.join("AppData")
                    .join("Local")
                    .join("Telos")
                    .join("logs")
            });

        AppDirs {
            data_dir,
            config_dir,
            cache_dir,
            logs_dir,
            is_portable: false,
        }
    }

    #[cfg(target_os = "macos")]
    {
        let data_dir = dirs::data_dir()
            .map(|p| p.join("Telos"))
            .unwrap_or_else(|| home.join("Library/Application Support/Telos"));
        let config_dir = data_dir.join("config");
        let cache_dir = dirs::cache_dir()
            .map(|p| p.join("Telos"))
            .unwrap_or_else(|| home.join("Library/Caches/Telos"));
        let logs_dir = home.join("Library/Logs/Telos");

        AppDirs {
            data_dir,
            config_dir,
            cache_dir,
            logs_dir,
            is_portable: false,
        }
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        // Linux / Unix / BSD: adhere to XDG Base Directory Specification
        let data_dir = std::env::var_os("XDG_DATA_HOME")
            .filter(|v| !v.is_empty())
            .map(|v| PathBuf::from(v).join(APP_NAME))
            .or_else(|| dirs::data_dir().map(|p| p.join(APP_NAME)))
            .unwrap_or_else(|| home.join(".local/share").join(APP_NAME));

        let config_dir = std::env::var_os("XDG_CONFIG_HOME")
            .filter(|v| !v.is_empty())
            .map(|v| PathBuf::from(v).join(APP_NAME))
            .or_else(|| dirs::config_dir().map(|p| p.join(APP_NAME)))
            .unwrap_or_else(|| home.join(".config").join(APP_NAME));

        let cache_dir = std::env::var_os("XDG_CACHE_HOME")
            .filter(|v| !v.is_empty())
            .map(|v| PathBuf::from(v).join(APP_NAME))
            .or_else(|| dirs::cache_dir().map(|p| p.join(APP_NAME)))
            .unwrap_or_else(|| home.join(".cache").join(APP_NAME));

        let logs_dir = std::env::var_os("XDG_STATE_HOME")
            .filter(|v| !v.is_empty())
            .map(|v| PathBuf::from(v).join(APP_NAME).join("logs"))
            .or_else(|| dirs::state_dir().map(|p| p.join(APP_NAME).join("logs")))
            .unwrap_or_else(|| home.join(".local/state").join(APP_NAME).join("logs"));

        AppDirs {
            data_dir,
            config_dir,
            cache_dir,
            logs_dir,
            is_portable: false,
        }
    }
}
