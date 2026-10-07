//! Error types for the WASM modding runtime and sandbox host.

use thiserror::Error;

/// Result type alias for mod operations.
pub type ModResult<T> = Result<T, ModError>;

/// Errors that can occur during mod compilation, instantiation, or execution.
#[derive(Debug, Error)]
pub enum ModError {
    /// Engine initialization failure.
    #[error("Failed to initialize WASM engine: {0}")]
    EngineInit(String),

    /// Component or module compilation failure.
    #[error("Failed to compile WASM module '{mod_id}': {message}")]
    Compile {
        /// Identifier of the mod.
        mod_id: String,
        /// Underlying compilation error message.
        message: String,
    },

    /// Linker or instantiation failure.
    #[error("Failed to instantiate WASM mod '{mod_id}': {message}")]
    Instantiation {
        /// Identifier of the mod.
        mod_id: String,
        /// Underlying instantiation error message.
        message: String,
    },

    /// Guest execution trap or panic.
    #[error("WASM mod '{mod_id}' trapped during execution: {message}")]
    Trap {
        /// Identifier of the mod.
        mod_id: String,
        /// Trap message.
        message: String,
    },

    /// Guest exhausted its allocated CPU instruction fuel budget.
    #[error("WASM mod '{mod_id}' exhausted its fuel budget ({fuel_limit} instructions)")]
    FuelExhausted {
        /// Identifier of the mod.
        mod_id: String,
        /// Configured fuel limit.
        fuel_limit: u64,
    },

    /// Memory table limit exceeded by guest.
    #[error("WASM mod '{mod_id}' exceeded memory allocation limit")]
    MemoryLimitExceeded {
        /// Identifier of the mod.
        mod_id: String,
    },

    /// Operation denied due to missing capability permission.
    #[error("Permission denied for mod '{mod_id}': capability '{permission}' is required")]
    Denied {
        /// Identifier of the mod.
        mod_id: String,
        /// Missing permission name.
        permission: &'static str,
    },

    /// Requested resource or entity was not found.
    #[error("Resource not found: {0}")]
    NotFound(String),

    /// Invalid argument passed across host-guest boundary.
    #[error("Invalid argument from mod '{mod_id}': {message}")]
    InvalidArgument {
        /// Identifier of the mod.
        mod_id: String,
        /// Validation failure reason.
        message: String,
    },

    /// General IO or filesystem error while reading mod package.
    #[error("IO error loading mod '{mod_id}': {error}")]
    Io {
        /// Identifier of the mod.
        mod_id: String,
        /// Underlying IO error.
        #[source]
        error: std::io::Error,
    },

    /// JavaScript runtime initialization or execution error.
    #[error("JavaScript mod error in '{mod_id}': {message}")]
    JsExecution {
        /// Identifier of the mod or script.
        mod_id: String,
        /// Error details.
        message: String,
    },

    /// JavaScript execution exceeded instruction fuel or wall-clock timeout budget.
    #[error("JavaScript script '{mod_id}' exceeded execution budget: {message}")]
    JsTimeout {
        /// Identifier of the mod or script.
        mod_id: String,
        /// Timeout / fuel detail.
        message: String,
    },

    /// JavaScript heap memory limit exceeded.
    #[error("JavaScript script '{mod_id}' exceeded memory quota")]
    JsMemoryLimitExceeded {
        /// Identifier of the mod or script.
        mod_id: String,
    },
}
