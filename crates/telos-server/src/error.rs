//! Server error types.

use std::net::AddrParseError;
use thiserror::Error;

use crate::config::ConfigError;
use crate::multi_world::WorldError;

/// Top-level errors occurring during server initialization, network binding, and execution.
#[derive(Debug, Error)]
pub enum ServerError {
    /// Network transport or Quinn error.
    #[error("Network error: {0}")]
    Network(#[from] telos_net::NetError),
    /// Standard I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Invalid IP socket address string.
    #[error("Invalid socket address '{0}': {1}")]
    InvalidAddress(String, AddrParseError),
    /// Multi-world error.
    #[error("World error: {0}")]
    World(#[from] WorldError),
    /// Persistent storage error.
    #[error("Storage error: {0}")]
    Storage(#[from] telos_storage::StorageError),
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),
}
