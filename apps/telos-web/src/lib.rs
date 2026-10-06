//! WebAssembly client wrapper for the voxel engine.
//!
//! Exposes a WebGL2-based runtime for the browser on `wasm32-unknown-unknown`.
//! On non-wasm targets, provides a stub so that workspace-level desktop
//! operations (`cargo test --workspace`, `cargo check`) proceed seamlessly.

#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(target_arch = "wasm32")]
pub use web::*;

#[cfg(not(target_arch = "wasm32"))]
/// Notice for non-wasm targets.
pub fn web_build_notice() {
    println!("voxel-web is designed to run in the browser via wasm32-unknown-unknown");
}
