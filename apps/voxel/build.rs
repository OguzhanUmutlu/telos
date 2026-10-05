//! Build script for compiling shaders to SPIR-V.

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../shaders/debug_triangle.vert");
    println!("cargo:rerun-if-changed=../../shaders/debug_triangle.frag");
    println!("cargo:rerun-if-changed=../../shaders/chunk.vert");
    println!("cargo:rerun-if-changed=../../shaders/chunk.frag");

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = Path::new(&out_dir);

    let shaders = [
        (
            "../../shaders/debug_triangle.vert",
            "debug_triangle.vert.spv",
        ),
        (
            "../../shaders/debug_triangle.frag",
            "debug_triangle.frag.spv",
        ),
        ("../../shaders/chunk.vert", "chunk.vert.spv"),
        ("../../shaders/chunk.frag", "chunk.frag.spv"),
    ];

    for (src, dst) in shaders {
        let dest_path = out_path.join(dst);
        let status = Command::new("glslc")
            .arg(src)
            .arg("-o")
            .arg(&dest_path)
            .status()
            .expect("Failed to execute glslc to compile shader");

        assert!(status.success(), "Failed to compile shader {src}");
    }
}
