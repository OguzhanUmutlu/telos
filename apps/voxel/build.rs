//! Build script for compiling shaders to SPIR-V.

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../shaders/debug_triangle.vert");
    println!("cargo:rerun-if-changed=../../shaders/debug_triangle.frag");
    println!("cargo:rerun-if-changed=../../shaders/chunk.vert");
    println!("cargo:rerun-if-changed=../../shaders/chunk.frag");
    println!("cargo:rerun-if-changed=../../shaders/lod.vert");
    println!("cargo:rerun-if-changed=../../shaders/lod.frag");
    println!("cargo:rerun-if-changed=../../shaders/highlight.vert");
    println!("cargo:rerun-if-changed=../../shaders/highlight.frag");
    println!("cargo:rerun-if-changed=../../shaders/cutout.vert");
    println!("cargo:rerun-if-changed=../../shaders/cutout.frag");
    println!("cargo:rerun-if-changed=../../shaders/chunk_t1.vert");
    println!("cargo:rerun-if-changed=../../shaders/chunk_t1.frag");
    println!("cargo:rerun-if-changed=../../shaders/translucent.vert");
    println!("cargo:rerun-if-changed=../../shaders/translucent.frag");
    println!("cargo:rerun-if-changed=../../shaders/hiz_generate.comp");
    println!("cargo:rerun-if-changed=../../shaders/cull_chunks.comp");
    println!("cargo:rerun-if-changed=../../shaders/cull_lod.comp");
    println!("cargo:rerun-if-changed=../../shaders/ui.vert");
    println!("cargo:rerun-if-changed=../../shaders/ui.frag");

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
        ("../../shaders/lod.vert", "lod.vert.spv"),
        ("../../shaders/lod.frag", "lod.frag.spv"),
        ("../../shaders/highlight.vert", "highlight.vert.spv"),
        ("../../shaders/highlight.frag", "highlight.frag.spv"),
        ("../../shaders/cutout.vert", "cutout.vert.spv"),
        ("../../shaders/cutout.frag", "cutout.frag.spv"),
        ("../../shaders/chunk_t1.vert", "chunk_t1.vert.spv"),
        ("../../shaders/chunk_t1.frag", "chunk_t1.frag.spv"),
        ("../../shaders/translucent.vert", "translucent.vert.spv"),
        ("../../shaders/translucent.frag", "translucent.frag.spv"),
        ("../../shaders/hiz_generate.comp", "hiz_generate.comp.spv"),
        ("../../shaders/cull_chunks.comp", "cull_chunks.comp.spv"),
        ("../../shaders/cull_lod.comp", "cull_lod.comp.spv"),
        ("../../shaders/ui.vert", "ui.vert.spv"),
        ("../../shaders/ui.frag", "ui.frag.spv"),
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
