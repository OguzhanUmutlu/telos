# voxel

A world-class, high-performance voxel engine and game written in **Rust** and **Vulkan**.

Its primary design goal is **extreme render distance at minimal resource cost**, using compressed chunk representations, GPU-driven rendering, and pixel-level far-field LOD.

## Project Documentation
- **Architecture & Binding Decisions (ADRs):** [`analysis/analysis.md`](analysis/analysis.md)
- **Agent Operating Manual & Workflow:** [`GEMINI.md`](GEMINI.md)
- **Roadmap & Phase Tracker:** [`todo.md`](todo.md)
- **Discipline Runbooks:** [`analysis/*/SKILL.md`](analysis/)

## Prerequisites
- **Rust:** 1.98.1+ stable (`rust-toolchain.toml` pinned)
- **Vulkan:** 1.3 compatible driver and loader
- **Shader tools:** `glslc` (shaderc) and `spirv-val` (spirv-tools)

Run the environment doctor to check your system:
```bash
cargo run -p xtask -- doctor
# or via alias
cargo doctor
```

## Running the Quality Pipeline
```bash
cargo run -p xtask -- ci
# or via alias
cargo ci
```

## Running the Server
```bash
cargo run -p voxel-server --release
```

## License
MIT License. See [LICENSE](LICENSE) for details.
Placeholder Classic Voxel assets live in `dev-assets/` and are strictly excluded from the repository.
