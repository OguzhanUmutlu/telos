//! Build automation, diagnostic tools, and CI runners for the voxel engine.

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use std::{
    path::Path,
    process::{Command, Stdio},
    time::Instant,
};

#[derive(Parser, Debug)]
#[command(
    name = "xtask",
    about = "Build automation and diagnostic tools for voxel"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run full continuous integration quality checks locally
    Ci,
    /// Inspect the local environment, drivers, compilers, and required tools
    Doctor,
    /// Compile and validate all GLSL shaders in shaders/ to SPIR-V
    Shaders,
    /// Compile voxel-web to WASM and bundle distribution files into web/dist
    Web,
    /// Capture a diagnostic screenshot with the voxel client
    Screenshot {
        /// Initial camera position as X,Y,Z
        #[arg(long, default_value = "207.9,174.4,190.7")]
        pos: String,

        /// Camera yaw in degrees
        #[arg(long, default_value_t = -90.0, allow_hyphen_values = true)]
        yaw: f32,

        /// Camera pitch in degrees
        #[arg(long, default_value_t = -15.0, allow_hyphen_values = true)]
        pitch: f32,

        /// Number of warmup frames before capture
        #[arg(long, default_value_t = 30)]
        frames: u32,

        /// Output PNG path
        #[arg(long, default_value = "dev-assets/screenshots/diagnostic.png")]
        output: String,

        /// Bypass Hi-Z culling
        #[arg(long, default_value_t = false)]
        no_cull: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Ci => run_ci()?,
        Commands::Doctor => run_doctor(),
        Commands::Shaders => compile_shaders()?,
        Commands::Web => run_web()?,
        Commands::Screenshot {
            pos,
            yaw,
            pitch,
            frames,
            output,
            no_cull,
        } => run_screenshot(&pos, yaw, pitch, frames, &output, no_cull)?,
    }

    Ok(())
}

fn run_ci() -> Result<()> {
    println!("============================================================");
    println!("            VOXEL CI QUALITY ASSURANCE PIPELINE             ");
    println!("============================================================");

    let steps: [(&str, &[&str]); 6] = [
        (
            "Checking formatting (cargo fmt --check)",
            &["cargo", "fmt", "--all", "--", "--check"],
        ),
        (
            "Linting workspace (cargo clippy -D warnings)",
            &[
                "cargo",
                "clippy",
                "--workspace",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ],
        ),
        ("Running workspace tests", &["cargo", "test", "--workspace"]),
        (
            "Compiling and validating shaders",
            &["cargo", "xtask", "shaders"],
        ),
        (
            "Building documentation (cargo doc --no-deps)",
            &["cargo", "doc", "--workspace", "--no-deps"],
        ),
        (
            "Verifying headless server build isolation",
            &["cargo", "build", "-p", "voxel-server"],
        ),
    ];

    let start_all = Instant::now();

    for (desc, cmd) in steps {
        println!("\n>> {desc}...");
        let step_start = Instant::now();

        let mut process = Command::new(cmd[0]);
        process.args(&cmd[1..]);

        let status = process
            .status()
            .with_context(|| format!("Failed to execute '{}'", cmd.join(" ")))?;

        if !status.success() {
            bail!("CI step failed: {desc}");
        }

        println!("   ✓ Passed ({:.2?})", step_start.elapsed());
    }

    // Optional cargo-deny check
    if is_command_available("cargo-deny") {
        println!("\n>> Checking licenses and bans (cargo deny check)...");
        let status = Command::new("cargo").args(["deny", "check"]).status()?;
        if !status.success() {
            bail!("cargo-deny check failed");
        }
        println!("   ✓ Passed");
    } else {
        println!("\n>> [NOTE] cargo-deny is not installed. Skipping license check.");
    }

    println!("\n============================================================");
    println!(
        "        ✓ ALL CI CHECKS PASSED ({:.2?})        ",
        start_all.elapsed()
    );
    println!("============================================================");

    Ok(())
}

fn run_doctor() {
    println!("============================================================");
    println!("               VOXEL ENVIRONMENT DOCTOR                     ");
    println!("============================================================");

    check_tool("rustc", &["--version"], true, "Install Rust via rustup");
    check_tool("cargo", &["--version"], true, "Install Cargo via rustup");
    check_tool(
        "glslc",
        &["--version"],
        true,
        "Install shaderc / glslc (e.g. sudo apt install glslc or libshaderc-dev)",
    );
    check_tool(
        "spirv-val",
        &["--version"],
        true,
        "Install spirv-tools (e.g. sudo apt install spirv-tools)",
    );
    check_tool(
        "vulkaninfo",
        &["--summary"],
        false,
        "Install vulkan-tools (sudo apt install vulkan-tools)",
    );
    check_tool(
        "cargo-nextest",
        &["--version"],
        false,
        "Install via: cargo install cargo-nextest --locked",
    );
    check_tool(
        "cargo-deny",
        &["--version"],
        false,
        "Install via: cargo install cargo-deny --locked",
    );

    println!("\n--- Asset Directories ---");
    let dev_assets = Path::new("dev-assets/classic-pack");
    if dev_assets.exists() {
        println!("✓ Placeholder assets found at {}", dev_assets.display());
    } else {
        println!("! Placeholder assets NOT found at {}", dev_assets.display());
    }

    let original_assets = Path::new("assets/voxel");
    if original_assets.exists() {
        println!(
            "✓ Original assets directory found at {}",
            original_assets.display()
        );
    } else {
        println!(
            "! Original assets directory NOT found at {}",
            original_assets.display()
        );
    }

    println!("============================================================");
}

fn compile_shaders() -> Result<()> {
    let shaders_dir = Path::new("shaders");
    if !shaders_dir.exists() {
        println!("No shaders directory found.");
        return Ok(());
    }

    let out_dir = Path::new("target/shaders");
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("Failed to create output directory: {}", out_dir.display()))?;

    let entries = std::fs::read_dir(shaders_dir)
        .with_context(|| format!("Failed to read directory: {}", shaders_dir.display()))?;

    let mut count = 0;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if !matches!(ext, "vert" | "frag" | "comp" | "geom" | "mesh" | "task") {
            continue;
        }

        let file_name = path
            .file_name()
            .context("Missing filename")?
            .to_string_lossy();
        let spv_file = out_dir.join(format!("{file_name}.spv"));

        println!("Compiling: {} -> {}", path.display(), spv_file.display());

        let glslc_status = Command::new("glslc")
            .arg(&path)
            .arg("-o")
            .arg(&spv_file)
            .status()
            .with_context(|| "Failed to execute 'glslc'. Is shaderc/glslc installed?")?;

        if !glslc_status.success() {
            bail!("Shader compilation failed for {}", path.display());
        }

        if is_command_available("spirv-val") {
            let val_status = Command::new("spirv-val")
                .arg(&spv_file)
                .status()
                .with_context(|| "Failed to execute 'spirv-val'")?;

            if !val_status.success() {
                bail!("SPIR-V validation failed for {}", spv_file.display());
            }
        }

        count += 1;
    }

    println!("✓ Successfully compiled and validated {count} shader(s)");
    Ok(())
}

fn is_command_available(cmd: &str) -> bool {
    Command::new("which")
        .arg(cmd)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn check_tool(name: &str, args: &[&str], required: bool, install_hint: &str) {
    print!("{name:<16} ");
    match Command::new(name).args(args).output() {
        Ok(out) if out.status.success() => {
            let first_line = String::from_utf8_lossy(&out.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            let msg = if first_line.is_empty() {
                String::from_utf8_lossy(&out.stderr)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string()
            } else {
                first_line
            };
            println!("✓ FOUND ({msg})");
        }
        _ => {
            if required {
                println!("✗ MISSING (REQUIRED) -> {install_hint}");
            } else {
                println!("- NOT FOUND (OPTIONAL) -> {install_hint}");
            }
        }
    }
}

fn run_web() -> Result<()> {
    println!("============================================================");
    println!("           VOXEL WEB COMPILATION & BUNDLE RUNNER            ");
    println!("============================================================");

    let start = Instant::now();

    // 1. Build release WASM for voxel-web
    println!("\n>> Compiling voxel-web (cargo build --target wasm32-unknown-unknown --release)...");
    let status = Command::new("cargo")
        .args([
            "build",
            "-p",
            "voxel-web",
            "--target",
            "wasm32-unknown-unknown",
            "--release",
        ])
        .status()
        .context("Failed to run cargo build for wasm32-unknown-unknown")?;

    if !status.success() {
        bail!("Failed to compile voxel-web to wasm32-unknown-unknown");
    }

    // 2. Prepare web/dist directory
    let dist_dir = Path::new("web/dist");
    if dist_dir.exists() {
        std::fs::remove_dir_all(dist_dir).context("Failed to clear web/dist directory")?;
    }
    std::fs::create_dir_all(dist_dir).context("Failed to create web/dist directory")?;

    // 3. Run wasm-bindgen
    println!("\n>> Running wasm-bindgen to produce JavaScript bindings...");
    let wasm_path = "target/wasm32-unknown-unknown/release/voxel_web.wasm";
    let status = Command::new("wasm-bindgen")
        .args([
            wasm_path,
            "--out-dir",
            "web/dist",
            "--target",
            "web",
            "--no-typescript",
        ])
        .status()
        .context("Failed to execute wasm-bindgen. Ensure wasm-bindgen is installed.")?;

    if !status.success() {
        bail!("wasm-bindgen failed to bundle voxel-web");
    }

    // 4. Copy static assets from web/static to web/dist
    println!("\n>> Copying static assets from web/static to web/dist...");
    let static_dir = Path::new("web/static");
    if static_dir.exists() {
        for entry in std::fs::read_dir(static_dir).context("Failed to read web/static")? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                let file_name = path.file_name().unwrap();
                let dest = dist_dir.join(file_name);
                std::fs::copy(&path, &dest).with_context(|| {
                    format!("Failed to copy {} to {}", path.display(), dest.display())
                })?;
                println!("   Copied {}", file_name.to_string_lossy());
            }
        }
    }

    let wasm_file = dist_dir.join("voxel_web_bg.wasm");
    let wasm_size = wasm_file.metadata().map_or(0, |m| m.len());
    println!("\n============================================================");
    println!("   ✓ WEB BUILD COMPLETE ({:.2?})", start.elapsed());
    println!("   WASM binary size: {:.1} KiB", wasm_size as f64 / 1024.0);
    println!("   Output directory: web/dist/");
    println!("   Host domain:      voxel.larvance.com");
    println!("============================================================");

    Ok(())
}

fn run_screenshot(
    pos: &str,
    yaw: f32,
    pitch: f32,
    frames: u32,
    output: &str,
    no_cull: bool,
) -> Result<()> {
    println!(">> Capturing diagnostic screenshot...");
    println!("   Position: {pos}");
    println!("   Yaw: {yaw}°, Pitch: {pitch}°");
    println!("   Warmup frames: {frames}");
    println!("   Output: {output}");
    if no_cull {
        println!("   Hi-Z culling: bypassed");
    }

    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--bin", "voxel", "--"]);
    cmd.arg("--pos").arg(pos);
    cmd.arg("--yaw").arg(yaw.to_string());
    cmd.arg("--pitch").arg(pitch.to_string());
    cmd.arg("--frames").arg(frames.to_string());
    cmd.arg("--screenshot").arg(output);
    if no_cull {
        cmd.arg("--no-cull");
    }

    let status = cmd
        .status()
        .context("Failed to run voxel client for screenshot")?;
    if !status.success() {
        bail!("Voxel client screenshot exited with failure status {status}");
    }

    println!("   ✓ Screenshot successfully saved to {output}");
    Ok(())
}
