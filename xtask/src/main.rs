//! Build automation, diagnostic tools, and CI runners for the Telos engine.

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
    about = "Build automation and diagnostic tools for Telos"
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
    /// Compile telos-web to WASM and bundle distribution files into web/dist
    Web,
    /// Capture a diagnostic screenshot with the telos client
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
    /// Build distribution packages (.deb, portable tarballs, `AppImage`)
    Dist {
        /// Target version string (defaults to Cargo.toml package version)
        #[arg(long)]
        version: Option<String>,

        /// Only build portable archives (.tar.gz and .zip)
        #[arg(long)]
        portable_only: bool,

        /// Skip building `AppImage` bundle
        #[arg(long)]
        no_appimage: bool,

        /// Skip building Debian packages (.deb)
        #[arg(long)]
        no_deb: bool,

        /// Output directory for distribution packages
        #[arg(long, default_value = "target/dist")]
        output_dir: String,
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
        Commands::Dist {
            version,
            portable_only,
            no_appimage,
            no_deb,
            output_dir,
        } => run_dist(
            version.as_deref(),
            portable_only,
            no_appimage,
            no_deb,
            &output_dir,
        )?,
    }

    Ok(())
}

fn run_ci() -> Result<()> {
    println!("============================================================");
    println!("            TELOS CI QUALITY ASSURANCE PIPELINE             ");
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
            &["cargo", "build", "-p", "telos-server-app"],
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
    println!("               TELOS ENVIRONMENT DOCTOR                     ");
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

    let original_assets = Path::new("assets/telos");
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
    println!("           TELOS WEB COMPILATION & BUNDLE RUNNER            ");
    println!("============================================================");

    let start = Instant::now();

    // 1. Build release WASM for telos-web
    println!("\n>> Compiling telos-web (cargo build --target wasm32-unknown-unknown --release)...");
    let status = Command::new("cargo")
        .args([
            "build",
            "-p",
            "telos-web",
            "--target",
            "wasm32-unknown-unknown",
            "--release",
        ])
        .status()
        .context("Failed to run cargo build for wasm32-unknown-unknown")?;

    if !status.success() {
        bail!("Failed to compile telos-web to wasm32-unknown-unknown");
    }

    // 2. Prepare web/dist directory
    let dist_dir = Path::new("web/dist");
    if dist_dir.exists() {
        std::fs::remove_dir_all(dist_dir).context("Failed to clear web/dist directory")?;
    }
    std::fs::create_dir_all(dist_dir).context("Failed to create web/dist directory")?;

    // 3. Run wasm-bindgen
    println!("\n>> Running wasm-bindgen to produce JavaScript bindings...");
    let wasm_path = "target/wasm32-unknown-unknown/release/telos_web.wasm";
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
        bail!("wasm-bindgen failed to bundle telos-web");
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

    let wasm_file = dist_dir.join("telos_web_bg.wasm");
    let wasm_size = wasm_file.metadata().map_or(0, |m| m.len());
    println!("\n============================================================");
    println!("   ✓ WEB BUILD COMPLETE ({:.2?})", start.elapsed());
    println!("   WASM binary size: {:.1} KiB", wasm_size as f64 / 1024.0);
    println!("   Output directory: web/dist/");
    println!("   Host domain:      telos.larvance.com");
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
    cmd.args(["run", "--bin", "telos", "--"]);
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
        .context("Failed to run telos client for screenshot")?;
    if !status.success() {
        bail!("Telos client screenshot exited with failure status {status}");
    }

    println!("   ✓ Screenshot successfully saved to {output}");
    Ok(())
}

#[allow(clippy::too_many_lines, clippy::similar_names)]
fn run_dist(
    version: Option<&str>,
    portable_only: bool,
    no_appimage: bool,
    no_deb: bool,
    output_dir: &str,
) -> Result<()> {
    let start = Instant::now();
    let ver = match version {
        Some(v) => v.trim_start_matches(['v', 'V']).to_string(),
        None => extract_workspace_version()?,
    };

    println!("============================================================");
    println!("       TELOS CROSS-PLATFORM DISTRIBUTION PACKAGER           ");
    println!("============================================================");
    println!("Target Version:   v{ver}");
    println!("Output Directory: {output_dir}");
    println!("Portable Only:    {portable_only}");
    if !portable_only {
        println!("Include Debian:   {}", !no_deb);
        println!("Include AppImage: {}", !no_appimage);
    }
    println!("------------------------------------------------------------");

    // 1. Build release binaries
    println!("\n>> 1. Building release binaries (telos, telos-server)...");
    let status = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--bin",
            "telos",
            "--bin",
            "telos-server",
        ])
        .status()
        .context("Failed to execute cargo build --release")?;
    if !status.success() {
        bail!("cargo build --release failed with status {status}");
    }

    let root_dir = std::env::current_dir()?;
    let dist_dir = root_dir.join(output_dir);
    std::fs::create_dir_all(&dist_dir)?;

    let client_bin = root_dir.join("target/release/telos");
    let server_bin = root_dir.join("target/release/telos-server");

    // 2. Build portable tarballs
    println!("\n>> 2. Assembling portable tarballs (.tar.gz)...");

    // Stage client
    let client_stage_dir = root_dir.join("target/tar-client");
    let client_stage = client_stage_dir.join(format!("telos-{ver}"));
    let _ = std::fs::remove_dir_all(&client_stage);
    std::fs::create_dir_all(&client_stage)?;

    std::fs::copy(&client_bin, client_stage.join("telos"))?;
    set_executable_perms(&client_stage.join("telos"))?;

    let desktop_file = root_dir.join("packaging/linux/telos.desktop");
    if desktop_file.exists() {
        std::fs::copy(&desktop_file, client_stage.join("telos.desktop"))?;
    }
    let icon_file = root_dir.join("packaging/linux/telos.png");
    if icon_file.exists() {
        std::fs::copy(&icon_file, client_stage.join("telos.png"))?;
    }

    let assets_dir = root_dir.join("assets/telos");
    if assets_dir.exists() {
        let client_assets = client_stage.join("assets/telos");
        copy_dir_all(&assets_dir, &client_assets)?;
    }

    let client_run_script = "#!/bin/sh\nDIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\nexport TELOS_ASSETS_DIR=\"$DIR/assets/telos\"\nexec \"$DIR/telos\" \"$@\"\n";
    std::fs::write(client_stage.join("run.sh"), client_run_script)?;
    set_executable_perms(&client_stage.join("run.sh"))?;

    let client_tar_name = format!("telos-{ver}-linux-x86_64.tar.gz");
    let client_tar_path = dist_dir.join(&client_tar_name);
    let tar_client_status = Command::new("tar")
        .args([
            "-czf",
            client_tar_path.to_str().unwrap(),
            "-C",
            client_stage_dir.to_str().unwrap(),
            &format!("telos-{ver}"),
        ])
        .status()
        .context("Failed to run tar for client archive")?;
    if !tar_client_status.success() {
        bail!("Failed to create client tarball");
    }
    println!("   ✓ Created client tarball: {client_tar_name}");

    // Stage server
    let server_stage_dir = root_dir.join("target/tar-server");
    let server_stage = server_stage_dir.join(format!("telos-server-{ver}"));
    let _ = std::fs::remove_dir_all(&server_stage);
    std::fs::create_dir_all(&server_stage)?;

    std::fs::copy(&server_bin, server_stage.join("telos-server"))?;
    set_executable_perms(&server_stage.join("telos-server"))?;

    let server_toml = root_dir.join("packaging/server.toml");
    if server_toml.exists() {
        std::fs::copy(&server_toml, server_stage.join("server.toml"))?;
    }

    let server_run_script = "#!/bin/sh\nDIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\nexec \"$DIR/telos-server\" \"$@\"\n";
    std::fs::write(server_stage.join("run.sh"), server_run_script)?;
    set_executable_perms(&server_stage.join("run.sh"))?;

    let server_tar_name = format!("telos-server-{ver}-linux-x86_64.tar.gz");
    let server_tar_path = dist_dir.join(&server_tar_name);
    let tar_server_status = Command::new("tar")
        .args([
            "-czf",
            server_tar_path.to_str().unwrap(),
            "-C",
            server_stage_dir.to_str().unwrap(),
            &format!("telos-server-{ver}"),
        ])
        .status()
        .context("Failed to run tar for server archive")?;
    if !tar_server_status.success() {
        bail!("Failed to create server tarball");
    }
    println!("   ✓ Created server tarball: {server_tar_name}");

    // 3. Debian packages
    if !portable_only && !no_deb {
        println!("\n>> 3. Building Debian packages (.deb)...");
        let deb_script = root_dir.join("packaging/linux/build-deb.sh");
        if deb_script.exists() {
            let status = Command::new("bash")
                .arg(&deb_script)
                .arg(&ver)
                .env("DIST_DIR", &dist_dir)
                .env("CLIENT_BIN", &client_bin)
                .env("SERVER_BIN", &server_bin)
                .status()
                .context("Failed to run packaging/linux/build-deb.sh")?;
            if !status.success() {
                bail!("build-deb.sh exited with failure: {status}");
            }
        } else {
            println!("   Skipping: packaging/linux/build-deb.sh not found");
        }
    }

    // 4. AppImage package
    if !portable_only && !no_appimage {
        println!("\n>> 4. Building Linux AppImage (.AppImage)...");
        let appimage_script = root_dir.join("packaging/linux/build-appimage.sh");
        if appimage_script.exists() {
            let status = Command::new("bash")
                .arg(&appimage_script)
                .arg(&ver)
                .env("DIST_DIR", &dist_dir)
                .env("CLIENT_BIN", &client_bin)
                .status()
                .context("Failed to run packaging/linux/build-appimage.sh")?;
            if !status.success() {
                bail!("build-appimage.sh exited with failure: {status}");
            }
        } else {
            println!("   Skipping: packaging/linux/build-appimage.sh not found");
        }
    }

    // 5. Generate SHA256SUMS and print table
    let mut entries: Vec<_> = std::fs::read_dir(&dist_dir)?
        .filter_map(std::result::Result::ok)
        .filter(|e| e.path().is_file() && e.file_name() != "SHA256SUMS")
        .collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);

    let mut checksum_lines = Vec::new();
    for entry in &entries {
        let path = entry.path();
        let file_name = entry.file_name();
        let hash = compute_sha256(&path);
        checksum_lines.push(format!("{hash}  {}", file_name.to_string_lossy()));
    }
    if !checksum_lines.is_empty() {
        std::fs::write(
            dist_dir.join("SHA256SUMS"),
            checksum_lines.join("\n") + "\n",
        )?;
    }

    println!(
        "\n=================================================================================================="
    );
    println!(
        "                        DISTRIBUTION PACKAGES GENERATED                                           "
    );
    println!(
        "=================================================================================================="
    );
    println!("{:<45} {:>10}  SHA256 Checksum", "Filename", "Size");
    println!("{:-<45} {:-<10}  {:-<64}", "", "", "");

    let mut all_entries: Vec<_> = std::fs::read_dir(&dist_dir)?
        .filter_map(std::result::Result::ok)
        .filter(|e| e.path().is_file())
        .collect();
    all_entries.sort_by_key(std::fs::DirEntry::file_name);

    for entry in all_entries {
        let path = entry.path();
        let file_name = entry.file_name();
        let size = path.metadata().map_or(0, |m| m.len());
        let hash = compute_sha256(&path);
        println!(
            "{:<45} {:>10}  {}",
            file_name.to_string_lossy(),
            format_bytes(size),
            hash
        );
    }
    println!(
        "=================================================================================================="
    );
    println!("✓ Packaging complete in {:.2?}", start.elapsed());
    println!("Output directory: {}", dist_dir.display());

    Ok(())
}

fn extract_workspace_version() -> Result<String> {
    let content = std::fs::read_to_string("Cargo.toml").context("Failed to read Cargo.toml")?;
    let mut in_workspace_package = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "[workspace.package]" {
            in_workspace_package = true;
            continue;
        } else if trimmed.starts_with('[') {
            in_workspace_package = false;
        }

        if in_workspace_package && trimmed.starts_with("version") {
            let (_, val) = trimmed
                .split_once('=')
                .context("Malformed version line in Cargo.toml")?;
            let v = val.trim().trim_matches('"').trim_matches('\'');
            return Ok(v.to_string());
        }
    }
    bail!("Could not find version under [workspace.package] in Cargo.toml")
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn set_executable_perms(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)?.permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_executable_perms(_path: &Path) -> Result<()> {
    Ok(())
}

fn compute_sha256(path: &Path) -> String {
    match Command::new("sha256sum").arg(path).output() {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            stdout
                .split_whitespace()
                .next()
                .unwrap_or("N/A")
                .to_string()
        }
        _ => "N/A".to_string(),
    }
}

#[allow(clippy::cast_precision_loss)]
fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;

    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}
