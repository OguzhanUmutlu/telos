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
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Ci => run_ci()?,
        Commands::Doctor => run_doctor(),
    }

    Ok(())
}

fn run_ci() -> Result<()> {
    println!("============================================================");
    println!("            VOXEL CI QUALITY ASSURANCE PIPELINE             ");
    println!("============================================================");

    let steps: [(&str, &[&str]); 5] = [
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
    let dev_assets = Path::new("dev-assets/classic-26.2");
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
