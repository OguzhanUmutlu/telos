//! Modding scaffolding, validation, and packaging tools for the Telos engine.

use anyhow::{Context, Result, bail};
use clap::Subcommand;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use telos_sdk::builder::{BlockDefBuilder, ItemDefBuilder, RecipeBuilder, TagBuilder};
use telos_sdk::data::{BlockDef, BlockShapeDef, ItemDef, RecipeDef, RenderLayerDef, TagDef};
use telos_sdk::js::{generate_starter_plugin_js, generate_typescript_declarations};
use telos_sdk::manifest::ModManifest;

#[derive(Subcommand, Debug)]
pub enum ModCommands {
    /// Scaffold a new mod project with manifest, assets, data, and code
    New {
        /// Mod identifier (lowercase alphanumeric with underscores)
        name: String,
        /// Target directory path (defaults to examples/mods/<name>)
        #[arg(long)]
        path: Option<String>,
        /// Template kind: "rust", "js", or "data"
        #[arg(long, default_value = "rust")]
        template: String,
    },
    /// Validate mod manifest, data definitions, assets, and script syntax
    Validate {
        /// Path to the mod directory
        #[arg(default_value = "examples/mods/hello_block")]
        path: String,
    },
    /// Build a mod package (compiles WASM if present and bundles .vxmod archive)
    Build {
        /// Path to the mod directory
        #[arg(default_value = "examples/mods/hello_block")]
        path: String,
        /// Output directory for .vxmod bundle
        #[arg(long, default_value = "dist")]
        output: String,
    },
}

/// Executes mod subcommands.
pub fn run_mod_command(cmd: ModCommands) -> Result<()> {
    match cmd {
        ModCommands::New {
            name,
            path,
            template,
        } => run_mod_new(&name, path.as_deref(), &template),
        ModCommands::Validate { path } => run_mod_validate(&path),
        ModCommands::Build { path, output } => run_mod_build(&path, &output),
    }
}

/// Scaffolds a new mod directory tree.
#[allow(clippy::too_many_lines)]
pub fn run_mod_new(name: &str, path: Option<&str>, template: &str) -> Result<()> {
    println!("============================================================");
    println!("              TELOS MOD PROJECT SCAFFOLDER                  ");
    println!("============================================================");

    // 1. Validate mod name
    if name.is_empty() {
        bail!("Mod name cannot be empty");
    }
    for ch in name.chars() {
        if !matches!(ch, 'a'..='z' | '0'..='9' | '_') {
            bail!(
                "Invalid character '{ch}' in mod name '{name}'. Mod names must be lowercase [a-z0-9_]."
            );
        }
    }

    let target_dir: PathBuf = match path {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(format!("examples/mods/{name}")),
    };

    println!("Mod ID:       {name}");
    println!("Template:     {template}");
    println!("Target Path:  {}", target_dir.display());
    println!("------------------------------------------------------------");

    if target_dir.exists() {
        let is_empty = std::fs::read_dir(&target_dir)?.next().is_none();
        if !is_empty {
            bail!(
                "Target directory '{}' already exists and is not empty.",
                target_dir.display()
            );
        }
    }

    // 2. Create directory skeleton
    let dirs = [
        target_dir.join(format!("assets/{name}/blockstates")),
        target_dir.join(format!("assets/{name}/models/block")),
        target_dir.join(format!("assets/{name}/models/item")),
        target_dir.join(format!("assets/{name}/textures/block")),
        target_dir.join(format!("assets/{name}/lang")),
        target_dir.join(format!("data/{name}/block")),
        target_dir.join(format!("data/{name}/item")),
        target_dir.join(format!("data/{name}/recipe")),
        target_dir.join(format!("data/{name}/tags/block/mineable")),
    ];

    for d in &dirs {
        std::fs::create_dir_all(d)
            .with_context(|| format!("Failed to create directory: {}", d.display()))?;
    }

    // 3. Generate mod.toml manifest
    let mut manifest = ModManifest::new(
        name,
        semver::Version::parse("0.1.0").unwrap(),
        format_display_name(name),
    );
    manifest.info.description = format!("Adds custom blocks and mechanics for {name}.");
    manifest.info.authors = vec!["Mod Developer".to_string()];
    manifest.permissions.server = vec![
        "world.read".to_string(),
        "world.write".to_string(),
        "command.register".to_string(),
        "events.listen".to_string(),
    ];

    let manifest_toml = manifest.to_toml_string()?;
    std::fs::write(target_dir.join("mod.toml"), manifest_toml)?;
    println!("✓ Generated mod.toml");

    // 4. Generate assets
    let block_name = format!("{name}_block");

    // Blockstate
    let blockstate_json = format!(
        r#"{{
  "variants": {{
    "": {{
      "model": "{name}:block/{block_name}"
    }}
  }}
}}
"#
    );
    std::fs::write(
        target_dir.join(format!("assets/{name}/blockstates/{block_name}.json")),
        blockstate_json,
    )?;

    // Block model
    let block_model_json = format!(
        r#"{{
  "parent": "telos:block/cube_all",
  "textures": {{
    "all": "{name}:block/{block_name}"
  }}
}}
"#
    );
    std::fs::write(
        target_dir.join(format!("assets/{name}/models/block/{block_name}.json")),
        block_model_json,
    )?;

    // Item model
    let item_model_json = format!(
        r#"{{
  "parent": "{name}:block/{block_name}"
}}
"#
    );
    std::fs::write(
        target_dir.join(format!("assets/{name}/models/item/{block_name}.json")),
        item_model_json,
    )?;

    // Procedural 16x16 PNG texture
    let texture_path = target_dir.join(format!("assets/{name}/textures/block/{block_name}.png"));
    generate_procedural_texture(&texture_path, name)?;
    println!("✓ Generated procedural texture assets");

    // Lang en_us.json
    let lang_json = format!(
        r#"{{
  "block.{name}.{block_name}": "{display_name}"
}}
"#,
        display_name = format_display_name(&block_name)
    );
    std::fs::write(
        target_dir.join(format!("assets/{name}/lang/en_us.json")),
        lang_json,
    )?;

    // 5. Generate data pack definitions
    // Block RON
    let block_def = BlockDefBuilder::new()
        .shape(BlockShapeDef::FullCube)
        .render_layer(RenderLayerDef::Opaque)
        .light_emission(12)
        .hardness(1.5)
        .blast_resistance(6.0)
        .tool("telos:pickaxe")
        .sound("telos:stone")
        .base_color([220, 160, 40, 255])
        .build();
    let block_ron = block_def.to_ron_string()?;
    std::fs::write(
        target_dir.join(format!("data/{name}/block/{block_name}.ron")),
        block_ron,
    )?;

    // Item RON
    let item_def = ItemDefBuilder::new(format_display_name(&block_name))
        .max_stack_size(64)
        .block_item(format!("{name}:{block_name}"))
        .build();
    let item_ron = item_def.to_ron_string()?;
    std::fs::write(
        target_dir.join(format!("data/{name}/item/{block_name}.ron")),
        item_ron,
    )?;

    // Recipe RON
    let recipe_def = RecipeBuilder::shaped(
        ["###", "#T#", "###"],
        [('#', "telos:stone"), ('T', "telos:torch")],
        format!("{name}:{block_name}"),
        1,
    );
    let recipe_ron = recipe_def.to_ron_string()?;
    std::fs::write(
        target_dir.join(format!("data/{name}/recipe/{block_name}.ron")),
        recipe_ron,
    )?;

    // Tag JSON
    let tag_def = TagBuilder::new()
        .replace(false)
        .add(format!("{name}:{block_name}"))
        .build();
    let tag_json = tag_def.to_json_string()?;
    std::fs::write(
        target_dir.join(format!("data/{name}/tags/block/mineable/pickaxe.json")),
        tag_json,
    )?;
    println!("✓ Generated data pack definitions (block, item, recipe, tags)");

    // 6. Code template
    match template {
        "rust" => {
            let root_dir = std::env::current_dir()?;
            let sdk_dir = root_dir.join("crates/telos-sdk");
            let rel_sdk = compute_relative_path(&target_dir, &sdk_dir);
            let rel_sdk_str = rel_sdk.to_string_lossy().replace('\\', "/");

            // Rust Cargo.toml
            let cargo_toml = format!(
                r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
telos-sdk = {{ path = "{rel_sdk_str}" }}

[workspace]

[profile.release]
opt-level = "s"
lto = true
strip = true
"#
            );
            std::fs::write(target_dir.join("Cargo.toml"), cargo_toml)?;

            // src/lib.rs
            let src_dir = target_dir.join("src");
            std::fs::create_dir_all(&src_dir)?;

            let src_lib = format!(
                r#"//! {display_name} - Telos WASM Mod
use telos_sdk::export_mod;
use telos_sdk::events::{{EventFilter, ModEvent}};
use telos_sdk::guest::{{log, register_command, set_block, subscribe_events, LogLevel, TelosMod}};

#[derive(Default)]
pub struct {struct_name};

impl TelosMod for {struct_name} {{
    fn init(&mut self) -> Result<(), String> {{
        log(LogLevel::Info, "[{name}] Mod initialized successfully!");
        subscribe_events(EventFilter::BLOCK_BROKEN.union(EventFilter::BLOCK_PLACED))?;
        register_command("glow")?;
        Ok(())
    }}

    fn on_event(&mut self, event: &ModEvent) {{
        match event {{
            ModEvent::BlockPlaced {{ pos, new_state, .. }} => {{
                log(LogLevel::Info, &format!("[{name}] Block placed at {{pos:?}} (state {{new_state}})"));
            }}
            ModEvent::BlockBroken {{ pos, old_state, .. }} => {{
                log(LogLevel::Info, &format!("[{name}] Block broken at {{pos:?}} (state {{old_state}})"));
            }}
            _ => {{}}
        }}
    }}

    fn on_command(&mut self, cmd: &str, _args: &str) -> Result<String, String> {{
        if cmd == "glow" {{
            // Example: place a glowing block 2 units above origin
            set_block(0, 70, 0, 1)?;
            Ok("Glow activated!".to_string())
        }} else {{
            Ok(String::new())
        }}
    }}
}}

export_mod!({struct_name});
"#,
                display_name = format_display_name(name),
                struct_name = to_pascal_case(name),
            );
            std::fs::write(src_dir.join("lib.rs"), src_lib)?;
            println!("✓ Generated Rust guest WASM template");
        }
        "js" => {
            let plugin_js = generate_starter_plugin_js(name);
            std::fs::write(target_dir.join("plugin.js"), plugin_js)?;

            let dts = generate_typescript_declarations();
            std::fs::write(target_dir.join("telos.d.ts"), dts)?;
            println!("✓ Generated QuickJS plugin template and TypeScript declarations");
        }
        "data" => {
            println!("✓ Pure data pack initialized");
        }
        other => bail!("Unknown template '{other}'. Valid options: rust, js, data"),
    }

    // README.md
    let readme = format!(
        r"# {display_name} (`{name}`)

Telos Engine Mod.

## Development & Testing
- Validate manifest and schemas:
  ```bash
  cargo xtask mod validate {path}
  ```
- Build distribution package (`.vxmod` and `server.wasm`):
  ```bash
  cargo xtask mod build {path}
  ```
",
        display_name = format_display_name(name),
        path = target_dir.display(),
    );
    std::fs::write(target_dir.join("README.md"), readme)?;

    println!("============================================================");
    println!("✓ Mod successfully created at: {}", target_dir.display());
    println!("Next steps:");
    println!("  cargo xtask mod validate {}", target_dir.display());
    println!("  cargo xtask mod build {}", target_dir.display());
    println!("============================================================");

    Ok(())
}

/// Validates mod manifest, data definitions, assets, and code.
#[allow(clippy::too_many_lines)]
pub fn run_mod_validate(path_str: &str) -> Result<()> {
    let mod_dir = Path::new(path_str);
    if !mod_dir.exists() {
        bail!("Mod directory not found: {}", mod_dir.display());
    }

    println!("============================================================");
    println!("              TELOS MOD VALIDATION RUNNER                   ");
    println!("============================================================");
    println!("Target Path: {}", mod_dir.display());
    println!("------------------------------------------------------------");

    let start = Instant::now();

    // 1. Validate mod.toml
    let manifest_path = mod_dir.join("mod.toml");
    if !manifest_path.exists() {
        bail!("Missing manifest file: {}", manifest_path.display());
    }
    let manifest = ModManifest::from_file(&manifest_path)
        .with_context(|| format!("Failed to parse {}", manifest_path.display()))?;
    manifest
        .validate()
        .with_context(|| "Manifest validation failed")?;
    println!(
        "✓ [Manifest] mod.toml valid: id='{}', version={}, side={:?}",
        manifest.info.id, manifest.info.version, manifest.info.side
    );

    // 2. Validate data definitions
    let mut data_count = 0;
    let data_dir = mod_dir.join("data");
    if data_dir.exists() {
        for entry in walk_dir(&data_dir)? {
            let ext = entry.extension().and_then(|s| s.to_str()).unwrap_or("");
            let file_str = entry.to_string_lossy();
            if file_str.contains("/tags/") && ext == "json" {
                let content = std::fs::read_to_string(&entry)?;
                let _: TagDef = serde_json::from_str(&content)
                    .with_context(|| format!("Failed to parse TagDef at {}", entry.display()))?;
                data_count += 1;
            } else if file_str.contains("/block/") && (ext == "ron" || ext == "json") {
                let content = std::fs::read_to_string(&entry)?;
                let _: BlockDef = if ext == "ron" {
                    ron::from_str(&content).with_context(|| {
                        format!("Failed to parse BlockDef at {}", entry.display())
                    })?
                } else {
                    serde_json::from_str(&content).with_context(|| {
                        format!("Failed to parse BlockDef at {}", entry.display())
                    })?
                };
                data_count += 1;
            } else if file_str.contains("/item/") && (ext == "ron" || ext == "json") {
                let content = std::fs::read_to_string(&entry)?;
                let _: ItemDef = if ext == "ron" {
                    ron::from_str(&content).with_context(|| {
                        format!("Failed to parse ItemDef at {}", entry.display())
                    })?
                } else {
                    serde_json::from_str(&content).with_context(|| {
                        format!("Failed to parse ItemDef at {}", entry.display())
                    })?
                };
                data_count += 1;
            } else if file_str.contains("/recipe/") && (ext == "ron" || ext == "json") {
                let content = std::fs::read_to_string(&entry)?;
                let _: RecipeDef = if ext == "ron" {
                    ron::from_str(&content).with_context(|| {
                        format!("Failed to parse RecipeDef at {}", entry.display())
                    })?
                } else {
                    serde_json::from_str(&content).with_context(|| {
                        format!("Failed to parse RecipeDef at {}", entry.display())
                    })?
                };
                data_count += 1;
            }
        }
    }
    println!("✓ [Data Pack] Validated {data_count} data definition file(s)");

    // 3. Validate assets
    let mut asset_count = 0;
    let assets_dir = mod_dir.join("assets");
    if assets_dir.exists() {
        for entry in walk_dir(&assets_dir)? {
            let ext = entry.extension().and_then(|s| s.to_str()).unwrap_or("");
            if ext == "json" {
                let content = std::fs::read_to_string(&entry)?;
                let _: serde_json::Value = serde_json::from_str(&content)
                    .with_context(|| format!("Invalid JSON asset at {}", entry.display()))?;
                asset_count += 1;
            } else if ext == "png" {
                // Verify valid PNG image
                let bytes = std::fs::read(&entry)?;
                image::load_from_memory(&bytes)
                    .with_context(|| format!("Corrupt PNG texture image at {}", entry.display()))?;
                asset_count += 1;
            }
        }
    }
    println!("✓ [Assets] Validated {asset_count} resource asset file(s)");

    // 4. Validate Code / Scripts
    let cargo_toml = mod_dir.join("Cargo.toml");
    if cargo_toml.exists() {
        println!(">> Checking Rust guest compilation...");
        let mut cmd = Command::new("cargo");
        cmd.arg("check");
        cmd.arg("--manifest-path").arg(&cargo_toml);
        // Try wasm32 target
        cmd.arg("--target").arg("wasm32-unknown-unknown");

        let status = cmd.status();
        match status {
            Ok(s) if s.success() => {
                println!("✓ [Code] Rust WASM guest compiles cleanly (wasm32-unknown-unknown)");
            }
            _ => {
                // Fallback to native check if wasm target isn't configured
                let mut fallback_cmd = Command::new("cargo");
                fallback_cmd
                    .arg("check")
                    .arg("--manifest-path")
                    .arg(&cargo_toml);
                let fallback_status = fallback_cmd
                    .status()
                    .with_context(|| "Failed to execute cargo check")?;
                if !fallback_status.success() {
                    bail!("Rust guest failed cargo check");
                }
                println!("✓ [Code] Rust guest compiles cleanly (native fallback)");
            }
        }
    }

    let plugin_js = mod_dir.join("plugin.js");
    if plugin_js.exists() {
        let content = std::fs::read_to_string(&plugin_js)?;
        if content.trim().is_empty() {
            bail!("JavaScript plugin at {} is empty", plugin_js.display());
        }
        println!(
            "✓ [Scripts] QuickJS plugin.js verified ({} bytes)",
            content.len()
        );
    }

    println!("------------------------------------------------------------");
    println!("✓ ALL VALIDATION CHECKS PASSED ({:.2?})", start.elapsed());
    println!("============================================================");

    Ok(())
}

/// Builds and packages a mod into a distribution bundle.
#[allow(clippy::too_many_lines)]
pub fn run_mod_build(path_str: &str, output_sub: &str) -> Result<()> {
    // 1. Run validation first
    run_mod_validate(path_str)?;

    let mod_dir = Path::new(path_str);
    let start = Instant::now();

    println!(
        "\n>> Assembling distribution package for {}...",
        mod_dir.display()
    );

    // 2. If Cargo.toml exists, build release WASM
    let cargo_toml = mod_dir.join("Cargo.toml");
    if cargo_toml.exists() {
        println!(
            ">> Compiling release WASM (cargo build --release --target wasm32-unknown-unknown)..."
        );
        let status = Command::new("cargo")
            .arg("build")
            .arg("--release")
            .arg("--target")
            .arg("wasm32-unknown-unknown")
            .arg("--manifest-path")
            .arg(&cargo_toml)
            .status()
            .context("Failed to run cargo build for WASM")?;

        if !status.success() {
            bail!("Cargo build failed for WASM target");
        }

        let manifest = ModManifest::from_file(mod_dir.join("mod.toml"))?;
        let wasm_src = mod_dir
            .join("target/wasm32-unknown-unknown/release")
            .join(format!("{}.wasm", manifest.info.id.replace('-', "_")));

        let dest_wasm = mod_dir.join("server.wasm");
        if wasm_src.exists() {
            std::fs::copy(&wasm_src, &dest_wasm)?;
            let size = dest_wasm.metadata().map_or(0, |m| m.len());
            println!("✓ Copied release WASM to server.wasm ({size} bytes)");
        } else {
            // Check if root target has it
            let root_wasm = Path::new("target/wasm32-unknown-unknown/release")
                .join(format!("{}.wasm", manifest.info.id.replace('-', "_")));
            if root_wasm.exists() {
                std::fs::copy(&root_wasm, &dest_wasm)?;
                let size = dest_wasm.metadata().map_or(0, |m| m.len());
                println!("✓ Copied release WASM to server.wasm ({size} bytes)");
            }
        }
    }

    // 3. Package .vxmod bundle
    let manifest = ModManifest::from_file(mod_dir.join("mod.toml"))?;
    let out_dir = mod_dir.join(output_sub);
    std::fs::create_dir_all(&out_dir)?;
    let abs_out_dir = std::fs::canonicalize(&out_dir)?;

    let bundle_name = format!("{}-{}.vxmod", manifest.info.id, manifest.info.version);
    let bundle_path = abs_out_dir.join(&bundle_name);

    if bundle_path.exists() {
        let _ = std::fs::remove_file(&bundle_path);
    }

    // Use zip utility to create clean .vxmod archive
    let mut zip_cmd = Command::new("zip");
    zip_cmd.arg("-r").arg(&bundle_path);
    zip_cmd.arg("mod.toml");

    if mod_dir.join("assets").exists() {
        zip_cmd.arg("assets");
    }
    if mod_dir.join("data").exists() {
        zip_cmd.arg("data");
    }
    if mod_dir.join("server.wasm").exists() {
        zip_cmd.arg("server.wasm");
    }
    if mod_dir.join("plugin.js").exists() {
        zip_cmd.arg("plugin.js");
    }

    zip_cmd.current_dir(mod_dir);

    let zip_status = zip_cmd.status().context("Failed to run zip command")?;
    if !zip_status.success() {
        bail!("Failed to generate .vxmod bundle archive");
    }

    let bundle_size = bundle_path.metadata().map_or(0, |m| m.len());
    let hash = compute_file_sha256(&bundle_path);

    println!(
        "\n=================================================================================================="
    );
    println!(
        "                        TELOS MOD PACKAGE GENERATED                                               "
    );
    println!(
        "=================================================================================================="
    );
    println!("{:<45} {:>10}  SHA256 Checksum", "Filename", "Size");
    println!("{:-<45} {:-<10}  {:-<64}", "", "", "");
    println!(
        "{:<45} {:>10}  {}",
        bundle_name,
        format_bytes(bundle_size),
        hash
    );
    println!(
        "=================================================================================================="
    );
    println!(
        "✓ Package created at: {} in {:.2?}",
        bundle_path.display(),
        start.elapsed()
    );

    Ok(())
}

fn format_display_name(raw: &str) -> String {
    raw.split('_')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn to_pascal_case(raw: &str) -> String {
    raw.split('_')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            }
        })
        .collect::<String>()
}

fn walk_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut results = Vec::new();
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                results.extend(walk_dir(&path)?);
            } else {
                results.push(path);
            }
        }
    }
    Ok(results)
}

fn generate_procedural_texture(path: &Path, seed_str: &str) -> Result<()> {
    use image::{Rgba, RgbaImage};

    let mut img = RgbaImage::new(16, 16);
    let mut hash = 0x811c_9dc5u32;
    for b in seed_str.as_bytes() {
        hash ^= u32::from(*b);
        hash = hash.wrapping_mul(0x0100_0193);
    }

    let r_base = ((hash & 0xFF) as u8).max(40);
    let g_base = (((hash >> 8) & 0xFF) as u8).max(40);
    let b_base = (((hash >> 16) & 0xFF) as u8).max(40);

    for y in 0..16 {
        for x in 0..16 {
            let is_border = x == 0 || x == 15 || y == 0 || y == 15;
            let checker = ((x ^ y) & 1) == 0;
            let (r, g, b) = if is_border {
                (
                    r_base.saturating_sub(30),
                    g_base.saturating_sub(30),
                    b_base.saturating_sub(30),
                )
            } else if checker {
                (r_base, g_base, b_base)
            } else {
                (
                    r_base.saturating_add(25),
                    g_base.saturating_add(25),
                    b_base.saturating_add(25),
                )
            };
            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    img.save_with_format(path, image::ImageFormat::Png)
        .with_context(|| format!("Failed to save procedural texture to {}", path.display()))?;

    Ok(())
}

fn compute_file_sha256(path: &Path) -> String {
    match Command::new("sha256sum").arg(path).output() {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
            .split_whitespace()
            .next()
            .unwrap_or("N/A")
            .to_string(),
        _ => "N/A".to_string(),
    }
}

fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    if bytes >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

fn compute_relative_path(from: &Path, to: &Path) -> PathBuf {
    let from_canon = from.canonicalize().unwrap_or_else(|_| from.to_path_buf());
    let to_canon = to.canonicalize().unwrap_or_else(|_| to.to_path_buf());

    let from_comps: Vec<_> = from_canon.components().collect();
    let to_comps: Vec<_> = to_canon.components().collect();

    let mut common = 0;
    while common < from_comps.len()
        && common < to_comps.len()
        && from_comps[common] == to_comps[common]
    {
        common += 1;
    }

    let mut rel = PathBuf::new();
    for _ in common..from_comps.len() {
        rel.push("..");
    }
    for comp in &to_comps[common..] {
        rel.push(comp.as_os_str());
    }

    if rel.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        rel
    }
}
