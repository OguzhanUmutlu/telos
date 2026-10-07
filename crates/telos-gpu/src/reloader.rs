//! Runtime shader compilation and hot-reloading using `glslc`.

use std::path::Path;
use std::process::Command;
use tracing::{error, info};

use crate::error::GpuError;
use crate::pipeline::ShaderModule;

/// Compiles GLSL shaders into SPIR-V bytecode and provides runtime pipeline reloading.
pub struct ShaderCompiler;

impl ShaderCompiler {
    /// Compiles GLSL source string to SPIR-V using `glslc`.
    pub fn compile_source(source: &str, stage: &str, name: &str) -> Result<Vec<u8>, GpuError> {
        let temp_dir = std::env::temp_dir();
        let src_path = temp_dir.join(format!("telos_{name}_{}.{stage}", std::process::id()));
        let spv_path = temp_dir.join(format!("telos_{name}_{}.{stage}.spv", std::process::id()));

        std::fs::write(&src_path, source)
            .map_err(|e| GpuError::Shader(format!("Failed to write shader source: {e}")))?;

        let output = Command::new("glslc")
            .arg(&src_path)
            .arg("-o")
            .arg(&spv_path)
            .output()
            .map_err(|e| GpuError::Shader(format!("Failed to execute glslc: {e}")))?;

        let _ = std::fs::remove_file(&src_path);

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!("Shader compilation error ({name}.{stage}):\n{stderr}");
            let _ = std::fs::remove_file(&spv_path);
            return Err(GpuError::Shader(format!("glslc failed:\n{stderr}")));
        }

        let spv_bytes = std::fs::read(&spv_path)
            .map_err(|e| GpuError::Shader(format!("Failed to read spv output: {e}")))?;
        let _ = std::fs::remove_file(&spv_path);
        Ok(spv_bytes)
    }

    /// Compiles a shader file on disk to SPIR-V using `glslc`.
    pub fn compile_file(path: &Path) -> Result<Vec<u8>, GpuError> {
        if !path.exists() {
            return Err(GpuError::Shader(format!(
                "Shader file not found: {}",
                path.display()
            )));
        }

        let temp_dir = std::env::temp_dir();
        let file_name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let spv_path = temp_dir.join(format!("telos_{}_{file_name}.spv", std::process::id()));

        let output = Command::new("glslc")
            .arg(path)
            .arg("-o")
            .arg(&spv_path)
            .output()
            .map_err(|e| GpuError::Shader(format!("Failed to execute glslc: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!("Shader compilation error ({}):\n{stderr}", path.display());
            let _ = std::fs::remove_file(&spv_path);
            return Err(GpuError::Shader(format!(
                "glslc failed for {}:\n{stderr}",
                path.display()
            )));
        }

        let spv_bytes = std::fs::read(&spv_path)
            .map_err(|e| GpuError::Shader(format!("Failed to read spv output: {e}")))?;
        let _ = std::fs::remove_file(&spv_path);
        info!("Compiled shader successfully: {}", path.display());
        Ok(spv_bytes)
    }

    /// Recompiles a shader file and creates a new Vulkan `ShaderModule`.
    pub fn compile_module(device: &ash::Device, path: &Path) -> Result<ShaderModule, GpuError> {
        let spv = Self::compile_file(path)?;
        ShaderModule::from_spv(device, &spv)
    }
}
