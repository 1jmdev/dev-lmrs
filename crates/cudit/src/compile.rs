use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use walkdir::WalkDir;

use crate::config::Config;
use crate::error::{CuditError, Result};

pub(crate) fn discover_cuda_files(kernels_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut cu_files = Vec::new();
    if kernels_dir.is_dir() {
        for entry in WalkDir::new(kernels_dir).sort_by_file_name() {
            let entry = entry.map_err(|source| CuditError::WalkDir {
                path: kernels_dir.to_path_buf(),
                source,
            })?;
            let path = entry.path();
            if entry.file_type().is_file()
                && path.extension().and_then(|ext| ext.to_str()) == Some("cu")
            {
                cu_files.push(path.to_path_buf());
            }
        }
    }
    cu_files.sort();
    Ok(cu_files)
}

pub(crate) fn compile_ptx(config: &Config, source_file: &Path, ptx_path: &Path) -> Result<()> {
    let source = fs::read_to_string(source_file).map_err(|err| CuditError::io(source_file, err))?;
    if source.contains("#include") {
        return compile_ptx_with_nvcc(config, source_file, ptx_path);
    }

    let mut options = CompileOptions {
        include_paths: config.include_paths.clone(),
        options: config.nvrtc_options.clone(),
        use_fast_math: config.use_fast_math,
        maxrregcount: config.max_register_count,
        name: source_file
            .file_name()
            .and_then(|name| name.to_str())
            .map(ToOwned::to_owned),
        ..Default::default()
    };
    if let Some(cap) = &config.compute_capability {
        options
            .options
            .push(format!("--gpu-architecture=compute_{cap}"));
    }

    let ptx = compile_ptx_with_opts(source, options).map_err(|error| CuditError::NvrtcCompile {
        source_file: source_file.to_path_buf(),
        error,
    })?;
    fs::write(ptx_path, ptx.to_src()).map_err(|err| CuditError::io(ptx_path, err))
}

fn compile_ptx_with_nvcc(config: &Config, source_file: &Path, ptx_path: &Path) -> Result<()> {
    let nvcc = std::env::var("CUDA_NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let mut args = vec![
        "--ptx".to_string(),
        source_file.to_string_lossy().into_owned(),
        "-o".to_string(),
        ptx_path.to_string_lossy().into_owned(),
    ];
    if let Some(cap) = &config.compute_capability {
        args.push(format!("-arch=sm_{cap}"));
    }
    for option in &config.nvrtc_options {
        args.push(option.clone());
    }
    for include_path in &config.include_paths {
        args.push(format!("-I{include_path}"));
    }

    run_command(source_file, &nvcc, &args)
}

pub(crate) fn compile_native_cuda(
    config: &Config,
    source_files: &[PathBuf],
    lib_name: &str,
) -> Result<Option<PathBuf>> {
    if source_files.is_empty() {
        return Ok(None);
    }

    let native_dir = config.out_dir.join("native_cuda");
    fs::create_dir_all(&native_dir).map_err(|err| CuditError::io(&native_dir, err))?;

    let nvcc = std::env::var("CUDA_NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let mut objects = Vec::new();
    for source_file in source_files {
        let stem = source_file
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| CuditError::InvalidKernelFileName(source_file.clone()))?;
        let obj = native_dir.join(format!("{stem}.o"));
        let mut args = vec![
            "-c".to_string(),
            source_file.to_string_lossy().into_owned(),
            "-o".to_string(),
            obj.to_string_lossy().into_owned(),
            "-Xcompiler".to_string(),
            "-fPIC".to_string(),
        ];
        if let Some(cap) = &config.compute_capability {
            args.push(format!("-arch=sm_{cap}"));
        }
        for include_path in &config.include_paths {
            args.push(format!("-I{include_path}"));
        }

        run_command(source_file, &nvcc, &args)?;
        objects.push(obj);
    }

    let shared = native_dir.join(format!("lib{lib_name}.so"));
    let nvcc = std::env::var("CUDA_NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let mut args = vec![
        "-shared".to_string(),
        "-o".to_string(),
        shared.to_string_lossy().into_owned(),
    ];
    args.extend(objects.iter().map(|obj| obj.to_string_lossy().into_owned()));
    args.push("-lcublas".to_string());
    args.push("-lcudart".to_string());
    run_command(&shared, &nvcc, &args)?;

    Ok(Some(native_dir))
}

pub(crate) fn read_cuda_source(source_file: &Path) -> Result<String> {
    fs::read_to_string(source_file).map_err(|err| CuditError::io(source_file, err))
}

fn run_command(source_file: &Path, program: &str, args: &[String]) -> Result<()> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|err| CuditError::io(source_file, err))?;
    if output.status.success() {
        return Ok(());
    }
    Err(CuditError::NativeCompile {
        source_file: source_file.to_path_buf(),
        program: program.to_string(),
        args: args.to_vec(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}
