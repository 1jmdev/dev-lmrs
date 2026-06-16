use std::fs;
use std::path::{Path, PathBuf};

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

pub(crate) fn read_cuda_source(source_file: &Path) -> Result<String> {
    fs::read_to_string(source_file).map_err(|err| CuditError::io(source_file, err))
}
