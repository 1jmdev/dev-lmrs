use std::fs;
use std::path::{Path, PathBuf};

use crate::compile::{compile_ptx, discover_cuda_files, read_cuda_source};
use crate::config::Config;
use crate::error::{CuditError, Result};
use crate::generate::generate_api;
use crate::model::KernelFile;
use crate::parser::parse_kernels;

/// Summary of files and kernels produced by a generation run.
#[derive(Debug)]
pub struct GeneratedApi {
    generated_file: PathBuf,
    ptx_files: Vec<PathBuf>,
    kernel_names: Vec<String>,
}

impl GeneratedApi {
    /// Returns the generated Rust source file path.
    ///
    /// ```no_run
    /// let api = lmrs_cudit::Config::from_env("kernels")?.generate()?;
    /// let generated = api.generated_file();
    /// assert!(generated.ends_with("gen_kernels.rs"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn generated_file(&self) -> &Path {
        &self.generated_file
    }

    /// Returns the generated PTX file paths.
    ///
    /// ```no_run
    /// let api = lmrs_cudit::Config::from_env("kernels")?.generate()?;
    /// for ptx in api.ptx_files() {
    ///     println!("{}", ptx.display());
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn ptx_files(&self) -> &[PathBuf] {
        &self.ptx_files
    }

    /// Returns the discovered CUDA kernel names.
    ///
    /// ```no_run
    /// let api = lmrs_cudit::Config::from_env("kernels")?.generate()?;
    /// println!("generated {} kernels", api.kernel_names().len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn kernel_names(&self) -> &[String] {
        &self.kernel_names
    }

    /// Returns the number of generated kernel methods.
    ///
    /// ```no_run
    /// let api = lmrs_cudit::Config::from_env("kernels")?.generate()?;
    /// assert_eq!(api.kernel_count(), api.kernel_names().len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn kernel_count(&self) -> usize {
        self.kernel_names.len()
    }
}

/// Compiles CUDA kernels to PTX and writes a generated tensor-first Rust API.
///
/// This is intended to run from a Cargo build script. The generated API depends on `cudarc` in the
/// consuming crate, but it does not depend on `lmrs_cudit` at runtime.
///
/// ```no_run
/// let config = lmrs_cudit::Config::from_env("kernels")?;
/// let generated = lmrs_cudit::generate(config)?;
/// println!("generated {} kernels", generated.kernel_count());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn generate(config: Config) -> Result<GeneratedApi> {
    if config.emit_cargo_directives {
        emit_static_cargo_directives(&config);
    }

    fs::create_dir_all(&config.out_dir).map_err(|err| CuditError::io(&config.out_dir, err))?;
    let cu_files = discover_cuda_files(&config.kernels_dir)?;
    let mut files = Vec::new();
    let mut ptx_files = Vec::new();

    for source_path in cu_files {
        if config.emit_cargo_directives {
            println!("cargo:rerun-if-changed={}", source_path.display());
        }

        let source = read_cuda_source(&source_path)?;
        let orig_stem = source_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| CuditError::InvalidKernelFileName(source_path.clone()))?
            .to_string();
        let ptx_name = format!("{orig_stem}.ptx");
        let ptx_path = config.out_dir.join(&ptx_name);

        compile_ptx(&config, &source_path, &ptx_path)?;
        ptx_files.push(ptx_path);

        let kernels =
            parse_kernels(&source).map_err(|(kernel, message)| CuditError::ParseKernel {
                source_file: source_path.clone(),
                kernel,
                message,
            })?;

        files.push(KernelFile {
            stem: sanitize(&orig_stem),
            ptx_name,
            kernels,
        });
    }

    let generated_source = generate_api(&config, &files);
    let generated_file = config.out_dir.join(&config.generated_file_name);
    fs::write(&generated_file, generated_source)
        .map_err(|err| CuditError::io(&generated_file, err))?;

    let kernel_names = files
        .iter()
        .flat_map(|file| file.kernels.iter().map(|kernel| kernel.name.clone()))
        .collect();

    Ok(GeneratedApi {
        generated_file,
        ptx_files,
        kernel_names,
    })
}

fn emit_static_cargo_directives(config: &Config) {
    println!("cargo:rerun-if-changed={}", config.kernels_dir.display());
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}
