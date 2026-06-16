use std::env;
use std::path::{Path, PathBuf};

use crate::error::{CuditError, Result};
use crate::pipeline::{GeneratedApi, generate};

/// Configuration for CUDA discovery, PTX compilation, and Rust API generation.
#[derive(Clone, Debug)]
pub struct Config {
    pub(crate) kernels_dir: PathBuf,
    pub(crate) out_dir: PathBuf,
    pub(crate) generated_file_name: String,
    pub(crate) compute_capability: Option<String>,
    pub(crate) include_paths: Vec<String>,
    pub(crate) nvrtc_options: Vec<String>,
    pub(crate) use_fast_math: Option<bool>,
    pub(crate) max_register_count: Option<usize>,
    pub(crate) cudarc_crate_path: String,
    pub(crate) emit_cargo_directives: bool,
}

impl Config {
    /// Creates a generator configuration from explicit kernel and output paths.
    ///
    /// Relative paths are resolved by the caller's current working directory. In Cargo build
    /// scripts, prefer [`Config::from_env`] so `OUT_DIR` and `CARGO_MANIFEST_DIR` are handled
    /// consistently.
    ///
    /// ```no_run
    /// let config = lmrs_cudit::Config::new("kernels", std::env::var("OUT_DIR")?);
    /// lmrs_cudit::generate(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(kernels_dir: impl Into<PathBuf>, out_dir: impl Into<PathBuf>) -> Self {
        Self {
            kernels_dir: kernels_dir.into(),
            out_dir: out_dir.into(),
            generated_file_name: "gen_kernels.rs".to_string(),
            compute_capability: None,
            include_paths: Vec::new(),
            nvrtc_options: Vec::new(),
            use_fast_math: None,
            max_register_count: None,
            cudarc_crate_path: "cudarc".to_string(),
            emit_cargo_directives: true,
        }
    }

    /// Creates a Cargo build-script configuration from environment variables.
    ///
    /// `kernels_dir` is resolved relative to `CARGO_MANIFEST_DIR` when it is not absolute. The
    /// generated Rust file and PTX files are written to `OUT_DIR`. `CUDA_COMPUTE_CAP` is read
    /// when present.
    ///
    /// ```no_run
    /// let config = lmrs_cudit::Config::from_env("kernels")?;
    /// lmrs_cudit::generate(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_env(kernels_dir: impl AsRef<Path>) -> Result<Self> {
        let manifest_dir = PathBuf::from(
            env::var_os("CARGO_MANIFEST_DIR")
                .ok_or(CuditError::MissingEnvVar("CARGO_MANIFEST_DIR"))?,
        );
        let out_dir =
            PathBuf::from(env::var_os("OUT_DIR").ok_or(CuditError::MissingEnvVar("OUT_DIR"))?);
        let kernels_dir = resolve_manifest_path(&manifest_dir, kernels_dir.as_ref());
        let mut config = Self::new(kernels_dir, out_dir);

        if let Ok(cap) = env::var("CUDA_COMPUTE_CAP") {
            config = config.compute_capability(cap);
        }
        Ok(config)
    }

    /// Overrides the directory that contains `.cu` kernel files.
    ///
    /// ```no_run
    /// let config = lmrs_cudit::Config::new("kernels", std::env::var("OUT_DIR")?)
    ///     .kernels_dir("cuda/kernels");
    /// lmrs_cudit::generate(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn kernels_dir(mut self, kernels_dir: impl Into<PathBuf>) -> Self {
        self.kernels_dir = kernels_dir.into();
        self
    }

    /// Overrides the directory where PTX files and the generated Rust file are written.
    ///
    /// ```no_run
    /// let config = lmrs_cudit::Config::new("kernels", "target/lmrs_cudit-out")
    ///     .out_dir("target/generated-cuda");
    /// lmrs_cudit::generate(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn out_dir(mut self, out_dir: impl Into<PathBuf>) -> Self {
        self.out_dir = out_dir.into();
        self
    }

    /// Overrides the generated Rust file name.
    ///
    /// The file is still written inside [`Config::out_dir`]. Consumer crates typically include it
    /// with `include!(concat!(env!("OUT_DIR"), "/generated_file_name.rs"))`.
    ///
    /// ```no_run
    /// let config = lmrs_cudit::Config::new("kernels", std::env::var("OUT_DIR")?)
    ///     .generated_file_name("cuda_api.rs");
    /// lmrs_cudit::generate(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn generated_file_name(mut self, generated_file_name: impl Into<String>) -> Self {
        self.generated_file_name = generated_file_name.into();
        self
    }

    /// Sets the CUDA compute capability passed to NVRTC as `--gpu-architecture=compute_<value>`.
    ///
    /// Use values such as `75`, `86`, or `90`. If unset, NVRTC chooses its default target.
    ///
    /// ```no_run
    /// let config = lmrs_cudit::Config::new("kernels", std::env::var("OUT_DIR")?)
    ///     .compute_capability("90");
    /// lmrs_cudit::generate(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn compute_capability(mut self, compute_capability: impl Into<String>) -> Self {
        self.compute_capability = Some(compute_capability.into());
        self
    }

    /// Clears any configured CUDA compute capability override.
    ///
    /// ```no_run
    /// let config = lmrs_cudit::Config::new("kernels", std::env::var("OUT_DIR")?)
    ///     .compute_capability("90")
    ///     .without_compute_capability();
    /// lmrs_cudit::generate(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn without_compute_capability(mut self) -> Self {
        self.compute_capability = None;
        self
    }

    /// Adds an include path passed to NVRTC.
    ///
    /// Use this when kernels include shared CUDA headers.
    ///
    /// ```no_run
    /// let config = lmrs_cudit::Config::new("kernels", std::env::var("OUT_DIR")?)
    ///     .include_path("kernels/include");
    /// lmrs_cudit::generate(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn include_path(mut self, include_path: impl Into<String>) -> Self {
        self.include_paths.push(include_path.into());
        self
    }

    /// Adds an arbitrary raw NVRTC option.
    ///
    /// The option is passed through to `cudarc::nvrtc::CompileOptions::options` unchanged.
    ///
    /// ```no_run
    /// let config = lmrs_cudit::Config::new("kernels", std::env::var("OUT_DIR")?)
    ///     .nvrtc_option("--std=c++17");
    /// lmrs_cudit::generate(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn nvrtc_option(mut self, option: impl Into<String>) -> Self {
        self.nvrtc_options.push(option.into());
        self
    }

    /// Enables or disables NVRTC `--use_fast_math`.
    ///
    /// ```no_run
    /// let config = lmrs_cudit::Config::new("kernels", std::env::var("OUT_DIR")?)
    ///     .use_fast_math(true);
    /// lmrs_cudit::generate(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn use_fast_math(mut self, use_fast_math: bool) -> Self {
        self.use_fast_math = Some(use_fast_math);
        self
    }

    /// Sets NVRTC `--maxrregcount`.
    ///
    /// ```no_run
    /// let config = lmrs_cudit::Config::new("kernels", std::env::var("OUT_DIR")?)
    ///     .max_register_count(64);
    /// lmrs_cudit::generate(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn max_register_count(mut self, max_register_count: usize) -> Self {
        self.max_register_count = Some(max_register_count);
        self
    }

    /// Sets the crate path used for `cudarc` in generated Rust code.
    ///
    /// Most crates should keep the default `cudarc`. Set this only when the consuming crate
    /// renames its `cudarc` dependency in `Cargo.toml`.
    ///
    /// ```no_run
    /// let config = lmrs_cudit::Config::new("kernels", std::env::var("OUT_DIR")?)
    ///     .cudarc_crate_path("cuda_driver");
    /// lmrs_cudit::generate(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn cudarc_crate_path(mut self, cudarc_crate_path: impl Into<String>) -> Self {
        self.cudarc_crate_path = cudarc_crate_path.into();
        self
    }

    /// Enables or disables Cargo `rerun-if-*` build-script directives.
    ///
    /// Build scripts usually leave this enabled. Disable it for non-Cargo tooling that reuses the
    /// generator directly.
    ///
    /// ```no_run
    /// let config = lmrs_cudit::Config::new("kernels", "target/lmrs_cudit-out")
    ///     .emit_cargo_directives(false);
    /// lmrs_cudit::generate(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn emit_cargo_directives(mut self, emit_cargo_directives: bool) -> Self {
        self.emit_cargo_directives = emit_cargo_directives;
        self
    }

    /// Runs CUDA compilation and Rust API generation with this configuration.
    ///
    /// This is equivalent to calling [`lmrs_cudit::generate`] with the same configuration.
    ///
    /// ```no_run
    /// lmrs_cudit::Config::from_env("kernels")?.generate()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn generate(self) -> Result<GeneratedApi> {
        generate(self)
    }

    /// Returns the configured kernel source directory.
    ///
    /// ```
    /// let config = lmrs_cudit::Config::new("kernels", "out");
    /// assert_eq!(config.kernels_dir_path(), std::path::Path::new("kernels"));
    /// ```
    pub fn kernels_dir_path(&self) -> &Path {
        &self.kernels_dir
    }

    /// Returns the configured output directory.
    ///
    /// ```
    /// let config = lmrs_cudit::Config::new("kernels", "out");
    /// assert_eq!(config.out_dir_path(), std::path::Path::new("out"));
    /// ```
    pub fn out_dir_path(&self) -> &Path {
        &self.out_dir
    }

    /// Returns the generated Rust file name.
    ///
    /// ```
    /// let config = lmrs_cudit::Config::new("kernels", "out");
    /// assert_eq!(config.generated_file_name_ref(), "gen_kernels.rs");
    /// ```
    pub fn generated_file_name_ref(&self) -> &str {
        &self.generated_file_name
    }

    /// Returns the configured CUDA compute capability, if any.
    ///
    /// ```
    /// let config = lmrs_cudit::Config::new("kernels", "out").compute_capability("90");
    /// assert_eq!(config.compute_capability_ref(), Some("90"));
    /// ```
    pub fn compute_capability_ref(&self) -> Option<&str> {
        self.compute_capability.as_deref()
    }

    /// Returns NVRTC include paths.
    ///
    /// ```
    /// let config = lmrs_cudit::Config::new("kernels", "out").include_path("kernels/include");
    /// assert_eq!(config.include_paths(), &["kernels/include"]);
    /// ```
    pub fn include_paths(&self) -> &[String] {
        &self.include_paths
    }

    /// Returns raw NVRTC options.
    ///
    /// ```
    /// let config = lmrs_cudit::Config::new("kernels", "out").nvrtc_option("--std=c++17");
    /// assert_eq!(config.nvrtc_options(), &["--std=c++17"]);
    /// ```
    pub fn nvrtc_options(&self) -> &[String] {
        &self.nvrtc_options
    }

    /// Returns the configured `--use_fast_math` setting.
    ///
    /// ```
    /// let config = lmrs_cudit::Config::new("kernels", "out").use_fast_math(true);
    /// assert_eq!(config.use_fast_math_ref(), Some(true));
    /// ```
    pub fn use_fast_math_ref(&self) -> Option<bool> {
        self.use_fast_math
    }

    /// Returns the configured `--maxrregcount` setting.
    ///
    /// ```
    /// let config = lmrs_cudit::Config::new("kernels", "out").max_register_count(64);
    /// assert_eq!(config.max_register_count_ref(), Some(64));
    /// ```
    pub fn max_register_count_ref(&self) -> Option<usize> {
        self.max_register_count
    }

    /// Returns the generated-code crate path used for `cudarc`.
    ///
    /// ```
    /// let config = lmrs_cudit::Config::new("kernels", "out");
    /// assert_eq!(config.cudarc_crate_path_ref(), "cudarc");
    /// ```
    pub fn cudarc_crate_path_ref(&self) -> &str {
        &self.cudarc_crate_path
    }
}

fn resolve_manifest_path(manifest_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        manifest_dir.join(path)
    }
}
