use std::io;
use std::path::PathBuf;

/// Result type used by `lmrs-cudit`.
pub type Result<T> = std::result::Result<T, CuditError>;

/// Error returned while discovering kernels, compiling PTX, or generating Rust code.
#[derive(Debug, thiserror::Error)]
pub enum CuditError {
    /// A required Cargo build-script environment variable was not present.
    #[error("missing environment variable `{0}`")]
    MissingEnvVar(&'static str),

    /// A file-system operation failed.
    #[error("I/O error at `{path}`: {source}")]
    Io {
        /// Path involved in the failed operation.
        path: PathBuf,
        /// Original I/O error.
        source: io::Error,
    },

    /// Recursive kernel discovery failed.
    #[error("failed to discover kernels below `{path}`: {source}")]
    WalkDir {
        /// Root path being traversed.
        path: PathBuf,
        /// Original traversal error.
        source: walkdir::Error,
    },

    /// A `.cu` file did not have a valid UTF-8 file stem.
    #[error("kernel file `{0}` has no valid UTF-8 file stem")]
    InvalidKernelFileName(PathBuf),

    /// NVRTC failed to compile a CUDA source file.
    #[error("NVRTC failed for `{source_file}`: {error:?}")]
    NvrtcCompile {
        /// CUDA source file being compiled.
        source_file: PathBuf,
        /// Original cudarc/NVRTC compilation error.
        error: cudarc::nvrtc::CompileError,
    },

    /// A CUDA kernel parameter could not be parsed into a Rust API type.
    #[error("failed to parse kernel `{kernel}` in `{source_file}`: {message}")]
    ParseKernel {
        /// CUDA source file being parsed.
        source_file: PathBuf,
        /// Kernel name.
        kernel: String,
        /// Human-readable parse failure.
        message: String,
    },
}

impl CuditError {
    /// Creates an I/O error with path context.
    ///
    /// ```
    /// let err = lmrs_cudit::CuditError::io("kernels", std::io::Error::from(std::io::ErrorKind::NotFound));
    /// assert!(err.to_string().contains("kernels"));
    /// ```
    pub fn io(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
