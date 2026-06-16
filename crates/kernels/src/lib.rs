//! Auto-generated safe wrappers over CUDA kernels discovered at build time.
//!
//! Usage:
//! ```no_run
//! let a = vec![1.0f32; 1024];
//! let b = vec![2.0f32; 1024];
//! let out = lmrs_kernels::vector_add(&a, &b, 1024)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Pull in everything build.rs emitted
include!(concat!(env!("OUT_DIR"), "/gen_kernels.rs"));
