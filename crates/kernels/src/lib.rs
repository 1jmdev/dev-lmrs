//! Auto-generated safe wrappers over CUDA kernels discovered at build time.
//!
//! Usage:
//! ```no_run
//! use cudarc::driver::DriverError;
//!
//! let rt = lmrs_kernels::Runtime::default()?;
//! let a = vec![1.0f32; 1024];
//! let b = vec![2.0f32; 1024];
//! let a = rt.upload(&a)?;
//! let b = rt.upload(&b)?;
//! let mut out = rt.zeros::<f32>(1024)?;
//! rt.vector_add(&a, &b, &mut out)?;
//! rt.synchronize()?;
//! # Ok::<(), DriverError>(())
//! ```

// Pull in everything build.rs emitted
include!(concat!(env!("OUT_DIR"), "/gen_kernels.rs"));
