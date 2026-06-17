//! Auto-generated safe wrappers over CUDA kernels discovered at build time.
//!
//! Usage:
//! ```no_run
//! use cudarc::driver::DriverError;
//! use lmrs_kernels::Runtime;
//!
//! let rt = Runtime::default()?;
//! let x = vec![half::f16::from_f32(1.0); 1024];
//! let x = rt.upload(&x)?;
//! let mut y = rt.zeros::<half::f16>(1024)?;
//! rt.fp16_silu(&mut y, &x)?;
//! rt.synchronize()?;
//! # Ok::<(), DriverError>(())
//! ```

// Pull in everything build.rs emitted
include!(concat!(env!("OUT_DIR"), "/gen_kernels.rs"));
