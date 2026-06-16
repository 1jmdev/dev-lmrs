//! Build-time CUDA tensor API generator.

mod compile;
mod config;
mod error;
mod generate;
mod model;
mod parser;
mod pipeline;

pub use config::Config;
pub use error::{CuditError, Result};
pub use pipeline::{GeneratedApi, generate};
