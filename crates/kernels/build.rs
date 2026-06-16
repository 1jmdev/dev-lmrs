use lmrs_cudit::{Config, generate};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let config = Config::from_env("kernels")?.generated_file_name("gen_kernels.rs");

    generate(config)?;
    Ok(())
}
