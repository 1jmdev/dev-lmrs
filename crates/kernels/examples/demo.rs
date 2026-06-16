use cudarc::driver::DriverError;
use lmrs_kernels::*;
use std::time::{Duration, Instant};

fn main() -> Result<(), DriverError> {
    println!("discovered kernels: {:?}", kernel_names());

    let n = 1024usize;
    let iters = 1_000usize;
    let a = vec![1.0f32; n];
    let b = vec![2.0f32; n];

    let regular = time_regular(&a, &b, n, iters)?;
    let graph = time_graph(&a, &b, n, iters)?;

    println!(
        "regular tensor launches: {:?} / iter",
        regular / iters as u32
    );
    println!("cuda graph replay:       {:?} / iter", graph / iters as u32);

    Ok(())
}

fn time_regular(a: &[f32], b: &[f32], n: usize, iters: usize) -> Result<Duration, DriverError> {
    let rt = Runtime::default()?;
    let d_a = rt.upload(a)?;
    let d_b = rt.upload(b)?;
    let mut d_out = rt.zeros::<f32>(n)?;
    let mut d_y = rt.zeros::<f32>(n)?;

    rt.synchronize()?;
    let start = Instant::now();
    for _ in 0..iters {
        rt.vector_add(&d_a, &d_b, &mut d_out)?;
        rt.saxpy(3.0, &d_out, &mut d_y)?;
        rt.scale(&mut d_y, 5.0)?;
    }
    rt.synchronize()?;
    Ok(start.elapsed())
}

fn time_graph(a: &[f32], b: &[f32], n: usize, iters: usize) -> Result<Duration, DriverError> {
    let rt = Runtime::default()?;
    let d_a = rt.upload(a)?;
    let d_b = rt.upload(b)?;
    let mut d_out = rt.zeros::<f32>(n)?;
    let mut d_y = rt.zeros::<f32>(n)?;

    rt.vector_add(&d_a, &d_b, &mut d_out)?;
    rt.saxpy(3.0, &d_out, &mut d_y)?;
    rt.scale(&mut d_y, 5.0)?;
    rt.synchronize()?;

    let graph = rt.capture_graph(|rt| {
        rt.vector_add(&d_a, &d_b, &mut d_out)?;
        rt.saxpy(3.0, &d_out, &mut d_y)?;
        rt.scale(&mut d_y, 5.0)
    })?;
    rt.synchronize()?;

    let start = Instant::now();
    for _ in 0..iters {
        graph.launch()?;
    }
    rt.synchronize()?;
    Ok(start.elapsed())
}
