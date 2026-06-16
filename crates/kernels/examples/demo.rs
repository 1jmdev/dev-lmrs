use cudarc::driver::DriverError;
use lmrs_kernels::*;

fn main() -> Result<(), DriverError> {
    println!("discovered kernels: {:?}", kernel_names());

    let n = 1024usize;

    let a = vec![1.0f32; n];
    let b = vec![2.0f32; n];
    let out = vector_add(&a, &b, n as i32)?;
    println!("vector_add[0..4] = {:?} (expect 3.0)", &out[..4]);

    let x = vec![1.0f32; n];
    let mut y = vec![1.0f32; n];
    saxpy(3.0, &x, &mut y, n as i32)?;
    println!("saxpy[0..4] = {:?} (expect 4.0)", &y[..4]);

    let mut data = vec![2.0f32; n];
    scale(&mut data, 5.0, n as i32)?;
    println!("scale[0..4] = {:?} (expect 10.0)", &data[..4]);

    let mut buf = vec![0.0f32; n];
    fill(&mut buf, 7.0, n as i32)?;
    println!("fill[0..4] = {:?} (expect 7.0)", &buf[..4]);

    Ok(())
}
