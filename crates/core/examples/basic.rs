use lmrs_core::{CudaDevice, Result, f16};

fn main() -> Result<()> {
    let device = CudaDevice::default()?;

    let rows = 2usize;
    let inner = 4usize;
    let cols = 3usize;

    let a: Vec<f16> = (0..rows * inner)
        .map(|i| f16::from_f32(i as f32 * 0.01))
        .collect();
    let b: Vec<f16> = (0..inner * cols)
        .map(|i| f16::from_f32((i as f32 + 1.0) * 0.02))
        .collect();
    let weight = vec![f16::from_f32(1.0); cols];

    let a = device.upload(&a, [rows, inner])?;
    let b = device.upload(&b, [inner, cols])?;
    let weight = device.upload(&weight, [cols])?;

    let out = a.matmul(&b)?.rms_norm(&weight, 1e-5)?.silu()?;
    device.synchronize()?;

    let preview: Vec<f32> = out.download()?.into_iter().map(|v| v.to_f32()).collect();
    println!("shape: {:?}, values: {preview:?}", out.shape());

    Ok(())
}
