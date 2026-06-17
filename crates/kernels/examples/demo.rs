use half::f16;
use lmrs_kernels::*;
use std::time::Instant;

fn main() {
    let rt = Runtime::default().unwrap();
    let tokens = 32usize;
    let hidden = 128usize;
    let ffn = 256usize;

    let activations: Vec<f16> = (0..tokens * hidden)
        .map(|i| f16::from_f32((((i * 13 + 17) % 97) as f32 / 48.0 - 1.0) * 0.013))
        .collect();
    let rms_weight = vec![f16::from_f32(1.0); hidden];
    let w_gate: Vec<f16> = (0..hidden * ffn)
        .map(|i| f16::from_f32((((i * 13 + 17) % 97) as f32 / 48.0 - 1.0) * 0.007))
        .collect();
    let w_up: Vec<f16> = (0..hidden * ffn)
        .map(|i| f16::from_f32((((i * 13 + 17) % 97) as f32 / 48.0 - 1.0) * 0.011))
        .collect();

    let x = rt.upload(&activations).unwrap();
    let rms_weight = rt.upload(&rms_weight).unwrap();
    let w_gate = rt.upload(&w_gate).unwrap();
    let w_up = rt.upload(&w_up).unwrap();

    let mut normed = rt.zeros::<f16>(tokens * hidden).unwrap();
    let mut gate = rt.zeros::<f16>(tokens * ffn).unwrap();
    let mut up = rt.zeros::<f16>(tokens * ffn).unwrap();
    let mut silu_gate = rt.zeros::<f16>(tokens * ffn).unwrap();
    let mut fused = rt.zeros::<f16>(tokens * ffn).unwrap();

    rt.fp16_matmul_nn(
        &mut gate,
        &x,
        &w_gate,
        tokens as i32,
        ffn as i32,
        hidden as i32,
    )
    .unwrap();
    rt.synchronize().unwrap();

    let graph = rt
        .capture_graph(|rt| {
            rt.fp16_rms_norm(&mut normed, &x, &rms_weight, hidden as i32, 1e-5)?;
            rt.fp16_matmul_nn(
                &mut gate,
                &normed,
                &w_gate,
                tokens as i32,
                ffn as i32,
                hidden as i32,
            )?;
            rt.fp16_matmul_nn(
                &mut up,
                &normed,
                &w_up,
                tokens as i32,
                ffn as i32,
                hidden as i32,
            )?;
            rt.fp16_silu(&mut silu_gate, &gate)?;
            rt.fp16_mul(&mut fused, &silu_gate, &up)?;
            Ok(())
        })
        .unwrap();

    rt.synchronize().unwrap();
    let start = Instant::now();
    graph.launch().unwrap();
    rt.synchronize().unwrap();

    let elapsed = start.elapsed();
    let out = rt.download(&fused).unwrap();
    let preview: Vec<f32> = out.iter().take(8).map(|v| v.to_f32()).collect();

    // Expected result: fused = silu(rms_norm(x) @ w_gate) * (rms_norm(x) @ w_up),
    // with output shape [tokens, ffn] = [32, 256]. The preview prints the first 8 fp16 values.
    println!("llm feed-forward block: {tokens}x{hidden} -> {tokens}x{ffn}");
    println!("elapsed: {:?}", elapsed);
    println!("output preview: {preview:?}");
}
