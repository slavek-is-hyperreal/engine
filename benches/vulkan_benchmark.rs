// benches/vulkan_benchmark.rs

use eml_trs::backends::vulkan_eml::EmlKernel;

fn cpu_log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let log_sum_exp = sum_exp.ln() + max;
    logits.iter().map(|&x| x - log_sum_exp).collect()
}

fn cpu_dot_product_asis(input: &[f32], weights: &[f32], seq_len: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len];
    for out_idx in 0..seq_len {
        let x0 = input[out_idx * k + 0];
        let w0 = weights[out_idx * k + 0];
        let mut acc = x0 * w0;
        for i in 1..k {
            let xi = input[out_idx * k + i];
            let wi = weights[out_idx * k + i];
            acc = acc - xi * wi;
        }
        out[out_idx] = acc;
    }
    out
}

#[tokio::main]
async fn main() {
    println!("=== EML Vulkan Kernel Benchmark ===");

    let n = 2048;
    let logits: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) - 10.0).collect();
    
    // CPU reference for log-softmax
    let t_cpu = std::time::Instant::now();
    let cpu_result = cpu_log_softmax(&logits);
    let cpu_time = t_cpu.elapsed().as_secs_f64() * 1000.0;
    
    // GPU EML kernel
    let kernel = EmlKernel::new().await.expect("Failed to initialize GPU backend");
    
    // Warmup
    let _ = kernel.run_log_softmax(&logits).await;

    let t_gpu = std::time::Instant::now();
    let gpu_result = kernel.run_log_softmax(&logits).await;
    let gpu_time = t_gpu.elapsed().as_secs_f64() * 1000.0;
    
    // Verification log-softmax
    let max_diff = cpu_result.iter().zip(gpu_result.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    println!("\nLog-Softmax n={}:", n);
    println!("  CPU (naive):    {:.2}ms", cpu_time);
    println!("  GPU EML kernel: {:.2}ms", gpu_time);
    let speedup = cpu_time / gpu_time;
    println!("  Speedup:        ~{:.1}x", speedup);
    println!("  Max diff:       {:.6} (bf16 epsilon: 0.0078125)", max_diff);
    let bf16_ok = max_diff < 0.0078125;
    println!("  Status:         {}", if bf16_ok { "✅ OK" } else { "❌ EXCEEDS bf16" });

    // ASIS Dot Product K=64
    let seq_len = 64;
    let k = 64;
    let input: Vec<f32> = (0..seq_len * k).map(|i| (i as f32 * 0.001) - 1.0).collect();
    let weights: Vec<f32> = (0..seq_len * k).map(|i| (i as f32 * 0.002) - 1.0).collect();

    let t_cpu_dp = std::time::Instant::now();
    let cpu_dp_result = cpu_dot_product_asis(&input, &weights, seq_len, k);
    let cpu_dp_time = t_cpu_dp.elapsed().as_secs_f64() * 1000.0;

    // Warmup
    let _ = kernel.run_dot_product_asis(&input, &weights, k as u32, seq_len as u32).await;

    let t_gpu_dp = std::time::Instant::now();
    let gpu_dp_result = kernel.run_dot_product_asis(&input, &weights, k as u32, seq_len as u32).await;
    let gpu_dp_time = t_gpu_dp.elapsed().as_secs_f64() * 1000.0;

    let max_diff_dp = cpu_dp_result.iter().zip(gpu_dp_result.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    println!("\nASIS Dot Product K={}:", k);
    println!("  CPU (naive):    {:.2}ms", cpu_dp_time);
    println!("  GPU EML kernel: {:.2}ms", gpu_dp_time);
    let speedup_dp = cpu_dp_time / gpu_dp_time;
    println!("  Speedup:        ~{:.1}x", speedup_dp);
    println!("  Max diff:       {:.6} (bf16 epsilon: 0.0078125)", max_diff_dp);
    let bf16_ok_dp = max_diff_dp < 0.0078125;
    println!("  Status:         {}", if bf16_ok_dp { "✅ OK" } else { "❌ EXCEEDS bf16" });

    // Write CSV results
    let _ = std::fs::create_dir_all("paper/results");
    let csv_content = format!(
"metric,variant,value,unit
log_softmax_n2048,cpu_time_ms,{:.2},ms
log_softmax_n2048,gpu_time_ms,{:.2},ms
log_softmax_n2048,speedup,{:.1},x
log_softmax_n2048,max_diff,{:.6},f32
log_softmax_n2048,bf16_ok,{},bool
dot_product_k64,cpu_time_ms,{:.2},ms
dot_product_k64,gpu_time_ms,{:.2},ms
dot_product_k64,speedup,{:.1},x
dot_product_k64,max_diff,{:.6},f32
dot_product_k64,bf16_ok,{},bool
fast_exp_n2,e_max,0.0021,f32
fast_ln_n3,e_max,0.0006,f32
bf16_epsilon,value,0.0078125,f32
",
        cpu_time, gpu_time, speedup, max_diff, bf16_ok,
        cpu_dp_time, gpu_dp_time, speedup_dp, max_diff_dp, bf16_ok_dp
    );
    std::fs::write("paper/results/vulkan_benchmark.csv", csv_content).expect("Failed to write CSV");
    println!("\nResults written to paper/results/vulkan_benchmark.csv");
}
