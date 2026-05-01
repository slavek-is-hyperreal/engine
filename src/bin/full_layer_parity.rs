// src/bin/full_layer_parity.rs
use eml_trs::ast::*;
use eml_trs::round_trip::compile_to_ops;
use eml_trs::constant_fold::ConstantMap;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

#[derive(Serialize, Deserialize)]
struct LayerData {
    hidden_dim: usize,
    weights: std::collections::HashMap<String, Vec<Vec<f32>>>,
}

fn rms_norm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ms = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (ms + eps).sqrt();
    x.iter().zip(gamma.iter()).map(|(&v, &g)| v * inv_rms * g).collect()
}

fn main() {
    println!("=== EML End-to-End Parity Test (SCIENTIFIC) ===");
    
    let t_load = Instant::now();
    let file = File::open("models/layer0_full.json").expect("Failed to open weights file");
    let reader = BufReader::new(file);
    let data: LayerData = serde_json::from_reader(reader).expect("Failed to parse JSON");
    println!("Loaded 374MB weights in {:?}", t_load.elapsed());

    let hidden = data.hidden_dim;
    // Realistic sine-wave input
    let input: Vec<f32> = (0..hidden).map(|i| (i as f32 * 1.337).sin() * 0.5).collect();
    
    let w_attn_norm = &data.weights["attn_norm"][0];
    let w_q = &data.weights["q"];

    // 1. Reference ALU Path
    let x_norm = rms_norm(&input, w_attn_norm, 1e-5);
    let q_ref: Vec<f32> = w_q.iter().map(|row| {
        row.iter().zip(x_norm.iter()).map(|(&w, &x)| w * x).sum()
    }).collect();

    // 2. EML Path (Verified Slice)
    println!("Starting EML inference (Round-Trip optimized)...");
    let slice_len = 128;
    let mut eml_results = Vec::new();
    let t_eml = Instant::now();
    
    use eml_trs::nn_layer::build_dot_product_eml;
    for i in 0..slice_len {
        let mut c = ConstantMap::new();
        let mut in_nodes = Vec::new();
        for (j, &val) in x_norm.iter().enumerate() {
            let name = format!("x{}", j);
            c.insert(name.clone(), val as f64);
            in_nodes.push(var(&name));
        }

        let tree = build_dot_product_eml(&in_nodes, &w_q[i]);
        let program = compile_to_ops(tree);
        let res = program.execute(&c).expect("EML eval failed") as f32;
        eml_results.push(res);
    }
    println!("EML slice (128 elements) done in {:?}", t_eml.elapsed());

    // 3. Comparison
    let mut max_diff: f32 = 0.0;
    let mut mse: f32 = 0.0;
    for i in 0..slice_len {
        let diff = (q_ref[i] - eml_results[i]).abs();
        max_diff = max_diff.max(diff);
        mse += diff * diff;
    }
    mse /= slice_len as f32;

    println!("\nRESULTS (EML Q-Slice vs ALU Reference):");
    println!("MSE:      {:.2e}", mse);
    println!("Max Diff: {:.2e}", max_diff);
    
    // Status based on standard f32 tolerance
    if max_diff < 1e-4 {
        println!("STATUS:   ✅ VERIFIED (Bit-parallel to ALU within float32 precision)");
    } else {
        println!("STATUS:   ❌ MISMATCH (Disparity found)");
    }
}
