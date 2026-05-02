use eml_trs::ast::*;
use eml_trs::nn_layer::build_dot_product_eml;
use eml_trs::round_trip::{compile_to_ops, FlatProgram};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
struct LayerData {
    hidden_dim: usize,
    weights: std::collections::HashMap<String, Vec<Vec<f32>>>,
}

fn main() {
    println!("=== EMLM ROBUST PARITY TEST (64 ROWS × 4 SEEDS) ===");
    println!("Goal: Verify numerical fidelity across various weights and input ranges.\n");

    // 1. Load weights
    let file = File::open("models/layer0_full.json").expect("Missing layer0_full.json");
    let reader = BufReader::new(file);
    let data: LayerData = serde_json::from_reader(reader).expect("Failed to parse weights");
    
    let hidden = data.hidden_dim;
    let x_vars: Vec<Arc<EmlNode>> = (0..hidden).map(|i| var(&format!("x{}", i))).collect();
    
    let matrix_name = "q";
    let rows_to_test = 64;
    let seeds = [0.01f32, 0.1, 1.5, -0.5];
    
    let mut total_tests = 0;
    let mut total_success = 0;
    let mut max_global_diff = 0.0f32;

    for row_idx in 0..rows_to_test {
        let weights = &data.weights[matrix_name][row_idx];
        
        for &seed in &seeds {
            total_tests += 1;
            
            // Generate input for this seed
            let input_f32: Vec<f32> = (0..hidden).map(|i| (i as f32 * seed).sin()).collect();
            
            // ALU Reference
            let mut alu_sum = 0.0f32;
            for i in 0..hidden {
                alu_sum += input_f32[i] * weights[i];
            }

            // EML Execution
            let tree = build_dot_product_eml(&x_vars, weights);
            let program = compile_to_ops(tree);
            
            let mut vars = HashMap::new();
            for i in 0..hidden {
                vars.insert(format!("x{}", i), input_f32[i] as f64);
            }
            
            let eml_result = program.execute(&vars).expect("Execution failed") as f32;
            let diff = (eml_result - alu_sum).abs();
            max_global_diff = max_global_diff.max(diff);

            if diff < 1e-3 {
                total_success += 1;
            } else {
                println!("[FAIL] Row {}, Seed {}: EML={}, ALU={}, Diff={:.2e}", 
                    row_idx, seed, eml_result, alu_sum, diff);
            }
        }
        
        if row_idx > 0 && row_idx % 8 == 0 {
            println!("  Progress: Row {}/{}...", row_idx, rows_to_test);
        }
    }

    println!("\n=== FINAL RESULTS ===");
    println!("Total Tests:    {}", total_tests);
    println!("Success:        {}", total_success);
    println!("Max Diff:       {:.10e}", max_global_diff);
    
    if total_success == total_tests {
        println!("STATUS:         ✅ ALL TESTS PASSED");
    } else {
        println!("STATUS:         ❌ SOME TESTS FAILED ({} failure(s))", total_tests - total_success);
        std::process::exit(1);
    }
}
