// src/bin/eml_parity.rs
//
// Parity verification binary for EML dot product.
// Reads JSON from stdin: {"inputs": [...], "weights": [...]}
// Writes JSON to stdout: {"eml_result": f64} or {"eml_result": null, "nan_reason": "..."}
//
// Used by scripts/verify_parity.py

use eml_trs::ast::*;
use eml_trs::constant_fold::{try_evaluate, ConstantMap};
use eml_trs::nn_layer::build_dot_product_eml;
use std::io::{self, Read};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("Failed to read stdin");

    // Parse input JSON manually (no serde dependency in core)
    let (inputs, weights) = parse_input(&input).unwrap_or_else(|e| {
        eprintln!("Parse error: {}", e);
        std::process::exit(1);
    });

    let k = inputs.len();
    if k == 0 || k != weights.len() {
        eprintln!("Invalid input: K={}, weights={}", k, weights.len());
        std::process::exit(1);
    }

    // Build EML tree
    let input_nodes: Vec<_> = (0..k).map(|i| var(&format!("x{}", i))).collect();
    let weights_f32: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
    let tree = build_dot_product_eml(&input_nodes, &weights_f32);

    // Optimize and compile (Round-Trip TRS is key for real-only stability)
    use eml_trs::round_trip::compile_to_ops;
    let program = compile_to_ops(tree);

    // Build constant map
    let mut consts = ConstantMap::new();
    for (i, &v) in inputs.iter().enumerate() {
        consts.insert(format!("x{}", i), v);
    }

    // Evaluate on optimized program
    match program.execute(&consts) {
        Some(result) if result.is_finite() => {
            println!("{{\"eml_result\": {:.15e}}}", result);
        }
        Some(result) => {
            // NaN or Inf
            println!(
                "{{\"eml_result\": null, \"nan_reason\": \"non-finite: {:.6e}\"}}",
                result
            );
        }
        None => {
            // execute returned None — still a failure, but less likely after TRS
            println!(
                "{{\"eml_result\": null, \"nan_reason\": \"domain error: execution returned None\"}}"
            );
        }
    }
}


fn parse_input(s: &str) -> Result<(Vec<f64>, Vec<f64>), String> {
    // Minimal JSON array parser — no external deps
    // Expects: {"inputs": [f64,...], "weights": [f64,...]}
    let s = s.trim();
    let inputs = extract_array(s, "inputs")?;
    let weights = extract_array(s, "weights")?;
    Ok((inputs, weights))
}

fn extract_array(s: &str, key: &str) -> Result<Vec<f64>, String> {
    let search = format!("\"{}\"", key);
    let start = s.find(&search)
        .ok_or_else(|| format!("Key '{}' not found", key))?;
    let rest = &s[start + search.len()..];
    let colon = rest.find(':')
        .ok_or_else(|| format!("No colon after '{}'", key))?;
    let rest = rest[colon + 1..].trim_start();
    let arr_start = rest.find('[')
        .ok_or_else(|| format!("No '[' for '{}'", key))?;
    let arr_end = rest.find(']')
        .ok_or_else(|| format!("No ']' for '{}'", key))?;
    let arr_str = &rest[arr_start + 1..arr_end];
    arr_str.split(',')
        .map(|x| x.trim().parse::<f64>()
            .map_err(|e| format!("Parse error for '{}': {}", key, e)))
        .collect()
}
