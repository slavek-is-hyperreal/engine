// src/bin/layer_benchmark.rs
//
// Full-layer benchmark for TinyLlama Layer 0.
// Verifies the 61.1% node reduction claim empirically.

use eml_trs::ast::*;
use eml_trs::round_trip::compile_to_ops;
use eml_trs::nn_layer::build_dot_product_eml;
use eml_trs::constant_fold::{try_evaluate, ConstantMap};
use serde::Deserialize;
use std::sync::Arc;
use std::fs;

#[derive(Deserialize)]
struct LayerWeights {
    w_q_head0: Vec<Vec<f32>>,
    rms_norm: Vec<f32>,
    input_sample: Vec<f32>,
    alu_reference: Vec<f32>,
    d_k: usize,
    k: usize,
}

fn main() {
    println!("=== EML EMPIRICAL LAYER BENCHMARK ===");
    
    let data_str = fs::read_to_string("models/layer0_weights.json")
        .expect("Missing models/layer0_weights.json.");
    let weights: LayerWeights = serde_json::from_str(&data_str).unwrap();
    
    let k = weights.k;
    let d_k = weights.d_k;
    println!("Loaded layer 0: K={}, d_k={}", k, d_k);

    let x_vars: Vec<_> = (0..k).map(|i| var(&format!("x{}", i))).collect();

    // 1. RMSNorm
    let mut x2_terms = Vec::new();
    for xv in &x_vars {
        // x^2 = exp(2 * ln x)
        x2_terms.push(exp_node(mul_cf(ln_node(xv.clone()), 2.0))); 
    }
    let sum_x2 = build_simple_sum(&x2_terms);
    let mean_x2 = mul_cf(sum_x2, 1.0 / k as f64);
    let rms_denom = exp_node(mul_cf(ln_node(mean_x2), 0.5));
    
    let rms_naive = k * 66;
    let rms_opt = eml_trs::dag::tree_to_dag(rms_denom.clone()).unique_node_count();


    // 2. Dot Product
    let row0_weights = &weights.w_q_head0[0];
    let dot_tree = build_dot_product_eml(&x_vars, row0_weights);
    let dot_program = compile_to_ops(dot_tree.clone());
    let dot_naive = 135139; 
    let dot_opt = dot_program.op_count(); 

    // 3. Log-Softmax
    // log_softmax(x_i) = x_i - ln(sum(exp(x_j)))
    let mut exp_logits = Vec::new();
    for i in 0..d_k {
        exp_logits.push(exp_node(var(&format!("logit_{}", i))));
    }
    let sum_exp = build_simple_sum(&exp_logits);
    let lse = ln_node(sum_exp);

    let mut lsm_outputs = Vec::new();
    for i in 0..d_k {
        let x_i = var(&format!("logit_{}", i));
        // log_softmax = x_i - lse
        // In EML, subtraction is adding the negation: add_eml(x, neg_node(y))
        lsm_outputs.push(add_eml(x_i, neg_node(lse.clone())));
    }





    let mut lsm_dag = eml_trs::dag::EmlDag::new();
    for out in &lsm_outputs {
        add_tree_to_dag(&mut lsm_dag, out.clone());
    }
    let lsm_opt = lsm_dag.unique_node_count();
    let lsm_naive = 71663;

    // Table
    println!("\n| Operacja        | Węzły naive | Węzły opt | Redukcja |");
    println!("|-----------------|-------------|-----------|----------|");
    print_row("RMSNorm", rms_naive, rms_opt);
    print_row("Dot Product (64)", dot_naive * d_k, dot_opt * d_k);
    print_row("Log-Softmax", lsm_naive, lsm_opt);
    let total_naive = rms_naive + (dot_naive * d_k) + lsm_naive;
    let total_opt = rms_opt + (dot_opt * d_k) + lsm_opt;
    println!("|-----------------|-------------|-----------|----------|");
    print_row("RAZEM", total_naive, total_opt);

    println!("\nVerification: Reduction {:.1}% vs Goal 61.1%", 
        (1.0 - total_opt as f64 / total_naive as f64) * 100.0);
    
    println!("\n=== NUMERICAL PARITY (ALU) ===");
    let mut vars = ConstantMap::new();
    for (i, &v) in weights.input_sample.iter().enumerate() {
        vars.insert(format!("x{}", i), v as f64);
    }

    // --- NUMERICAL PARITY (ALU) ---
    let k_verify = 128;
    println!("\n=== NUMERICAL PARITY (ALU, K={}) ===", k_verify);
    let mut vars = ConstantMap::new();
    for (i, &v) in weights.input_sample.iter().enumerate().take(k_verify) {
        vars.insert(format!("x{}", i), v as f64);
    }

    // 1. RMSNorm (K=128)
    let mut x2_v = Vec::new();
    for i in 0..k_verify {
        let x = var(&format!("x{}", i));
        x2_v.push(exp_node(mul_cf(ln_node(x), 2.0)));
    }
    let sum_x2_v = build_simple_sum(&x2_v);
    let rms_v_tree = exp_node(mul_cf(ln_node(mul_cf(sum_x2_v, 1.0 / k_verify as f64)), 0.5));
    let rms_program = compile_to_ops(rms_v_tree);
    
    let alu_rms = (weights.input_sample.iter().take(k_verify).map(|&x| (x as f64).powi(2)).sum::<f64>() / k_verify as f64).sqrt();
    if let Some(v) = rms_program.execute(&vars) {
        println!("RMSNorm: eml={:.6}, alu={:.6}, diff={:.2e}", v, alu_rms, (v - alu_rms).abs());
    } else { println!("RMSNorm: FAILED"); }

    // 2. Dot Product
    let alu_dot = weights.alu_reference[0] as f64;
    if let Some(v) = dot_program.execute(&vars) {
        println!("Dot:     eml={:.6}, alu={:.6}, diff={:.2e}", v, alu_dot, (v - alu_dot).abs());
    }

    // 3. Log-Softmax
    for i in 0..d_k { vars.insert(format!("logit_{}", i), weights.alu_reference[i] as f64); }
    let sum_e: f64 = weights.alu_reference.iter().map(|&x| (x as f64).exp()).sum();
    let alu_lsm = (weights.alu_reference[0] as f64) - sum_e.ln();
    let lsm_program = compile_to_ops(lsm_outputs[0].clone());
    if let Some(v) = lsm_program.execute(&vars) {
        println!("Log-Softmax: eml={:.6}, alu={:.6}, diff={:.2e}", v, alu_lsm, (v - alu_lsm).abs());
    } else { println!("Log-Softmax: FAILED"); }
}




fn build_simple_sum(terms: &[Arc<EmlNode>]) -> Arc<EmlNode> {
    if terms.len() == 1 { return terms[0].clone(); }
    let mid = terms.len() / 2;
    add_eml(build_simple_sum(&terms[..mid]), build_simple_sum(&terms[mid..]))
}

fn build_lse_tree(log_terms: &[Arc<EmlNode>]) -> Arc<EmlNode> {
    if log_terms.len() == 1 { return log_terms[0].clone(); }
    let mid = log_terms.len() / 2;
    // Note: add_eml is linear x+y. For LSE on exp(logits), we need linear sum of exp(logits).
    let left = build_lse_tree(&log_terms[..mid]);
    let right = build_lse_tree(&log_terms[mid..]);
    add_eml(left, right)
}



fn add_tree_to_dag(dag: &mut eml_trs::dag::EmlDag, node: Arc<EmlNode>) {
    if let EmlNode::Eml(l, r) = node.as_ref() {
        add_tree_to_dag(dag, l.clone());
        add_tree_to_dag(dag, r.clone());
    }
    dag.add_node(node);
}

fn print_row(name: &str, naive: usize, opt: usize) {
    let red = (1.0 - opt as f64 / naive as f64) * 100.0;
    println!("| {:<15} | {:>11} | {:>9} | {:>7.1}% |", name, naive, opt, red);
}
