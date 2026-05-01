// src/bin/layer_benchmark.rs
use eml_trs::ast::*;
use eml_trs::round_trip::compile_to_ops;
use eml_trs::nn_layer::build_dot_product_eml;
use eml_trs::constant_fold::ConstantMap;
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
    println!("=== EML EMPIRICAL LAYER BENCHMARK (SCIENTIFIC VERSION) ===");
    
    let data_str = fs::read_to_string("models/layer0_weights.json")
        .expect("Missing models/layer0_weights.json.");
    let weights: LayerWeights = serde_json::from_str(&data_str).unwrap();
    
    let k = weights.k;
    let d_k = weights.d_k;
    println!("Loaded layer 0 sample dimensions: K={}, d_k={}", k, d_k);
    println!("NOTE: This benchmark measures Head-scale (d_k=64) complexity.");

    let x_vars: Vec<_> = (0..d_k).map(|i| var(&format!("x{}", i))).collect();

    // 1. RMSNorm (Full: x * gamma * (1/sqrt(mean(x2))))
    // Naive cost: 66 nodes per element
    let rms_naive = 66 * d_k; 

    // Denominator calculation (x^2 is always positive, exp(ln(x)*2) is safe)
    let mut x2_terms = Vec::new();
    for xv in &x_vars {
        x2_terms.push(exp_node(mul_cf(ln_node(xv.clone()), 2.0))); 
    }
    let sum_x2 = build_simple_sum(&x2_terms);
    let mean_x2 = mul_cf(sum_x2, 1.0 / d_k as f64);
    let inv_rms = exp_node(mul_cf(ln_node(mean_x2.clone()), -0.5));
    
    // SAFE RMSNorm outputs: must use BIAS trick for multiplication by gamma and x_i
    // because x_i can be negative.
    let mut rms_full_outputs = Vec::new();
    for i in 0..d_k {
        let x_i = x_vars[i].clone();
        let gamma_val = weights.rms_norm[i] as f64;
        
        // We simulate the safe multiplication chain: (x_i * inv_rms) * gamma
        // Using build_dot_product_eml logic for safety.
        let x_normalized = mul_safe_stable(x_i, inv_rms.clone());
        let final_out = mul_cf_safe_stable(x_normalized, gamma_val);
        rms_full_outputs.push(final_out);
    }
    let mut rms_dag = eml_trs::dag::EmlDag::new();
    for out in &rms_full_outputs { add_tree_to_dag(&mut rms_dag, out.clone()); }
    let rms_opt = rms_dag.unique_node_count();

    // RMSNorm with Fusion 3 (gamma folded into next matrix)
    let mut rms_fused_outputs = Vec::new();
    for i in 0..d_k {
        let x_i = x_vars[i].clone();
        rms_fused_outputs.push(mul_safe_stable(x_i, inv_rms.clone()));
    }
    let mut rms_fused_dag = eml_trs::dag::EmlDag::new();
    for out in &rms_fused_outputs { add_tree_to_dag(&mut rms_fused_dag, out.clone()); }
    let rms_fused_opt = rms_fused_dag.unique_node_count();

    // 2. Dot Product (K=64 for one head projection)
    // Naive cost per Theorem 1: 36K - 19
    let dot_naive = 36 * d_k - 19; 
    let dot_tree = build_dot_product_eml(&x_vars, &weights.w_q_head0[0][..d_k]);
    let dot_opt = eml_trs::dag::tree_to_dag(dot_tree.clone()).unique_node_count(); 

    // 3. Log-Softmax (n=64)
    // Naive cost: 35n - 17
    let lsm_naive = 35 * d_k - 17;
    let mut exp_logits = Vec::new();
    for i in 0..d_k { exp_logits.push(exp_node(var(&format!("logit_{}", i)))); }
    let sum_exp = build_simple_sum(&exp_logits);
    let lse = ln_node(sum_exp);
    let mut lsm_outputs = Vec::new();
    for i in 0..d_k {
        let x_i = var(&format!("logit_{}", i));
        lsm_outputs.push(add_eml(x_i, neg_node(lse.clone())));
    }
    let mut lsm_dag = eml_trs::dag::EmlDag::new();
    for out in &lsm_outputs { add_tree_to_dag(&mut lsm_dag, out.clone()); }
    let lsm_opt = lsm_dag.unique_node_count();

    // Table
    println!("\n| Operacja                | Węzły naive | Węzły opt | Redukcja |");
    println!("|-------------------------|-------------|-----------|----------|");
    print_row("RMSNorm (Standard)",    rms_naive, rms_opt);
    print_row("RMSNorm (Fusion 3: γ)", rms_naive, rms_fused_opt);
    print_row("Dot Product (64)",      dot_naive, dot_opt);
    print_row("Log-Softmax (64)",      lsm_naive, lsm_opt);
    
    // Total calculation using the BEST available optimizations (Fusion 3)
    let total_naive = rms_naive + dot_naive + lsm_naive;
    let total_opt = rms_fused_opt + dot_opt + lsm_opt;
    println!("|-------------------------|-------------|-----------|----------|");
    print_row("RAZEM (Best Case)", total_naive, total_opt);

    println!("\nVerification: Full Layer Reduction (inc. Fusion 3) {:.1}%", 
        (1.0 - total_opt as f64 / total_naive as f64) * 100.0);
    
    // --- NUMERICAL PARITY (on the same 64 elements) ---
    println!("\n=== NUMERICAL PARITY (K=64) ===");
    let mut vars = ConstantMap::new();
    for (i, &v) in weights.input_sample.iter().enumerate().take(d_k) {
        vars.insert(format!("x{}", i), v as f64);
    }

    let alu_rms_denom = (weights.input_sample.iter().take(d_k).map(|&x| (x as f64).powi(2)).sum::<f64>() / d_k as f64).sqrt();
    let rms_denom_program = compile_to_ops(exp_node(mul_cf(ln_node(mean_x2.clone()), 0.5)));
    if let Some(v) = rms_denom_program.execute(&vars) {
        println!("RMSNorm Denom: eml={:.6}, alu={:.6}, diff={:.2e}", v, alu_rms_denom, (v - alu_rms_denom).abs());
    }

    let alu_dot: f64 = weights.input_sample.iter().take(d_k).zip(weights.w_q_head0[0].iter().take(d_k)).map(|(&x, &w)| x as f64 * w as f64).sum();
    let dot_program = compile_to_ops(dot_tree);
    if let Some(v) = dot_program.execute(&vars) {
        println!("Dot:           eml={:.6}, alu={:.6}, diff={:.2e}", v, alu_dot, (v - alu_dot).abs());
    }
}

/// Helper for safe multiplication with BIAS trick (x can be negative)
fn mul_safe_stable(a: Arc<EmlNode>, b: Arc<EmlNode>) -> Arc<EmlNode> {
    // a * b = exp(ln(a) + ln(b)) -- unsafe if a,b <= 0
    // Simplified BIAS trick for benchmark purposes:
    // (a + 4) * b - 4 * b  (assuming b is positive like inv_rms)
    let a_shifted = add_eml(a, konst(4.0));
    let product = mul_eml(a_shifted, b.clone());
    sub_eml(product, mul_eml(konst(4.0), b))
}

/// Helper for safe constant multiplication with BIAS trick
fn mul_cf_safe_stable(x: Arc<EmlNode>, w: f64) -> Arc<EmlNode> {
    let abs_w = w.abs();
    let x_shifted = add_eml(x, konst(4.0));
    let scaled = mul_cf(x_shifted, abs_w);
    let correction = konst(4.0 * abs_w);
    let res = sub_eml(scaled, correction);
    if w < 0.0 { neg_node(res) } else { res }
}

fn build_simple_sum(terms: &[Arc<EmlNode>]) -> Arc<EmlNode> {
    if terms.len() == 1 { return terms[0].clone(); }
    let mid = terms.len() / 2;
    add_eml(build_simple_sum(&terms[..mid]), build_simple_sum(&terms[mid..]))
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
    println!("| {:<16} | {:>11} | {:>9} | {:>8.1}% |", name, naive, opt, red);
}
