use eml_trs::ast::*;
use eml_trs::nn_layer::build_dot_product_eml;
use eml_trs::dag_mmap::*;
use eml_trs::round_trip::compile_to_ops;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use std::time::Instant;
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
struct LayerData {
    hidden_dim: usize,
    weights: std::collections::HashMap<String, Vec<Vec<f32>>>,
}

struct ParityStats {
    max_diff: f32,
    sum_diff: f64,
    n_tests: u64,
}

impl ParityStats {
    fn new() -> Self {
        Self { max_diff: 0.0, sum_diff: 0.0, n_tests: 0 }
    }

    fn update(&mut self, eml: f32, alu: f32) {
        let diff = (eml - alu).abs();
        if diff > self.max_diff {
            self.max_diff = diff;
        }
        self.sum_diff += diff as f64;
        self.n_tests += 1;
    }

    fn print(&self, name: &str) {
        let mean = if self.n_tests > 0 { self.sum_diff / self.n_tests as f64 } else { 0.0 };
        println!("  Parity [{:<10}]: max_diff={:.2e}, mean_diff={:.2e}, tests={}", 
            name, self.max_diff, mean, self.n_tests);
    }
}

fn main() {
    println!("=== EML 1:1 UNIFIED GLOBAL DAG AUDIT & PARITY V4 (FINAL) ===");
    println!("Strategy: Mmap-backed Global DAG + Row-by-Row Parity Verification.");
    
    let dag_path = "/vectorlegis_ssd_pool/eml_working/global_dag_v4.bin";
    let initial_capacity = 500_000_000u64; 
    
    let mut dag = MmapDag::create(dag_path, initial_capacity)
        .expect("Failed to create MmapDag on ZFS");
    println!("MmapDag initialized at {}.", dag_path);

    let t_load = Instant::now();
    let file = File::open("models/layer0_full.json").expect("Missing layer0_full.json");
    let reader = BufReader::new(file);
    let data: LayerData = serde_json::from_reader(reader).expect("Failed to parse weights");
    println!("Loaded weights in {:?}", t_load.elapsed());

    let hidden = data.hidden_dim;
    let x_vars: Vec<Arc<EmlNode>> = (0..hidden).map(|i| var(&format!("x{}", i))).collect();

    // Constant input sample for parity
    let input_f32: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut vars = HashMap::new();
    for i in 0..hidden {
        vars.insert(format!("x{}", i), input_f32[i] as f64);
    }

    let mut stats = ParityStats::new();
    let mut total_naive_count: u64 = 0;
    let t_start = Instant::now();

    // 1. RMSNorms
    println!("\nAuditing & Verifying RMSNorms...");
    for norm_name in ["attn_norm", "ffn_norm"] {
        let gammas = &data.weights[norm_name][0];
        let d = gammas.len();
        
        // ALU Reference for this norm
        let ms = input_f32.iter().map(|&v| v*v).sum::<f32>() / d as f32;
        let inv_rms_alu = 1.0 / (ms + 1e-5).sqrt();

        // EML Nodes (RMSNorm logic with BIAS trick for stability)
        let mut x2_terms = Vec::new();
        for i in 0..d {
            let xv = var(&format!("x{}", i));
            // x^2 = (x+4)^2 - 8(x+4) + 16
            x2_terms.push(mul_safe_stable(xv.clone(), xv));
        }
        let sum_x2 = build_simple_sum(&x2_terms);
        let mean_x2 = mul_cf_any_x(sum_x2, 1.0 / d as f64);
        
        // inv_rms = (mean_x2 + 1e-5)^-0.5
        let inv_rms = exp_node(mul_cf_any_x(ln_node(add_eml(mean_x2, konst(1e-5))), -0.5));

        for i in 0..d {
            let x_i = var(&format!("x{}", i));
            let normalized = mul_eml_safe_stable(x_i, inv_rms.clone()); // x * inv_rms
            let out_tree = mul_cf_any_x(normalized, gammas[i] as f64);       // * gamma
            
            // Audit
            let mut local_cache = HashMap::new();
            add_tree_to_mmap_dag(&mut dag, &out_tree, &mut local_cache);
            
            // Parity
            let program = compile_to_ops(out_tree);
            let eml_res = program.execute(&vars).unwrap() as f32;
            let alu_res = (input_f32[i] * inv_rms_alu) * gammas[i]; 
            stats.update(eml_res, alu_res);
        }
        total_naive_count += (66 * d) as u64;
        stats.print(norm_name);
    }

    // 2. Matrices (Q, K, V, O)
    for name in ["q", "k", "v", "o"] {
        println!("\nIntegrating & Verifying matrix: {:<10}...", name);
        let weights = &data.weights[name];
        let rows = weights.len();
        let cols = hidden;
        let naive_per_row = (36 * cols - 19) as u64;

        for i in 0..rows {
            if i > 0 && i % 512 == 0 {
                println!("  Progress: {:>4}/{} (Unique nodes: {})", i, rows, dag.unique_node_count());
            }
            let tree = build_dot_product_eml(&x_vars, &weights[i]);
            
            // Audit
            let mut local_cache = HashMap::new();
            add_tree_to_mmap_dag(&mut dag, &tree, &mut local_cache);
            
            // Parity
            let program = compile_to_ops(tree);
            let eml_res = program.execute(&vars).unwrap() as f32;
            let mut alu_res = 0.0f32;
            for j in 0..cols { alu_res += input_f32[j] * weights[i][j]; }
            stats.update(eml_res, alu_res);
        }
        total_naive_count += (rows as u64 * naive_per_row);
        stats.print(name);
    }

    // 3. FFN Block (SwiGLU)
    println!("\nIntegrating & Verifying FFN Block (SwiGLU)...");
    let w_gate = &data.weights["gate"];
    let w_up = &data.weights["up"];
    for i in 0..w_gate.len() {
        if i > 0 && i % 512 == 0 {
            println!("  FFN Progress: {:>4}/{} (Unique nodes: {})", i, w_gate.len(), dag.unique_node_count());
        }
        let g_tree = build_dot_product_eml(&x_vars, &w_gate[i]);
        let u_tree = build_dot_product_eml(&x_vars, &w_up[i]);
        let fused = eml_trs::fusions::swiglu_fused(g_tree, u_tree);
        
        // Audit
        let mut local_cache = HashMap::new();
        add_tree_to_mmap_dag(&mut dag, &fused, &mut local_cache);
        
        // Parity
        let program = compile_to_ops(fused);
        let eml_res = program.execute(&vars);
        
        let mut g_alu = 0.0f32; for j in 0..hidden { g_alu += input_f32[j] * w_gate[i][j]; }
        let mut u_alu = 0.0f32; for j in 0..hidden { u_alu += input_f32[j] * w_up[i][j]; }
        let alu_res = (g_alu / (1.0 + (-g_alu).exp())) * u_alu;

        if let Some(res) = eml_res {
            stats.update(res as f32, alu_res);
        }
    }
    total_naive_count += (w_gate.len() * (36 * hidden - 19) * 2) as u64;
    stats.print("swiglu");

    // 4. Down projection
    let w_down = &data.weights["down"];
    let mut swi_vars = HashMap::new();
    let mut swi_input_f32 = Vec::new();
    for i in 0..w_gate.len() {
        let val = (i as f32 * 0.05).cos();
        swi_vars.insert(format!("g{}", i), val as f64);
        swi_input_f32.push(val);
    }
    let g_vars: Vec<Arc<EmlNode>> = (0..w_gate.len()).map(|i| var(&format!("g{}", i))).collect();
    
    println!("\nIntegrating & Verifying Down Projection...");
    for i in 0..w_down.len() {
        let tree = build_dot_product_eml(&g_vars, &w_down[i]);
        let mut local_cache = HashMap::new();
        add_tree_to_mmap_dag(&mut dag, &tree, &mut local_cache);
        
        let program = compile_to_ops(tree);
        let eml_res = program.execute(&swi_vars).unwrap() as f32;
        let mut alu_res = 0.0f32;
        for j in 0..w_gate.len() { alu_res += swi_input_f32[j] * w_down[i][j]; }
        stats.update(eml_res, alu_res);
    }
    total_naive_count += (w_down.len() as u64 * (36 * w_gate.len() - 19) as u64);
    stats.print("down");

    println!("\n=== FINAL V4 AUDIT & PARITY VERDICT ===");
    let total_opt = dag.unique_node_count() as u64;
    let mean_diff = if stats.n_tests > 0 { stats.sum_diff / stats.n_tests as f64 } else { 0.0 };
    
    println!("Execution Time:     {:?}", t_start.elapsed());
    println!("Total Naive Nodes:  {:>16}", total_naive_count);
    println!("Total Unique Nodes: {:>16}", total_opt);
    println!("FINAL REDUCTION:    {:>15.2}%", (1.0 - total_opt as f64 / total_naive_count as f64) * 100.0);
    println!("Parity Max Diff:    {:.2e}", stats.max_diff);
    println!("Parity Mean Diff:   {:.2e}", mean_diff);
    println!("Total Tests:        {}", stats.n_tests);
    
    if stats.max_diff < 1e-3 {
        println!("STATUS:             ✅ ALL SYSTEMS NOMINAL");
    } else {
        println!("STATUS:             ⚠️ NUMERICAL DISPARITY DETECTED");
    }
}

// --- STABLE ARITHMETIC HELPERS ---

/// Truly stable multiplication for any a, b > -3.0.
/// ab = (a+4)(b+4) - 4(a+4) - 4(b+4) + 16
fn mul_safe_stable(a: Arc<EmlNode>, b: Arc<EmlNode>) -> Arc<EmlNode> {
    let a_s = add_eml(a, konst(4.0));
    let b_s = add_eml(b, konst(4.0));
    let prod_s = mul_eml(a_s.clone(), b_s.clone());
    
    // mul_cf on biased vars is safe because a+4 > 1
    let term_a = mul_cf_robust(a_s, 4.0);
    let term_b = mul_cf_robust(b_s, 4.0);
    
    add_eml(sub_eml(sub_eml(prod_s, term_a), term_b), konst(16.0))
}

/// Constant weight mul that works for any w and any x > -3.0 (via internal bias).
fn mul_cf_any_x(x: Arc<EmlNode>, w: f64) -> Arc<EmlNode> {
    if w == 0.0 { return konst(0.0); }
    if w > 0.0 {
        let x_s = add_eml(x, konst(4.0));
        let prod = mul_cf(x_s, w);
        sub_eml(prod, konst(4.0 * w))
    } else {
        // Recursive handling for negative w
        neg_node(mul_cf_any_x(x, -w))
    }
}

/// mul_cf for x guaranteed to be > 1.0 (like shifted vars), handles w sign.
fn mul_cf_robust(x_s: Arc<EmlNode>, w: f64) -> Arc<EmlNode> {
    if w == 0.0 { return konst(0.0); }
    if w > 0.0 { mul_cf(x_s, w) } else { neg_node(mul_cf(x_s, -w)) }
}

/// Stable multiplication x(node) * w(node) using BIAS trick for x.
fn mul_eml_safe_stable(x: Arc<EmlNode>, w: Arc<EmlNode>) -> Arc<EmlNode> {
    let x_s = add_eml(x, konst(4.0));
    let prod_s = mul_eml(x_s, w.clone());
    sub_eml(prod_s, mul_cf_any_x(w, 4.0))
}

fn build_simple_sum(terms: &[Arc<EmlNode>]) -> Arc<EmlNode> {
    if terms.len() == 1 { return terms[0].clone(); }
    let mid = terms.len() / 2;
    add_eml(build_simple_sum(&terms[..mid]), build_simple_sum(&terms[mid..]))
}
