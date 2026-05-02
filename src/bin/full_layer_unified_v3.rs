use eml_trs::ast::*;
use eml_trs::nn_layer::build_dot_product_eml;
use eml_trs::dag_mmap::*;
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

fn main() {
    println!("=== EML 1:1 UNIFIED GLOBAL DAG AUDIT V3 (Out-of-Core) ===");
    println!("Strategy: Mmap-backed Global DAG on ZFS.");
    println!("Deduplication: Structural Hashing + Per-Tree Local Cache.\n");
    
    let dag_path = "/vectorlegis_ssd_pool/eml_working/global_dag_v3.bin";
    let initial_capacity = 500_000_000u64; // 500M nodes = ~12 GB
    
    let mut dag = MmapDag::create(dag_path, initial_capacity)
        .expect("Failed to create MmapDag on ZFS");
    println!("MmapDag initialized at {}. Capacity: {} nodes.", dag_path, initial_capacity);

    let t_load = Instant::now();
    let file = File::open("models/layer0_full.json").expect("Missing layer0_full.json");
    let reader = BufReader::new(file);
    let data: LayerData = serde_json::from_reader(reader).expect("Failed to parse weights");
    println!("Loaded 374MB weights in {:?}", t_load.elapsed());

    let hidden = data.hidden_dim;
    let x_vars: Vec<Arc<EmlNode>> = (0..hidden).map(|i| var(&format!("x{}", i))).collect();

    let mut total_naive_count: u64 = 0;
    let t_audit = Instant::now();

    // 1. RMSNorm Audit
    println!("Auditing RMSNorm (Fusion 3)...");
    let mut audit_norm = |dag: &mut MmapDag, name: &str, weights: &[f32]| {
        let d = weights.len();
        let mut x2_terms = Vec::new();
        for i in 0..d {
            let xv = var(&format!("x{}", i));
            x2_terms.push(exp_node(mul_cf(ln_node(xv), 2.0)));
        }
        let mean_x2 = mul_cf(build_simple_sum(&x2_terms), 1.0 / d as f64);
        let inv_rms = exp_node(mul_cf(ln_node(mean_x2), -0.5));

        for i in 0..d {
            let x_i = var(&format!("x{}", i));
            let out = mul_safe_stable(x_i, inv_rms.clone());
            let mut local_cache = HashMap::new();
            add_tree_to_mmap_dag(dag, &out, &mut local_cache);
        }
        println!("  Norm {} integrated.", name);
        (66 * d) as u64
    };

    total_naive_count += audit_norm(&mut dag, "attn_norm", &data.weights["attn_norm"][0]);
    total_naive_count += audit_norm(&mut dag, "ffn_norm", &data.weights["ffn_norm"][0]);

    // 2. Matrix Audit
    let mut integrate_matrix = |dag: &mut MmapDag, name: &str, weights: &Vec<Vec<f32>>, inputs: &Vec<Arc<EmlNode>>| {
        let rows = weights.len();
        let cols = inputs.len();
        let naive_per_row = (36 * cols - 19) as u64;
        
        println!("Integrating matrix: {:<10} ({} rows)...", name, rows);
        for i in 0..rows {
            if i > 0 && i % 1024 == 0 {
                println!("  Progress: {:>4}/{} (Unique nodes: {})", i, rows, dag.unique_node_count());
            }
            let tree = build_dot_product_eml(inputs, &weights[i]);
            let mut local_cache = HashMap::new();
            add_tree_to_mmap_dag(dag, &tree, &mut local_cache);
        }
        (rows as u64 * naive_per_row) as u64
    };

    for proj in ["q", "k", "v", "o"] {
        total_naive_count += integrate_matrix(&mut dag, proj, &data.weights[proj], &x_vars);
    }

    // 3. FFN Audit
    println!("Integrating FFN Block (SwiGLU)...");
    let w_gate = &data.weights["gate"];
    let w_up = &data.weights["up"];
    let w_down = &data.weights["down"];
    for i in 0..w_gate.len() {
        if i > 0 && i % 1024 == 0 {
            println!("  FFN Progress: {:>4}/{} (Unique nodes: {})", i, w_gate.len(), dag.unique_node_count());
        }
        let g = build_dot_product_eml(&x_vars, &w_gate[i]);
        let u = build_dot_product_eml(&x_vars, &w_up[i]);
        let fused = eml_trs::fusions::swiglu_fused(g, u);
        let mut local_cache = HashMap::new();
        add_tree_to_mmap_dag(&mut dag, &fused, &mut local_cache);
    }
    total_naive_count += (w_gate.len() * (36 * hidden - 19) * 2) as u64;

    let gate_vars: Vec<Arc<EmlNode>> = (0..w_gate.len()).map(|i| var(&format!("g{}", i))).collect();
    total_naive_count += integrate_matrix(&mut dag, "down", w_down, &gate_vars);

    println!("\n=== FINAL UNIFIED V3 AUDIT VERDICT ===");
    let total_opt = dag.unique_node_count() as u64;
    println!("Execution Time:     {:?}", t_audit.elapsed());
    println!("Total Naive Nodes:  {:>16}", total_naive_count);
    println!("Total Unique Nodes: {:>16}", total_opt);
    println!("Sharing Savings:    {:>16} nodes", dag.sharing_savings());
    println!("FINAL REDUCTION:    {:>15.2}%", (1.0 - total_opt as f64 / total_naive_count as f64) * 100.0);
    println!("SSD Storage used:   {:.2} GB", (total_opt * 24) as f64 / 1e9);
}

fn mul_safe_stable(a: Arc<EmlNode>, b: Arc<EmlNode>) -> Arc<EmlNode> {
    let a_shifted = add_eml(a, konst(4.0));
    let product = mul_eml(a_shifted, b.clone());
    sub_eml(product, mul_eml(konst(4.0), b))
}

fn build_simple_sum(terms: &[Arc<EmlNode>]) -> Arc<EmlNode> {
    if terms.len() == 1 { return terms[0].clone(); }
    let mid = terms.len() / 2;
    add_eml(build_simple_sum(&terms[..mid]), build_simple_sum(&terms[mid..]))
}
