// benches/tinyllama_ops.rs

use eml_trs::cost_model::CostModel;

/// TinyLlama Parameters
const HIDDEN_DIM: usize = 4096;
const FFN_DIM: usize = 11008;
const SEQ_LEN: usize = 2048;
const N_HEADS: usize = 32;
const D_K: usize = 64;
const N_LAYERS: usize = 22;

fn main() {
    println!("=== TinyLlama EML Cost Analysis ===");
    println!("Model: TinyLlama 1.1B");
    println!("hidden={}, ffn={}, seq={}, heads={}, d_k={}",
        HIDDEN_DIM, FFN_DIM, SEQ_LEN, N_HEADS, D_K);
    println!();

    // MatMul
    println!("--- MatMul (x @ W, hidden->ffn) ---");
    let dot_naive = CostModel::dot_product_naive(HIDDEN_DIM);
    let dot_asis = CostModel::dot_product_asis(HIDDEN_DIM);
    let dot_cf = CostModel::dot_product_cf_asis(HIDDEN_DIM);
    println!("Dot product K={}: naive={}, ASIS={}, ASIS+CF={}",
        HIDDEN_DIM, dot_naive, dot_asis, dot_cf);
    println!("Reduction ASIS vs naive: {:.1}%",
        (dot_naive - dot_asis) as f64 / dot_naive as f64 * 100.0);
    println!("Reduction CF+ASIS vs naive: {:.1}%",
        (dot_naive - dot_cf) as f64 / dot_naive as f64 * 100.0);

    // Softmax vs Log-Softmax
    println!();
    println!("--- Softmax vs Log-Softmax (seq_len={}) ---", SEQ_LEN);
    let softmax = CostModel::softmax_dag(SEQ_LEN);
    let log_softmax = CostModel::log_softmax_dag(SEQ_LEN);
    println!("Softmax DAG: {} nodes", softmax);
    println!("Log-Softmax DAG: {} nodes (native EML!)", log_softmax);
    println!("Log-Softmax savings: {:.1}%",
        (softmax - log_softmax) as f64 / softmax as f64 * 100.0);

    // SiLU / Sigmoid
    println!();
    println!("--- Activations ---");
    println!("Sigmoid(x): {} nodes", CostModel::sigmoid_cost());
    println!("SiLU(x) naive: 67 nodes");
    println!("SiLU(x) optimized: {} nodes (= Sigmoid!)",
        CostModel::silu_cost());

    // RMSNorm
    println!();
    println!("--- RMSNorm (d={}) ---", HIDDEN_DIM);
    println!("RMSNorm DAG: {} nodes", CostModel::rmsnorm_dag(HIDDEN_DIM));

    // RoPE
    println!();
    println!("--- RoPE (d_k={}, seq={}) ---", D_K, SEQ_LEN);
    let rope_total = D_K * CostModel::rope_element_cost_cf() * SEQ_LEN;
    println!("RoPE per token (after CF): {} nodes",
        D_K * CostModel::rope_element_cost_cf());
    println!("RoPE full sequence (1 head): {} nodes", rope_total);

    // Attention
    println!();
    println!("--- Attention (1 head, seq={}, d_k={}) ---", SEQ_LEN, D_K);
    let qkt = SEQ_LEN * SEQ_LEN * CostModel::dot_product_naive(D_K);
    let sm = SEQ_LEN * CostModel::softmax_dag(SEQ_LEN);
    let sv = SEQ_LEN * D_K * (SEQ_LEN * 17 + (SEQ_LEN - 1) * 19);
    let total_head = qkt + sm + sv;
    println!("Q*K^T: {} nodes ({:.1}%)", qkt,
        qkt as f64 / total_head as f64 * 100.0);
    println!("Softmax: {} nodes ({:.1}%)", sm,
        sm as f64 / total_head as f64 * 100.0);
    println!("Scores*V: {} nodes ({:.1}%)", sv,
        sv as f64 / total_head as f64 * 100.0);
    println!("Total 1 head: {} nodes", total_head);
    println!("Total 32 heads: {} nodes", total_head * N_HEADS);

    // Full Layer Optimization
    println!();
    println!("--- Full Layer (Naive vs Optimized) ---");
    let layer_naive = CostModel::tinyllama_layer_naive();
    let layer_opt = CostModel::tinyllama_layer_optimized();
    println!("Naive: {} nodes", layer_naive);
    println!("Optimized (CF+ASIS+DAG+Fusions): {} nodes", layer_opt);
    println!("Reduction: {:.1}%", CostModel::tinyllama_layer_reduction());

    println!();
    println!("--- Attention Lower Bound ---");
    let lb = CostModel::attention_lower_bound(SEQ_LEN, D_K);
    println!("Theoretical lower bound Omega(n^2d): {} nodes", lb);

    // Save CSV
    save_csv(dot_naive, dot_asis, dot_cf, softmax, log_softmax, total_head, layer_naive, layer_opt);
}

fn save_csv(
    dot_naive: usize, dot_asis: usize, dot_cf: usize,
    softmax: usize, log_softmax: usize, attention_head: usize,
    layer_naive: u64, layer_opt: u64
) {
    use std::fs::File;
    use std::io::Write;

    std::fs::create_dir_all("paper/results").unwrap();
    let mut f = File::create("paper/results/tinyllama_optimized.csv").unwrap();
    writeln!(f, "operation,variant,cost_nodes,reduction_pct").unwrap();
    writeln!(f, "dot_product_K4096,naive,{},0.0", dot_naive).unwrap();
    writeln!(f, "dot_product_K4096,asis,{},{:.2}", dot_asis,
        (dot_naive - dot_asis) as f64 / dot_naive as f64 * 100.0).unwrap();
    writeln!(f, "dot_product_K4096,asis_cf,{},{:.2}", dot_cf,
        (dot_naive - dot_cf) as f64 / dot_naive as f64 * 100.0).unwrap();
    writeln!(f, "softmax_n2048,naive_dag,{},0.0", softmax).unwrap();
    writeln!(f, "softmax_n2048,log_softmax_dag,{},{:.2}", log_softmax,
        (softmax - log_softmax) as f64 / softmax as f64 * 100.0).unwrap();
    writeln!(f, "attention_1head,total,{},0.0", attention_head).unwrap();
    writeln!(f, "full_layer,naive,{},0.0", layer_naive).unwrap();
    writeln!(f, "full_layer,optimized,{},{:.2}", layer_opt,
        CostModel::tinyllama_layer_reduction()).unwrap();
    println!("\nSaved results to: paper/results/tinyllama_optimized.csv");
}
