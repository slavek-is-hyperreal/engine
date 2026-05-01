// src/bin/eml_benchmark.rs
//
// Benchmark: TinyLlama Q-projection — ALU vs EML
//
// Uruchomienie:
//   cargo run --bin eml_benchmark --release -- models/layer0_weights.json
//
// Mierzy:
//   ALU:  tradycyjny f32 dot product
//   EML:  build_dot_product_eml + rewrite + try_evaluate
//
// Wyniki zapisywane na bieżąco do models/benchmark_results.txt

use std::time::Instant;
use std::io::Write;
use eml_trs::ast::var;
use eml_trs::nn_layer::build_dot_product_eml;
use eml_trs::trs::rewrite;
use eml_trs::constant_fold::{try_evaluate, ConstantMap};

#[derive(serde::Deserialize)]
struct Weights {
    k: usize,
    d_k: usize,
    w_q_head0: Vec<Vec<f32>>,
    input_sample: Vec<f32>,
    alu_reference: Vec<f32>,
}

fn alu_dot(x: &[f32], w: &[f32]) -> f32 {
    x.iter().zip(w.iter()).map(|(a, b)| a * b).sum()
}

fn main() {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "models/layer0_weights.json".to_string());

    println!("Loading {}...", path);
    let data = std::fs::read_to_string(&path)
        .expect("Cannot read weights JSON — run scripts/extract_weights.py first");
    let weights: Weights = serde_json::from_str(&data)
        .expect("Invalid JSON format");

    let k   = weights.k;
    let d_k = weights.d_k;
    let x   = &weights.input_sample;
    println!("Weights loaded: K={}, d_k={}, rows={}", k, d_k, weights.w_q_head0.len());

    // ============================================================
    // ALU BASELINE — tradycyjny f32 matmul
    // ============================================================
    println!("\n--- ALU matmul (baseline) ---");
    let t0 = Instant::now();
    let alu_results: Vec<f32> = weights.w_q_head0.iter()
        .map(|row| alu_dot(x, row))
        .collect();
    let alu_time = t0.elapsed();

    println!("  Czas:        {:.3} ms", alu_time.as_secs_f64() * 1000.0);
    println!("  result[0]:   {:.8}", alu_results[0]);
    println!("  result[1]:   {:.8}", alu_results[1]);

    // Weryfikacja z ALU reference z Pythona
    let python_ref = &weights.alu_reference;
    let max_alu_diff: f32 = alu_results.iter()
        .zip(python_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  Python parity diff: {:.2e}  {}", max_alu_diff,
        if max_alu_diff < 1e-3 { "✅" } else { "⚠️ " });

    // ============================================================
    // EML — symboliczne drzewo + TRS + ewaluacja
    // ============================================================
    println!("\n--- EML matmul (symbolic tree + TRS + evaluate) ---");
    println!("  K={}, d_k={} wierszy — to może zająć dużo czasu...", k, d_k);
    println!("  Postęp zapisywany do models/benchmark_results.txt");

    // Zmienne symboliczne — budowane raz, dzielone między wierszami
    let x_vars: Vec<_> = (0..k).map(|i| var(&format!("x{}", i))).collect();

    // ConstantMap — budowana raz
    let mut const_map = ConstantMap::new();
    for (i, &v) in x.iter().enumerate() {
        const_map.insert(format!("x{}", i), v as f64);
    }

    let mut out_file = std::fs::File::create("models/benchmark_results.txt")
        .expect("Cannot create results file");
    writeln!(out_file, "row,alu,eml,diff,build_ms,rewrite_ms,eval_ms,nodes_before,nodes_after,reduction_pct").unwrap();


    let mut eml_results = Vec::with_capacity(d_k);
    let mut total_build   = std::time::Duration::ZERO;
    let mut total_rewrite = std::time::Duration::ZERO;
    let mut total_eval    = std::time::Duration::ZERO;

    let mut last_nodes_before = 0;
    let mut last_nodes_after = 0;
    let mut last_reduction = 0.0;

    for (row_idx, row_weights) in weights.w_q_head0.iter().enumerate() {
        // Budowa drzewa EML
        let t_build = Instant::now();
        let tree = build_dot_product_eml(&x_vars, row_weights);
        let dt_build = t_build.elapsed();

        let nodes_before = tree.node_count();

        // TRS do fixpoint
        let t_rewrite = Instant::now();
        let tree_opt = rewrite(tree);
        let dt_rewrite = t_rewrite.elapsed();

        let nodes_after = tree_opt.node_count();
        let reduction = 1.0 - (nodes_after as f64 / nodes_before as f64);

        last_nodes_before = nodes_before;
        last_nodes_after = nodes_after;
        last_reduction = reduction;

        // Ewaluacja (Optimized path using Round-Trip TRS)
        use eml_trs::round_trip::compile_to_ops;
        let t_eval = Instant::now();
        let program = compile_to_ops(tree_opt);
        let eml_val = program.execute(&const_map)
            .map(|v| v as f32)
            .unwrap_or(f32::NAN);
        let dt_eval = t_eval.elapsed();


        total_build   += dt_build;
        total_rewrite += dt_rewrite;
        total_eval    += dt_eval;

        let diff = (eml_val - alu_results[row_idx]).abs();
        eml_results.push(eml_val);

        // Zapisz wiersz wyników
        writeln!(out_file,
            "{},{:.8},{:.8},{:.2e},{:.3},{:.3},{:.3},{},{},{:.1}",
            row_idx,
            alu_results[row_idx],
            eml_val,
            diff,
            dt_build.as_secs_f64() * 1000.0,
            dt_rewrite.as_secs_f64() * 1000.0,
            dt_eval.as_secs_f64() * 1000.0,
            nodes_before,
            nodes_after,
            reduction * 100.0,
        ).unwrap();
        out_file.flush().unwrap();

        // Print progress every 8 rows
        if row_idx % 8 == 0 || row_idx == d_k - 1 {
            let elapsed = total_build + total_rewrite + total_eval;
            let remaining = if row_idx > 0 {
                elapsed / (row_idx as u32 + 1) * (d_k as u32 - row_idx as u32 - 1)
            } else {
                std::time::Duration::ZERO
            };
            eprintln!("[{:3}/{}] diff={:.2e} nodes={}/{} ({:.1}%) elapsed={:.1}s ETA={:.0}s",
                row_idx + 1, d_k, diff,
                nodes_after, nodes_before, reduction * 100.0,
                elapsed.as_secs_f64(),
                remaining.as_secs_f64());
        }
    }


    // ============================================================
    // PODSUMOWANIE
    // ============================================================
    let eml_total = total_build + total_rewrite + total_eval;
    let max_diff: f32 = eml_results.iter()
        .zip(alu_results.iter())
        .filter(|(e, _)| !e.is_nan())
        .map(|(e, a)| (e - a).abs())
        .fold(0.0f32, f32::max);
    let nan_count = eml_results.iter().filter(|v| v.is_nan()).count();
    let bf16_eps  = 0.0078125f32;

    println!("\n=== WYNIKI BENCHMARK ===");
    println!("TinyLlama Q-projection, layer 0, head 0");
    println!("K={}, d_k={} wierszy", k, d_k);
    println!();
    println!("--- ALU (f32 tradycyjny) ---");
    println!("  Czas:           {:.4} ms", alu_time.as_secs_f64() * 1000.0);
    println!("  result[0]:      {:.8}", alu_results[0]);
    println!();
    println!("--- EML (symbolic tree + TRS + evaluate) ---");
    println!("  build_tree:     {:.1} ms", total_build.as_secs_f64() * 1000.0);
    println!("  rewrite (TRS):  {:.1} ms", total_rewrite.as_secs_f64() * 1000.0);
    println!("  evaluate:       {:.1} ms", total_eval.as_secs_f64() * 1000.0);
    println!("  TOTAL:          {:.1} ms", eml_total.as_secs_f64() * 1000.0);
    println!("  result[0]:      {:.8}", eml_results[0]);
    println!();
    println!("--- Parytet ---");
    println!("  max_diff:       {:.2e}", max_diff);
    println!("  BF16 eps:       {:.7}", bf16_eps);
    println!("  Status:         {}", if max_diff < bf16_eps { "✅ OK" } else { "⚠️  POZA ZAKRESEM" });
    println!("  NaN count:      {}/{}", nan_count, d_k);
    println!();
    println!("--- Redukcja węzłów (Dot Product K=2048) ---");
    println!("  Nodes before TRS: {}", last_nodes_before);
    println!("  Nodes after TRS:  {}", last_nodes_after);
    println!("  TRS eliminuje {:.1}% węzłów w dot product z float32 wagami", last_reduction * 100.0);
    println!("  (UWAGA: Teoretyczne 61.1% dotyczy całych warstw z fuzją Log-Softmax/RMSNorm)");


    if alu_time.as_nanos() > 0 {
        let overhead = eml_total.as_secs_f64() / alu_time.as_secs_f64();
        println!("EML overhead vs ALU: {:.0}×", overhead);
        println!("(overhead jest oczekiwany — EML symbolic path na CPU);");
        println!(" na GPU WGSL każdy węzeł drzewa = 1 FMA cycle)");
    }
    println!("\nSzczegóły w: models/benchmark_results.txt");
}
