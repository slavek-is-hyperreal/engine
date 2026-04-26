// cargo run --features gguf --bin real_tinyllama -- models/tinyllama.gguf
fn main() {
    #[cfg(feature = "gguf")] run();
    #[cfg(not(feature = "gguf"))]
    eprintln!("Uruchom: cargo run --features gguf --bin real_tinyllama -- model.gguf");
}

#[cfg(feature = "gguf")]
fn run() {
    use eml_trs::cost_model::CostModel;
    use eml_trs::loader::gguf::GgufLoader;
    use eml_trs::nn_layer::{preprocess_wq_offline, build_and_optimize_sample, measure_costs};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Użycie: real_tinyllama <model.gguf>");
        eprintln!("Pobierz: huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \\");
        eprintln!("  tinyllama-1.1b-chat-v1.0.F16.gguf --local-dir ./models");
        std::process::exit(1);
    }

    println!("=== TinyLlama EML Analysis ===\nModel: {}\n", args[1]);
    let t0 = Instant::now();
    let mut loader = GgufLoader::open(&args[1])
        .unwrap_or_else(|e| { eprintln!("Błąd: {}", e); std::process::exit(1); });

    let mut names = loader.tensor_names();
    names.sort();
    println!("Tensory: {}", names.len());
    println!("Przykłady: {:?}\n", &names[..names.len().min(5)]);

    let layer = loader.load_layer(0)
        .unwrap_or_else(|e| { eprintln!("Błąd: {}", e); std::process::exit(1); });
    println!("Warstwa 0: {:.0}ms", t0.elapsed().as_secs_f64()*1000.0);
    println!("  W_Q: {} W_gate: {}\n", layer.w_q.len(), layer.w_gate.len());

    println!("=== Koszty EML ===");
    let (naive,asis,cf) = measure_costs(4096);
    println!("K=4096: naive={} asis={} ({:.1}%) cf={} ({:.1}%)",
        naive, asis, (naive-asis) as f64/naive as f64*100.0,
        cf, (naive-cf) as f64/naive as f64*100.0);
    println!("Warstwa: naive={} opt={} red={:.1}%",
        CostModel::tinyllama_layer_naive(),
        CostModel::tinyllama_layer_optimized(),
        CostModel::tinyllama_layer_reduction());
    println!("Dolna granica Ω(n²d): {}\n", CostModel::attention_lower_bound(2048,64));

    println!("=== Offline Preprocessing ===");
    let t1 = Instant::now();
    let hidden = layer.rms_att_weight.len();
    let w_pre = preprocess_wq_offline(&layer.w_q, &layer.rms_att_weight, 64, hidden);
    println!("γ + 1/√dk → W_Q: {:.0}ms\n", t1.elapsed().as_secs_f64()*1000.0);

    println!("=== Próbka K=16 ===");
    let w_s: Vec<f32> = w_pre.iter().take(16).copied().collect();
    let t2 = Instant::now();
    let r = build_and_optimize_sample(&w_s, 16);
    println!("Czas: {:.0}ms", t2.elapsed().as_secs_f64()*1000.0);
    println!("  naive={} cf={} trs={} red={:.1}% depth={}",
        r.nodes_naive, r.nodes_after_cf, r.nodes_after_trs,
        r.reduction_pct, r.sample_tree.depth());

    fn stats(name: &str, w: &[f32]) {
        let min=w.iter().cloned().fold(f32::INFINITY,f32::min);
        let max=w.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
        let mean=w.iter().sum::<f32>()/w.len() as f32;
        let neg=w.iter().filter(|&&x|x<0.0).count();
        println!("  {} [{}]: min={:.4} max={:.4} mean={:.4} neg={:.0}%",
            name, w.len(), min, max, mean, neg as f64/w.len() as f64*100.0);
    }
    println!("\n=== Statystyki Wag ===");
    stats("W_Q", &layer.w_q);
    stats("W_Q preprocessed", &w_pre);
    stats("γ RMSNorm", &layer.rms_att_weight);

    use std::fs::File; use std::io::Write;
    std::fs::create_dir_all("paper/results").unwrap();
    let mut f=File::create("paper/results/real_tinyllama.csv").unwrap();
    writeln!(f,"metric,value").unwrap();
    writeln!(f,"sample_k,{}",r.sample_k).unwrap();
    writeln!(f,"sample_naive,{}",r.nodes_naive).unwrap();
    writeln!(f,"sample_after_cf,{}",r.nodes_after_cf).unwrap();
    writeln!(f,"sample_after_trs,{}",r.nodes_after_trs).unwrap();
    writeln!(f,"sample_reduction_pct,{:.2}",r.reduction_pct).unwrap();
    writeln!(f,"layer_naive,{}",CostModel::tinyllama_layer_naive()).unwrap();
    writeln!(f,"layer_optimized,{}",CostModel::tinyllama_layer_optimized()).unwrap();
    writeln!(f,"layer_reduction_pct,{:.1}",CostModel::tinyllama_layer_reduction()).unwrap();
    println!("\nZapisano: paper/results/real_tinyllama.csv");
}
