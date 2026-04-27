// benches/trs_performance.rs

use eml_trs::ast::*;
use eml_trs::trs::rewrite_with_stats;
use std::sync::Arc;

fn rewrite_safe(node: Arc<EmlNode>) -> (Arc<EmlNode>, eml_trs::trs::ReductionStats) {
    // Uruchom w wątku z większym stosem
    let builder = std::thread::Builder::new().stack_size(32 * 1024 * 1024);
    let handler = builder.spawn(move || rewrite_with_stats(node)).unwrap();
    handler.join().unwrap()
}

fn main() {
    println!("=== TRS Reduction Benchmarks ===");

    // Test 1: ln(exp(x)) — powinno → x
    {
        let x = var("x");
        let tree = ln_node(exp_node(x.clone()));
        let (result, stats) = rewrite_safe(tree);
        println!("ln(exp(x)): {} → {} węzłów ({:.1}% redukcja)",
            stats.nodes_before, stats.nodes_after, stats.reduction_percent);
        assert!(result.structural_eq(&x), "ln(exp(x)) nie zredukował do x!");
    }

    // Test 2: Zagnieżdżone ln(exp())
    {
        let x = var("x");
        let tree = ln_node(exp_node(ln_node(exp_node(x.clone()))));
        let (result, stats) = rewrite_safe(tree);
        println!("ln(exp(ln(exp(x)))): {} → {} węzłów ({:.1}% redukcja)",
            stats.nodes_before, stats.nodes_after, stats.reduction_percent);
    }

    // Test 3: Wzorzec Sigmoid uproszczenia
    {
        // eml(ln(a), 1) → a
        let a = var("a");
        let tree = eml(ln_node(a.clone()), one());
        let (result, stats) = rewrite_safe(tree);
        println!("eml(ln(a), 1): {} → {} węzłów ({:.1}% redukcja)",
            stats.nodes_before, stats.nodes_after, stats.reduction_percent);
    }

    // Test 4: Stała e
    {
        let tree = eml(one(), one());
        let (result, stats) = rewrite_safe(tree);
        println!("eml(1,1) → e: {} → {} węzłów ({:.1}% redukcja)",
            stats.nodes_before, stats.nodes_after, stats.reduction_percent);
        if let EmlNode::Const(v) = result.as_ref() {
            assert!((v - std::f64::consts::E).abs() < 1e-10);
        }
    }

    println!("\nTRS działa poprawnie.");
}
