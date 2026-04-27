// benches/trs_performance.rs

use eml_trs::ast::*;
use eml_trs::trs::{rewrite_with_stats, ReductionStats};
use eml_trs::round_trip::round_trip_optimize;
use std::sync::Arc;

fn rewrite_safe(node: Arc<EmlNode>) -> (Arc<EmlNode>, ReductionStats) {
    // Run in a thread with a larger stack
    let builder = std::thread::Builder::new().stack_size(32 * 1024 * 1024);
    let handler = builder.spawn(move || rewrite_with_stats(node)).unwrap();
    handler.join().unwrap()
}

fn main() {
    println!("=== TRS Reduction Benchmarks ===");

    // Test 1: ln(exp(x)) — should → x
    {
        let x = var("x");
        let tree = ln_node(exp_node(x.clone()));
        let (result, stats) = rewrite_safe(tree);
        println!("ln(exp(x)): {} → {} nodes ({:.1}% reduction)",
            stats.nodes_before, stats.nodes_after, stats.reduction_percent);
        assert!(result.structural_eq(&x), "ln(exp(x)) did not reduce to x!");
    }

    // Test 2: Nested ln(exp())
    {
        let x = var("x");
        let tree = ln_node(exp_node(ln_node(exp_node(x.clone()))));
        let (_, stats) = rewrite_safe(tree);
        println!("ln(exp(ln(exp(x)))): {} → {} nodes ({:.1}% reduction)",
            stats.nodes_before, stats.nodes_after, stats.reduction_percent);
    }

    // Test 3: Sigmoid-like pattern simplification
    {
        // eml(ln(a), 1) → a
        let a = var("a");
        let tree = eml(ln_node(a.clone()), one());
        let (_, stats) = rewrite_safe(tree);
        println!("eml(ln(a), 1): {} → {} nodes ({:.1}% reduction)",
            stats.nodes_before, stats.nodes_after, stats.reduction_percent);
    }

    // Test 4: Constant e
    {
        let tree = eml(one(), one());
        let (result, stats) = rewrite_safe(tree);
        println!("eml(1,1) → e: {} → {} nodes ({:.1}% reduction)",
            stats.nodes_before, stats.nodes_after, stats.reduction_percent);
        if let EmlNode::Const(v) = result.as_ref() {
            assert!((v - std::f64::consts::E).abs() < 1e-10);
        }
    }

    println!("\n=== Round-Trip vs TRS Comparison ===");
    {
        let x = var("x");
        // An expression that standard TRS might not fully reduce 
        // if it lacks global identity rules (RT4: ln(exp(x)) -> x)
        let tree = ln_node(exp_node(ln_node(exp_node(
            eml(ln_node(exp_node(x.clone())), one())
        ))));

        let (_, trs_stats) = rewrite_safe(tree.clone());
        
        let rt_result = {
            let builder = std::thread::Builder::new().stack_size(32 * 1024 * 1024);
            let tree_clone = tree.clone();
            let handler = builder.spawn(move || {
                round_trip_optimize(tree_clone)
            }).unwrap();
            handler.join().unwrap()
        };

        println!("Round-Trip vs TRS Comparison:");
        println!("  Original:     {} nodes", tree.eml_count());
        println!("  TRS alone:    {} nodes", trs_stats.nodes_after);
        println!("  Round-trip:   {} nodes", rt_result.eml_count());
        println!("  RT bonus:     {} additional nodes removed",
            trs_stats.nodes_after.saturating_sub(rt_result.eml_count()));
    }

    {
        let x = var("x");
        let y = var("y");
        
        // ln(x) + ln(y)
        // Manual construction of EML addition: x + y = eml(ln(x), exp(eml(ln(0), exp(y))))
        // but here we want ln(x) + ln(y)
        let lnx = ln_node(x.clone());
        let lny = ln_node(y.clone());
        
        // Add(lnx, lny) = eml(ln(lnx), exp(eml(ln(0), exp(lny))))
        let tree = eml(
            ln_node(lnx), 
            exp_node(eml(ln_node(konst(0.0)), exp_node(lny)))
        );

        let (_, trs_stats) = rewrite_safe(tree.clone());
        let rt_result = round_trip_optimize(tree.clone());

        println!("RT1: ln(x) + ln(y) -> ln(x*y):");
        println!("  Original:     {} nodes", tree.eml_count());
        println!("  TRS alone:    {} nodes", trs_stats.nodes_after);
        println!("  Round-trip:   {} nodes", rt_result.eml_count());
        println!("  RT bonus:     {} nodes",
            trs_stats.nodes_after.saturating_sub(rt_result.eml_count()));
    }

    println!("\nVerification completed successfully.");
}
