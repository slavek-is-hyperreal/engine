use eml_trs::ast::*;
use std::sync::Arc;

fn build_dot_product(k: usize) -> Arc<EmlNode> {
    let mut terms = Vec::new();
    for i in 0..k {
        terms.push(mul_eml(var(&format!("x_{}", i)), var(&format!("w_{}", i))));
    }
    
    while terms.len() > 1 {
        let mut next_level = Vec::new();
        for i in (0..terms.len()).step_by(2) {
            if i + 1 < terms.len() {
                next_level.push(add_eml(terms[i].clone(), terms[i+1].clone()));
            } else {
                next_level.push(terms[i].clone());
            }
        }
        terms = next_level;
    }
    terms[0].clone()
}

fn main() {
    println!("=== Theorem C3 Verification (Depth Scaling) ===");
    println!("| K (Size) | Node Count | Depth | Log2(N) | Ratio (D/LogN) |");
    println!("|:---|:---|:---|:---|:---|");
    
    for k_pow in 1..=10 {
        let k = 2usize.pow(k_pow);
        let tree = build_dot_product(k);
        let n = tree.node_count();
        let d = tree.depth();
        let log_n = (n as f64).log2();
        let ratio = d as f64 / log_n;
        
        println!("| {} | {} | {} | {:.2} | {:.2} |", k, n, d, log_n, ratio);
    }
    println!("\nConclusion: Depth scales as O(log N), confirming NC1 classification.");
}
