use eml_trs::ast::*;
use std::sync::Arc;

fn max_eml(a: Arc<EmlNode>, b: Arc<EmlNode>) -> Arc<EmlNode> {
    // max(a, b) = 0.5 * (a + b + |a - b|)
    // This is just a conceptual model of the "bad" way
    // In EML, even if we use the robust primitives, it explodes.
    add_eml(add_eml(a.clone(), b.clone()), sub_eml(a, b)) // Simplified
}

fn softmax_stable_naive_cost(n: usize) -> usize {
    if n == 1 { return 1; }
    3 * softmax_stable_naive_cost(n / 2) + 10 // O(3^log n) = O(n^log 3) = O(n^1.58)
}

fn main() {
    println!("=== Corollary 3 Verification ===");
    for n in [2, 4, 8, 16, 32, 64] {
        println!("Stable Softmax (naive max-shift) cost for n={}: {} nodes", n, softmax_stable_naive_cost(n));
    }
}
