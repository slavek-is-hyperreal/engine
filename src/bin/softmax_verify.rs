
fn softmax_stable_naive_cost(n: usize) -> usize {
    if n == 1 { return 1; }
    3 * softmax_stable_naive_cost(n / 2) + 10 // O(3^log n) = O(n^1.58)
}

fn main() {
    println!("=== Corollary 3 Verification ===");
    for n in [2, 4, 8, 16, 32, 64] {
        println!("Stable Softmax (naive max-shift) cost for n={}: {} nodes", n, softmax_stable_naive_cost(n));
    }
}
