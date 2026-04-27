// benches/proc_macro_bench.rs
//
// Measures instruction count reduction from #[eml_optimize]

use eml_trs_macro::eml_optimize;

// Without macro: baseline
fn gaussian_naive(x: f64) -> f64 {
    (-x * x).exp()
}

// With macro: EML-optimized (Phase 1: identity + inline)
#[eml_optimize]
fn gaussian_eml(x: f64) -> f64 {
    (-x * x).exp()
}

fn main() {
    println!("=== EML Proc Macro Optimization ===");
    
    // Verify correctness
    let x = 1.5f64;
    let naive = gaussian_naive(x);
    let eml = gaussian_eml(x);
    
    assert!((naive - eml).abs() < 1e-10);
    println!("gaussian({}) = {:.6}", x, naive);
    println!("Results match: ✅");
    println!();
    
    println!("EML form of exp(-x^2):");
    println!("  eml(eml(1, eml(eml(1, eml(eml(1,x),eml(x,1))), 1)), 1)");
    println!();
    
    println!("To measure instruction count reduction, use 'cargo asm':");
    println!("  cargo asm eml_trs::gaussian_naive");
    println!("  cargo asm eml_trs::gaussian_eml");
}
