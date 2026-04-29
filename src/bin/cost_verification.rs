use eml_trs::cost_model::CostModel;

fn main() {
    println!("=== Cost Model Verification ===");
    
    let k = 64;
    let naive = CostModel::dot_product_naive(k);
    let asis = CostModel::dot_product_asis(k);
    let cf_asis = CostModel::dot_product_cf_asis(k);
    
    println!("K=64 Dot Product Costs:");
    println!("  Naive:    {}", naive);
    println!("  ASIS:     {}", asis);
    println!("  CF+ASIS:  {}", cf_asis);
    
    let reduction_asis = (naive as f64 - asis as f64) / naive as f64;
    println!("ASIS Reduction: {:.2}% (Expected ~22.2%)", reduction_asis * 100.0);
    
    let reduction_total = (naive as f64 - cf_asis as f64) / naive as f64;
    println!("Total Reduction (CF+ASIS): {:.2}% (Expected ~61%)", reduction_total * 100.0);
    
    println!("\n=== Operation Fusions ===");
    println!("SwiGLU Fused Cost: {} nodes (Expected 32 per dim)", CostModel::swiglu_fused_cost());
    println!("Residual Fused Cost: {} node (Expected 1)", CostModel::residual_fused_cost());
}
