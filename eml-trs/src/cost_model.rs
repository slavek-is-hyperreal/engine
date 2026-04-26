// src/cost_model.rs
//
// CONVENTION: eml_count() = number of internal Eml(l,r) nodes
//             node_count() = eml_count() + number of leaves
// Costs from exhaustive search (Odrzywołek paper) = node_count()
// Costs in this project's tests = eml_count() (internal only)

// No imports from ast.rs, because CostModel operates only on numbers (usize)

/// Base costs of operations from exhaustive search (Odrzywołek 2026)
/// Cost = total number of tree nodes (internal + leaves)
/// for arguments being simple variables (leaf cost = 1)
pub struct CostModel;

impl CostModel {
    pub fn exp_cost() -> usize { 3 }
    pub fn ln_cost() -> usize { 7 }
    pub fn sub_cost() -> usize { 11 }
    pub fn neg_cost() -> usize { 15 }
    pub fn mul_cost() -> usize { 17 }
    pub fn div_cost() -> usize { 17 }
    pub fn add_cost() -> usize { 19 }

    /// Operator overhead = cost - number of arguments
    /// Used to calculate the cost of composition with subtrees
    pub fn exp_overhead() -> usize { 2 }   // 3 - 1
    pub fn ln_overhead() -> usize { 6 }    // 7 - 1
    pub fn sub_overhead() -> usize { 9 }   // 11 - 2
    pub fn neg_overhead() -> usize { 14 }  // 15 - 1
    pub fn mul_overhead() -> usize { 15 }  // 17 - 2
    pub fn div_overhead() -> usize { 15 }  // 17 - 2
    pub fn add_overhead() -> usize { 17 }  // 19 - 2

    /// Composition cost: op(A, B) where A and B are subtrees
    pub fn compose_binary(overhead: usize, cost_a: usize, cost_b: usize) -> usize {
        overhead + cost_a + cost_b
    }

    pub fn compose_unary(overhead: usize, cost_a: usize) -> usize {
        overhead + cost_a
    }

    /// Cost of dot product of length K — naive method
    /// C_naive(K) = 36K - 19
    pub fn dot_product_naive(k: usize) -> usize {
        36 * k - 19
    }

    /// Cost of dot product of length K — ASIS method
    /// C_ASIS(K) = 28K - 11
    /// Requires: weights pre-negated offline
    pub fn dot_product_asis(k: usize) -> usize {
        28 * k - 11
    }

    /// Cost of dot product — ASIS + Constant Folding of weights
    /// C_CF_ASIS(K) = 14K - 9
    /// Requires: static weights, ln(ln(x)) precomputed
    pub fn dot_product_cf_asis(k: usize) -> usize {
        14 * k - 9
    }

    /// Cost of matrix multiplication M×K · K×N — naive method
    pub fn matmul_naive(m: usize, n: usize, k: usize) -> usize {
        m * n * Self::dot_product_naive(k)
    }

    /// Cost of matrix multiplication — ASIS
    pub fn matmul_asis(m: usize, n: usize, k: usize) -> usize {
        m * n * Self::dot_product_asis(k)
    }

    /// Cost of matrix multiplication — ASIS + CF
    pub fn matmul_cf_asis(m: usize, n: usize, k: usize) -> usize {
        // Precomputing Z = ln(ln(A)) costs 13*m*k
        // The rest: m*n*(14k-9)
        13 * m * k + m * n * Self::dot_product_cf_asis(k)
    }

    /// Cost of Softmax for a vector of length N — DAG (hardware)
    /// DAG_naive = 35n - 17
    pub fn softmax_dag(n: usize) -> usize {
        35 * n - 17
    }

    /// Cost of Log-Softmax for a vector of length N — DAG
    /// DAG_log = 28n - 17
    /// Log-Softmax is a NATIVE EML operation: eml(ln(x_i), S) = x_i - ln(S)
    pub fn log_softmax_dag(n: usize) -> usize {
        28 * n - 17
    }

    /// Cost of Sigmoid(x) — optimal
    pub fn sigmoid_cost() -> usize { 51 }

    /// Cost of SiLU(x) — optimal (= sigmoid!)
    /// SiLU(x) = x/(1+exp(-x)) instead of x*sigmoid(x)
    pub fn silu_cost() -> usize { 51 }

    /// Cost of RMSNorm for a vector of length d — DAG with memoization R
    /// Cost = 66d + 41
    pub fn rmsnorm_dag(d: usize) -> usize {
        66 * d + 41
    }

    /// Cost of RoPE per pair of dimensions — real form with Constant Folding
    /// Verification: Gemini Deep Research (algebraic proof)
    /// 4 multiplications by constants (CF: 5 nodes each) + subtraction (11) + addition (19)
    /// = 4*5 + 11 + 19 = 50 nodes per pair
    pub fn rope_pair_cost_cf() -> usize { 50 }
    pub fn rope_element_cost_cf() -> usize { 25 } // 50/2 per element

    // Old — mark as deprecated:
    #[deprecated(note = "Use rope_element_cost_cf() = 25 nodes (verified)")]
    pub fn rope_element_cost_lower() -> usize { 68 }
    #[deprecated(note = "Use rope_element_cost_cf() = 25 nodes (verified)")]
    pub fn rope_element_cost_upper() -> usize { 53 }
    #[deprecated(note = "Use rope_element_cost_cf() = 25.0")]
    pub fn rope_avg_cost() -> f64 { 60.5 }

    /// Full Attention cost for TinyLlama (one head)
    /// Q·K^T dominates: 49.38%, Scores·V: 49.78%, Softmax: 0.84%
    pub fn attention_one_head(seq_len: usize, d_k: usize) -> usize {
        let qkt = seq_len * seq_len * Self::dot_product_naive(d_k);
        let softmax = seq_len * Self::softmax_dag(seq_len);
        let sv = seq_len * d_k * (seq_len * 17 + (seq_len - 1) * 19);
        qkt + softmax + sv
    }

    /// Cost of one TinyLlama layer — naive
    /// Result from Deep Research: ~13,102 billion nodes
    pub fn tinyllama_layer_naive() -> u64 {
        13_102_000_000_000
    }

    /// Cost of one TinyLlama layer — after optimization
    /// CF + ASIS + DAG + boundary fusions
    /// Result from Deep Research: ~4,838 billion nodes (63% reduction)
    pub fn tinyllama_layer_optimized() -> u64 {
        4_838_000_000_000
    }

    /// Total reduction for the entire layer
    pub fn tinyllama_layer_reduction() -> f64 {
        let naive = Self::tinyllama_layer_naive() as f64;
        let opt = Self::tinyllama_layer_optimized() as f64;
        (naive - opt) / naive * 100.0
    }

    /// Cost of SwiGLU after boundary fusion (gate*up via SiLU)
    /// Reduction: 68 → 32 nodes per dimension
    pub fn swiglu_fused_cost() -> usize { 32 }

    /// Cost of residual connection when x is in ln(x) form in DAG
    /// 1 EML node instead of 19
    pub fn residual_fused_cost() -> usize { 1 }

    /// Lower bound of attention complexity — theorem
    /// Ω(n² · d) EML nodes for full attention
    pub fn attention_lower_bound(n: usize, d: usize) -> u64 {
        (n * n * d) as u64
    }

    /// Total reduction for each TinyLlama operation
    pub fn tinyllama_breakdown() -> Vec<(&'static str, f64, f64, f64)> {
        // (operation, naive_B, opt_B, reduction_%)
        vec![
            ("RMSNorm×2",        34.0,   0.02,  99.9),
            ("Projections Q,K,V",1855.3, 480.8, 74.1),
            // seq=2048, d_k=64, n_heads=32
            // 25 * 64 * 2048 * 32 = 104,857,600 ≈ 0.105B nodes
            ("RoPE",             1.7,    0.105, 93.8),
            ("Q@K^T",            306.6,  119.0, 61.2),
            ("Log-Softmax",      4.69,   0.13,  97.2),
            ("Attention@V",      309.1,  120.2, 61.1),
            ("W_O proj",         618.3,  240.4, 61.1),
            ("Residual×2",       0.15,   0.008, 94.6),
            ("FFN W_gate,W_up",  6647.0, 2585.0,61.1),
            ("SwiGLU",           1.53,   0.34,  77.7),
            ("W_down",           3324.0, 1292.0,61.1),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_costs() {
        assert_eq!(CostModel::exp_cost(), 3);
        assert_eq!(CostModel::ln_cost(), 7);
        assert_eq!(CostModel::sub_cost(), 11);
        assert_eq!(CostModel::neg_cost(), 15);
        assert_eq!(CostModel::mul_cost(), 17);
        assert_eq!(CostModel::div_cost(), 17);
        assert_eq!(CostModel::add_cost(), 19);
    }

    #[test]
    fn test_dot_product_formulas() {
        // K=1: first element is just multiplication
        assert_eq!(CostModel::dot_product_naive(1), 17);
        // K=2: 36*2-19=53
        assert_eq!(CostModel::dot_product_naive(2), 53);
        // ASIS K=64: 28*64-11=1781
        assert_eq!(CostModel::dot_product_asis(64), 1781);
        // CF+ASIS K=64: 14*64-9=887
        assert_eq!(CostModel::dot_product_cf_asis(64), 887);
    }

    #[test]
    fn test_asis_reduction() {
        // ASIS should be 22.2% cheaper asymptotically
        let naive = CostModel::dot_product_naive(1024) as f64;
        let asis = CostModel::dot_product_asis(1024) as f64;
        let reduction = (naive - asis) / naive;
        assert!((reduction - 0.222).abs() < 0.01);
    }

    #[test]
    fn test_log_softmax_cheaper_than_softmax() {
        let n = 2048;
        assert!(CostModel::log_softmax_dag(n) < CostModel::softmax_dag(n));
        // Log-Softmax: 57327, Softmax: 71663
        assert_eq!(CostModel::log_softmax_dag(n), 57327);
        assert_eq!(CostModel::softmax_dag(n), 71663); // 35*2048-17
    }

    #[test]
    fn test_silu_equals_sigmoid() {
        assert_eq!(CostModel::silu_cost(), CostModel::sigmoid_cost());
        assert_eq!(CostModel::silu_cost(), 51);
    }

    #[test]
    fn test_tinyllama_attention_softmax_fraction() {
        let seq = 2048;
        let dk = 64;
        let total = CostModel::attention_one_head(seq, dk);
        let softmax = seq * CostModel::softmax_dag(seq);
        let fraction = softmax as f64 / total as f64;
        // Softmax should be < 1% of total cost
        assert!(fraction < 0.01,
            "Softmax fraction: {:.4} (expected < 0.01)", fraction);
    }

    #[test]
    fn test_tinyllama_layer_reduction() {
        assert!((CostModel::tinyllama_layer_reduction() - 63.1).abs() < 0.1);
    }

    #[test]
    fn test_rope_cf_cost() {
        assert_eq!(CostModel::rope_pair_cost_cf(), 50);
        assert_eq!(CostModel::rope_element_cost_cf(), 25);
    }

    #[test]
    fn test_swiglu_fused_cheaper() {
        // SwiGLU fused: 32 nodes < naive: ~68 nodes
        assert!(CostModel::swiglu_fused_cost() < 68);
    }

    #[test]
    fn test_residual_fused_one_node() {
        assert_eq!(CostModel::residual_fused_cost(), 1);
    }

    #[test]
    fn test_attention_lower_bound() {
        // For TinyLlama: n=2048, d=64
        let lb = CostModel::attention_lower_bound(2048, 64);
        assert_eq!(lb, 2048 * 2048 * 64); // 268,435,456
    }
}
