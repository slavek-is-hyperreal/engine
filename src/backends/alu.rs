// src/backends/alu.rs

/// Choice of backend for a given tree node.
#[derive(Debug, Clone, PartialEq)]
pub enum BackendChoice {
    /// Use EML (fast_exp/fast_ln via FMA)
    Eml,
    /// Use classic ALU (add, mul, FMA)
    Alu,
}

/// Analyzes EML tree and decides which backend to use
/// for each subgraph.
pub fn choose_backend(node: &crate::ast::EmlNode) -> BackendChoice {
    use crate::ast::EmlNode;

    match node {
        // Pure addition/multiplication -> ALU
        // (Detected via cost: if node is dot product without non-linearities)
        EmlNode::Eml(_, _) => {
            // If it's an exp/ln pattern without nested variables -> EML
            // If it's an accumulation pattern -> ALU
            // Simplification: always EML for now
            BackendChoice::Eml
        }
        _ => BackendChoice::Alu,
    }
}

/// Operation cost in ALU (in cycles, not EML nodes)
pub struct AluCost;

impl AluCost {
    pub fn add_cycles() -> usize { 1 }
    pub fn mul_cycles() -> usize { 4 }
    pub fn fma_cycles() -> usize { 4 }
    pub fn dot_product_cycles(k: usize) -> usize {
        // k multiplications + k-1 additions, mostly via FMA
        k * 4 // approximation for FMA
    }
}
