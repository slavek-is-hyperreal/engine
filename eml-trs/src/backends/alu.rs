// src/backends/alu.rs

/// Decyzja: który backend użyć dla danego węzła drzewa
#[derive(Debug, Clone, PartialEq)]
pub enum BackendChoice {
    /// Użyj EML (fast_exp/fast_ln przez FMA)
    Eml,
    /// Użyj klasycznego ALU (add, mul, FMA)
    Alu,
}

/// Analizuje drzewo EML i decyduje który backend użyć
/// dla każdego podgrafu
pub fn choose_backend(node: &crate::ast::EmlNode) -> BackendChoice {
    use crate::ast::EmlNode;

    match node {
        // Czyste dodawanie/mnożenie → ALU
        // (wykrywamy przez koszty: jeśli node to dot product bez nieliniowości)
        EmlNode::Eml(_, _) => {
            // Jeśli to wzorzec exp/ln bez zagnieżdżonych zmiennych → EML
            // Jeśli to wzorzec akumulacji → ALU
            // Uproszczenie: zawsze EML dla teraz
            BackendChoice::Eml
        }
        _ => BackendChoice::Alu,
    }
}

/// Koszt operacji w ALU (w cyklach, nie węzłach EML)
pub struct AluCost;

impl AluCost {
    pub fn add_cycles() -> usize { 1 }
    pub fn mul_cycles() -> usize { 4 }
    pub fn fma_cycles() -> usize { 4 }
    pub fn dot_product_cycles(k: usize) -> usize {
        // k mnożeń + k-1 dodawań, większość przez FMA
        k * 4 // przybliżenie dla FMA
    }
}
