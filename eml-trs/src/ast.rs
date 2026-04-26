// src/ast.rs
//
// KONWENCJA: eml_count() = liczba węzłów wewnętrznych Eml(l,r)
//            node_count() = eml_count() + liczba liści
// Koszty z exhaustive search (paper Odrzywołka) = node_count()
// Koszty w testach tego projektu = eml_count() (tylko wewnętrzne)

use std::sync::Arc;
// HashMap usunięty

/// Węzeł drzewa EML.
/// Gramatyka: S → 1 | eml(S, S)
#[derive(Debug, Clone, PartialEq)]
pub enum EmlNode {
    /// Liść: stała 1
    One,
    /// Liść: nazwana zmienna (np. "x", "w1", "a_0_1")
    Var(String),
    /// Liść: stała numeryczna (waga po constant folding)
    Const(f64),
    /// Węzeł wewnętrzny: eml(lewy, prawy)
    Eml(Arc<EmlNode>, Arc<EmlNode>),
}

impl EmlNode {
    /// Liczba węzłów w drzewie (wszystkich: wewnętrznych + liści)
    pub fn node_count(&self) -> usize {
        match self {
            EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_) => 1,
            EmlNode::Eml(l, r) => 1 + l.node_count() + r.node_count(),
        }
    }

    /// Liczba węzłów eml (tylko wewnętrznych, nie liści)
    pub fn eml_count(&self) -> usize {
        match self {
            EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_) => 0,
            EmlNode::Eml(l, r) => 1 + l.eml_count() + r.eml_count(),
        }
    }

    /// Głębokość drzewa
    pub fn depth(&self) -> usize {
        match self {
            EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_) => 0,
            EmlNode::Eml(l, r) => 1 + l.depth().max(r.depth()),
        }
    }

    /// Czy drzewo jest liściem
    pub fn is_leaf(&self) -> bool {
        matches!(self, EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_))
    }

    /// Czy dwa drzewa są strukturalnie identyczne
    pub fn structural_eq(&self, other: &EmlNode) -> bool {
        match (self, other) {
            (EmlNode::One, EmlNode::One) => true,
            (EmlNode::Var(a), EmlNode::Var(b)) => a == b,
            (EmlNode::Const(a), EmlNode::Const(b)) => (a - b).abs() < 1e-10,
            (EmlNode::Eml(l1, r1), EmlNode::Eml(l2, r2)) => {
                l1.structural_eq(l2) && r1.structural_eq(r2)
            }
            _ => false,
        }
    }
}

/// Konstruktory pomocnicze — używaj tych zamiast pisać Arc::new ręcznie
pub fn one() -> Arc<EmlNode> { Arc::new(EmlNode::One) }
pub fn var(name: &str) -> Arc<EmlNode> { Arc::new(EmlNode::Var(name.to_string())) }
pub fn konst(v: f64) -> Arc<EmlNode> { Arc::new(EmlNode::Const(v)) }
pub fn eml(l: Arc<EmlNode>, r: Arc<EmlNode>) -> Arc<EmlNode> {
    Arc::new(EmlNode::Eml(l, r))
}

/// Makra dla podstawowych operacji — zbudowane z węzłów eml(x,y)
/// exp(x) = eml(x, 1)
pub fn exp_node(x: Arc<EmlNode>) -> Arc<EmlNode> {
    eml(x, one())
}

/// ln(x) = eml(1, eml(eml(1, x), 1))
pub fn ln_node(x: Arc<EmlNode>) -> Arc<EmlNode> {
    eml(one(), eml(eml(one(), x), one()))
}

/// Negacja w EML: -x = 15 węzłów
/// Konstrukcja z exhaustive search
pub fn neg_node(_x: Arc<EmlNode>) -> Arc<EmlNode> {
    // -x = eml(ln(1/e), x) gdzie ln(1/e) = -1
    // = eml(eml(1, eml(eml(1,1),1)), x) ... sprawdź
    // Tymczasowo przez stałą:
    // eml(Const(-1.0_as_EML), exp(x))
    // Prawidłowa 15-węzłowa forma z exhaustive search:
    // TODO: wyznacz z papera Odrzywołka
    panic!("neg_node: niezaimplementowane. Użyj asis_preprocess_weights() offline.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaf_count() {
        assert_eq!(one().node_count(), 1);
        assert_eq!(var("x").node_count(), 1);
    }

    #[test]
    fn test_exp_node_cost() {
        // exp(x) = eml(x, 1) = 3 węzły: eml + x + 1
        let e = exp_node(var("x"));
        assert_eq!(e.node_count(), 3);
        assert_eq!(e.eml_count(), 1);
    }

    #[test]
    fn test_ln_node_cost() {
        // ln(x) = 7 węzłów
        let l = ln_node(var("x"));
        assert_eq!(l.node_count(), 7);
        assert_eq!(l.eml_count(), 3);
    }

    #[test]
    fn test_structural_eq() {
        let a = exp_node(var("x"));
        let b = exp_node(var("x"));
        let c = exp_node(var("y"));
        assert!(a.structural_eq(&b));
        assert!(!a.structural_eq(&c));
    }
}
