// src/ast.rs
//
// CONVENTION: eml_count() = number of internal Eml(l,r) nodes
//             node_count() = eml_count() + number of leaves
// Costs from exhaustive search (Odrzywołek paper) = node_count()
// Costs in this project's tests = eml_count() (internal only)

use std::sync::Arc;
// HashMap removed

/// EML tree node.
/// Grammar: S → 1 | eml(S, S)
#[derive(Debug, Clone, PartialEq)]
pub enum EmlNode {
    /// Leaf: constant 1
    One,
    /// Leaf: named variable (e.g. "x", "w1", "a_0_1")
    Var(String),
    /// Leaf: numeric constant (weight after constant folding)
    Const(f64),
    /// Internal node: eml(left, right)
    Eml(Arc<EmlNode>, Arc<EmlNode>),
}

impl EmlNode {
    /// Number of nodes in the tree (all: internal + leaves)
    pub fn node_count(&self) -> usize {
        match self {
            EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_) => 1,
            EmlNode::Eml(l, r) => 1 + l.node_count() + r.node_count(),
        }
    }

    /// Number of eml nodes (internal only, no leaves)
    pub fn eml_count(&self) -> usize {
        match self {
            EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_) => 0,
            EmlNode::Eml(l, r) => 1 + l.eml_count() + r.eml_count(),
        }
    }

    /// Depth of the tree
    pub fn depth(&self) -> usize {
        match self {
            EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_) => 0,
            EmlNode::Eml(l, r) => 1 + l.depth().max(r.depth()),
        }
    }

    /// Whether the tree is a leaf
    pub fn is_leaf(&self) -> bool {
        matches!(self, EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_))
    }

    /// Whether two trees are structurally identical
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

/// Helper constructors — use these instead of writing Arc::new manually
pub fn one() -> Arc<EmlNode> { Arc::new(EmlNode::One) }
pub fn var(name: &str) -> Arc<EmlNode> { Arc::new(EmlNode::Var(name.to_string())) }
pub fn konst(v: f64) -> Arc<EmlNode> { Arc::new(EmlNode::Const(v)) }
pub fn eml(l: Arc<EmlNode>, r: Arc<EmlNode>) -> Arc<EmlNode> {
    Arc::new(EmlNode::Eml(l, r))
}

/// Macros for basic operations — built from eml(x,y) nodes
/// exp(x) = eml(x, 1)
pub fn exp_node(x: Arc<EmlNode>) -> Arc<EmlNode> {
    eml(x, one())
}

/// ln(x) = eml(1, eml(eml(1, x), 1))
pub fn ln_node(x: Arc<EmlNode>) -> Arc<EmlNode> {
    eml(one(), eml(eml(one(), x), one()))
}

/// Negation in EML: -x = 15 nodes
/// Construction from exhaustive search
pub fn neg_node(_x: Arc<EmlNode>) -> Arc<EmlNode> {
    // -x = eml(ln(1/e), x) where ln(1/e) = -1
    // = eml(eml(1, eml(eml(1,1),1)), x) ... verify
    // Temporarily via constant:
    // eml(Const(-1.0_as_EML), exp(x))
    // Correct 15-node form from exhaustive search:
    // Pending: awaiting exhaustive search result from Odrzywołek (2026).
    panic!("neg_node: unimplemented. Use asis_preprocess_weights() offline.")
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
        // exp(x) = eml(x, 1) = 3 nodes: eml + x + 1
        let e = exp_node(var("x"));
        assert_eq!(e.node_count(), 3);
        assert_eq!(e.eml_count(), 1);
    }

    #[test]
    fn test_ln_node_cost() {
        // ln(x) = 7 nodes
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
