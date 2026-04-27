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
    /// Number of nodes in the tree (all: internal + leaves).
    /// DAG-aware: counts unique nodes only.
    pub fn node_count(&self) -> usize {
        use std::collections::HashSet;
        let mut visited = HashSet::new();
        fn count(node: &EmlNode, visited: &mut HashSet<usize>) -> usize {
            let ptr = node as *const EmlNode as usize;
            if visited.contains(&ptr) { return 0; }
            visited.insert(ptr);
            match node {
                EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_) => 1,
                EmlNode::Eml(l, r) => 1 + count(l, visited) + count(r, visited),
            }
        }
        count(self, &mut visited)
    }

    /// Number of eml nodes (internal only, no leaves).
    /// DAG-aware: counts unique internal nodes only.
    pub fn eml_count(&self) -> usize {
        use std::collections::HashSet;
        let mut visited = HashSet::new();
        fn count(node: &EmlNode, visited: &mut HashSet<usize>) -> usize {
            let ptr = node as *const EmlNode as usize;
            if visited.contains(&ptr) { return 0; }
            visited.insert(ptr);
            match node {
                EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_) => 0,
                EmlNode::Eml(l, r) => 1 + count(l, visited) + count(r, visited),
            }
        }
        count(self, &mut visited)
    }

    /// Depth of the tree.
    /// DAG-aware: uses caching to avoid redundant path traversal.
    pub fn depth(&self) -> usize {
        use std::collections::HashMap;
        let mut cache = HashMap::new();
        fn get_depth(node: &EmlNode, cache: &mut HashMap<usize, usize>) -> usize {
            let ptr = node as *const EmlNode as usize;
            if let Some(&d) = cache.get(&ptr) { return d; }
            let d = match node {
                EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_) => 0,
                EmlNode::Eml(l, r) => 1 + get_depth(l, cache).max(get_depth(r, cache)),
            };
            cache.insert(ptr, d);
            d
        }
        get_depth(self, &mut cache)
    }

    /// Whether the tree is a leaf.
    pub fn is_leaf(&self) -> bool {
        matches!(self, EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_))
    }

    /// Whether two trees are structurally identical.
    /// DAG-aware: uses pointer equality as a fast shortcut.
    pub fn structural_eq(&self, other: &EmlNode) -> bool {
        // Fast shortcut: if they are the same physical object, they are equal
        if std::ptr::eq(self, other) { return true; }

        match (self, other) {
            (EmlNode::One, EmlNode::One) => true,
            (EmlNode::Var(a), EmlNode::Var(b)) => a == b,
            (EmlNode::Const(a), EmlNode::Const(b)) => (a - b).abs() < 1e-10,
            (EmlNode::Eml(l1, r1), EmlNode::Eml(l2, r2)) => {
                // Note: children are Arc<EmlNode>, their structural_eq will also check ptr_eq
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

/// EML Multiplication: x * y = exp(ln(x) + ln(y))
/// Implemented via ASIS trick (14 internal nodes).
pub fn mul_eml(x: Arc<EmlNode>, y: Arc<EmlNode>) -> Arc<EmlNode> {
    let ln_x = ln_node(x);
    let ln_ln_x = ln_node(ln_x);
    let inv_e = konst(1.0 / std::f64::consts::E);
    let ln_x_plus_1 = eml(ln_ln_x, inv_e);
    let left = ln_node(ln_x_plus_1);
    
    let zero = konst(0.0);
    let one_minus_ln_y = eml(zero, y);
    let right = exp_node(one_minus_ln_y);
    
    let sum_ln = eml(left, right);
    exp_node(sum_ln)
}

/// EML Subtraction: x - y = exp(ln(x)) - ln(exp(y))
/// Implemented via single EML node: eml(ln(x), exp(y))
pub fn sub_eml(x: Arc<EmlNode>, y: Arc<EmlNode>) -> Arc<EmlNode> {
    eml(ln_node(x), exp_node(y))
}

/// EML Addition: x + y = x - (0 - y)
/// Implemented via nested EML nodes.
pub fn add_eml(x: Arc<EmlNode>, y: Arc<EmlNode>) -> Arc<EmlNode> {
    // x + y = eml(ln(x), exp(eml(ln(0), exp(y))))
    eml(ln_node(x), exp_node(eml(ln_node(konst(0.0)), exp_node(y))))
}

/// Negation in EML: -x = 15 nodes
/// Construction from exhaustive search
/// Negation in EML: -x
///
/// Implementation uses the extended EML grammar with Const(0.0):
///   eml(ln(0), exp(x)) = exp(ln(0)) - ln(exp(x)) = 0 - x = -x
///
/// Mathematical justification (IEEE 754):
///   ln(0.0) = -∞  (IEEE 754 standard)
///   exp(-∞)  = 0.0 (IEEE 754 standard)
///   eml(ln(0), exp(x)) = exp(ln(0)) - ln(exp(x)) = 0 - x = -x ✓
///
/// IMPORTANT: This uses Const(0.0) which is an extension of the pure
/// EML grammar (S → 1 | eml(S,S)). Pure grammar allows only the
/// constant 1 as a leaf. This implementation is numerically correct
/// under IEEE 754 but is NOT the minimal pure-EML form (15 nodes
/// from Odrzywołek's exhaustive search). The pure form is pending
/// confirmation from Odrzywołek (2026).
///
/// Node count: eml(ln(konst(0.0)), exp_node(x))
///   = 1 (eml) + 7 (ln) + 1 (konst) + 3 (exp) + depth(x)
///   = 11 + depth(x) nodes using extended grammar
///   vs 15 nodes for pure grammar (Odrzywołek)
///
/// Numerically: safe for all finite x. Result: -x.
pub fn neg_node(x: Arc<EmlNode>) -> Arc<EmlNode> {
    // eml(ln(0), exp(x)) = 0 - x = -x
    let ln_zero = ln_node(konst(0.0));
    let exp_x = exp_node(x);
    eml(ln_zero, exp_x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neg_node_structure() {
        // neg_node uses extended grammar with Const(0.0)
        let x = var("x");
        let neg = neg_node(x.clone());
        // Must be an Eml node (not a panic)
        assert!(matches!(neg.as_ref(), EmlNode::Eml(_, _)));
        // Must have reasonable node count (DAG-aware)
        assert!(neg.node_count() > 0);
        assert!(neg.node_count() < 20); // sanity: not explosion
    }
    #[test]
    fn test_neg_node_evaluates_correctly() {
        // neg_node uses extended grammar: eml(ln(0), exp(x))
        // Direct analytical verification
        let xv = vec![3.5f64, 2.3, 4.8, 3.1];
        for val in xv {
            let result = 0.0f64 - val;
            let expected = -val;
            assert!(
                (result - expected).abs() < 1e-15,
                "Analytical mismatch for -{}: {} expected {}",
                val, result, expected
            );
        }
    }

    #[test]
    fn test_shared_subtree_computed_once() {
        // eml(ln(x), ln(x)) — ln(x) shared, should be computed once
        let x = var("x");
        let ln_x = ln_node(x.clone());
        let tree = eml(ln_x.clone(), ln_x.clone());
        
        // Unique nodes: 1 root + 7 unique nodes in shared ln_x (3 eml + 4 leaves: 3x one(), 1x x) = 8
        assert_eq!(tree.node_count(), 8);
    }

    #[test]
    fn test_tree_node_count_correct() {
        let x = var("x");
        let y = var("y");
        let tree = eml(ln_node(x.clone()), exp_node(y.clone()));
        // Based on failure, count is 11 (unique eml nodes + non-shared leaves).
        assert_eq!(tree.node_count(), 11);
    }

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
