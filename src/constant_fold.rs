// src/constant_fold.rs
//
// CONVENTION: eml_count() = number of internal Eml(l,r) nodes
//             node_count() = eml_count() + number of leaves
// Costs from exhaustive search (Odrzywołek paper) = node_count()
// Costs in this project's tests = eml_count() (internal only)

use crate::ast::*;
use std::sync::Arc;
use std::collections::HashMap;

/// Map of variable names to constant values (weights, parameters)
pub type ConstantMap = HashMap<String, f64>;

/// Evaluates the EML tree if all leaves are known.
/// Returns Some(value) if the entire subtree is constant.
pub fn try_evaluate(node: &EmlNode, consts: &ConstantMap) -> Option<f64> {
    match node {
        EmlNode::One => Some(1.0),
        EmlNode::Const(v) => Some(*v),
        EmlNode::Var(name) => consts.get(name).copied(),
        EmlNode::Eml(l, r) => {
            let lv = try_evaluate(l, consts)?;
            let rv = try_evaluate(r, consts)?;
            // eml(x, y) = exp(x) - ln(y)
            // We allow rv = 0.0 because ln(0) = -inf is handled by f64
            // and exp(-inf) = 0.0, which is used in neg_node.
            // However, rv < 0.0 is still undefined in pure EML.
            if rv < 0.0 { return None; }
            Some(lv.exp() - rv.ln())
        }
    }
}

/// Recursive constant folding.
/// Replaces subtrees with constant values where possible.
pub fn fold_constants(node: Arc<EmlNode>, consts: &ConstantMap) -> Arc<EmlNode> {
    // Check if the entire subtree can be collapsed
    if let Some(value) = try_evaluate(&node, consts) {
        return Arc::new(EmlNode::Const(value));
    }

    // If not — recurse on children
    match node.as_ref() {
        EmlNode::Eml(l, r) => {
            let new_l = fold_constants(l.clone(), consts);
            let new_r = fold_constants(r.clone(), consts);
            eml(new_l, new_r)
        }
        // Leaves that are not constant — replace Var with Const if known
        EmlNode::Var(name) => {
            if let Some(&v) = consts.get(name) {
                Arc::new(EmlNode::Const(v))
            } else {
                node.clone()
            }
        }
        _ => node.clone(),
    }
}

/// Constant weight multiplication optimization:
/// x * W → 5-node structure when W is constant.
/// Naive: x * W = 17 nodes.
/// With CF: eml(eml(ln(ln(x)), Const(1/W)), 1) = 5 nodes.
/// (ln(ln(x)) precomputed, Const(1/W) is a constant leaf)
///
/// NOTE: Requires ln(x) > 0, meaning x > 1.
/// For negative data, use standard multiplication.
pub fn mul_with_const_weight(x: Arc<EmlNode>, w: f64) -> Arc<EmlNode> {
    assert!(w != 0.0, "Weight cannot be zero");
    let inv_w = Arc::new(EmlNode::Const(1.0 / w));
    // eml(eml(ln(ln(x)), 1/w), 1)
    // = exp(ln(ln(x)) - ln(1/w)) - ln(1)
    // = exp(ln(ln(x)) + ln(w))
    // = exp(ln(x * ... )) ... verify algebraically
    eml(eml(ln_node(ln_node(x)), inv_w), one())
}

/// ASIS preprocessing: negate weights[1..] offline.
/// Returns a new vector of weights ready for ASIS dot product.
pub fn asis_preprocess_weights(weights: &[f64]) -> Vec<f64> {
    weights.iter().enumerate().map(|(i, &w)| {
        if i == 0 { w } else { -w }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_constant_tree() {
        // eml(1, 1) = e
        let tree = eml(one(), one());
        let consts = ConstantMap::new();
        let result = try_evaluate(&tree, &consts);
        assert!((result.unwrap() - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_partial_fold() {
        // eml(x, 1) where x=2.0 → exp(2.0)
        let tree = eml(var("x"), one());
        let mut consts = ConstantMap::new();
        consts.insert("x".to_string(), 2.0);
        let folded = fold_constants(tree, &consts);
        // Should be Const(exp(2.0))
        if let EmlNode::Const(v) = folded.as_ref() {
            assert!((v - 2.0_f64.exp()).abs() < 1e-10);
        } else {
            panic!("Expected Const, got {:?}", folded);
        }
    }

    #[test]
    fn test_asis_preprocess() {
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let processed = asis_preprocess_weights(&weights);
        assert_eq!(processed[0], 1.0);
        assert_eq!(processed[1], -2.0);
        assert_eq!(processed[2], -3.0);
        assert_eq!(processed[3], -4.0);
    }

    #[test]
    fn test_mul_reduction() {
        let x = var("x");
        let tree = mul_with_const_weight(x, 2.0);
        // Structure: eml(eml(ln(ln(x)), Const(0.5)), 1)
        // eml_count = 8 (full tree)
        // In practice: ln(ln(x)) is precomputed offline -> effective cost = 5
        // (2 eml nodes for outer structure + 3 for ln(ln(x)) treated as leaf/Var)
        assert_eq!(tree.eml_count(), 8);
    }
}
