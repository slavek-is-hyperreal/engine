// src/asis.rs
//
// CONVENTION: eml_count() = number of internal Eml(l,r) nodes
//             node_count() = eml_count() + number of leaves
// Costs from exhaustive search (Odrzywołek paper) = node_count()
// Costs in this project's tests = eml_count() (internal only)

use crate::ast::*;
// use crate::cost_model::CostModel; // Unused import
use std::sync::Arc;

/// Builds EML tree for ASIS dot product
/// inputs: input variables [x1, x2, ..., xK]
/// weights: weights [w1, -w2, -w3, ..., -wK] (pre-negated by CF)
///
/// Result: A₁B₁ - Ã₂B₂ - Ã₃B₃ - ...
/// where Ãₖ = -Aₖ for k≥2 (pre-negated offline)
pub fn build_asis_dot_product(
    inputs: &[Arc<EmlNode>],
    weights: &[Arc<EmlNode>],
) -> Arc<EmlNode> {
    assert_eq!(inputs.len(), weights.len());
    assert!(!inputs.is_empty());

    // Step 1: first product A₁B₁
    let first = mul_eml(inputs[0].clone(), weights[0].clone());

    // Steps 2..K: accumulation via subtraction (ASIS)
    // C = A₁B₁ - Ã₂B₂ - Ã₃B₃ - ...
    inputs[1..].iter().zip(weights[1..].iter())
        .fold(first, |acc, (x, w)| {
            let product = mul_eml(x.clone(), w.clone());
            sub_eml(acc, product)  // subtraction instead of addition!
        })
}

/// Verifies that ASIS yields the same result as naive dot product
/// (for specific numerical values)
pub fn verify_asis_correctness(
    inputs: &[f64],
    weights: &[f64],
) -> bool {
    use crate::constant_fold::asis_preprocess_weights;

    // Naive: Σ(aᵢ * bᵢ)
    let naive: f64 = inputs.iter().zip(weights.iter())
        .map(|(a, b)| a * b)
        .sum();

    // ASIS: A₁B₁ - (-A₂)B₂ - (-A₃)B₃ - ...
    let asis_weights = asis_preprocess_weights(weights);
    let first = inputs[0] * asis_weights[0];
    let asis: f64 = inputs[1..].iter().zip(asis_weights[1..].iter())
        .fold(first, |acc, (x, w)| acc - (x * w));

    (naive - asis).abs() < 1e-10
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asis_correctness_k2() {
        assert!(verify_asis_correctness(&[1.0, 2.0], &[3.0, 4.0]));
        // 1*3 + 2*4 = 11
        // ASIS: 1*3 - (-2)*4 = 3 - (-8) = 11 ✓
    }

    #[test]
    fn test_asis_correctness_k4() {
        let inputs = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![0.5, 1.5, 2.5, 3.5];
        assert!(verify_asis_correctness(&inputs, &weights));
    }

    #[test]
    fn test_asis_tree_size() {
        // ASIS tree K=2 should be smaller than naive
        let inputs: Vec<_> = (0..2).map(|i| var(&format!("x{}", i))).collect();
        let weights: Vec<_> = (0..2).map(|i| var(&format!("w{}", i))).collect();
        let tree = build_asis_dot_product(&inputs, &weights);
        let asis_cost = tree.eml_count();
        // ASIS K=2 with mul_eml=14 nodes: A1*B1(14) - A2*B2(14)
        // sub = 11, total = 14 + 14 + 11 = 39. (Naive ~ 53)
        assert!(asis_cost <= 53, "ASIS cost {} > naive 53", asis_cost);
    }

    #[test]
    fn test_mul_eml_correct() {
        use crate::constant_fold::try_evaluate;
        use crate::constant_fold::ConstantMap;

        let x = var("x");
        let _y = var("y");
        // Use canonical mul_cf from ast.rs (PAPER.md §4.2, Theorem 3)
        let tree_xy = eml(
            mul_cf(x, 1.0), // x * 1.0 as a proxy — mul_cf(x, w) not mul_cf(x,y)
            one(),
        );
        // Direct: mul_cf(x, 4.0) for x=3.0 → should give 12.0
        let tree = mul_cf(var("x"), 4.0);
        let mut consts = ConstantMap::new();
        consts.insert("x".to_string(), 3.0);

        let result = try_evaluate(&tree, &consts).unwrap();
        assert!((result - 12.0).abs() < 1e-8, "Expected 3*4=12, got {}", result);
        let _ = tree_xy; // suppress unused warning
    }

    #[test]
    fn test_mul_eml_node_count() {
        // mul_cf runtime tree has 8 internal eml nodes:
        //   eml(eml(ln(ln(x)), Const(1/w)), one())
        //   = 1(root) + 1(inner eml) + 3(ln(ln(x))) + 3(ln(x)) ← wait, ln_node adds 3 each
        //   Actual count: ln(x)=3, ln(ln(x))=6, eml(lnln,inv)=7, eml(…,1)=8
        // The "5 nodes" from PAPER.md §4.2 is the *effective* cost when
        // ln(ln(x)) is precomputed offline as a constant (== treated as a leaf).
        // At runtime the full tree is 8 internal nodes. See also constant_fold.rs:134.
        let tree = mul_cf(var("x"), 2.0);
        assert_eq!(tree.eml_count(), 8, "mul_cf runtime tree has {} eml nodes, expected 8", tree.eml_count());
    }


    #[test]
    fn test_mul_eml_positive_only() {
        // NOTE: mul_cf works only for x > 0 (PAPER.md §4.2 precondition)
        // For negative activations use ALU backend
        use crate::constant_fold::try_evaluate;
        use crate::constant_fold::ConstantMap;

        let mut consts = ConstantMap::new();
        consts.insert("x".to_string(), -2.0);
        let tree = mul_cf(var("x"), 3.0);
        // We expect None because ln(ln(-2.0)) is NaN
        assert!(try_evaluate(&tree, &consts).is_none(),
            "mul_cf does not work for negative x — use ALU backend");
    }
}
