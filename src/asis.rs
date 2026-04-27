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

    // Multiplication macro: x * y = 14 nodes (internal)
    fn mul_eml(x: Arc<EmlNode>, y: Arc<EmlNode>) -> Arc<EmlNode> {
        // x * y = exp(ln(x) + ln(y))
        // ln(x) + ln(y) via ASIS trick:
        //   ln(x) + 1 = eml(ln(ln(x)), 1/e)   [because exp(ln(ln(x))) - ln(1/e) = ln(x)+1]
        //   1 - ln(y) = eml(0, y)              [because exp(0) - ln(y) = 1 - ln(y)]
        //   (ln(x)+1) - (1-ln(y)) = ln(x) + ln(y) ✓
        // Cost: eml_count() = 14 (internal nodes)
        // Equivalent to node_count() ≈ 29 (all nodes including leaves)
        
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

    // Subtraction macro: x - y = 11 nodes
    fn sub_eml(x: Arc<EmlNode>, y: Arc<EmlNode>) -> Arc<EmlNode> {
        eml(ln_node(x), exp_node(y))
    }

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
        let y = var("y");
        // Use the same function as inside `build_asis_dot_product`, but define it locally:
        fn mul_eml_local(x: Arc<EmlNode>, y: Arc<EmlNode>) -> Arc<EmlNode> {
            let ln_x = ln_node(x);
            let ln_ln_x = ln_node(ln_x);
            let inv_e = konst(1.0 / std::f64::consts::E);
            let ln_x_plus_1 = eml(ln_ln_x, inv_e);
            let left = ln_node(ln_x_plus_1);
            let zero = konst(0.0);
            let one_minus_ln_y = eml(zero, y);
            let right = exp_node(one_minus_ln_y);
            exp_node(eml(left, right))
        }
        
        let tree = mul_eml_local(x, y);
        let mut consts = ConstantMap::new();
        consts.insert("x".to_string(), 3.0);
        consts.insert("y".to_string(), 4.0);
        
        let result = try_evaluate(&tree, &consts).unwrap();
        assert!((result - 12.0).abs() < 1e-8, "Expected 3*4=12, got {}", result);
    }

    #[test]
    fn test_mul_eml_node_count() {
        fn mul_eml_local(x: Arc<EmlNode>, y: Arc<EmlNode>) -> Arc<EmlNode> {
            let ln_x = ln_node(x);
            let ln_ln_x = ln_node(ln_x);
            let inv_e = konst(1.0 / std::f64::consts::E);
            let left = ln_node(eml(ln_ln_x, inv_e));
            let right = exp_node(eml(konst(0.0), y));
            exp_node(eml(left, right))
        }
        let tree = mul_eml_local(var("x"), var("y"));
        assert!(tree.eml_count() <= 17, "mul_eml has {} nodes, expected <= 17", tree.eml_count());
        assert_eq!(tree.eml_count(), 14, "Optimized to 14!");
    }

    #[test]
    fn test_mul_eml_positive_only() {
        // NOTE: mul_eml works only for x,y > 0
        // For negative values use asis_preprocess_weights (offline pre-negation)
        // or classical multiplication in ALU backend
        use crate::constant_fold::try_evaluate;
        use crate::constant_fold::ConstantMap;
        
        fn mul_eml_local(x: Arc<EmlNode>, y: Arc<EmlNode>) -> Arc<EmlNode> {
            let ln_x = ln_node(x);
            let ln_ln_x = ln_node(ln_x);
            let inv_e = konst(1.0 / std::f64::consts::E);
            let ln_x_plus_1 = eml(ln_ln_x, inv_e);
            let left = ln_node(ln_x_plus_1);
            let zero = konst(0.0);
            let one_minus_ln_y = eml(zero, y);
            let right = exp_node(one_minus_ln_y);
            exp_node(eml(left, right))
        }
        
        let mut consts = ConstantMap::new();
        consts.insert("x".to_string(), -2.0);
        consts.insert("y".to_string(), 3.0);
        let tree = mul_eml_local(var("x"), var("y"));
        // We expect None because ln(-2.0) is NaN
        assert!(try_evaluate(&tree, &consts).is_none(),
            "mul_eml does not work for negative values — use ALU backend");
    }
}
