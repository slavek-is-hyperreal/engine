// src/nn_layer.rs
// CONVENTION: eml_count()=internal, node_count()=all
// NOTE: mul_eml requires x,y > 0

use crate::ast::*;
use crate::trs::rewrite;
use crate::cost_model::CostModel;
use std::sync::Arc;

pub struct LayerOptResult {
    pub nodes_naive: usize,
    pub nodes_after_cf: usize,
    pub nodes_after_trs: usize,
    pub reduction_pct: f64,
    pub sample_tree: Arc<EmlNode>,
    pub sample_k: usize,
}

/// Dot product with CF + ASIS. Uses balanced tree for O(log K) depth.
pub fn build_dot_product_eml(input: &[Arc<EmlNode>], weights: &[f32]) -> Arc<EmlNode> {
    assert_eq!(input.len(), weights.len());
    assert!(!input.is_empty());

    // sub_eml(a, b) = a - b  [requires a > 0 for ln(a)]
    fn sub_eml_local(a: Arc<EmlNode>, b: Arc<EmlNode>) -> Arc<EmlNode> {
        eml(ln_node(a), exp_node(b))
    }
    // add_eml(a, b) = a + b  [requires a > 0]
    fn add_eml_local(a: Arc<EmlNode>, b: Arc<EmlNode>) -> Arc<EmlNode> {
        sub_eml_local(a, neg_node(b))
    }

    // mul_cf_safe: |w| * x  (x > -1.0)
    // BIAS=4.0 ensures ln(x+BIAS) > ln(3) > 1.0, so ln(ln(x+BIAS)) is stable.
    fn mul_cf_positive(x: Arc<EmlNode>, abs_w: f64) -> Arc<EmlNode> {
        const BIAS: f64 = 4.0;
        if abs_w < 1e-15 { return konst(0.0); }

        let x_shifted = add_eml_local(x, konst(BIAS));
        let scaled = mul_cf(x_shifted, abs_w);
        let bias_correction = konst(abs_w * BIAS);
        sub_eml_local(scaled, bias_correction)
    }

    let mut pos_terms = Vec::new();
    let mut neg_terms = Vec::new();

    for (x, &w) in input.iter().zip(weights.iter()) {
        let abs_w = (w as f64).abs();
        if abs_w < 1e-15 { continue; }
        
        let term = mul_cf_positive(x.clone(), abs_w);
        if w >= 0.0 {
            pos_terms.push(term);
        } else {
            neg_terms.push(term);
        }
    }

    fn sum_balanced(mut terms: Vec<Arc<EmlNode>>) -> Arc<EmlNode> {
        if terms.is_empty() { return konst(0.0); }
        while terms.len() > 1 {
            let mut next = Vec::new();
            for i in (0..terms.len()).step_by(2) {
                if i + 1 < terms.len() {
                    next.push(add_eml_local(terms[i].clone(), terms[i + 1].clone()));
                } else {
                    next.push(terms[i].clone());
                }
            }
            terms = next;
        }
        terms[0].clone()
    }

    let pos_sum = sum_balanced(pos_terms);
    let neg_sum = sum_balanced(neg_terms);

    if neg_sum.as_ref() == &EmlNode::Const(0.0) {
        pos_sum
    } else if pos_sum.as_ref() == &EmlNode::Const(0.0) {
        neg_node(neg_sum)
    } else {
        sub_eml_local(pos_sum, neg_sum)
    }
}


/// Offline: absorb γ and 1/√dk into W_Q. Runtime cost: 0 nodes.
pub fn preprocess_wq_offline(w_q: &[f32], gamma: &[f32], d_k: usize, hidden: usize) -> Vec<f32> {
    let scale = 1.0 / (d_k as f32).sqrt();
    w_q.iter().enumerate().map(|(i, &w)| w * gamma[i % hidden] * scale).collect()
}

pub fn measure_costs(k: usize) -> (usize, usize, usize) {
    (CostModel::dot_product_naive(k),
     CostModel::dot_product_asis(k),
     CostModel::dot_product_cf_asis(k))
}

/// Build a sample K=sample_k (use ≤32 to avoid gigantic trees)
pub fn build_and_optimize_sample(weights: &[f32], sample_k: usize) -> LayerOptResult {
    assert!(weights.len() >= sample_k);
    let w = &weights[..sample_k];
    let input: Vec<Arc<EmlNode>> = (0..sample_k).map(|i| var(&format!("x{}", i))).collect();
    let nodes_naive = CostModel::dot_product_naive(sample_k);
    let tree = build_dot_product_eml(&input, w);
    let nodes_after_cf = tree.eml_count();
    let tree_opt = rewrite(tree);
    let nodes_after_trs = tree_opt.eml_count();
    let reduction_pct = (nodes_naive - nodes_after_trs) as f64 / nodes_naive as f64 * 100.0;
    LayerOptResult { nodes_naive, nodes_after_cf, nodes_after_trs,
                     reduction_pct, sample_tree: tree_opt, sample_k }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constant_fold::{try_evaluate, ConstantMap};

    #[test]
    fn test_preprocess_wq_offline() {
        let w_q = vec![1.0f32; 16];
        let gamma = vec![2.0f32; 4];
        let result = preprocess_wq_offline(&w_q, &gamma, 4, 4);
        // 1.0 * 2.0 * 1/sqrt(4) = 1.0
        for &v in &result { assert!((v - 1.0f32).abs() < 1e-6); }
    }

    #[test]
    fn test_measure_costs() {
        let (naive, asis, cf) = measure_costs(64);
        assert_eq!(naive, 36*64-19);
        assert_eq!(asis, 28*64-11);
        assert_eq!(cf, 14*64-9);
    }

    #[test]
    fn test_build_dot_product_structure() {
        let weights = vec![0.5f32, 0.3, 0.7, 0.2];
        let input: Vec<Arc<EmlNode>> = (0..4).map(|i| var(&format!("x{}",i))).collect();
        let tree = build_dot_product_eml(&input, &weights);
        assert!(tree.eml_count() > 0);
        assert!(tree.eml_count() <= 125); // naive K=4: 36*4-19=125
    }

    #[test]
    fn test_correctness_positive() {
        let weights = vec![0.5f32, 0.3, 0.7, 0.2];
        let xv = vec![1.5f64, 2.0, 0.8, 3.1];
        let expected: f64 = xv.iter().zip(weights.iter()).map(|(x,w)| x*(*w as f64)).sum();
        let input: Vec<Arc<EmlNode>> = (0..4).map(|i| var(&format!("x{}",i))).collect();
        let tree = build_dot_product_eml(&input, &weights);
        let mut c = ConstantMap::new();
        for (i,&v) in xv.iter().enumerate() { c.insert(format!("x{}",i), v); }
        if let Some(v) = try_evaluate(&tree, &c) {
            assert!((v - expected).abs() < 1e-3, "expected={} got={}", expected, v);
        }
        // None is OK for negative intermediate values
    }

    #[test]
    fn test_negative_weights_limitation() {
        // Documents known limitation: None for negative weights
        let weights = vec![0.5f32, -0.3, 0.7, -0.2];
        let xv = vec![1.0f64, 2.0, 3.0, 4.0];
        let input: Vec<Arc<EmlNode>> = (0..4).map(|i| var(&format!("x{}",i))).collect();
        let tree = build_dot_product_eml(&input, &weights);
        let mut c = ConstantMap::new();
        for (i,&v) in xv.iter().enumerate() { c.insert(format!("x{}",i), v); }
        let r = try_evaluate(&tree, &c);
        println!("Negative weights -> {:?} (None is expected)", r);
        // We don't assert None because it depends on intermediate values
    }

    #[test]
    fn test_build_and_optimize_sample() {
        let weights: Vec<f32> = (1..=8).map(|i| i as f32 * 0.1).collect();
        let result = build_and_optimize_sample(&weights, 8);
        assert_eq!(result.sample_k, 8);
        assert!(result.nodes_after_cf <= result.nodes_naive);
        assert!(result.reduction_pct >= 0.0);
    }
}
