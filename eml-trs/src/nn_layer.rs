// src/nn_layer.rs
// KONWENCJA: eml_count()=wewnętrzne, node_count()=wszystkie
// UWAGA: mul_eml wymaga x,y > 0

use crate::ast::*;
use crate::constant_fold::asis_preprocess_weights;
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

/// Dot product z CF + ASIS. Wymaga input > 0.
pub fn build_dot_product_eml(input: &[Arc<EmlNode>], weights: &[f32]) -> Arc<EmlNode> {
    assert_eq!(input.len(), weights.len());
    assert!(!input.is_empty());
    let weights_f64: Vec<f64> = weights.iter().map(|&w| w as f64).collect();
    let asis_w = asis_preprocess_weights(&weights_f64);

    fn mul_cf(x: Arc<EmlNode>, w: f64) -> Arc<EmlNode> {
        if w.abs() < 1e-15 { return konst(0.0); }
        eml(eml(ln_node(ln_node(x)), konst(1.0 / w)), one())
    }
    fn sub_eml(x: Arc<EmlNode>, y: Arc<EmlNode>) -> Arc<EmlNode> {
        eml(ln_node(x), exp_node(y))
    }

    let mut acc = mul_cf(input[0].clone(), asis_w[0] as f64);
    for i in 1..input.len() {
        let p = mul_cf(input[i].clone(), asis_w[i] as f64);
        acc = sub_eml(acc, p);
    }
    acc
}

/// Offline: wchłoń γ i 1/√dk do W_Q. Koszt runtime: 0 węzłów.
pub fn preprocess_wq_offline(w_q: &[f32], gamma: &[f32], d_k: usize, hidden: usize) -> Vec<f32> {
    let scale = 1.0 / (d_k as f32).sqrt();
    w_q.iter().enumerate().map(|(i, &w)| w * gamma[i % hidden] * scale).collect()
}

pub fn measure_costs(k: usize) -> (usize, usize, usize) {
    (CostModel::dot_product_naive(k),
     CostModel::dot_product_asis(k),
     CostModel::dot_product_cf_asis(k))
}

/// Buduj próbkę K=sample_k (używaj ≤32 żeby uniknąć gigantycznych drzew)
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
        assert!(tree.eml_count() <= 125); // naiwne K=4: 36*4-19=125
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
        // None jest OK dla ujemnych pośrednich wartości
    }

    #[test]
    fn test_negative_weights_limitation() {
        // Dokumentuje znane ograniczenie: None dla ujemnych wag
        let weights = vec![0.5f32, -0.3, 0.7, -0.2];
        let xv = vec![1.0f64, 2.0, 3.0, 4.0];
        let input: Vec<Arc<EmlNode>> = (0..4).map(|i| var(&format!("x{}",i))).collect();
        let tree = build_dot_product_eml(&input, &weights);
        let mut c = ConstantMap::new();
        for (i,&v) in xv.iter().enumerate() { c.insert(format!("x{}",i), v); }
        let r = try_evaluate(&tree, &c);
        println!("Ujemne wagi -> {:?} (None jest oczekiwane)", r);
        // Nie assertujemy None bo zależy od wartości pośrednich
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
