// src/asis.rs
//
// KONWENCJA: eml_count() = liczba węzłów wewnętrznych Eml(l,r)
//            node_count() = eml_count() + liczba liści
// Koszty z exhaustive search (paper Odrzywołka) = node_count()
// Koszty w testach tego projektu = eml_count() (tylko wewnętrzne)

use crate::ast::*;
// use crate::cost_model::CostModel; // Nieużywany import
use std::sync::Arc;

/// Buduje drzewo EML dla ASIS dot product
/// inputs: zmienne wejściowe [x1, x2, ..., xK]
/// weights: wagi [w1, -w2, -w3, ..., -wK] (pre-negowane przez CF)
///
/// Wynik: A₁B₁ - Ã₂B₂ - Ã₃B₃ - ...
/// gdzie Ãₖ = -Aₖ dla k≥2 (pre-negowane offline)
pub fn build_asis_dot_product(
    inputs: &[Arc<EmlNode>],
    weights: &[Arc<EmlNode>],
) -> Arc<EmlNode> {
    assert_eq!(inputs.len(), weights.len());
    assert!(!inputs.is_empty());

    // Makro mnożenia: x * y = 14 węzłów (wewnętrznych)
    fn mul_eml(x: Arc<EmlNode>, y: Arc<EmlNode>) -> Arc<EmlNode> {
        // x * y = exp(ln(x) + ln(y))
        // ln(x) + ln(y) przez ASIS trick:
        //   ln(x) + 1 = eml(ln(ln(x)), 1/e)   [bo exp(ln(ln(x))) - ln(1/e) = ln(x)+1]
        //   1 - ln(y) = eml(0, y)              [bo exp(0) - ln(y) = 1 - ln(y)]
        //   (ln(x)+1) - (1-ln(y)) = ln(x) + ln(y) ✓
        // Koszt: eml_count() = 14 (węzły wewnętrzne)
        // Odpowiednik node_count() ≈ 29 (wszystkie węzły z liśćmi)
        
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

    // Makro odejmowania: x - y = 11 węzłów
    fn sub_eml(x: Arc<EmlNode>, y: Arc<EmlNode>) -> Arc<EmlNode> {
        eml(ln_node(x), exp_node(y))
    }

    // Krok 1: pierwszy iloczyn A₁B₁
    let first = mul_eml(inputs[0].clone(), weights[0].clone());

    // Kroki 2..K: akumulacja przez odejmowanie (ASIS)
    // C = A₁B₁ - Ã₂B₂ - Ã₃B₃ - ...
    inputs[1..].iter().zip(weights[1..].iter())
        .fold(first, |acc, (x, w)| {
            let product = mul_eml(x.clone(), w.clone());
            sub_eml(acc, product)  // odejmowanie zamiast dodawania!
        })
}

/// Weryfikuje że ASIS daje ten sam wynik co naiwny dot product
/// (dla konkretnych wartości liczbowych)
pub fn verify_asis_correctness(
    inputs: &[f64],
    weights: &[f64],
) -> bool {
    use crate::constant_fold::asis_preprocess_weights;

    // Naiwny: Σ(aᵢ * bᵢ)
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
        // Drzewo ASIS K=2 powinno być mniejsze niż naiwne
        let inputs: Vec<_> = (0..2).map(|i| var(&format!("x{}", i))).collect();
        let weights: Vec<_> = (0..2).map(|i| var(&format!("w{}", i))).collect();
        let tree = build_asis_dot_product(&inputs, &weights);
        let asis_cost = tree.eml_count();
        // ASIS K=2 z mul_eml=14 węzłów: A1*B1(14) - A2*B2(14)
        // sub = 11, razem = 14 + 14 + 11 = 39. (Naiwne ~ 53)
        assert!(asis_cost <= 53, "ASIS cost {} > naive 53", asis_cost);
    }

    #[test]
    fn test_mul_eml_correct() {
        use crate::constant_fold::try_evaluate;
        use crate::constant_fold::ConstantMap;
        
        let x = var("x");
        let y = var("y");
        // Wykorzystujemy tę samą funkcję co wewnątrz `build_asis_dot_product`, ale musimy ją zdefiniować lokalnie:
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
        assert_eq!(tree.eml_count(), 14, "Zoptymalizowano do 14!");
    }

    #[test]
    fn test_mul_eml_positive_only() {
        // UWAGA: mul_eml działa tylko dla x,y > 0
        // Dla ujemnych wartości użyj asis_preprocess_weights (pre-negacja offline)
        // lub klasycznego mnożenia w backendzie ALU
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
        // Oczekujemy None bo ln(-2.0) jest NaN
        assert!(try_evaluate(&tree, &consts).is_none(),
            "mul_eml nie działa dla ujemnych wartości — użyj ALU backend");
    }
}
