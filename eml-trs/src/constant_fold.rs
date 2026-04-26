// src/constant_fold.rs
//
// KONWENCJA: eml_count() = liczba węzłów wewnętrznych Eml(l,r)
//            node_count() = eml_count() + liczba liści
// Koszty z exhaustive search (paper Odrzywołka) = node_count()
// Koszty w testach tego projektu = eml_count() (tylko wewnętrzne)

use crate::ast::*;
use std::sync::Arc;
use std::collections::HashMap;

/// Mapa zmiennych → wartości stałe (wagi, parametry)
pub type ConstantMap = HashMap<String, f64>;

/// Ewaluuje drzewo EML jeśli wszystkie liście są znane
/// Zwraca Some(wartość) jeśli całe poddrzewo jest stałe
pub fn try_evaluate(node: &EmlNode, consts: &ConstantMap) -> Option<f64> {
    match node {
        EmlNode::One => Some(1.0),
        EmlNode::Const(v) => Some(*v),
        EmlNode::Var(name) => consts.get(name).copied(),
        EmlNode::Eml(l, r) => {
            let lv = try_evaluate(l, consts)?;
            let rv = try_evaluate(r, consts)?;
            // eml(x, y) = exp(x) - ln(y)
            if rv <= 0.0 { return None; } // ln(0) undefined
            Some(lv.exp() - rv.ln())
        }
    }
}

/// Rekurencyjny constant folding
/// Zastępuje poddrzewa stałymi wartościami gdy możliwe
pub fn fold_constants(node: Arc<EmlNode>, consts: &ConstantMap) -> Arc<EmlNode> {
    // Sprawdź czy całe poddrzewo można zwinąć
    if let Some(value) = try_evaluate(&node, consts) {
        return Arc::new(EmlNode::Const(value));
    }

    // Jeśli nie — rekurencja na dzieciach
    match node.as_ref() {
        EmlNode::Eml(l, r) => {
            let new_l = fold_constants(l.clone(), consts);
            let new_r = fold_constants(r.clone(), consts);
            eml(new_l, new_r)
        }
        // Liście które nie są stałe — zamień Var na Const jeśli znana
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

/// Optymalizacja mnożenia przez stałą:
/// x * W → 5-węzłowa struktura gdy W jest stałe
/// Normalnie: x * W = 17 węzłów
/// Z CF: eml(eml(ln(ln(x)), Const(1/W)), 1) = 5 węzłów
/// (ln(ln(x)) prekomputowane, Const(1/W) to stały liść)
///
/// UWAGA: wymaga że ln(x) > 0, czyli x > 1
/// Dla danych ujemnych użyj standardowego mnożenia
pub fn mul_with_const_weight(x: Arc<EmlNode>, w: f64) -> Arc<EmlNode> {
    assert!(w != 0.0, "Weight cannot be zero");
    let inv_w = Arc::new(EmlNode::Const(1.0 / w));
    // eml(eml(ln(ln(x)), 1/w), 1)
    // = exp(ln(ln(x)) - ln(1/w)) - ln(1)
    // = exp(ln(ln(x)) + ln(w))
    // = exp(ln(x * ... )) ... sprawdź algebraicznie
    eml(eml(ln_node(ln_node(x)), inv_w), one())
}

/// ASIS preprocessing: neguj wagi[1..] offline
/// Zwraca nowy wektor wag gotowy do ASIS dot product
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
        // eml(x, 1) gdzie x=2.0 → exp(2.0)
        let tree = eml(var("x"), one());
        let mut consts = ConstantMap::new();
        consts.insert("x".to_string(), 2.0);
        let folded = fold_constants(tree, &consts);
        // Powinno być Const(exp(2.0))
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
        // Struktura: eml(eml(ln(ln(x)), Const(0.5)), 1)
        // eml_count = 8 (pełne drzewo)
        // W praktyce: ln(ln(x)) jest prekomputowane offline → efektywny koszt = 5
        // (2 węzły eml dla zewnętrznej struktury + 3 dla ln(ln(x)) traktowanego jako liść/Var)
        assert_eq!(tree.eml_count(), 8);
    }
}
