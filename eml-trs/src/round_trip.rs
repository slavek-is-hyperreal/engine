// src/round_trip.rs
//
// KONWENCJA: eml_count() = liczba węzłów wewnętrznych Eml(l,r)
//            node_count() = eml_count() + liczba liści
// Koszty z exhaustive search (paper Odrzywołka) = node_count()
// Koszty w testach tego projektu = eml_count() (tylko wewnętrzne)

use crate::ast::*;
use crate::trs::rewrite;
use std::sync::Arc;

/// Reguły przepisywania z EML do klasycznej matematyki
/// Używane w fazie "wyciągania" przed klasycznym TRS
pub enum ClassicalOp {
    Exp(Arc<EmlNode>),      // exp(x)
    Ln(Arc<EmlNode>),       // ln(x)
    Sub(Arc<EmlNode>, Arc<EmlNode>), // x - y
    Add(Arc<EmlNode>, Arc<EmlNode>), // x + y (po rozpoznaniu wzorca)
    Var(String),
    Const(f64),
}

/// Rozpoznaje klasyczne operacje w drzewie EML
pub fn recognize_classical(node: &Arc<EmlNode>) -> Option<ClassicalOp> {
    use crate::trs::{is_exp_pattern, is_ln_pattern};
    
    // eml(x, 1) → Exp(x)
    if let EmlNode::Eml(l, r) = node.as_ref() {
        if matches!(r.as_ref(), EmlNode::One) {
            return Some(ClassicalOp::Exp(l.clone()));
        }
    }
    
    // ln(x) wzorzec
    if let Some(inner) = is_ln_pattern(node) {
        return Some(ClassicalOp::Ln(inner));
    }
    
    // eml(ln(a), exp(b)) → Sub(a, b)
    if let EmlNode::Eml(l, r) = node.as_ref() {
        if let (Some(a), Some(b)) = (is_ln_pattern(l), is_exp_pattern(r)) {
            return Some(ClassicalOp::Sub(a, b));
        }
    }
    
    None
}

/// Główna funkcja round-trip:
/// 1. TRS w EML
/// 2. Rozpoznaj wzorce klasyczne
/// 3. Zastosuj klasyczne tożsamości
/// 4. Przepisz z powrotem do EML
/// 5. TRS w EML znowu
pub fn round_trip_optimize(node: Arc<EmlNode>) -> Arc<EmlNode> {
    // Krok 1: TRS w EML
    let after_eml_trs = rewrite(node);
    
    // Krok 2-3: TODO klasyczne tożsamości
    // Na razie zwróć wynik po EML TRS
    after_eml_trs
}
