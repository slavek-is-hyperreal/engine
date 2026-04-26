// src/fusions.rs
//
// KONWENCJA: eml_count() = liczba węzłów wewnętrznych Eml(l,r)
//            node_count() = eml_count() + liczba liści
// Koszty z exhaustive search (paper Odrzywołka) = node_count()
// Koszty w testach tego projektu = eml_count() (tylko wewnętrzne)

use crate::ast::*;
use std::sync::Arc;

/// FUZJA 1: SwiGLU styk — gate*up przez SiLU
/// SiLU(gate) * up = gate*up / (1+exp(-gate))
/// Reprezentacja: eml(ln(gate) + ln(up), 1+exp(-gate))
/// Redukcja: 68 → 32 węzłów
pub(crate) fn swiglu_fused(gate: Arc<EmlNode>, up: Arc<EmlNode>) -> Arc<EmlNode> {
    // Krok 1: ln(gate) + ln(up) = ln(gate*up)
    // Używamy tożsamości z mul_eml: ln(x*y) przez 14-węzłową strukturę
    // Ale potrzebujemy samego ln(gate*up), nie exp(ln(gate*up))
    // Więc: ln(gate*up) = ln_node(mul_eml_inner(gate, up))
    // gdzie mul_eml_inner zwraca wyrażenie PRZED exp_node()
    
    // Alternatywnie: ln(gate) + ln(up) przez bezpośrednią sumę logarytmów
    // ln(g) + ln(u) = eml(ln(ln(g)), exp(eml(0, u)))  [ze struktury mul_eml]
    
    let ln_g = ln_node(gate.clone());
    let ln_ln_g = ln_node(ln_g);
    let inv_e = Arc::new(EmlNode::Const(1.0 / std::f64::consts::E));
    let left = ln_node(eml(ln_ln_g, inv_e));
    let right = exp_node(eml(Arc::new(EmlNode::Const(0.0)), up));
    let ln_gate_times_up = eml(left, right); // ln(gate*up)
    
    // Krok 2: mianownik = 1 + exp(-gate)
    // exp(-gate) = eml(eml(1, eml(eml(1,gate),1)), 1) = 1/gate... nie
    // -gate jako negacja EML (15 węzłów):
    // -gate = eml(eml(eml(1,eml(eml(1,1),1)),eml(gate,1)),1) ... złożone
    // Uproszczenie: użyj konst(-1.0) i mul przez gate... też złożone
    // 
    // Na razie: todo z poprawnym komentarzem matematycznym
    // Pełna implementacja wymaga negacji w EML
    
    let _ = ln_gate_times_up;
    unimplemented!(
        "swiglu_fused: wymaga negacji EML dla -gate w mianowniku.\
         Matematyka: eml(ln(gate*up), 1+exp(-gate)).\
         Zaimplementuj po dodaniu neg_eml() do ast.rs"
    )
}

/// FUZJA 2: Residual connection gdy poprzednia operacja trzyma ln(x) w DAG
/// x + out → 1 węzeł EML gdy x jest w formie ln(x)
/// eml(ln(x), exp(-out)) = exp(ln(x)) - ln(exp(-out)) = x - (-out) = x + out
pub fn residual_fused(ln_x: Arc<EmlNode>, neg_out: Arc<EmlNode>) -> Arc<EmlNode> {
    // x + out = eml(ln(x), exp(-out))
    // UWAGA: neg_out musi być -out (pre-negowane przez wywołującego)
    eml(ln_x, exp_node(neg_out))
}

/// FUZJA 3: RMSNorm γ wchłonięte do W_Q
/// (x ⊙ γ) @ W_Q = x @ (diag(γ) · W_Q)
/// Preprocessing offline: W_Q_new[i,j] = γ[j] * W_Q[i,j]
/// Koszt runtime: 0 węzłów dla całego mnożenia przez γ
pub fn rmsnorm_gamma_fold(gamma: &[f64], w_q: &[Vec<f64>]) -> Vec<Vec<f64>> {
    w_q.iter().map(|row| {
        row.iter().zip(gamma.iter())
            .map(|(w, g)| w * g)
            .collect()
    }).collect()
}

/// FUZJA 4: Skalowanie 1/√d_k wchłonięte do W_Q
/// W_Q_scaled = W_Q / √d_k
/// Koszt runtime: 0 węzłów dla skalowania macierzy scores
pub fn scale_weight_fold(w_q: &[Vec<f64>], d_k: usize) -> Vec<Vec<f64>> {
    let scale = 1.0 / (d_k as f64).sqrt();
    w_q.iter().map(|row| {
        row.iter().map(|w| w * scale).collect()
    }).collect()
}
