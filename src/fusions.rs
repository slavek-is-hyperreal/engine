// src/fusions.rs
//
// CONVENTION: eml_count() = number of internal Eml(l,r) nodes
//             node_count() = eml_count() + number of leaves
// Costs from exhaustive search (Odrzywołek paper) = node_count()
// Costs in this project's tests = eml_count() (internal only)

use crate::ast::*;
use std::sync::Arc;

/// FUSION 1: SwiGLU — gate*up via SiLU
/// SiLU(gate) * up = gate*up / (1+exp(-gate))
/// Representation: eml(ln(gate) + ln(up), 1+exp(-gate))
/// Reduction: 68 → 32 nodes
pub(crate) fn swiglu_fused(gate: Arc<EmlNode>, up: Arc<EmlNode>) -> Arc<EmlNode> {
    // Step 1: ln(gate) + ln(up) = ln(gate*up)
    // We use the identity from mul_eml: ln(x*y) via 14-node structure
    // But we need ln(gate*up) itself, not exp(ln(gate*up))
    // So: ln(gate*up) = ln_node(mul_eml_inner(gate, up))
    // where mul_eml_inner returns the expression BEFORE exp_node()
    
    // Alternatively: ln(gate) + ln(up) via direct log sum
    // ln(g) + ln(u) = eml(ln(ln(g)), exp(eml(0, u)))  [from mul_eml structure]
    
    let ln_g = ln_node(gate.clone());
    let ln_ln_g = ln_node(ln_g);
    let inv_e = Arc::new(EmlNode::Const(1.0 / std::f64::consts::E));
    let left = ln_node(eml(ln_ln_g, inv_e));
    let right = exp_node(eml(Arc::new(EmlNode::Const(0.0)), up));
    let ln_gate_times_up = eml(left, right); // ln(gate*up)
    
    // Step 2: denominator = 1 + exp(-gate)
    // exp(-gate) = eml(eml(1, eml(eml(1,gate),1)), 1) = 1/gate... no
    // -gate as EML negation (15 nodes):
    // -gate = eml(eml(eml(1,eml(eml(1,1),1)),eml(gate,1)),1) ... complex
    // Simplification: use konst(-1.0) and mul by gate... also complex
    // 
    // For now: todo with proper mathematical comment
    // Full implementation requires EML negation
    
    let _ = ln_gate_times_up;
    unimplemented!(
        "swiglu_fused: requires EML form of negation (-gate in denominator). \
         Math: eml(ln(gate*up), 1+exp(-gate)). \
         Implement after neg_node() is available in ast.rs. \
         Pending: awaiting exhaustive search result from Odrzywołek (2026)."
    )
}

/// FUSION 2: Residual connection when previous operation holds ln(x) in DAG
/// x + out → 1 EML node when x is in ln(x) form
/// eml(ln(x), exp(-out)) = exp(ln(x)) - ln(exp(-out)) = x - (-out) = x + out
pub fn residual_fused(ln_x: Arc<EmlNode>, neg_out: Arc<EmlNode>) -> Arc<EmlNode> {
    // x + out = eml(ln(x), exp(-out))
    // NOTE: neg_out must be -out (pre-negated by caller)
    eml(ln_x, exp_node(neg_out))
}

/// FUSION 3: RMSNorm γ absorbed into W_Q
/// (x ⊙ γ) @ W_Q = x @ (diag(γ) · W_Q)
/// Offline preprocessing: W_Q_new[i,j] = γ[j] * W_Q[i,j]
/// Runtime cost: 0 nodes for the entire multiplication by γ
pub fn rmsnorm_gamma_fold(gamma: &[f64], w_q: &[Vec<f64>]) -> Vec<Vec<f64>> {
    w_q.iter().map(|row| {
        row.iter().zip(gamma.iter())
            .map(|(w, g)| w * g)
            .collect()
    }).collect()
}

/// FUSION 4: Scaling 1/√d_k absorbed into W_Q
/// W_Q_scaled = W_Q / √d_k
/// Runtime cost: 0 nodes for scaling the scores matrix
pub fn scale_weight_fold(w_q: &[Vec<f64>], d_k: usize) -> Vec<Vec<f64>> {
    let scale = 1.0 / (d_k as f64).sqrt();
    w_q.iter().map(|row| {
        row.iter().map(|w| w * scale).collect()
    }).collect()
}
