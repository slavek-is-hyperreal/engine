// src/fusions.rs
//
// Advanced operation fusions for Transformer layers.
// Fusions reduce node count by canceling out ln/exp pairs at boundaries.

use crate::ast::*;
use std::sync::Arc;

/// FUSION 1: SwiGLU — gate * up via SiLU
/// SiLU(gate) * up = (gate * up) / (1 + exp(-gate))
///
/// Mathematical derivation in EML:
/// A = gate * up
/// B = 1 + exp(-gate)
/// Result = A / B = exp(ln A - ln B)
/// 
/// In EML terms:
/// ln_A = ln(gate * up)
/// ln_ratio = eml(ln(ln_A), B) = exp(ln(ln A)) - ln(B) = ln A - ln B
/// Final = exp(ln_ratio) = exp(ln A - ln B) = A / B
///
/// CAUTION: This fusion requires (gate * up) > 1.0 because it uses ln(ln(gate * up)).
/// If the product is <= 1.0, ln(gate * up) is <= 0, and the second ln() is undefined.
/// For general range, use standard non-fused implementation or Round-Trip TRS.
pub fn swiglu_fused(gate: Arc<EmlNode>, up: Arc<EmlNode>) -> Arc<EmlNode> {
    // A = gate * up
    let gate_times_up = mul_eml(gate.clone(), up);
    let ln_a = ln_node(gate_times_up);

    // B = 1 + exp(-gate)
    let neg_gate = neg_node(gate);
    let exp_neg_gate = exp_node(neg_gate);
    let b = add_eml(one(), exp_neg_gate);

    // ln_ratio = ln(A) - ln(B) = eml(ln(ln A), B)
    let ln_ratio = eml(ln_node(ln_a), b);
    
    // Result = exp(ln_ratio)
    exp_node(ln_ratio)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constant_fold::ConstantMap;

    #[test]
    fn test_swiglu_fused_correctness() {
        let gate_node = var("gate");
        let up_node = var("up");
        let fused = swiglu_fused(gate_node, up_node);

        let mut c = ConstantMap::new();
        // Test for gate=3.0, up=2.0 (gate*up = 6.0 > 1.0)
        let gate_v = 3.0f64;
        let up_v = 2.0f64;
        c.insert("gate".to_string(), gate_v);
        c.insert("up".to_string(), up_v);

        let expected = gate_v * up_v / (1.0 + (-gate_v).exp());

        use crate::round_trip::compile_to_ops;
        let program = compile_to_ops(fused);
        let result = program.execute(&c)
            .expect("Should evaluate for gate * up > 1.0");
            
        assert!(
            (result - expected).abs() < 1e-6,
            "SwiGLU mismatch: expected {:.8}, got {:.8}", expected, result
        );
    }
}
