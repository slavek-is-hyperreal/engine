// src/cost_model.rs
//
// KONWENCJA: eml_count() = liczba węzłów wewnętrznych Eml(l,r)
//            node_count() = eml_count() + liczba liści
// Koszty z exhaustive search (paper Odrzywołka) = node_count()
// Koszty w testach tego projektu = eml_count() (tylko wewnętrzne)

// Brak importów z ast.rs, ponieważ CostModel operuje tylko na liczbach (usize)

/// Koszty bazowe operacji z exhaustive search (Odrzywołek 2026)
/// Koszt = całkowita liczba węzłów drzewa (wewnętrzne + liście)
/// dla argumentów będących prostymi zmiennymi (koszt liścia = 1)
pub struct CostModel;

impl CostModel {
    pub fn exp_cost() -> usize { 3 }
    pub fn ln_cost() -> usize { 7 }
    pub fn sub_cost() -> usize { 11 }
    pub fn neg_cost() -> usize { 15 }
    pub fn mul_cost() -> usize { 17 }
    pub fn div_cost() -> usize { 17 }
    pub fn add_cost() -> usize { 19 }

    /// Narzut operatora (overhead) = koszt - liczba argumentów
    /// Używany do obliczania kosztu kompozycji z poddrzewami
    pub fn exp_overhead() -> usize { 2 }   // 3 - 1
    pub fn ln_overhead() -> usize { 6 }    // 7 - 1
    pub fn sub_overhead() -> usize { 9 }   // 11 - 2
    pub fn neg_overhead() -> usize { 14 }  // 15 - 1
    pub fn mul_overhead() -> usize { 15 }  // 17 - 2
    pub fn div_overhead() -> usize { 15 }  // 17 - 2
    pub fn add_overhead() -> usize { 17 }  // 19 - 2

    /// Koszt kompozycji: op(A, B) gdzie A i B są poddrzewami
    pub fn compose_binary(overhead: usize, cost_a: usize, cost_b: usize) -> usize {
        overhead + cost_a + cost_b
    }

    pub fn compose_unary(overhead: usize, cost_a: usize) -> usize {
        overhead + cost_a
    }

    /// Koszt iloczynu skalarnego długości K — metoda naiwna
    /// C_naive(K) = 36K - 19
    pub fn dot_product_naive(k: usize) -> usize {
        36 * k - 19
    }

    /// Koszt iloczynu skalarnego długości K — metoda ASIS
    /// C_ASIS(K) = 28K - 11
    /// Wymaga: wagi pre-negowane offline
    pub fn dot_product_asis(k: usize) -> usize {
        28 * k - 11
    }

    /// Koszt iloczynu skalarnego — ASIS + Constant Folding wag
    /// C_CF_ASIS(K) = 14K - 9
    /// Wymaga: wagi statyczne, ln(ln(x)) prekomputowane
    pub fn dot_product_cf_asis(k: usize) -> usize {
        14 * k - 9
    }

    /// Koszt mnożenia macierzy M×K · K×N — metoda naiwna
    pub fn matmul_naive(m: usize, n: usize, k: usize) -> usize {
        m * n * Self::dot_product_naive(k)
    }

    /// Koszt mnożenia macierzy — ASIS
    pub fn matmul_asis(m: usize, n: usize, k: usize) -> usize {
        m * n * Self::dot_product_asis(k)
    }

    /// Koszt mnożenia macierzy — ASIS + CF
    pub fn matmul_cf_asis(m: usize, n: usize, k: usize) -> usize {
        // Prekomputacja Z = ln(ln(A)) kosztuje 13*m*k
        // Reszta: m*n*(14k-9)
        13 * m * k + m * n * Self::dot_product_cf_asis(k)
    }

    /// Koszt Softmax dla wektora długości N — DAG (sprzętowy)
    /// DAG_naive = 35n - 17
    pub fn softmax_dag(n: usize) -> usize {
        35 * n - 17
    }

    /// Koszt Log-Softmax dla wektora długości N — DAG
    /// DAG_log = 28n - 17
    /// Log-Softmax jest NATYWNĄ operacją EML: eml(ln(x_i), S) = x_i - ln(S)
    pub fn log_softmax_dag(n: usize) -> usize {
        28 * n - 17
    }

    /// Koszt Sigmoid(x) — optymalny
    pub fn sigmoid_cost() -> usize { 51 }

    /// Koszt SiLU(x) — optymalny (= sigmoid!)
    /// SiLU(x) = x/(1+exp(-x)) zamiast x*sigmoid(x)
    pub fn silu_cost() -> usize { 51 }

    /// Koszt RMSNorm dla wektora długości d — DAG z memoizacją R
    /// Cost = 66d + 41
    pub fn rmsnorm_dag(d: usize) -> usize {
        66 * d + 41
    }

    /// Koszt RoPE per para wymiarów — forma rzeczywista z Constant Folding
    /// Weryfikacja: Gemini Deep Research (dowód algebraiczny)
    /// 4 mnożenia przez stałe (CF: 5 węzłów każde) + odejmowanie (11) + dodawanie (19)
    /// = 4*5 + 11 + 19 = 50 węzłów per para
    pub fn rope_pair_cost_cf() -> usize { 50 }
    pub fn rope_element_cost_cf() -> usize { 25 } // 50/2 per element

    // Stare — oznacz jako deprecated:
    #[deprecated(note = "Użyj rope_element_cost_cf() = 25 węzłów (zweryfikowane)")]
    pub fn rope_element_cost_lower() -> usize { 68 }
    #[deprecated(note = "Użyj rope_element_cost_cf() = 25 węzłów (zweryfikowane)")]
    pub fn rope_element_cost_upper() -> usize { 53 }
    #[deprecated(note = "Użyj rope_element_cost_cf() = 25.0")]
    pub fn rope_avg_cost() -> f64 { 60.5 }

    /// Koszt pełnego Attention dla TinyLlama (jedna głowa)
    /// Q·K^T dominuje: 49.38%, Scores·V: 49.78%, Softmax: 0.84%
    pub fn attention_one_head(seq_len: usize, d_k: usize) -> usize {
        let qkt = seq_len * seq_len * Self::dot_product_naive(d_k);
        let softmax = seq_len * Self::softmax_dag(seq_len);
        let sv = seq_len * d_k * (seq_len * 17 + (seq_len - 1) * 19);
        qkt + softmax + sv
    }

    /// Koszt jednej warstwy TinyLlama — naiwny
    /// Wynik z Deep Research: ~13,102 miliardów węzłów
    pub fn tinyllama_layer_naive() -> u64 {
        13_102_000_000_000
    }

    /// Koszt jednej warstwy TinyLlamy — po optymalizacji
    /// CF + ASIS + DAG + fuzje styków
    /// Wynik z Deep Research: ~4,838 miliardów węzłów (63% redukcja)
    pub fn tinyllama_layer_optimized() -> u64 {
        4_838_000_000_000
    }

    /// Redukcja dla całej warstwy
    pub fn tinyllama_layer_reduction() -> f64 {
        let naive = Self::tinyllama_layer_naive() as f64;
        let opt = Self::tinyllama_layer_optimized() as f64;
        (naive - opt) / naive * 100.0
    }

    /// Koszt SwiGLU po fuzji styku (gate*up przez SiLU)
    /// Redukcja: 68 → 32 węzłów per wymiar
    pub fn swiglu_fused_cost() -> usize { 32 }

    /// Koszt residual connection gdy x w formie ln(x) w DAG
    /// 1 węzeł EML zamiast 19
    pub fn residual_fused_cost() -> usize { 1 }

    /// Dolna granica złożoności attention — twierdzenie
    /// Ω(n² · d) węzłów EML dla full attention
    pub fn attention_lower_bound(n: usize, d: usize) -> u64 {
        (n * n * d) as u64
    }

    /// Całkowita redukcja dla każdej operacji TinyLlamy
    pub fn tinyllama_breakdown() -> Vec<(&'static str, f64, f64, f64)> {
        // (operacja, naiwny_B, opt_B, redukcja_%)
        vec![
            ("RMSNorm×2",        34.0,   0.02,  99.9),
            ("Projekcje Q,K,V",  1855.3, 480.8, 74.1),
            // seq=2048, d_k=64, n_heads=32
            // 25 * 64 * 2048 * 32 = 104,857,600 ≈ 0.105B węzłów
            ("RoPE",             1.7,    0.105, 93.8),
            ("Q@K^T",            306.6,  119.0, 61.2),
            ("Log-Softmax",      4.69,   0.13,  97.2),
            ("Attention@V",      309.1,  120.2, 61.1),
            ("W_O proj",         618.3,  240.4, 61.1),
            ("Residual×2",       0.15,   0.008, 94.6),
            ("FFN W_gate,W_up",  6647.0, 2585.0,61.1),
            ("SwiGLU",           1.53,   0.34,  77.7),
            ("W_down",           3324.0, 1292.0,61.1),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_costs() {
        assert_eq!(CostModel::exp_cost(), 3);
        assert_eq!(CostModel::ln_cost(), 7);
        assert_eq!(CostModel::sub_cost(), 11);
        assert_eq!(CostModel::neg_cost(), 15);
        assert_eq!(CostModel::mul_cost(), 17);
        assert_eq!(CostModel::div_cost(), 17);
        assert_eq!(CostModel::add_cost(), 19);
    }

    #[test]
    fn test_dot_product_formulas() {
        // K=1: pierwszy element tylko mnożenie
        assert_eq!(CostModel::dot_product_naive(1), 17);
        // K=2: 36*2-19=53
        assert_eq!(CostModel::dot_product_naive(2), 53);
        // ASIS K=64: 28*64-11=1781
        assert_eq!(CostModel::dot_product_asis(64), 1781);
        // CF+ASIS K=64: 14*64-9=887
        assert_eq!(CostModel::dot_product_cf_asis(64), 887);
    }

    #[test]
    fn test_asis_reduction() {
        // ASIS powinien być 22.2% tańszy asymptotycznie
        let naive = CostModel::dot_product_naive(1024) as f64;
        let asis = CostModel::dot_product_asis(1024) as f64;
        let reduction = (naive - asis) / naive;
        assert!((reduction - 0.222).abs() < 0.01);
    }

    #[test]
    fn test_log_softmax_cheaper_than_softmax() {
        let n = 2048;
        assert!(CostModel::log_softmax_dag(n) < CostModel::softmax_dag(n));
        // Log-Softmax: 57327, Softmax: 71663
        assert_eq!(CostModel::log_softmax_dag(n), 57327);
        assert_eq!(CostModel::softmax_dag(n), 71663); // 35*2048-17
    }

    #[test]
    fn test_silu_equals_sigmoid() {
        assert_eq!(CostModel::silu_cost(), CostModel::sigmoid_cost());
        assert_eq!(CostModel::silu_cost(), 51);
    }

    #[test]
    fn test_tinyllama_attention_softmax_fraction() {
        let seq = 2048;
        let dk = 64;
        let total = CostModel::attention_one_head(seq, dk);
        let softmax = seq * CostModel::softmax_dag(seq);
        let fraction = softmax as f64 / total as f64;
        // Softmax powinien być < 1% całkowitego kosztu
        assert!(fraction < 0.01,
            "Softmax fraction: {:.4} (expected < 0.01)", fraction);
    }

    #[test]
    fn test_tinyllama_layer_reduction() {
        assert!((CostModel::tinyllama_layer_reduction() - 63.1).abs() < 0.1);
    }

    #[test]
    fn test_rope_cf_cost() {
        assert_eq!(CostModel::rope_pair_cost_cf(), 50);
        assert_eq!(CostModel::rope_element_cost_cf(), 25);
    }

    #[test]
    fn test_swiglu_fused_cheaper() {
        // SwiGLU fused: 32 węzły < naiwne: ~68 węzłów
        assert!(CostModel::swiglu_fused_cost() < 68);
    }

    #[test]
    fn test_residual_fused_one_node() {
        assert_eq!(CostModel::residual_fused_cost(), 1);
    }

    #[test]
    fn test_attention_lower_bound() {
        // Dla TinyLlamy: n=2048, d=64
        let lb = CostModel::attention_lower_bound(2048, 64);
        assert_eq!(lb, 2048 * 2048 * 64); // 268,435,456
    }
}
