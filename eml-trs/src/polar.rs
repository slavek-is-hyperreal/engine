// src/polar.rs
//
// KONWENCJA: eml_count() = liczba węzłów wewnętrznych Eml(l,r)
//            node_count() = eml_count() + liczba liści
// Koszty z exhaustive search (paper Odrzywołka) = node_count()
// Koszty w testach tego projektu = eml_count() (tylko wewnętrzne)
//
// HIPOTEZA: Unifikacja TurboQuant + RoPE + EML w układzie biegunowym
//
// W układzie biegunowym wektor q = (r, φ) gdzie r = |q|, φ = arg(q).
// W EML: ln(q) = ln(r) + iφ — to jest NATURALNA reprezentacja biegunowa.
//
// TurboQuant (Google 2026) przechowuje wektory jako (r, φ).
// RoPE to rotacja: q * exp(imθ) = (r, φ + mθ).
// W EML: ln(q) + imθ = ln(r) + i(φ + mθ) — czyste dodawanie do kąta.
//
// WNIOSEK: Jeśli przechowujemy wagi w reprezentacji (ln(r), φ):
// - RoPE = addytywna operacja na φ = 0 węzłów EML
// - Iloczyn skalarny Q·K^T = operacja na magnitudach i kątach
// - TurboQuant kompresja jest naturalna (oddzielne kanały r i φ)
//
// To jest nowy kierunek badań: EML-compatible polar positional encoding.
// Weryfikacja: czy model wytrenowany z tym kodowaniem ma tę samą jakość?

pub struct PolarVector {
    pub ln_r: f64,  // ln(magnituda)
    pub phi: f64,   // kąt (faza)
}

impl PolarVector {
    pub fn from_cartesian(re: f64, im: f64) -> Self {
        let r = (re * re + im * im).sqrt();
        Self {
            ln_r: r.ln(),
            phi: im.atan2(re),
        }
    }
    
    /// RoPE to czyste dodawanie do kąta — 0 węzłów EML
    pub fn apply_rope(&self, position_angle: f64) -> Self {
        Self {
            ln_r: self.ln_r,           // magnituda bez zmian
            phi: self.phi + position_angle, // kąt += mθ
        }
    }
    
    /// Iloczyn skalarny dwóch wektorów biegunowych
    /// q·k = r_q * r_k * cos(φ_q - φ_k)
    ///      = exp(ln_r_q + ln_r_k) * cos(φ_q - φ_k)
    pub fn dot(&self, other: &Self) -> f64 {
        let ln_r_sum = self.ln_r + other.ln_r;
        let delta_phi = self.phi - other.phi;
        ln_r_sum.exp() * delta_phi.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_cartesian() {
        // re=3, im=4 -> r=5, phi=atan2(4,3)
        let v = PolarVector::from_cartesian(3.0, 4.0);
        assert!((v.ln_r - 5.0f64.ln()).abs() < 1e-6);
        assert!((v.phi - 4.0f64.atan2(3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_apply_rope() {
        let v = PolarVector::from_cartesian(1.0, 0.0); // r=1, phi=0
        let v_rot = v.apply_rope(std::f64::consts::PI / 2.0); // obrót o 90 deg
        assert!((v_rot.ln_r - 0.0).abs() < 1e-6);
        assert!((v_rot.phi - std::f64::consts::PI / 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let v1 = PolarVector::from_cartesian(3.0, 4.0);
        let v2 = PolarVector::from_cartesian(5.0, 12.0);
        let dot = v1.dot(&v2);
        // klasyczny dot: 3*5 + 4*12 = 15 + 48 = 63
        assert!((dot - 63.0).abs() < 1e-6);
    }
}
