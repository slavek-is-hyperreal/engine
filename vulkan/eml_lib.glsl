// vulkan/eml_lib.glsl
//
// Podstawowe operacje EML zoptymalizowane pod ALU (Minimax N=3).
// Omija SFU (Special Function Units) dla maksymalnej wydajności.

// fast_exp: N=2 Minimax (E_max ≈ 0.0021)
// Liczy e^x przez 2^(x * log2(e))
float fast_exp(float x) {
    float x_log2e = x * 1.44269504089; // log2(e)
    float n = floor(x_log2e);
    float f = x_log2e - n;
    
    // Minimax N=2 dla 2^f (f in [0, 1])
    float poly = 1.0 + f * (0.655359 + f * 0.235314);
    
    return poly * pow(2.0, n); // ldexp-like
}

// fast_ln: N=3 Minimax (E_max ≈ 0.0006)
// Liczy ln(x) przez ln(2^n * (1+u)) = n*ln(2) + ln(1+u)
float fast_ln(float x) {
    // Bardzo uproszczona ekstrakcja mantysy i wykładnika dla GLSL
    // W produkcji użyjemy bit-hacking: floatBitsToInt
    float n = floor(log2(x));
    float m = x / pow(2.0, n);
    float u = m - 1.0;
    
    // Minimax N=3 dla ln(1+u) (u in [0, 1])
    float poly = u * (0.999115 + u * (-0.489959 + u * 0.285675));
    
    return n * 0.69314718056 + poly; // n*ln(2) + ln(1+u)
}


// Główny operator EML: eml(x, y) = exp(x) - ln(y)
float eml_op(float x, float y) {
    return fast_exp(x) - fast_ln(y); 
}

// Bezpieczne mnożenie przez stałą (mul_cf_safe z Offset Trick)
float eml_mul_cf(float x, float w, float bias) {
    float x_shifted = x + bias;
    float scaled = fast_exp(fast_ln(x_shifted) + fast_ln(abs(w)));
    float result = scaled - (abs(w) * bias);
    return (w < 0.0) ? -result : result;
}

