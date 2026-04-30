import math
import numpy as np
import struct

# ============================================================
# SET A: src/backends/eml_kernels.wgsl
# ============================================================
A_EXP = (1.00247605, 0.65104678, 0.34400111)
A_LN  = (0.98771793, -0.40916155, 0.11513792)

# ============================================================
# SET B: src/backends/wgsl.rs
# ============================================================
B_EXP = (0.9981335, 0.6552899, 0.3444342)
B_LN  = (0.9981084, -0.4788506, 0.1740927)

LOG2_E = 1.4426950408889634
LN_2   = 0.6931471805599453

def fast_exp(x, coeffs):
    a0, a1, a2 = coeffs
    x = max(-87.0, min(87.0, x))
    w = x * LOG2_E
    i = math.floor(w)
    f = w - i
    p = a0 + f * (a1 + f * a2)
    return p * (2.0 ** i)

def fast_ln(x, coeffs):
    c1, c2, c3 = coeffs
    if x <= 0: return -1e38
    bits = struct.unpack('!I', struct.pack('!f', float(np.float32(x))))[0]
    e = ((bits >> 23) & 0xFF) - 127
    m_bits = (bits & 0x7FFFFF) | 0x3F800000
    m = struct.unpack('!f', struct.pack('!I', m_bits))[0]
    u = m - 1.0
    poly = u * (c1 + u * (c2 + u * c3))
    return e * LN_2 + poly

def verify_set(name, exp_c, ln_c):
    print(f"\n--- {name} ---")
    # EXP
    xs = np.linspace(-5.0, 5.0, 10000)
    max_rel_err = max(abs(fast_exp(x, exp_c) - math.exp(x)) / math.exp(x) for x in xs)
    print(f"  fast_exp  max relative error: {max_rel_err:.6f}  (BF16 eps = 0.0078125)  {'OK' if max_rel_err < 0.0078 else 'FAIL'}")
    # LN
    xs = np.linspace(0.01, 20.0, 10000)
    max_abs_err = max(abs(fast_ln(x, ln_c) - math.log(x)) for x in xs)
    print(f"  fast_ln   max absolute error: {max_abs_err:.6f}  (BF16 eps = 0.0078125)  {'OK' if max_abs_err < 0.0078 else 'FAIL'}")

if __name__ == "__main__":
    print("Comparing two Minimax coefficient sets found in the codebase:")
    verify_set("SET A: eml_kernels.wgsl", A_EXP, A_LN)
    verify_set("SET B: wgsl.rs",          B_EXP, B_LN)
    print("\nConclusion: Use the set with smaller errors.")
