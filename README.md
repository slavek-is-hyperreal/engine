# eml-trs

**Algebraic compression of neural network inference graphs via EML Term Rewriting.**

[![Rust](https://img.shields.io/badge/rust-2021-orange)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-PAPER.md-green)](PAPER.md)

---

In 2026, Odrzywołek (Jagiellonian University) proved that a single binary operator

```
eml(x, y) = exp(x) - ln(y)
```

together with the constant `1`, generates all elementary functions
([arXiv:2603.21852v2](https://arxiv.org/abs/2603.21852)).

This means every neural network computation graph can be expressed as a
**uniform binary tree** with one node type — and uniform trees can be
compressed algebraically in ways heterogeneous graphs cannot.

`eml-trs` implements this compression pipeline for neural network inference.

---

## Key Results on TinyLlama 1.1B

| Operation | Naive (B nodes) | Optimized (B nodes) | Reduction |
|:----------|---------------:|--------------------:|:---------:|
| RMSNorm ×2 | 0.00027 | 0.00002 | **92.6%** |
| Q, K, V projections | 927.47 | 240.40 | **74.1%** |
| RoPE | 0.10 | 0.01 | **90.0%** |
| Q@K^T | 306.60 | 119.00 | **61.2%** |
| Log-Softmax | 4.69 | 3.75 | **20.0%** |
| SwiGLU | 1.53 | 0.34 | **77.7%** |
| **Full layer** | **5,896** | **2,300** | **61.0%** |

### Surprising findings

**Log-Softmax is a native EML operation** — $O(1)$ amortized nodes per output element:
```
log_softmax(xᵢ) = xᵢ − ln(S) = eml(ln(exp(xᵢ)), S)
```
Stable Softmax via `max()` requires O(3ⁿ) nodes. Use Log-Softmax.

**The cost hierarchy inverts** — in EML, addition (19 nodes) costs more
than multiplication (17 nodes). Softmax costs **0.84%** of Attention;
matrix multiplication dominates **99%**. Classical GPU optimization
priorities are reversed.

**Approximation is lossless** — fast_exp via 2-term Minimax FMA has
error 0.0021, below bf16 machine epsilon (0.0078). The approximation
is exact relative to the model's own weight precision.

**EML inference ∈ NC1** — the optimized DAG maps to a Tree Straight-Line
Program (TSLP). By Ganardi, Jeż & Lohrey (JACM 2021), this can be balanced
to O(log N) evaluation depth with constant size overhead, placing EML
network inference in the parallel complexity class NC1.

**Zero label overhead** — EML has exactly one internal node type (|Σ| = 1).
Label entropy per node: log₂(1) = 0 bits. The full graph topology encodes
in 2N bits (balanced parentheses), near the Catalan number lower bound of
≈ 2N − 1.5·log₂(N) bits. ONNX pays ~7.6 bits/node for operator labels. EML pays 0.

---

## How It Works

Four independent optimizations compose to 61.0% reduction:

**1. ASIS (Subtractive Inner Product)**
Pre-negate frozen weights offline. Replace additions (19 nodes) with
subtractions (11 nodes) at inference time. Zero accuracy loss.
```
Σ aᵢwᵢ = a₁w̃₁ − a₂w̃₂ − ... − aₖw̃ₖ    (w̃ₖ = −wₖ for k≥2)
C_ASIS(K) = 28K − 11   vs   C_naive(K) = 36K − 19   →   −22.2%
```

**2. Constant Folding of Weights**
Frozen weights become constant leaves. Multiplication by a constant W
collapses from 17 nodes to 5:
```
x · W = eml(eml(ln(ln(x)), 1/W), 1)     [requires x > 0]
C_CF_ASIS(K) = 14K − 9   →   −61.1% vs naive
```

**3. DAG with Common Subexpression Elimination**
Shared subexpressions (e.g. ln(x) appearing multiple times) are computed
once and referenced. The CSE DAG is isomorphic to LZ dictionary compression
on the serialized tree (Theorem C1).

**4. Operation Fusion at Layer Boundaries**
- RMSNorm γ absorbed into W_Q offline → 0 runtime nodes
- Attention scaling 1/√dₖ absorbed into W_Q → 0 runtime nodes
- Residual connection when x is in log-domain → 1 node instead of 19

---

## Vulkan GPU Kernels

Fast EML kernels for AMD GCN 2.0 (R7 260X) via wgpu/WGSL:

```wgsl
// fast_exp: 2-term Minimax, E_max ≈ 0.0021 < bf16 epsilon
// fast_ln:  3-term Minimax, E_max ≈ 0.0006 < bf16 epsilon
fn eml(x: f32, y: f32) -> f32 { fast_exp(x) - fast_ln(y) }
```

Both functions use FMA instructions (4 cycles) instead of SFU transcendentals
(16 cycles) — 4× speedup per EML node. Combined with 61.0% node reduction:
theoretical **~10.2× speedup** vs naive EML.

---

## Quick Start

```bash
# Run all tests
cargo test

# Analytical cost model for TinyLlama
cargo run --bin tinyllama_costs

# TRS benchmark (ln(exp(x)) → x, etc.)
cargo run --bin trs_bench

# Numerical verification (EML tree vs CPU reference)
cargo run --bin verify_output

# GPU benchmark (requires Vulkan)
cargo run --bin vulkan_benchmark

# Real TinyLlama benchmark (requires model file)
# Download: huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
#     tinyllama-1.1b-chat-v1.0.F16.gguf --local-dir ./models
cargo run --features gguf --bin real_tinyllama -- models/tinyllama-1.1b-chat-v1.0.F16.gguf
```

---

## Implementation

```
src/
├── ast.rs           EML binary tree (Arc<EmlNode>, one/var/konst/eml)
├── cost_model.rs    Analytical cost formulas from Odrzywołek (2026)
├── trs.rs           Bottom-up TRS to fixpoint (6 rewriting rules)
├── constant_fold.rs Symbolic constant folding for frozen weights
├── asis.rs          ASIS dot product construction + correctness verification
├── dag.rs           DAG with Common Subexpression Elimination
├── fusions.rs       Layer boundary fusions (RMSNorm, scaling, residual)
├── nn_layer.rs      EML tree builder for transformer layers
├── round_trip.rs    Round-trip optimization (EML → classical → EML)
├── polar.rs         Polar coordinate hypothesis (TurboQuant + RoPE + EML)
└── backends/
    ├── alu.rs           Classical ALU fallback for negative values
    ├── vulkan_eml.rs    wgpu/Vulkan async compute pipeline
    └── eml_kernels.wgsl fast_exp + fast_ln + log_softmax + dot_product_asis
└── loader/
    └── gguf.rs      Minimal GGUF loader (F32/F16, zero external deps)
```

**Zero external dependencies** in the core algebraic engine.
Vulkan backend uses: `wgpu`, `bytemuck`, `tokio`, `futures`.

---

## Theoretical Results

Full proofs in [PAPER.md](PAPER.md) and Appendices A–C.

| Result | Statement |
|:-------|:----------|
| **Theorem 2** (ASIS) | `C_ASIS(K) = 28K−11`, saving 22.2% asymptotically |
| **Theorem 3** (CF) | Multiplication by constant W: 17 → 5 EML nodes |
| **Theorem 4** (Log-Softmax) | `log_softmax(xᵢ) = eml(ln(exp(xᵢ)), S)` — O(1) amortized |
| **Theorem 5** (Lower Bound) | Full attention requires Ω(n²d) EML dependencies |
| **Theorem C1** (LZ≅CSE) | CSE in EML DAG ≅ LZ dictionary compression |
| **Theorem C2** (TRS⊂HRG) | EML TRS is a subclass of Hyperedge Replacement Grammars |
| **Theorem C3** (NC1) | EML inference ∈ NC1, evaluation depth O(log N) |
| **Theorem C4** (Succinct) | EML topology encodes in 2N bits, zero label overhead |

---

## Status

| Component | Status |
|:----------|:------:|
| Core TRS (6 rules) | ✅ Complete |
| Constant Folding | ✅ Complete |
| ASIS dot product | ✅ Complete |
| DAG / CSE | ✅ Complete |
| Cost model (TinyLlama) | ✅ Complete |
| GGUF loader (F32/F16) | ✅ Complete |
| Vulkan kernels (WGSL) | ✅ Complete |
| Numerical verification | ✅ Complete |
| `neg_node` (EML form) | ✅ Complete |
| SwiGLU fusion | ✅ Complete |
| Round-trip optimization | ✅ Complete |
| EML-TSLP Compression | ✅ Complete |
| Performance Benchmarking | ✅ Complete |
| Academic Paper Audit | ✅ Repair Complete |

---

## License

MIT — see [LICENSE](LICENSE).

Based on the mathematical foundation by Andrzej Odrzywołek
([arXiv:2603.21852v2](https://arxiv.org/abs/2603.21852)),
Jagiellonian University, Kraków, 2026.