# Appendix F: Generalization Theory, EML Attractors, and Polar Unification

> **Status: Theoretical extensions — unverified empirically on eml-trs.**
> Section F.1 (SLT/LLC correlation) relies on Lan et al. (2026) results
> on Pythia models, not on our own measurements. Section F.2 (Gumbel-Softmax)
> extends the EML-PTQ framework in `scripts/eml_ptq.py` but is not yet
> implemented. Section F.3 (attractors beyond ternary) is a theoretical
> projection. Section F.4 extends Section 8 of the main paper.

---

## F.1 Singular Learning Theory as Formal Foundation for Hypothesis C5

Hypothesis C5 (Section C.5) proposes that EML node count $C_{\text{eml}}$
correlates with generalization capacity via the MDL principle. This appendix
provides a more formal foundation through Singular Learning Theory (SLT).

**The LLC–compressibility connection.** Neural networks are *singular*
models: their Fisher information matrix is degenerate, meaning optimal
parameters form manifolds (flat minima) rather than isolated points. The
Local Learning Coefficient (LLC) measures the effective geometric dimension
of the loss landscape at a solution — lower LLC means the network has
converged to a lower-dimensional singularity, indicating it has learned a
simpler rule rather than memorizing training noise.

Lan et al. (2026) measure compressibility across Pythia models (70M–6.9B
parameters) and find that LLC estimates are **linearly correlated** with
compressibility across quantization and tensor factorization methods. This
provides the missing empirical link: models that compress well (low
$C_{\text{eml}}$) are precisely those that converge to low-LLC singularities
in the loss landscape.

**Why $C_{\text{eml}}$ is a natural LLC proxy.** The four-step EML
complexity measurement procedure (constant folding → ASIS → DAG/CSE →
TRS fixpoint) mirrors the structure of SLT analysis:

| EML Compression Step | SLT Interpretation |
|:---------------------|:-------------------|
| Constant Folding of weights | Identification of posterior parameter distribution |
| ASIS pre-negation | Reduction of optimization noise in dot product |
| DAG/CSE construction | Locating flat minima with degenerate parameter space |
| TRS to fixpoint | Estimation of effective dimensionality (LLC) |

The key argument: a model with low LLC will naturally develop weights that
produce short EML trees — because low LLC implies the model has internalized
a simple generating rule, and simple rules have simple EML representations.
This makes $C_{\text{eml}}$ a **deterministic, post-training** generalization
metric computable without a validation set.

**Tightened PAC-Bayes bound via $C_{\text{eml}}$.** Replacing the
classical bit-length description $|\sigma|$ with $C_{\text{eml}}$ in the
Dziugaite–Roy (2017) framework:

$$L \leq \hat{L} + \sqrt{\frac{C_{\text{eml}} + \log(1/\delta)}{2n}}$$

This bound is tighter than classical approaches because $C_{\text{eml}}$
penalizes algebraic redundancy rather than raw parameter count. A weight
value of $1.0$ activates the TRS identity rule (cost: 0 nodes); a weight
of $0.0$ eliminates entire DAG branches (cost: 0 nodes); a float32 weight
far from any EML attractor costs up to 17 nodes. The prior distribution
$P$ in the PAC-Bayes framework is naturally peaked at EML attractors,
so the KL divergence $D_{KL}(Q \| P)$ penalizes not merely Euclidean
distance from attractors but **algebraic complexity** of the departure.

This extends Hinton & Van Camp (1993)'s original MDL framing of PAC-Bayes
to the continuous algebraic domain, providing what we believe to be the
first PAC-Bayes bound grounded in a formal, computable algebraic IR.

---

## F.2 Gumbel-Softmax: Differentiable Surrogate for $C_{\text{eml}}$

The EML-PTQ objective in `scripts/eml_ptq.py` uses a Straight-Through
Estimator (STE) to backpropagate through the discrete EML cost function.
STE introduces gradient bias and cannot guarantee convergence to global
optima. This section describes a rigorous differentiable alternative.

**Setup.** For each weight $w_{ij}$, define a discrete set of EML
attractors $A = \{a_1, \ldots, a_m\}$ ordered by $C_{\text{eml}}(a_k)$.
Maintain learnable logit vectors $\mathbf{g} \in \mathbb{R}^m$ per weight.

**Gumbel-Softmax relaxation.** Replace argmax sampling with:

$$\tilde{\pi}_k = \frac{\exp\bigl((\log \pi_k + g_k)/\tau\bigr)}
{\sum_{j=1}^{m} \exp\bigl((\log \pi_j + g_j)/\tau\bigr)}$$

where $g_k \sim \text{Gumbel}(0,1)$ and $\tau$ is a temperature parameter
annealed toward zero during training. The forward-pass weight becomes the
convex combination $\tilde{w}_{ij} = \sum_k \tilde{\pi}_k a_k$.

**Differentiable EML cost.** The non-differentiable $C_{\text{eml}}(w)$
becomes a differentiable expected cost:

$$\tilde{\mathcal{C}}_{\text{eml}}(w_{ij}) = \sum_{k=1}^{m}
\tilde{\pi}_k \cdot C_{\text{eml}}(a_k)$$

This is fully differentiable with respect to logits $\pi$, enabling
gradient-based optimization of the EML-PTQ objective without STE bias.

**Updated EML-PTQ objective:**

$$\mathcal{L}_{\text{EML-PTQ}} = \|WX - \tilde{W}X\|_2^2
+ \lambda \sum_{i,j} \tilde{\mathcal{C}}_{\text{eml}}(w_{ij})$$

As $\tau \to 0$: $\tilde{\pi}$ converges to a one-hot vector over attractors,
recovering hard quantization. During training, the smooth landscape allows
standard Adam/SGD to find attractor assignments that minimize both
reconstruction loss and algebraic complexity simultaneously.

**Comparison of gradient estimators:**

| Estimator | Differentiable? | Convergence guarantee | Memory overhead |
|:----------|:---------------:|:---------------------:|:---------------:|
| STE (current) | Pseudo | None (gradient bias) | $O(1)$ |
| **Gumbel-Softmax** | **Yes** | **Asymptotic** | $O(m)$ logits per weight |
| L1 + proximal projection | Approximate | Heuristic | Low |

**Implementation path.** Modify `scripts/eml_ptq.py`:
1. Replace `round()` quantization with Gumbel-Softmax sampling
2. Precompute `C_eml_costs: Dict[float, int]` for each attractor
3. Add temperature schedule: $\tau_t = \tau_0 \cdot e^{-\alpha t}$
4. Track attractor convergence per epoch (existing metric)

---

## F.3 EML Attractors Beyond the Ternary Set

Section C.6 and 10.1 focus on ternary attractors $\{-1, 0, 1\}$. The
full EML attractor landscape is richer.

**Definition.** An EML attractor is any weight value $w \in \mathbb{R}$
where $C_{\text{eml}}(w)$ achieves a local minimum relative to neighboring
float32 values, due to constant folding collapsing algebraic structure.

**Known attractors:**

| Weight value | Why it's an attractor | $C_{\text{eml}}$ cost |
|:------------|:----------------------|:---------------------:|
| $0$ | Eliminates entire DAG branch | 0 |
| $1$ | Identity rule: $x \cdot 1 = x$ | 0 |
| $-1$ | ASIS pre-negation offline | 0 |
| $e \approx 2.718$ | $\mathrm{eml}(1,1) = e$, zero-cost constant | ~2 |
| $1/e \approx 0.368$ | $1/w = e$, absorbed by $\mathrm{eml}(1,1)$ | ~2 |
| $\ln 2 \approx 0.693$ | Appears in fast\_exp/fast\_ln Minimax constants | ~3 |
| $\log_2 e \approx 1.443$ | `LOG2_E` constant in hardware kernels | ~3 |
| Dyadic fractions $k/2^n$ | Match Horner evaluation fixed-points | ~4–8 |

**Implication for quantization.** Standard int8 quantization distributes
256 levels uniformly on $[-\text{max}, +\text{max}]$. EML-guided
quantization would instead cluster levels around the attractor set —
denser near $\{0, \pm 1, \pm e, \pm 1/e\}$ and sparser elsewhere.
This is a non-uniform, algebraically motivated quantization scheme.

**Connection to Minimax approximation.** The fast\_exp Minimax
coefficients in `eml_kernels.wgsl` ($A_0 \approx 1.002$, $A_1 \approx
0.651$, $A_2 \approx 0.344$) are themselves near-attractors: weights
trained to match these values would benefit from hardware acceleration
via the same FMA pipeline used for the approximation itself. This
creates a feedback loop: EML-aware training → weights converge to
Minimax coefficients → inference uses hardware-optimized paths.

---

## F.4 Polar Unification: EML, TurboQuant, and PoPE

Section 8 of the main paper describes the polar coordinate connection
at high level. This appendix makes the three-way algebraic unification
precise.

**The log-domain identity.** Any complex vector $\mathbf{q} = r e^{i\phi}$
satisfies:

$$\ln(\mathbf{q}) = \ln(r) + i\phi$$

In EML: $\ln(r)$ is the magnitude channel (real-valued, subject to EML
TRS), and $\phi$ is the phase channel (angular offset, modified by RoPE).

**TurboQuant — PolarQuant mechanism** (Google Research, ICLR 2026):
Applies random orthogonal rotation to equalize distributions, then
quantizes magnitude $r$ and phase $\phi$ separately using Lloyd-Max
centroids. A one-bit Quantized Johnson-Lindenstrauss (QJL) parity term
compensates residual error at logit computation. Result: 3–4 bits per
KV cache dimension with zero retraining, $\sim4.7\times$ VRAM reduction
verified on LLaMA-3.1-8B and Qwen3-14B.

**PoPE** (Gopalakrishnan et al., 2025): Separates "what" (magnitude,
via softplus to enforce $r \geq 0$) from "where" (phase $\phi$, modified
by additive positional offset $m\theta$). RoPE rotation becomes:

$$\ln(r) + i(\phi + m\theta)$$

In EML: the positional update is **pure phase addition** — zero EML nodes,
zero algebraic cost. Compare with RoPE which requires full trigonometric
matrix multiplication.

**Three-way pipeline:**

```
KV Cache (TurboQuant):  [ln(r) quantized 3-bit] + [φ quantized 3-bit]
                                    ↓
Position update (PoPE):  φ ← φ + mθ  (additive, 0 EML nodes)
                                    ↓
Attention (EML):         eml(ln(r_q) + ln(r_k), exp(-φ_diff))
                         = r_q · r_k · e^{φ_diff}  (dot product in log-domain)
```

**What this eliminates:**

| Classical approach | Unified EML+PoPE+TurboQuant |
|:-------------------|:----------------------------|
| Cartesian-to-polar conversion | Never needed — polar throughout |
| RoPE trigonometric multiply | Phase addition (0 EML nodes) |
| float16 KV cache | 3-bit polar quantization |
| Catastrophic cancellation in dot product | Log-domain subtraction (ASIS) |

**Status.** This unification is theoretical. The three systems were
developed independently and have not been jointly implemented. The
algebraic convergence is exact; the engineering integration requires:
- A PoPE-native attention kernel operating in log-domain
- TurboQuant KV cache decoded directly to EML log-representation
- ASIS dot product operating on $\ln(r)$ components directly

This represents a complete redesign of the transformer attention
computation, not an incremental optimization.

---

## F.5 Summary: Three Contributions of This Appendix

**F.1** formalizes Hypothesis C5 through SLT: LLC linearly correlates
with compressibility on Pythia 70M–6.9B, and $C_{\text{eml}}$ is a
natural, computable LLC proxy. The PAC-Bayes bound with
$|\sigma| = C_{\text{eml}}$ penalizes algebraic redundancy rather
than raw parameter count — tighter than classical approaches.

**F.2** replaces STE in EML-PTQ with Gumbel-Softmax relaxation,
making the EML complexity penalty fully differentiable and convergent.
Expected cost $\tilde{\mathcal{C}}_{\text{eml}} = \sum_k \tilde{\pi}_k
C_{\text{eml}}(a_k)$ enables gradient-based attractor optimization.

**F.3** extends the EML attractor set beyond $\{-1, 0, 1\}$ to include
$e$, $1/e$, $\ln 2$, $\log_2 e$, and Minimax Horner fixed-points.
These define a non-uniform, algebraically motivated quantization grid.

**F.4** makes the EML–TurboQuant–PoPE three-way unification precise:
all three systems operate naturally in the log-polar domain, enabling
a unified inference pipeline with 0-node positional updates, 3-bit
KV cache, and ASIS log-domain attention computation.

---

*Sources: Lan et al. (2026), "Compressibility Measures Complexity: MDL
Meets SLT"; Dziugaite & Roy (2017), "Computing Nonvacuous Generalization
Bounds for Deep (Stochastic) Neural Networks with Many Parameters";
Hinton & Van Camp (1993), "Keeping Neural Networks Simple"; Willsey
et al. (2021), "egg: Fast and Extensible Equality Saturation";
Gopalakrishnan et al. (2025), "PoPE: Polar Coordinate Position Embeddings";
Google Research (2026), "TurboQuant".*
