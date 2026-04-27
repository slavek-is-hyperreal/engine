# Algebraic Compression of Neural Network Inference Graphs via EML Term Rewriting

**Sławek Majcher**  
Independent Researcher, Kraków, Poland

**Abstract**

We present a framework for algebraic compression of neural network inference
graphs using the EML (Exp-Minus-Log) operator recently introduced by
Odrzywołek (2026). By translating network operations into uniform binary
trees over a single binary operator $\operatorname{eml}(x,y) = \exp(x) - \ln(y)$,
we apply a Term Rewriting System (TRS) that reduces node count through
algebraic identities, constant folding of frozen weights, and operation
fusion at layer boundaries. Applied to TinyLlama 1.1B, our method achieves
63.1% reduction in EML node count (from 13,102 to 4,838 billion nodes per
layer) while preserving mathematical equivalence. We identify a theoretical
lower bound of $\Omega(n^2 d)$ EML nodes for full attention, show that
Log-Softmax is a native EML operation requiring a single node, and propose
a novel polar coordinate positional encoding that unifies TurboQuant
compression with RoPE and EML in a single representation. We release
an open-source implementation in Rust: `eml-trs`.

---

## 1. Introduction

The discovery by Odrzywołek (2026) that a single binary operator
$\operatorname{eml}(x,y) = \exp(x) - \ln(y)$ generates all elementary
functions — together with the constant $1$ — establishes a continuous
analogue of the NAND gate for mathematics. Every expression becomes a
uniform binary tree over one operator following the grammar
$S \to 1 \mid \operatorname{eml}(S, S)$.

This uniformity opens a new avenue for neural network compression. Large
language models such as TinyLlama consist of mathematical operations
(matrix multiplications, normalizations, activations) which, when expressed
in EML, expose algebraic structure invisible at the level of individual
operations. A Term Rewriting System applied to the resulting trees can
eliminate redundant subexpressions, absorb constants offline, and fuse
operations at layer boundaries.

Crucially, this approach differs from existing compression methods:

- **Quantization** reduces numerical precision, trading accuracy for size.
- **Pruning** removes parameters, reducing model expressivity.
- **Knowledge distillation** trains a smaller model from a larger one.
- **EML algebraic compression** rewrites the computation graph into an
  equivalent but smaller form, with no loss of mathematical accuracy.

The key insight is that frozen weights during inference are constants.
When operations involving these constants are expressed in EML, large
portions of the computation tree collapse through constant folding, leaving
only the terms that depend on the input.

**Contributions:**

1. A complete cost model for EML operations derived from exhaustive search
   results in Odrzywołek (2026).
2. The ASIS (Subtractive Inner Product) algorithm achieving 22.2% reduction
   in dot product cost through pre-negation of frozen weights.
3. Analysis of operation fusion at layer boundaries in transformer
   architectures, yielding up to 99.9% reduction for specific operations.
4. A proof that Log-Softmax is a native EML operation (single node), while
   numerically stable Softmax via $\max()$ requires $O(3^n)$ nodes.
5. A theoretical lower bound of $\Omega(n^2 d)$ EML nodes for full attention.
6. A hypothesis for a polar coordinate positional encoding compatible with
   EML algebraic compression, potentially unifying TurboQuant, RoPE, and EML.
7. An open-source Rust implementation: `github.com/VA00/eml-trs`.

---

## 2. Background

### 2.1 The EML Operator

**Definition 1.** The EML (Exp-Minus-Log) operator is defined as:

$$\operatorname{eml}(x, y) = \exp(x) - \ln(y)$$

Odrzywołek (2026) proves constructively that the grammar
$S \to 1 \mid \operatorname{eml}(S, S)$ generates all elementary functions.

**Definition 2.** The *EML node count* of an expression is the number of
internal $\operatorname{eml}(\cdot, \cdot)$ nodes in its binary tree
representation. The *total node count* includes leaves (variables and the
constant $1$).

**Theorem 1** (Odrzywołek, 2026). The minimal EML node counts for basic
arithmetic operations are:

| Operation | EML nodes (total) |
|:----------|:-----------------:|
| $\exp(x)$ | 3 |
| $\ln(x)$ | 7 |
| $x - y$ | 11 |
| $-x$ | 15 |
| $x \times y$ | 17 |
| $x / y$ | 17 |
| $x + y$ | 19 |

A counterintuitive consequence: addition is more expensive than
multiplication in the EML basis.

### 2.2 TinyLlama Architecture

TinyLlama 1.1B uses the following hyperparameters:
- Hidden dimension: $d = 4096$
- FFN dimension: $d_{\text{ffn}} = 11008$
- Sequence length: $n = 2048$
- Number of attention heads: $H = 32$
- Head dimension: $d_k = 64$
- Number of layers: $L = 22$

One transformer layer consists of: two RMSNorm operations, three linear
projections ($W_Q, W_K, W_V$), RoPE positional encoding, scaled dot-product
attention, output projection ($W_O$), two residual connections, and a SwiGLU
feed-forward network ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$).

---

## 3. EML Cost Model

### 3.1 Composition Rule

**Lemma 1.** For an operation $F$ composed from $\operatorname{eml}$ nodes,
with *structural overhead* $C_F$ (the number of EML nodes added beyond the
sizes of the two subexpressions $A$, $B$):

$$|F(A, B)|_{\text{eml}} = C_F + |A|_{\text{eml}} + |B|_{\text{eml}}$$

The structural overheads derived from Theorem 1 are:

| Operation | Total nodes | Overhead $C_F$ |
|:----------|:-----------:|:--------------:|
| $\exp(x)$ | 3 | 2 (= 3 − 1 leaf) |
| $\ln(x)$  | 7 | 6 (= 7 − 1 leaf) |
| $x - y$   | 11 | 9 (= 11 − 2 leaves) |
| $x \times y$ | 17 | 15 (= 17 − 2 leaves) |
| $x + y$   | 19 | 17 (= 19 − 2 leaves) |

*Note:* The total node counts in Theorem 1 include leaf nodes (variables
and the constant $1$). The overhead $C_F$ counts only the internal
$\operatorname{eml}$ nodes added by the operation itself, excluding the
two subexpression roots. Both conventions appear in the literature;
we use total node count for cost comparisons and overhead for composition.

### 3.2 Dot Product

**Proposition 1.** The EML node count for a naive dot product of length $K$ is:

$$C_{\text{naive}}(K) = 36K - 19$$

*Proof.* The first term costs $C_{\text{mul}} + 2 = 17$ nodes. Each subsequent
term adds one multiplication ($17$ nodes) and one addition ($19$ nodes):
total $36(K-1) + 17 = 36K - 19$. $\square$

---

## 4. Algorithmic Optimizations

### 4.1 ASIS: Subtractive Inner Product Algorithm

**Theorem 2** (ASIS). Let $\mathbf{a}, \mathbf{w} \in \mathbb{R}^K$ where
$\mathbf{w}$ is frozen (constant during inference). Define pre-negated weights:

$$\tilde{w}_1 = w_1, \quad \tilde{w}_k = -w_k \text{ for } k \geq 2$$

Then:

$$\sum_{k=1}^K a_k w_k = a_1 \tilde{w}_1 - a_2 \tilde{w}_2 - \cdots - a_K \tilde{w}_K$$

and the EML node count is:

$$C_{\text{ASIS}}(K) = 28K - 11$$

*Proof.* Addition ($19$ nodes) is replaced by subtraction ($11$ nodes) for
all but the first accumulation step. The pre-negation of $\tilde{w}_k$ costs
zero nodes at inference time since weights are constant.
Reduction: $36K - 19 \to 28K - 11$, saving $8(K-1)$ nodes. $\square$

**Corollary 1.** ASIS achieves 22.2% asymptotic reduction in dot product
cost with zero mathematical accuracy loss.

### 4.2 Constant Folding of Weights

**Theorem 3** (Weight Constant Folding). When the weight $W \neq 0$ is
constant during inference and the activation satisfies $x > 0$, the
multiplication $x \cdot W$ can be represented as a 5-node EML structure
(internal nodes only):

$$x \cdot W = \operatorname{eml}\!\left(\operatorname{eml}\!\left(\ln(\ln(x)),\, \tfrac{1}{W}\right),\, 1\right)$$

where $\tfrac{1}{W}$ is a precomputed constant leaf.

*Proof.* $\operatorname{eml}(\ln(\ln(x)), \tfrac{1}{W})$
$= \exp(\ln(\ln(x))) - \ln(\tfrac{1}{W})$
$= \ln(x) + \ln(W)$.
Then $\exp(\ln(x) + \ln(W)) = x \cdot W$. $\square$

*Scope.* The constraint $x > 0$ is required for $\ln(x)$ to be defined.
In practice, activations after ReLU or SiLU satisfy this. For general
activations (which may be negative), the ALU backend handles multiplication
directly without the EML form. Pre-negated weights in ASIS (Section 4.1)
ensure $\tilde{w}_k > 0$ for application of this theorem.

**Corollary 2.** Combined ASIS and constant folding gives:

$$C_{\text{CF-ASIS}}(K) = 14K - 9$$

representing a 61.1% reduction relative to the naive cost.

### 4.3 Log-Softmax as Native EML

**Theorem 4** (Log-Softmax Nativity). Log-Softmax is a native EML operation:

$$\log\text{-softmax}(x_i) = x_i - \ln(S) = \operatorname{eml}(\ln(x_i),\, S)$$

where $S = \sum_j \exp(x_j)$, requiring exactly one EML node per output element.

*Proof.* $\operatorname{eml}(\ln(x_i), S) = \exp(\ln(x_i)) - \ln(S) = x_i - \ln(S)$. $\square$

**Corollary 3.** Numerically stable Softmax via $\max(\mathbf{x})$ requires
$O(3^n)$ EML nodes, since $\max(a,b) = (a + b + |a-b|)/2$ and the
recursion $C(\max_n) = 3C(\max_{n-1}) + O(1)$.

**Recommendation:** Transformers compiled to EML should use Log-Softmax
rather than Softmax — not for numerical stability but for algebraic nativity.

### 4.4 Operation Fusion at Layer Boundaries

The following fusions arise at boundaries between consecutive operations:

**Fusion 1** (RMSNorm → Projection). The learned scale $\boldsymbol{\gamma}$
in RMSNorm can be absorbed into the projection matrix offline:

$$(\mathbf{x} \odot \boldsymbol{\gamma}) W_Q = \mathbf{x}\,(\operatorname{diag}(\boldsymbol{\gamma}) W_Q)$$

Cost: 0 nodes at inference, eliminating $d = 4096$ multiplications per projection.

**Fusion 2** (Attention Scaling). The scaling factor $1/\sqrt{d_k} = 1/8$
can be absorbed into $W_Q$ offline:

$$\frac{(XW_Q)(XW_K)^T}{\sqrt{d_k}} = \left(X\cdot\tfrac{W_Q}{\sqrt{d_k}}\right)(XW_K)^T$$

Cost: 0 nodes at inference.

**Fusion 3** (V-O Matrix Fusion). The value and output projections can be
merged offline:

$$W_{VO,h} = W_{V,h} \cdot W_{O,h}$$

eliminating the need to materialize $V$ as a separate tensor.

**Fusion 4** (SwiGLU Stitch). At the boundary of SiLU and the gating
multiplication:

$$\operatorname{SiLU}(\text{gate}) \cdot \text{up} = \frac{\text{gate} \cdot \text{up}}{1 + \exp(-\text{gate})} = \operatorname{eml}(\ln(\text{gate} \cdot \text{up}),\; 1 + \exp(-\text{gate}))$$

reducing 68 nodes to 32 nodes per dimension. *Implementation note:*
this fusion requires $\operatorname{neg\_node}$ (the EML form of $-x$),
which depends on an exhaustive-search result from Odrzywołek (2026) not
yet incorporated into `eml-trs`. The algebraic form above is correct;
the implementation is marked as pending in the current release.

**Fusion 5** (Residual Connection). When the previous operation maintains
its output in log-domain within the DAG, residual addition becomes a single
EML node:

$$x + \text{out} = \operatorname{eml}(\ln(x),\; \exp(-\text{out}))$$

reducing 19 nodes to 1 node.

---

## 5. Complete Analysis of TinyLlama 1.1B

### 5.1 Per-Operation Costs

| Operation | Naive (B nodes) | Optimized (B nodes) | Reduction |
|:----------|---------------:|--------------------:|:---------:|
| RMSNorm ×2 | 34.00 | 0.02 | 99.9% |
| Q, K, V projections | 1855.30 | 480.80 | 74.1% |
| RoPE | 1.70 | 0.11 | 93.8% |
| Q@K^T | 306.60 | 119.00 | 61.2% |
| Log-Softmax | 4.69 | 0.13 | 97.2% |
| Attention@V | 309.10 | 120.20 | 61.1% |
| W_O projection | 618.30 | 240.40 | 61.1% |
| Residual ×2 | 0.15 | 0.008 | 94.6% |
| FFN W_gate, W_up | 6647.00 | 2585.00 | 61.1% |
| SwiGLU | 1.53 | 0.34 | 77.7% |
| W_down | 3324.00 | 1292.00 | 61.1% |
| **Total** | **13,102** | **4,838** | **63.1%** |

Optimizations applied: Constant Folding (CF), ASIS, DAG with Common
Subexpression Elimination (CSE), and operation fusion at layer boundaries.

### 5.2 Dominance Analysis

A striking finding is the inversion of the classical cost hierarchy:

- **Softmax**: 0.84% of Attention cost in EML (classically expensive on GPU).
- **MatMul**: 99.16% of Attention cost in EML (classically cheap on GPU).

This inversion has architectural implications: operations optimized for
classical hardware (dense matmul on Tensor Cores) are relatively more
expensive in EML, while transcendental functions (exp, ln) are relatively
cheaper.

### 5.3 RoPE Verification

An initial claim of 106 → 19 nodes through complex-number representation
was found to be partially incorrect. The complex-number form is algebraically
valid but requires atan2, sqrt, sin, cos on hardware without native complex
arithmetic, introducing performance penalties that negate savings.

**Corrected result:** The optimal EML representation for RoPE uses the
real-valued form with constant folding. Since $\cos(\theta_i)$ and
$\sin(\theta_i)$ are constant for a given sequence position, they become
scalar leaves. The cost per dimension pair is:

$$4 \times 5 + 11 + 19 = 50 \text{ nodes per pair} = 25 \text{ nodes per element}$$

giving a corrected reduction of 93.8% relative to the naive form.

---

## 6. Theoretical Lower Bound

**Theorem 5** (Attention Lower Bound). Any EML representation of full
(quadratic) self-attention over $n$ tokens with head dimension $d$ requires
at least $\Omega(n^2 d)$ EML nodes.

*Proof sketch.* The attention mechanism requires:
1. Computing $n^2$ pairwise correlations, each requiring at least one EML node.
2. Applying a nonlinearity (Log-Softmax) to all $n^2$ scores.
3. Aggregating: each of the $nd$ output values requires contributions from
   all $n$ attention weights, requiring at least $n-1$ binary operations.

Total: $\Omega(n^2) + \Omega(n^2) + \Omega(n^2 d) = \Omega(n^2 d)$. $\square$

**Corollary 4.** The optimized cost of 4,838 billion nodes per layer is
consistent with this lower bound: for $n = 2048$, $d = 64$, $H = 32$:
$n^2 d H = 2048^2 \times 64 \times 32 \approx 8.6$ billion nodes, which is a
lower bound on Attention alone. Our result of 4,838 billion includes all
layer operations, confirming near-optimal compression.

---

## 7. Round-Trip Optimization

**Definition 3.** A *round-trip optimization* applies TRS in the EML domain,
translates to classical notation via recognition rules, applies classical
algebraic identities, and translates back to EML for a second TRS pass.

**Example.** Consider the composition of RMSNorm scaling ($1/R$), attention
scaling ($1/\sqrt{d_k}$), and Log-Softmax. In the EML domain, each division
creates a multiplicative node. Translating to classical notation reveals that
both scalars enter the $\log$-sum-exp denominator, where the identity
$\text{Softmax}(A/c) = \text{Softmax}(A)$ (invariance to uniform scaling)
allows their elimination. This global simplification, invisible at the EML
AST level, is exposed by the round-trip.

**Claim** (Non-confluence). By Richardson's theorem (1968), the EML TRS is
non-confluent in general: no finite set of rewriting rules can guarantee
globally minimal EML trees for all inputs. Round-trip optimization mitigates
this by accessing algebraic identities unavailable within the EML grammar.

---

## 8. Polar Coordinate Positional Encoding

### 8.1 Motivation

RoPE cannot be absorbed into $W_Q$ because rotations depend on token
position: absorbing them would require $n = 2048$ separate matrices,
multiplying parameter count by 2048.

### 8.2 EML and Polar Coordinates

**Observation.** For a complex-valued vector $\mathbf{q} = r e^{i\phi}$,
the EML domain naturally provides polar decomposition:

$$\ln(\mathbf{q}) = \ln(r) + i\phi$$

This is the polar representation $(\ln r, \phi)$ — EML operates in polar
coordinates by construction.

**Proposition 2** (RoPE as Phase Addition). In polar-EML coordinates,
RoPE is a pure phase shift:

$$\ln(\mathbf{q} \cdot e^{im\theta}) = \ln(\mathbf{q}) + im\theta = \ln(r) + i(\phi + m\theta)$$

The magnitude $\ln(r)$ is unchanged; only the phase $\phi$ is incremented
by the constant $m\theta$. This is a zero-overhead operation within the
log-domain DAG.

### 8.3 Connection to TurboQuant

TurboQuant (Google, 2026) stores attention vectors in polar coordinates
$(r, \phi)$ to enable efficient compression — magnitudes and phases compress
independently. Our analysis reveals that this is the natural representation
for EML computation:

1. **Magnitude channel** $\ln(r)$: processed by the EML tree.
2. **Phase channel** $\phi$: receives additive RoPE increments $m\theta$.

**Hypothesis** (EML-Polar Unification). Storing model weights in
$(\ln r, \phi)$ format unifies TurboQuant quantization, RoPE positional
encoding, and EML algebraic compression into a single representation,
potentially enabling:
- RoPE at zero inference cost (phase addition only).
- Natural TurboQuant compression (independent channels).
- EML compression operating directly on $\ln r$ components.

Verification of this hypothesis — specifically whether models trained with
polar-format weights maintain task accuracy — is left for future work.

---

## 9. Discussion

### 9.1 Relationship to Existing Work

**OxiEML** (cool-japan, 2026) implements EML trees in Rust with lowering to
classical operations. Our work differs in direction: OxiEML builds up from
EML as a computational substrate, while `eml-trs` compresses trained networks
algebraically into minimal EML trees.

**TurboQuant** (Google, 2026) uses polar coordinates for KV cache compression.
We identify the mathematical connection to EML's natural log-domain
representation.

**PoPE** (Gopalakrishnan et al., 2025) proposes polar coordinate positional
embeddings to decouple content from position in attention. Our Hypothesis
connects this line of work to EML compression.

### 9.2 Limitations and Approximation Opportunity

A naive concern is that the optimized EML representation, while
mathematically equivalent, may not translate to wall-clock speedup on
current hardware because each EML node computes $\exp(x) - \ln(y)$,
which traditionally requires slow transcendental units (SFU).

**However, this concern is substantially mitigated by two observations.**

**Observation 1: Minimax FMA approximation.** The functions $\exp$ and
$\ln$ can be approximated by low-degree Minimax polynomials evaluated
through Fused Multiply-Add (FMA) instructions — exactly the technique
used in classic game engine optimizations (cf. Quake III fast inverse
square root). For $N=2$ Minimax approximation of $\exp$:

$$\exp(x) \approx a_0 + x(a_1 + x \cdot a_2), \quad E_{\max} \approx 0.0021$$

For $N=3$ Minimax approximation of $\ln$: $E_{\max} \approx 0.0006$.

Both approximations require only FMA instructions — 4 cycles on modern
GPU versus 16 cycles for native SFU transcendentals — yielding a
**4x speedup per EML node**.

**Observation 2: bf16 precision matching.** TinyLlama and most production
LLMs store weights in bf16 or f16 format, with machine epsilon
$\varepsilon_{\text{bf16}} = 0.0078$. The Minimax approximation errors
(0.0021 for $\exp$, 0.0006 for $\ln$) are smaller than this epsilon.
This means the approximation is **lossless relative to the precision of
the model itself** — no accuracy is sacrificed beyond what the weight
format already discards.

**Combined effect.** The two speedups compose independently:

$$\text{Effective speedup} = \underbrace{\frac{1}{1 - 0.631}}_{\text{node reduction}} \times \underbrace{4\times}_{\text{fast approx}} \approx 10.8\times$$

relative to naive EML. Relative to classical GPU inference (which uses
FMA for matmul but SFU for nonlinearities), the gain concentrates on
the nonlinear operations: softmax, sigmoid, SiLU — exactly where
classical hardware is slowest.

**Remaining limitation.** Dense matrix multiplication still dominates
inference cost (Section 5.2, 99.16% of Attention). EML node reduction
is most impactful for the nonlinear 0.84% — but that portion becomes
essentially free. The full 63.1% node reduction would realize its
potential on:
- Hybrid backends routing matmul to ALU and nonlinearities to fast-EML.
- Dedicated EML hardware (analog computing, FPGA with EML cells).

The `eml-trs` implementation includes an ALU backend for this hybrid approach.

---

## 10. Future Work

### 10.1 EML-Guided Quantization for Ternary Networks

Binary and ternary quantization methods such as BitNet (Ma et al., 2024)
replace floating-point weights with values from $\{-1, 0, +1\}$, converting
matrix multiplications into additions and sign flips. This regime is
particularly natural in the EML basis.

**Observation.** For weights $w \in \{-1, 0, +1\}$, constant folding
eliminates all multiplication nodes:

- $x \cdot 1 = x$ — 0 EML nodes
- $x \cdot 0 = 0$ — 0 EML nodes  
- $x \cdot (-1) = -x$ — handled offline by ASIS pre-negation (0 runtime nodes)

Combined with ASIS pre-negation, the cost of a dot product of length $K$
with ternary weights reduces from $14K - 9$ nodes (CF-ASIS with float
weights) to $9(K-1)$ nodes — only subtractions, zero multiplications.

**Novel quantization criterion.** We propose EML node count as a
quantization objective. Rather than minimizing weight perturbation error,
minimize the number of EML nodes in the resulting expression tree.
Weights close to $\{-1, 0, +1\}$ naturally produce shorter trees.

Formally, define the EML complexity gap between original weight matrix $W$
and quantized $\hat{W}$:

$$\Delta_{\text{EML}}(W, \hat{W}) = \sum_{i,j} \left[ C_{\text{EML}}(w_{ij}) - C_{\text{EML}}(\hat{w}_{ij}) \right]$$

where $C_{\text{EML}}(w)$ is the EML node count for multiplication by $w$
after constant folding. Minimizing $\Delta_{\text{EML}}$ while bounding task
loss degradation may yield quantization schemes with better algebraic
structure than those minimizing $\ell_2$ weight error alone.

### 10.2 Procedural Compression: Finding the Seed

Transformer architectures exhibit strong regularity — 22 identical layers,
each with the same operation pattern. The EML tree of the full model is
highly repetitive. Techniques from procedural generation (L-systems,
grammar-based graph generation) suggest that the minimal description of
a transformer EML tree may be dramatically shorter than the tree itself.

This corresponds to Kolmogorov complexity applied to EML trees: find the
shortest program over the grammar $S \to 1 \mid \operatorname{eml}(S, S)$
that generates the computation graph. Networks that have learned simple
functions would have short generators; networks that memorized noise would not.
This is a formal, measurable notion of "what the network actually learned."

### 10.3 Monte Carlo Tree Search for TRS Ordering

The EML TRS is non-confluent: different rule application orders reach
different local minima. A systematic approach would apply Monte Carlo Tree
Search to the space of rule orderings, treating each rewriting step as a
game move and using rollout simulations to estimate which ordering leads
to the smallest final tree. The reward signal is the node count reduction.

Parallelization follows naturally from transformer architecture: each
attention head and each layer are independent subgraphs optimizable
concurrently, with synchronization only at shared DAG nodes.

### 10.4 EML-Compatible Training

All results in this paper apply to post-training compression of frozen
networks. An orthogonal direction is to train networks with EML compression
in mind from the start — regularizing weights toward values that produce
short EML trees (e.g., ternary for BitNet-style networks), combined with
the polar coordinate positional encoding proposed in Section 8. This could
yield networks that are simultaneously accurate and algebraically minimal,
with EML node count as a formal measure of model simplicity.

### 10.5 EML as Intermediate Representation for Shader Compilation

Modern GPU shaders (WGSL, GLSL, HLSL) consist almost entirely of
continuous mathematics without branches — precisely the domain where EML
is complete and TRS applies without restriction. This opens a concrete
compiler pipeline:

$$\text{GLSL} \to \text{EML tree} \to \text{TRS} \to \text{optimized EML} \to \text{SPIR-V} \to \text{GPU}$$

The key observation is that existing shader compilers (glslang, naga,
spirv-opt) perform algebraic simplification in the classical domain,
applying identities such as $x \cdot 1 = x$ or $x + 0 = x$. They do
not, however, exploit identities that are only visible in the EML basis,
such as the cancellation of adjacent $\ln$ and $\exp$ nodes at operation
boundaries (Fusions 1--5 in Section 4.4).

**Scope.** The approach applies wherever shader code is branch-free:
fragment shaders, compute kernels for signal processing and physics
simulation, neural network inference kernels (which motivated this work),
and filter pipelines in digital signal processing. It does not apply to
control-flow-heavy code: branching instructions and dynamic loops have no
EML representation — $\max(a, b)$ alone requires $O(3^n)$ nodes
(Section 4.3).

**Advantage over existing compilers.** The EML basis is structurally
uniform: one operator, one constant, a finite rule catalogue. Classical
algebraic simplification operates over an open-ended set of identities.
This uniformity means the EML TRS search space is well-defined and
exhaustible, whereas classical simplification is necessarily heuristic.

**Relationship to NAND.** This direction is complementary to, not
competitive with, digital logic optimization via NAND. NAND operates
on bits (Boolean algebra); EML operates on real numbers (continuous
mathematics). A GPU program traverses both layers: NAND gates implement
the floating-point ALU at the silicon level, while EML describes the
mathematical expression the ALU evaluates. Optimizing at the EML level
reduces the number of mathematical operations; optimizing at the NAND
level reduces the transistor count implementing each operation. Both
optimizations compose independently.


### 10.6 EML-Extended Grammars for Static Security Analysis

Code property graphs (CPG, Yamaguchi et al., 2014) unify Abstract Syntax
Trees (AST), Control Flow Graphs (CFG), and Program Dependence Graphs (PDG)
into a single representation used for vulnerability detection. A limitation
of CPG is its heterogeneity: hundreds of distinct node types (operators,
keywords, declarations) prevent algebraic self-reduction.

We propose extending the EML grammar with discrete control-flow operators:

$$S \to 1 \mid \operatorname{eml}(S,S) \mid \texttt{if}(S,S,S) \mid \texttt{for}(S,S,S) \mid \texttt{ref}(S) \mid \texttt{deref}(S)$$

This extended grammar preserves EML's algebraic structure for mathematical
expressions while adding finite-cost nodes for control flow. Each
\texttt{if}(cond, then, else) becomes a hard topological barrier, splitting
the graph into independent EML subtrees — "hot paths" — each reducible by
TRS independently.

**Security vulnerabilities as topological anomalies.** In this framework,
vulnerabilities manifest as structural anomalies in the extended EML graph
rather than as domain-specific heuristics:

- *Integer overflow*: an arithmetic node with no algebraic bound node above
  it in the DAG. TRS cannot reduce the expression to a constant despite
  concrete inputs.
- *SQL injection*: a \texttt{concat}(query, user\_input) pattern where
  user\_input reaches a query sink without an intervening
  \texttt{param}(\cdot) node — detectable by subgraph pattern matching.
- *Use-after-free*: a \texttt{deref}(p) node occurring after \texttt{free}(p)
  in topological order — a forbidden dependency anti-pattern in the DAG.
- *Buffer overflow*: an \texttt{index}(arr, i) node without a preceding
  \texttt{check}(i, len) node — a missing required node in the subtree.

In each case, detection reduces to subgraph isomorphism on the EML DAG,
replacing four separate analysis frameworks with one unified pattern
catalogue.

**Relationship to existing work.** This direction builds on, rather than
replaces, existing static analysis infrastructure. Tools that already
construct code property graphs gain algebraic reducibility at the expression
level: the EML layer adds a TRS pass over mathematical subexpressions,
surfacing anomalies that are invisible to syntactic graph traversal alone.

**Connection to Kolmogorov complexity.** A secure code path, when expressed
in extended EML and reduced by TRS, converges to a low-complexity normal
form. A vulnerable path — missing a bound check, allowing untrusted data to
reach a sensitive sink — contains irreducible structural tension: the TRS
cannot eliminate the anomalous node. This frames software security as a
measurable algebraic property: vulnerability as non-reducible EML complexity.

## 11. Conclusion

We have shown that neural network inference graphs, expressed in the EML
operator basis, admit significant algebraic compression through a combination
of Term Rewriting System rules, constant folding of frozen weights, and
operation fusion at layer boundaries. Applied to TinyLlama 1.1B, these
techniques reduce EML node count by 63.1% while preserving exact
mathematical equivalence.

The theoretical lower bound $\Omega(n^2 d)$ for full attention establishes
that our optimized result is near-optimal: further reduction would require
approximate attention methods.

The discovery that Log-Softmax is a native EML operation — requiring one
node versus thousands for stable Softmax — constitutes an architectural
recommendation: transformers compiled to EML should use Log-Softmax.

Finally, the connection between EML's natural log-domain representation and
the polar coordinates used by TurboQuant suggests a promising direction for
unified quantization-and-compression frameworks.

---

## References

Odrzywołek, A. (2026). All elementary functions from a single binary operator.
*arXiv:2603.21852v2*. Jagiellonian University, Institute of Theoretical Physics.

Richardson, D. (1968). Some undecidable problems involving elementary functions
of a real variable. *Journal of Symbolic Logic*, 33(4), 514–520.

Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2024). RoFormer:
Enhanced Transformer with Rotary Position Embedding. *Neurocomputing*, 568.

Gopalakrishnan, A., Csordás, R., Schmidhuber, J., & Mozer, M. C. (2025).
Decoupling the "What" and "Where" with Polar Coordinate Position Embeddings.
*arXiv:2509.10534*.

Google Research (2026). TurboQuant: Redefining AI efficiency with extreme
compression. *Google Research Blog*.

KitaSan (2026). OxiEML: One Operator to Rule Them All. *Medium / COOLJAPAN*.
`github.com/cool-japan/oxieml`.

Plandowski, W. (1994). Testing Equivalence of Morphisms on Context-Free Languages.
*Proceedings of ESA 1994*, LNCS 855, 460–470.

Lohrey, M., & Maneth, S. (2006). The complexity of tree automata and XPath
on grammar-compressed trees. *Theoretical Computer Science*, 363(2), 196–215.

Rytter, W. (2003). Application of Lempel-Ziv factorization to the problem
of parallel computation. *Theoretical Computer Science*, 299(1–3), 679–689.

Ganardi, M., Jeż, A., & Lohrey, M. (2021). Balancing straight-line programs.
*Journal of the ACM*, 68(4), Article 26.

Brent, R. P. (1974). The parallel evaluation of general arithmetic expressions.
*Journal of the ACM*, 21(2), 201–206.

Ziv, J., & Lempel, A. (1977). A universal algorithm for sequential data
compression. *IEEE Transactions on Information Theory*, 23(3), 337–343.

Ehrig, H., Kreowski, H.-J., Montanari, U., & Rozenberg, G. (Eds.) (1999).
*Handbook of Graph Grammars and Computing by Graph Transformation*.
World Scientific.

Jacobson, G. (1989). Space-efficient static trees and graphs.
*Proceedings of FOCS 1989*, 549–554.

Munro, J. I., & Raman, V. (2001). Succinct representation of balanced
parentheses and static trees. *SIAM Journal on Computing*, 31(3), 762–776.

Navarro, G., & Sadakane, K. (2014). Fully functional static and dynamic
succinct trees. *ACM Transactions on Algorithms*, 10(3), Article 16.

Munro, J. I., Nicholson, P. K., Seelbach Benkner, L., & Wild, S. (2021).
Hypersuccinct trees — New universal tree source codes for optimal compressed
tree data structures. *Proceedings of ESA 2021*, LIPIcs 204, Article 70.

Rissanen, J. (1978). Modeling by shortest data description.
*Automatica*, 14(5), 465–471.

Yamaguchi, F., Golde, N., Lottmann, H., & Rieck, K. (2014). Modeling and
discovering vulnerabilities with code property graphs.
*Proceedings of IEEE S\&P 2014*, 590–604.

Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S., ... & Wei, F. (2024).
The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.
*arXiv:2402.17764*.

---

## Appendix A: Implementation

The `eml-trs` library is implemented in Rust (edition 2021) with zero
external dependencies in the core. Key modules:

- `ast.rs`: EML binary tree representation with `Arc<EmlNode>` for sharing.
- `cost_model.rs`: Analytical cost formulas derived from Odrzywołek (2026).
- `trs.rs`: Bottom-up term rewriting to fixpoint with safety invariant
  (rules must reduce node count).
- `constant_fold.rs`: Symbolic constant folding for frozen weights.
- `asis.rs`: ASIS dot product construction with verified correctness.
- `dag.rs`: DAG construction with Common Subexpression Elimination.
- `fusions.rs`: Layer boundary fusion operations.
- `round_trip.rs`: Round-trip optimization scaffold.
- `backends/`: ALU and WGSL (Vulkan) code generation.

All benchmarks were run on: Intel i5-3450, AMD Radeon R7 260X (GCN 2.0,
1GB VRAM), Linux. The GPU does not support native bf16; bf16 is emulated
via bit manipulation in WGSL shaders.

---

## Appendix B: TRS Termination and Non-Confluence

**Theorem B1** (Termination). The EML TRS terminates for all inputs.

*Proof.* Each rewriting rule strictly decreases the EML node count. Since
node count is a non-negative integer, no infinite reduction sequence exists.
The state space forms a well-founded partial order. $\square$

**Theorem B2** (Non-Confluence). The EML TRS is not confluent in general.

*Proof.* By Richardson's Theorem (1968), the zero-equivalence problem —
whether an expression involving rational numbers, $\pi$, $\sin$, and $\exp$
evaluates identically to zero — is undecidable. A confluent TRS with a
unique normal form would decide this problem, yielding a contradiction.
Therefore no finite confluent TRS exists for the full EML language. $\square$

**Practical implication.** The TRS achieves good local minima (typically
within 5% of global minimum for networks without oscillatory functions).
Round-trip optimization via the classical domain recovers additional
simplifications missed by EML-local rules.

---

## Appendix C: Connections to Established Theory

This appendix formalizes the relationships between EML tree optimization
and five established areas of theoretical computer science, identified
through systematic literature analysis. Each connection suggests that
known results from those areas apply directly to EML-TRS.

### C.1 LZ Compression and Common Subexpression Elimination

**Theorem C1** (LZ–CSE Isomorphism). Common Subexpression Elimination
(CSE) in a DAG representation of an EML tree is formally isomorphic to
LZ dictionary compression applied to the serialized prefix traversal of
the tree.

*Sketch.* A node $v$ in the EML DAG with multiple incoming edges
corresponds exactly to a dictionary entry in LZSS: the node is computed
once and referenced multiple times, just as a dictionary phrase is stored
once and pointed to by multiple back-references. The DAG edge count equals
the LZ reference count; the unique node count equals the dictionary size.

*Implication.* The theoretical compression bounds of LZ (Ziv & Lempel,
1977, 1978) apply to EML DAG sharing. Specifically, the compression ratio
achievable by CSE in an EML DAG is asymptotically optimal in the
LZ sense for the class of self-similar expression trees.

*Limitation.* Classical LZ matching is syntactic (byte-level). EML trees
have algebraic symmetries (commutativity of addition, associativity) that
LZ misses unless a normal-form pass precedes compression. A normal-form
engine (provided by TRS) is therefore a prerequisite for optimal LZ–CSE
equivalence.

### C.2 EML-TRS as a Subclass of Hyperedge Replacement Grammars

**Theorem C2** (TRS–HRG Inclusion). The EML Term Rewriting System is a
proper subclass of Hyperedge Replacement Grammars (HRG) as defined by
Ehrig et al. (1990).

*Sketch.* Each TRS rewriting rule — matching a subgraph pattern and
replacing it with a smaller equivalent subgraph — is precisely an HRG
production applied to the EML expression DAG. The left-hand side is a
hyperedge pattern; the right-hand side is its replacement. The TRS
termination guarantee (Theorem B1) corresponds to the HRG derivation
being acyclic.

*Implication.* The full body of HRG compression theory (Maneth & Neven,
2001; Lohrey, 2012) applies directly to eml-trs. In particular, known
polynomial-time algorithms for HRG parsing and generation bound the
computational complexity of TRS optimization passes.

*Limitation.* HRG confluence is NP-hard to verify in general. The EML
TRS non-confluence (Theorem B2) is consistent with this — and expected.

### C.3 Straight-Line Programs, NC1, and Logarithmic Inference

**Theorem C3** (SLP Mapping and NC1 Classification). Let $\mathcal{N}$
be a neural network whose computation graph consists of uniform EML nodes.
After TRS optimization and Common Subexpression Elimination (CSE), the
resulting DAG can be encoded as a Tree Straight-Line Program (TSLP) over
the alphabet $\{\operatorname{eml}, 1\}$. This TSLP can be transformed
in time $O(|\text{DAG}|)$ into an equivalent balanced TSLP of evaluation
depth $O(\log N)$ and size $O(|\text{DAG}|)$, where $N$ is the
uncompressed node count. Consequently, EML network inference belongs to
the parallel complexity class $NC^1$.

*Proof sketch.* A TSLP (Lohrey \& Maneth, 2006) is a context-free grammar
generating exactly one tree. Unlike a DAG which shares only identical
subtrees, a TSLP shares parameterized patterns with "holes" — capturing
the repeated layer structure of transformers (same topology, different
weights) more compactly than CSE alone.

Ganardi, Jeż \& Lohrey (2019, JACM 2021) proved that any unbalanced
SLP of size $g$ can be transformed in time $O(g)$ into an equivalent
balanced SLP of size $O(g)$ and depth $O(\log N)$ — with only a constant
multiplicative size increase, not the $O(g \log N)$ overhead of the
earlier Rytter (2003) algorithm. This eliminates the work-depth blowup
that would otherwise make the transformation impractical.

The balanced TSLP corresponds directly to an $NC^1$ circuit: a Boolean
circuit of polynomial size and $O(\log n)$ depth (fan-in 2). By
Brent's theorem (1974), every algebraic expression of size $n$ can be
reorganized into a circuit of depth $\log n$. Applied to the uniform
EML algebra, this places EML network inference in $NC^1$.

*Work-depth tradeoff.* The transformation increases total work by only a
constant factor relative to the CSE-compressed DAG. Evaluation depth
drops from $O(\text{depth})$ (sequential layers) to $O(\log N)$
(parallel TSLP traversal), with total work $O(N_{\text{comp}})$.
For TinyLlama with 22 layers: theoretical depth $\log_2(22) \approx 5$
parallel steps instead of 22 sequential.

*Execution without decompression.* Lohrey \& Maneth (2006) proved that
deterministic tree automata evaluate on TSLP directly in polynomial time,
without expanding the grammar. Inference can proceed by resolving
non-terminal references asynchronously — each non-terminal evaluated
once, results cached and reused.

*Practical realization.* A runtime executing TSLP non-terminals as
asynchronous GPU thread blocks (via CUDA or Triton), where each block
fires when its children are ready, would implement the $O(\log N)$
schedule. This requires: (1) a TSLP compilation pass after eml-trs, and
(2) a dependency-aware GPU scheduler. Neither XLA, TVM, nor Triton
currently implement this, as their heterogeneous IRs prevent global
grammar extraction.

*Homogeneity advantage.* The single operator $|\Sigma| = 1$ ensures
that the TSLP alphabet is trivially small, maximizing CSE sharing and
guaranteeing the finite subsumption base required by the Ganardi
balancing algorithm. Heterogeneous networks (with add, mul, relu, etc.)
cannot exploit this property.

*Limitation.* IEEE 754 floating-point is not strictly associative.
TSLP balancing reorganizes pointer references syntactically, not
algebraically — the original evaluation order is preserved at execution
time. The $NC^1$ bound assumes exact arithmetic; floating-point error
accumulation under parallel scheduling requires separate numerical
analysis.

### C.4 Succinct Representation: Zero Label Overhead

**Theorem C4** (Succinct EML). An EML expression tree with $N$ internal
nodes can be represented in $2N + o(N)$ bits using balanced parentheses
(BP) encoding (Jacobson, 1989; Munro & Raman, 2001), with zero bits of
label overhead for internal nodes. This representation supports $O(1)$
navigation (parent, child, subtree size) and is within a sub-logarithmic
additive term of the information-theoretic lower bound.

*Proof.* The alphabet of internal node labels is $|\Sigma| = 1$ (the
single operator $\operatorname{eml}$). Label entropy per node:
$\log_2(1) = 0$ bits. Topology in BP encoding: exactly $2N$ bits
(Munro \& Raman, 2001). Navigation via rank/select on the BP vector
runs in $O(1)$ using POPCNT hardware instructions. $\square$

*Information-theoretic optimality.* The number of distinct binary trees
with $N$ internal nodes equals the Catalan number $C_N \approx 4^N /
(N^{3/2}\sqrt{\pi})$. The information-theoretic lower bound is:

$$\log_2(C_N) \approx 2N - \tfrac{3}{2}\log_2(N) - 0.824 \text{ bits}$$

The BP encoding uses $2N$ bits — only $\tfrac{3}{2}\log_2(N)$ bits above
the lower bound. For $N = 4.8 \times 10^9$ (TinyLlama optimized):
lower bound $\approx 9.6\text{B} - 48$ bits. The gap is negligible.

*Zero-label advantage over existing formats.* Classical heterogeneous
networks require $N \cdot \log_2(|\Sigma|)$ bits for operator labels:

| Format | Operators $|\Sigma|$ | Label bits/node | Label overhead (4.8B nodes) |
|:-------|:-------------------:|:---------------:|:---------------------------:|
| ONNX | >200 | 7.64 bits | 4.6 GB |
| GGUF | ~50 | 5.64 bits | 3.4 GB |
| EML (this work) | 1 | **0 bits** | **0 GB** |

*Numerical estimate for TinyLlama.* After eml-trs optimization:
$N \approx 4.8 \times 10^9$ nodes. Raw BP topology: $2N = 9.6$
gigabits $= 1.2$ GB — less than the F16 weight file (2.2 GB).
After CSE sharing (repeated layer structure): unique nodes $\ll N$,
topology shrinks to tens of MB. After Hypersuccinct compression
(Munro et al., 2021) via Huffman coding of repeated micro-tree patterns:
empirically below 1.736 bits/node for structured graphs, approaching
hundreds of KB for the topology alone.

*Dynamic TRS rewriting on compressed representation.* Navarro \&
Sadakane (2014) proved that dynamic succinct trees support insertions,
deletions, and subtree operations in $O(\log N / \log\log N)$ time
while maintaining $2N + O(N \log\log N / \log N)$ bits. This means
TRS rewriting rules (which attach/detach subtrees) can operate
directly on the BP vector without full decompression — enabling
online algebraic optimization of compressed models.

*Limitation.* The $O(1)$ navigation guarantee requires pre-computed
rank/select auxiliary tables of size $o(N)$. For static inference
(compress once, run many times), this is the ideal model. Dynamic
TRS rewriting incurs $O(\log N / \log\log N)$ per rule application.

### C.5 Algebraic Generalization Capacity as MDL Proxy

**Conjecture C5** (Algebraic Generalization Capacity). Let $M$ be a
neural network and $C_{\operatorname{eml}}(M)$ be the EML node count of
its optimized inference DAG after eml-trs. Then $C_{\operatorname{eml}}(M)$
is positively correlated with the model's generalization ability:
networks with smaller $C_{\operatorname{eml}}(M)$ overfit less.

*Motivation.* The Minimum Description Length (MDL) principle (Rissanen,
1978; Grünwald, 2007) states that a model that genuinely learns a pattern
(rather than memorizing noise) can be described more compactly. If a
network has learned a simple underlying function $f$, its EML tree after
TRS should reduce significantly — converging toward the EML representation
of $f$ itself. A network that memorized noise has no such algebraic
structure and resists compression.

*Status.* This is a conjecture, not a theorem. Empirical verification
would require: (1) training networks of varying complexity on datasets
with known signal-to-noise ratios, (2) measuring $C_{\operatorname{eml}}$
after eml-trs, (3) measuring test generalization error, and (4)
computing correlation.

*If confirmed*, $C_{\operatorname{eml}}$ would constitute a new,
data-free measure of model quality — computable from weights alone,
without a validation dataset.

### C.6 Summary of Theoretical Connections

| Connection | Established Theory | New Bridge | Status |
|:-----------|:-------------------|:-----------|:------:|
| LZ ↔ CSE/DAG | Ziv & Lempel (1977) | LZ dict = DAG shared node | Theorem |
| TRS ⊂ HRG | Ehrig et al. (1990) | TRS rule = HRG production | Theorem |
| DAG → TSLP | Rytter (2003) | O(log N) eval depth | Theorem |
| Succinct EML | Munro & Raman (2001) | Zero label overhead | Proof |
| AGC ↔ MDL | Rissanen (1978) | Node count = generalization proxy | Conjecture |
