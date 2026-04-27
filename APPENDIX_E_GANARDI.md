# Appendix E: Ganardi TSLP Balancing for EML Graphs

> **Status: Future Work — not yet implemented.**
> This appendix documents the design and expected properties of the
> Ganardi balancing algorithm for EML graphs, based on deep research
> into Ganardi, Jeż & Lohrey (JACM 2021). Implementation is estimated
> at 75–110 engineering hours. Expected outcome: TinyLlama depth
> reduction from 789 waves (pre-balancing) to ~10–20 waves.

---

## E.1 Why Rytter Is Insufficient

The prior state of the art for SLP balancing (Rytter, 2003) achieves
$O(\log N)$ depth but at cost $O(g \log N)$ grammar size. For |Σ|=1 —
even with the homogeneous EML alphabet — this overhead does **not**
disappear. The reason: even when all internal nodes are identical
(one operator type), the AVL rotation mechanism still requires new
non-terminal rules at each split boundary. The count of these rules
grows with $\log N$.

For TinyLlama: $g \approx 5 \times 10^9$ nodes per layer,
$\log N \approx 35$ (for $N \approx 10^{11}$ expanded nodes).
Rytter would require $\sim 175\,\text{GB}$ grammar representation
per layer — a practical dead end.

Ganardi, Jeż & Lohrey (JACM 2021) resolve this by replacing AVL
rotations with **Symmetric Centroid Decomposition (SCD)**, achieving
$O(g)$ grammar size with the same $O(\log N)$ depth guarantee.

---

## E.2 The SCD Algorithm for |Σ| = 1

The algorithm operates on the TSLP grammar extracted by `grammar.rs`
(a flat `Vec<(NodeId, TslpRhs)>`). Five deterministic steps:

**Step 1 — Compute weighted path measures $\pi$.**

For each node $v$, compute two values via two DAG traversals:

$$\pi(r, v) = \text{number of times } v \text{ is visited in the fully expanded tree, counting from root } r$$
$$\pi(v, W) = \text{total size of the subtree rooted at } v \text{ in the expanded (uncompressed) tree}$$

$\pi(v, W)$ is computed bottom-up in $O(g)$:
leaf nodes have $\pi = 1$; internal node $\mathrm{eml}(l, r)$ has
$\pi = \pi(l, W) + \pi(r, W) + 1$.

$\pi(r, v)$ is computed top-down in $O(g)$: root has $\pi = 1$;
for node $v$ with parent $u$, $\pi(r, v) = \pi(r, u)$
(each parent visit generates one visit to each child).

Both stored as `HashMap<NodeId, u64>` — no Arc mutation required.

**Step 2 — Assign logarithmic labels $\lambda$.**

$$\lambda(v) = \bigl(\lfloor \log_2 \pi(r, v) \rfloor,\; \lfloor \log_2 \pi(v, W) \rfloor\bigr)$$

This buckets nodes by their "depth class" and "subtree class". Nodes
in the same bucket belong to the same level of the compression
hierarchy. Stored as `HashMap<NodeId, (u8, u8)>`.

**Step 3 — Extract SCD edges $E_\text{scd}$.**

An edge $(u \to v)$ is a **centroid edge** iff $\lambda(u) = \lambda(v)$.

Key theorem: each node has at most one outgoing and one incoming
centroid edge. This partitions the DAG into a set of disjoint
**SC-paths** — chains of nodes with identical $\lambda$ labels.

For |Σ| = 1: each context along an SC-path has exactly one of two
forms: $\mathrm{eml}(\square, C)$ or $\mathrm{eml}(B, \square)$.
The subsumption base is $O(1)$ — no type dispatch needed.

**Step 4 — Balance each SC-path as a tournament tree.**

Each SC-path is a chain of length $L$ — equivalent to the sequential
accumulation problem solved by `parallel_prefix.rs`. The balancing
step replaces the chain with a binary tournament tree of depth
$O(\log L)$, using the same divide-and-conquer as `parallel_prefix_sum`.

New `NodeId` values (beyond `max(existing_ids)`) are allocated for
each new internal node. No existing `Arc<EmlNode>` is mutated.

**Step 5 — Reconstruct the grammar.**

Replace sequential SC-path edges with references to the new balanced
subtrees. The final `TslpGrammar` has:
- Same $O(g)$ total productions (new nodes replace old chain nodes)
- Grammar depth $O(\log N)$ — Ganardi's main guarantee

---

## E.3 Rust Implementation Architecture

Three phases, cleanly separated to avoid Arc lifetime conflicts:

**Phase 1 — Extraction** (already implemented in `grammar.rs`)
```
Arc<EmlNode> DAG  →  TslpGrammar { Vec<(NodeId, TslpRhs)>, start }
```
`NodeId = Arc::as_ptr(...) as usize`. All pointer arithmetic done
once; original `Arc` objects remain untouched throughout.

**Phase 2 — SCD Transformation** (to implement in `ganardi.rs`)
```
TslpGrammar  →  TslpGrammar (balanced)
```
Operates entirely on `Vec`, `HashMap<NodeId, _>`, and `usize`.
No `Arc` involved. Steps 1–5 above. Arena allocation: new nodes
get `NodeId = existing_max + 1, + 2, ...`

Key data structures:
```rust
struct GanardiState {
    grammar: TslpGrammar,
    pi_down: HashMap<NodeId, u64>,  // π(r, v)
    pi_up:   HashMap<NodeId, u64>,  // π(v, W)
    lambda:  HashMap<NodeId, (u8, u8)>,
    sc_paths: Vec<Vec<NodeId>>,     // extracted centroid paths
}
```

**Phase 3 — Reconstruction** (new `src/tslp/ganardi.rs`)
```
TslpGrammar (balanced)  →  Arc<EmlNode> (new DAG)
```
Bottom-up traversal of balanced grammar. Each `NodeId` maps to a
freshly allocated `Arc::new(EmlNode::Eml(...))`. Shared nodes
(same `NodeId` referenced by multiple productions) map to a single
`Arc` stored in `HashMap<NodeId, Arc<EmlNode>>`.

No `RwLock`, no `Mutex`. Thread-safe because construction is
purely bottom-up with no back-references.

---

## E.4 Numerical Stability After Balancing

Balancing changes associativity of EML operations. Since IEEE 754
floating-point is not associative, the rebalanced graph may produce
slightly different numerical results than the original.

**Error bound:** For a balanced tree of depth $D = O(\log N)$:

$$\text{accumulated error} \leq D \cdot \varepsilon_\text{machine} = O(\log N \cdot u)$$

For $D \approx 20$ and $\varepsilon_{\text{bf16}} = 0.0078$:
$$\text{max error} \approx 20 \times 0.0078 = 0.156$$

This is larger than the Minimax approximation errors ($E_\text{max}
\approx 0.002$) but remains within the rounding tolerance of bf16
weights — the network already operates with precision $\approx 0.008$
per weight. The balancing error is absorbed by quantization noise.

**Stochastic verification protocol** (to implement in test suite):
1. Sample 10,000 random input vectors from $U[-1, 1]^d$
2. Evaluate original (unbalanced) EML graph in `f64`
3. Evaluate balanced EML graph in `f64`
4. Assert: $\|\text{original} - \text{balanced}\|_2 < \varepsilon_{\text{bf16}} \cdot \sqrt{d}$

**Critical warning — neg\_node:** Nodes using `Const(0.0)` in
`eml(ln(0), exp(x))` for negation must not be placed adjacent to
other `ln`-domain operations during SCD path construction.
If the balanced tree places `ln(0) = -∞` as input to another `eml`
node expecting $y > 0$, the result is `NaN`. The implementation
must detect and isolate `neg_node` patterns before SCD extraction.

---

## E.5 Implementation Roadmap

Estimated effort for one experienced Rust developer:

| Task | Hours | Notes |
|:-----|------:|:------|
| $\pi$ computation (bottom-up + top-down) | 10–15 | Two `HashMap` traversals over flat `Vec` |
| $\lambda$ labeling and $E_\text{scd}$ extraction | 15–20 | Group nodes by `(u8,u8)` bucket |
| SC-path tournament tree construction | 20–30 | Hardest step; reuse `parallel_prefix_sum` logic |
| Bottom-up Arc reconstruction | 15–25 | Arena pattern; `HashMap<NodeId, Arc<EmlNode>>` |
| Stochastic verification + integration | 15–20 | Test suite, benchmarks, paper results |
| **Total** | **75–110** | **2–3 weeks dedicated** |

**Three implementation pitfalls to avoid:**

*Pitfall 1 — Dictionary explosion.* When building tournament tree
nodes in Step 4, each new `NodeId` must be deduplicated via a
`HashMap`. Without deduplication, identical subtrees are duplicated,
restoring the $O(g \log N)$ size of Rytter.

*Pitfall 2 — $O(V^2)$ topological sort.* Computing $\pi(r, v)$
top-down requires an inverted adjacency list (parent → children).
Without pre-allocation, naive lookup is $O(V^2)$.
Fix: build `HashMap<NodeId, Vec<NodeId>>` of children during
grammar extraction.

*Pitfall 3 — neg\_node NaN propagation.* Nodes containing
`Const(0.0)` as the left child of `ln_node` produce $-\infty$.
If such a node appears on an SC-path and is combined with adjacent
nodes expecting $y > 0$, the result propagates `NaN` silently.
Fix: mark `neg_node` roots as SC-path boundaries (non-centroid edges).

---

## E.6 Expected Results After Implementation

Based on the SCD depth guarantee and current measurements:

| Metric | Pre-balancing | Post-Ganardi (projected) |
|:-------|:-------------|:------------------------|
| TinyLlama 22 layers, K=64 | 789 waves | ~10–20 waves |
| Single dot product K=64 | 32 waves (parallel prefix) | 6–8 waves |
| Grammar size | $g$ nodes | $O(g)$ — same order |
| Numerical error | 0 (exact) | $\leq 0.156$ (within bf16) |
| Implementation time | — | 75–110 hours |

The projected 10–20 waves for TinyLlama represents the full empirical
realization of Theorem C3: EML inference ∈ NC1, achieving $O(\log N)$
evaluation depth for the complete transformer inference graph.

---

*Source: Deep Research report on Ganardi TSLP Balancing (April 2026).
Algorithm: Ganardi, Jeż & Lohrey, "Balancing straight-line programs,"
Journal of the ACM 68(4), 2021. doi:10.1145/3457389.*
