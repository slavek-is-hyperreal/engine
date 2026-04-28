# Appendix G: Hardware Roadmap — From RTL to Silicon

> **Status: Design study and roadmap — no physical silicon exists.**
> Appendix D describes the RTL prototype (theoretical). This appendix
> extends it with concrete synthesis targets, resource analysis, and
> a milestone-based adoption path informed by RISC-V and TPU history.
> Key correction from Appendix D: SKY130 achieves 70–120 MHz (not
> 200–500 MHz as estimated there — that figure applies to 28nm CMOS).

---

## G.1 The Strategic Question: Itanium or RISC-V?

Intel Itanium failed because it transferred optimization complexity to
the compiler — requiring static scheduling of VLIW/EPIC slots that
compilers could not reliably fill. EML faces an analogous risk: if the
TRS cannot automatically discover cross-layer identities from arbitrary
network graphs, the programmer bears the optimization burden.

The RISC-V analogy is more optimistic. RISC-V succeeded because:
- Clean-slate ISA, no legacy constraints
- Open license, zero entry barrier
- Modular extensions (base + V + B + custom)
- First tapeout in 18 months (UC Berkeley → STMicroelectronics 28nm
  FD-SOI, sponsored)

EML shares the first three properties. The fourth requires deliberate
execution. This appendix maps that execution.

---

## G.2 Corrected Resource Analysis

Appendix D estimated 450 MHz for `eml_cell` in 28nm CMOS. This is
accurate for that process. The table below adds SKY130 (open-source
tapeout target) and FPGA alternatives:

### eml_cell (single node)

| Platform | LUT / GE | DSP | Fmax | Notes |
|:---------|:--------:|:---:|-----:|:------|
| 28nm CMOS ASIC | ~550 LUT eq. | 4 | ~450 MHz | Appendix D estimate |
| **SKY130 (130nm)** | **~20k GE** | 0 (synthesized) | **70–120 MHz** | OpenLane, sky130_fd_sc_hd |
| Arty A7-100T (Artix-7) | ~550 LUT | 4 | ~150 MHz | Xilinx Vivado |
| Intel Stratix 10 | ~400 LUT | 2–4 (hardened FP32) | ~300 MHz | VP DSP Float32 native |

### eml_dot_product_64 (tournament tree, K=64)

| Platform | LUT | DSP | Fmax | Area / Cost |
|:---------|----:|:---:|-----:|:-----------|
| 28nm CMOS ASIC | ~11,440 LUT eq. | 64 | ~200–300 MHz | ~0.07 mm² |
| **SKY130 (130nm)** | **150k–250k GE** | 0 | **70–100 MHz** | **~1.0–1.6 mm²** |
| **Arty A7-100T** | **~11,440 (18% LUT)** | **64 (26.67% DSP)** | **~150 MHz** | **<$300 board** |
| Alveo U250 | ~11,440 per instance | 64 per instance | ~300 MHz | 150+ parallel instances |

**The Arty A7 number is the key result:** 18% LUT and 26.67% DSP leaves
>70% of the FPGA free for test infrastructure (DDR3 controller, UART,
soft-core RISC-V manager). A working K=64 dot product demo fits on a
$300 development board.

---

## G.3 Why GPU SFU Is the Bottleneck EML Eliminates

A100 (Ampere) SFU throughput: **16 ops/cycle per SM** (CUDA C++
Programming Guide, arithmetic-throughput table). FP32 FMA throughput:
**64 ops/cycle per SM** (**4× higher** than SFU). Every call to `exp()` or `log()` in a transformer kernel
dispatches to SFU, creating a 4× throughput asymmetry.

*Note:* 512 ops/cycle/SM is the Tensor Core FP16 throughput (4 TC ×
128 FP16 FMA/cycle), not FP32 FMA. The correct FP32 FMA rate is 64,
giving a 4× SFU bottleneck ratio, not 32×.

H100 (Hopper) inherits this division. Additional latency: HBM3 cache
miss costs ~500 cycles. The structural SFU bottleneck is architectural,
not a bug — it reflects the cost of implementing transcendental functions
in silicon.

**EML eliminates SFU dispatch entirely.** The `fast_exp` and `fast_ln`
Minimax approximations (N=2 and N=3) execute as sequences of FMA
operations — same units that run at 512 ops/cycle. Log-Softmax, which
on A100 requires: N SFU exp() + reduction + N SFU log() + N division,
becomes: N parallel FMA chains in `eml_cell` + tournament adder tree.
No SFU invocations. $O(\log N)$ FMA cycles total.

The concrete comparison for Log-Softmax on N=2048 (TinyLlama context):

| Implementation | SFU calls | Approximate cycles |
|:--------------|:---------:|:------------------:|
| CUDA baseline (A100) | 2 × 2048 = 4096 | ~4096 × 16 = ~65k |
| **EML native** | **0** | **O(log 2048) × 4 ≈ 44** |

The SFU bottleneck ratio is 4× (64 FP32 FMA/cycle vs 16 SFU/cycle per SM).
EML eliminates all SFU dispatches, converting nonlinear operations to
pure FMA sequences and capturing the full 4× throughput advantage
for transcendental-heavy kernels like Log-Softmax.

---

## G.4 Synthesis Path: SKY130 via OpenLane

For the academic tapeout milestone, the OpenLane flow on SKY130:

```
eml_cell.v + eml_dot_product_64.v
        ↓  (fix 4 known bugs first — see Appendix D.4)
   Yosys + ABC
   Technology mapping → sky130_fd_sc_hd
        ↓
   OpenLane
   1. Floorplan + PDN (Power Distribution Network)
      - Metal layers 4+5 for VDD/VSS
      - IR drop < 5% VDD
      - Decap cells for switching noise
   2. Placement (RePLace: global → detailed)
      - Kogge-Stone tree: symmetric placement critical
   3. Clock Tree Synthesis (TritonCTS)
      - 4-stage eml_cell pipeline + 6-stage subtraction tree
      - Hold violations: buffer insertion required
   4. Routing (FASTRoute + TritonRoute)
      - Antenna diodes for long gate nets
   5. Sign-off: OpenSTA (timing) + Magic (DRC) + Netgen (LVS)
   6. Export: GDSII
```

**Known challenge:** Float32 multipliers synthesized from RTL (no
hardened DSP in SKY130) require Wallace tree multipliers. Each 8×8
mantissa multiplier = ~500 GE. For 64 instances: ~32k GE just for
multipliers. This is why the estimate is 150k–250k GE total.

**Realistic Fmax on SKY130:** 70–100 MHz. This validates the RTL
architecture and the float_to_int floor bug, but does not demonstrate
production performance. That requires 28nm or FPGA.

**Submission path:** Efabless ceased operations in March 2025.
The current path to open-source silicon is **ChipFoundry** (successor),
offering chipIgnite shuttle runs on SKY130 at approximately **$14,950
per project**, with 2–3 shuttles per year. Turnaround is typically
5–9 months from GDSII submission to packaged chips.

Alternative: **Tiny Tapeout** (tinytapeout.com) offers lower-cost
multi-project submissions, though the program experienced disruptions
in 2025 (TT08/09 delayed; exploring IHP 130nm and GF180 processes).

---

## G.5 FPGA Proof-of-Concept: Arty A7-100T

**Board:** Digilent Arty A7-100T, Xilinx XC7A100TCSG324-1, ~$230

| Resource | Available | eml_dot_product_64 | Remaining |
|:---------|----------:|:------------------:|----------:|
| LUT (6-input) | 63,400 | 11,440 (18%) | 51,960 (82%) |
| Flip-Flops | 126,800 | <5,000 est. (<4%) | >96% |
| DSP48E1 | 240 | 64 (26.7%) | 176 (73.3%) |
| Block RAM | 607.5 KB | ~64 KB (buffer) | ~543 KB |

Remaining resources accommodate:
- MIG DDR3 controller (for loading weight matrices)
- UART/Ethernet for result streaming
- Soft-core RISC-V (e.g., VexRiscv) for orchestration
- Additional `eml_cell` instances for Log-Softmax

**Demo target:** Load K=64 weight matrix via UART → compute dot product
via EML tournament tree → stream result back → compare with reference
Python implementation. Measure wall-clock cycles. Report: 15 cycles
vs naive 128 cycles (8.5× measured speedup).

This is sufficient for a conference demo or hardware paper supplement.

---

## G.6 Commercial Accelerator Integration

Three platforms allow EML integration without custom ASIC:

**Tenstorrent (Wormhole/Grayskull — Tensix cores):**
Each Tensix core contains 5 RISC-V controllers + programmable SFPU
(Special Function Processing Unit). SFPU is programmed via SFPI
(C/C++ inline assembly). EML Minimax kernels (`fast_exp`, `fast_ln`,
tournament reduction) map directly to SFPU microcodes. No silicon
change required — firmware update sufficient.

**Groq (Tensor Streaming Processor):**
TSP is fully deterministic — compiler assigns every operation to an
exact lane and cycle. The 4-stage `eml_cell` pipeline and 15-cycle
`eml_dot_product_64` have provably fixed latencies. This matches
Groq's compilation model perfectly: static scheduling with zero
runtime dispatch overhead. Integration = compiler backend addition.

**Cerebras (Wafer-Scale Engine):**
~900k processing elements, programmed in CSL (Cerebras Software Language).
Custom EML kernels in CSL would reduce attention layer message-passing
across the wafer by eliminating SFU-equivalent transcendental calls.
Integration = CSL kernel library.

---

## G.7 Milestone Roadmap

Ordered by effort and dependency:

| # | Milestone | Effort | Dependency | Validates |
|:--|:----------|:------:|:----------:|:----------|
| 1 | Fix 4 RTL bugs (Appendix D.4) | 2 days | None | Appendix D correctness |
| 2 | Arty A7 synthesis + demo | 2–4 weeks | Milestone 1 | 8.5× hardware speedup |
| 3 | SKY130 tapeout (MPW shuttle) | 3–6 months | Milestone 1 | Physical silicon viability |
| 4 | Tenstorrent SFPI microcodes | 4–8 weeks | Milestone 2 | Commercial adoption path |
| 5 | EML-TRS → Groq compiler backend | 3–6 months | Paper published | Deterministic ISA fit |
| 6 | 28nm ASIC (commercial fab) | 2–3 years | Milestone 3 | Production performance |

**Critical path:** Milestone 2 (Arty A7 demo) is the minimum viable
hardware result for academic credibility. It requires fixing the RTL
bugs and running Vivado synthesis — approximately 2–4 weeks of work.

The RISC-V precedent: 18 months from paper to first tapeout.
The TPU precedent: 15 months from decision to production deployment.
EML Milestone 3 (SKY130) is achievable in the same timeframe with
the MPW shuttle program.

---

## G.8 The "Itanium Test"

EML passes if and only if:

**Condition 1 (compiler automation):** The TRS must discover
cross-layer identities (Fusions 1–5) automatically from an arbitrary
ONNX/GGUF graph without programmer annotation. If programmers must
manually mark fusion boundaries, EML is Itanium.

**Condition 2 (zero programmer burden for correctness):** The
numerical error from Minimax approximation and ASIS pre-negation must
be bounded and automatic. If programmers must reason about error
accumulation, EML is Itanium.

**Current status:** Condition 1 — the TRS in `trs.rs` applies rules
automatically, bottom-up to fixpoint. The proc\_macro `#[eml_optimize]`
handles CPU-side annotation. Condition 2 — error bounds are proven
(< bf16 epsilon throughout). Both conditions are met at the software
level. The hardware compiler (Milestone 5) would need to extend this
to the RTL domain.

**Verdict:** EML is architecturally closer to RISC-V than Itanium.
The grammar $S \to 1 \mid \mathrm{eml}(S,S)$ is as minimal as RV32I.
The TRS is the compiler. The burden is on the TRS, not the programmer.
Whether the TRS is powerful enough is an empirical question — one that
the roadmap above is designed to answer.
