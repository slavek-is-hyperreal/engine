# Plan Strategiczny EML-TRS: Akceleracja przez Minimax FMA + Kompresja TSLP

## Cel Projektu (z PAPER.md)

EML-TRS to algebraiczny komprestor grafów sieci neuronowych.
PAPER.md (§9.2) identyfikuje dwa niezależnie składające się mechanizmy przyspieszenia:

1. **Redukcja węzłów przez TRS** — mniej węzłów do obliczenia
2. **Minimax FMA zamiast SFU** — tańsze obliczenie per węzeł

> [!IMPORTANT]
> Konkretne wartości przyspieszenia z PAPER.md (redukcja węzłów, speedup per EML node)
> są **teoretycznymi oszacowaniami**, nie zweryfikowanymi eksperymentalnie.
> Celem projektu jest ich empiryczne potwierdzenie lub korektacja przez benchmark.

Główna idea z §9.2 jest strukturalnie poprawna: EML eliminuje wywołania SFU
zastępując je sekwencją FMA. To jest **motywacja projektu**, nie fakt do cytowania.

---

## Dlaczego SFU to Problem i jak Go Rozwiązujemy

Każde wywołanie `exp()` lub `ln()` w GPU trafia na **SFU (Special Function Units)**,
które są rzadszym i wolniejszym zasobem niż zwykłe rdzenie ALU (FMA).

**EML eliminuje SFU** — zastępując je sekwencją FMA przez aproksymację Minimax
N=2/3 (PAPER.md §9.1, §10.1 — "Quake-style EML Kernels"):

```
exp(x) przez SFU:          wolniejsze, bottleneck przy masowym użyciu
exp(x) Minimax N=2 FMA:    szybsze, pełna przepustowość ALU
```

**Konkretny zysk (do zmierzenia przez benchmark)**: PAPER.md szacuje ~4× per węzeł,
ale jest to wartość do empirycznej weryfikacji na docelowym sprzęcie.

Dla Log-Softmax (PAPER.md §4.3 + Appendix G.3): teoretycznie operacja natywna EML
bez żadnych wywołań SFU. Rzeczywisty zysk = przedmiot benchmarku.

---

## Archiwizacja RTL

```bash
git mv rtl/ archive/rtl/
```

RTL (eml_cell.v, eml_dot_product_64.v, eml_log_softmax.v) → `archive/rtl/`.
Koncepcja Minimax z RTL była poprawna. Realizacja w software (GLSL/WGSL) jest
szybsza do wdrożenia i przenośna na AMD/NVIDIA/Intel bez tapeoutu.

---

## Priorytet 1: Minimax FMA Kernels (PAPER.md §9.2, §10.1)

### [NEW] `vulkan/eml_cell_minimax.comp` — Fused EML node

Aproksymacja Minimax N=2/3 zastępuje SFU przez sekwencję FMA.
Błąd E_max ≈ 0.002 < ε_BF16 = 0.0078 — niewidoczny w modelach BF16.

```glsl
// Quake-style EML kernel: exp i ln przez bit-hacking + Horner FMA
// Odpowiednik fast inverse square root, ale dla exp/ln.
// Źródło: PAPER.md Section 10.1 "Quake-Style EML Kernels"

const float LN2   = 0.6931471805f;
const float LOG2E = 1.4426950408f;

// Minimax N=2 dla exp (E_max ≈ 0.0021, PAPER.md §9.2)
float fast_exp(float x) {
    int i   = int(floor(x * LOG2E));
    float f = x - float(i) * LN2;              // f ∈ [-0.5, 0.5]
    // Horner: a0 + f*(a1 + f*a2) — 2 FMA, zero SFU
    float p = 1.0017247f + f * (0.6566422f + f * 0.3546143f);
    return uintBitsToFloat((i + 127) << 23) * p;
}

// Minimax N=3 dla ln (E_max ≈ 0.0006, PAPER.md §9.2)
float fast_ln(float y) {
    int   e = (floatBitsToInt(y) >> 23) - 127;
    float m = uintBitsToFloat((floatBitsToInt(y) & 0x007FFFFFu) | 0x3F800000u);
    float u = m - 1.0f;
    // Horner: u*(c1 + u*(c2 + u*c3)) — 3 FMA, zero SFU
    float q = u * (0.9978116f + u * (-0.2118863f + u * 0.0562745f));
    return float(e) * LN2 + q;
}

// Fused EML operator — jeden kernel, zero memory round-trip między exp i ln
float eml_op(float x, float y) { return fast_exp(x) - fast_ln(y); }
```

### [NEW] `vulkan/eml_dot_product.comp` — ASIS tournament tree

- Wczytaj aktywacje `x[i]` i pre-negowane wagi ASIS `w_tilde[i]` z SSBO
- Dla `i=0`: term = `mul_cf(x[0], w[0])` = `eml(eml(ln(ln(x[0])), 1/w[0]), 1)`
- Dla `i≥1`: term = `mul_cf(x[i], -w[i])` (waga już pre-negowana offline)
- Redukcja: `result = term[0] - term[1] - term[2] - ...` w `shared memory`
- Drzewo turniejowe O(log K) głębokości (PAPER.md §4.1, Theorem 2)

### [NEW] `vulkan/eml_log_softmax.comp` — Native EML Log-Softmax

Sekcja 4.3 PAPER.md: Log-Softmax jest **natywną** operacją EML.
Fuzja: `eml(a, b*S)` zamiast `eml(a,b) - ln(S)` — zero dodatkowych węzłów.

### [NEW] `vulkan/eml_runtime.rs`

- `wgpu` inicjalizacja (Vulkan/Metal/DX12 — jeden kod)
- Pre-ładowanie wag z pre-negacją ASIS offline
- Dispatch shaderów per warstwa TinyLlama
- Benchmark: cycles per layer, porównanie z NumPy baseline

---

## Priorytet 2: EML jako IR dla Shaderów (PAPER.md §10.7)

Sekcja 10.7 PAPER.md: docelowa ścieżka kompilatora:

```
GLSL/WGSL → EML tree → TRS rewrite → optimized EML → SPIR-V → GPU
```

Metryka sukcesu: redukcja liczby `OpExtInst` (SFU) w output SPIR-V.

### [NEW] `src/backends/naga_eml.rs`

- Parse WGSL → naga `Module`
- Identyfikacja branch-free podgrafów matematycznych
- Translacja do EML, `trs::rewrite()`, translacja z powrotem
- Emisja zoptymalizowanego SPIR-V przez naga backend

### [NEW] `src/backends/proc_macro.rs`

- Makro `#[eml_optimize]` dla CPU hot paths w Rust
- Parsowanie przez `syn`, emit przez `quote!`
- Zgodne z PAPER.md §10.9

---

## Priorytet 3: Naprawy Rdzenia (blokerów parytetu)

Bez tych napraw nie ma parytetu EML z TinyLlama:

### 3.1 Niespójność `mul_eml` — asis.rs vs ast.rs

`build_asis_dot_product` (asis.rs:26) wywołuje `mul_eml` z `ast.rs`,
ale testy asis.rs (linie 97-130) weryfikują lokalną `mul_eml_local`.
→ Przenieść `mul_cf` jako publiczną funkcję do `ast.rs`.

### 3.2 `mul_cf` — obsługa aktywacji `x ∈ (0, 1)`

PAPER.md §4.2 Theorem 3 wymaga `x > 0`. Po SiLU aktywacje mogą być w `(0, 1)`,
co spełnia ten warunek (nie wymaga `x > 1`). Problem dotyczy tylko `x ≤ 0`.
Sprawdzić, czy `mul_cf_safe` z offsetem jest naprawdę potrzebny.

### 3.3 Przepisanie `ganardi.rs` — eliminacja OOM (PAPER.md §10.4)

Sekcja 10.4 PAPER.md: Ganardi jest **najwyższym priorytetem implementacyjnym**.
Aktualny `rebuild_tree` = O(N) pamięci = crash dla TinyLlama.
Plan: DAG-native SCD w O(g log g) czasie, O(g) pamięci.

---

## Kolejność Wykonania

```
Tydzień 1:
  [ ] git mv rtl/ archive/rtl/
  [ ] Naprawa niespójności mul_eml (asis.rs + ast.rs)
  [ ] cargo test — wszystkie muszą przejść
  [ ] scripts/verify_parity.py — zero NaN dla K=64

Tydzień 2:
  [ ] eml_cell_minimax.comp — fast_exp + fast_ln Minimax
  [ ] eml_dot_product.comp — ASIS tournament tree
  [ ] eml_log_softmax.comp — native EML fusion
  [ ] wgpu runtime (eml_runtime.rs)
  [ ] Benchmark: EML Vulkan vs NumPy — ms/layer

Tydzień 3:
  [ ] Przepisanie ganardi.rs (faza A: π-measures na DAG)
  [ ] Przepisanie ganardi.rs (faza B: rebalance bez dekompresji)
  [ ] Test: depth < O(log K) po balansowaniu

Tydzień 4 (stretch):
  [ ] naga_eml.rs — WGSL → EML → TRS → SPIR-V
  [ ] Metryka: redukcja OpExtInst w output SPIR-V
```

---

## Oczekiwany Efekt (do weryfikacji benchmarkiem)

PAPER.md §9.2 przewiduje łączne przyspieszenie z obu mechanizmów.
Wartości MUSZĄ być zmierzone empirycznie — nie cytujemy ich jako fakt:

| Mechanizm | Przewidywanie z PAPER.md | Status |
|---|---|---|
| Redukcja węzłów przez TRS | Znacząca (% do zmierzenia) | ❌ niezweryfikowane |
| Minimax FMA vs SFU per węzeł | Szybsze (× do zmierzenia) | ❌ niezweryfikowane |
| Log-Softmax native EML | O(1) amortized, zero SFU | ❌ niezweryfikowane |
| Ganardi O(log N) depth | Masowy paralelizm NC^1 | ❌ nie zaimplementowane |

**Cel benchmarku**: zastąpić każde ❌ konkretnym pomiarem `ms/layer` na
standard. GPU (np. AMD Radeon R7 260X z PAPER.md Appendix A) vs NumPy.
