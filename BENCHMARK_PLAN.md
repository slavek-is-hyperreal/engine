# Plan: TinyLlama ALU vs EML Benchmark

## Cel
Załadować wagi z `models/tinyllama-f16.gguf` (Python + gguf lib → JSON),
wykonać jedno przejście **tradycyjnego matmul (ALU f32)** i jedno
**EML matmul (symboliczne drzewo + TRS + ewaluacja)** na tej samej
warstwie, porównać wyniki numerycznie i zmierzyć czas.

---

## Faza 1: Sprzątanie kodu (najpierw)

### Usuwamy całkowicie:
- `src/loader/` — zastępujemy Pythonem, własny GGUF loader wyrzucamy
- `src/backends/wgsl.rs` — gorsze współczynniki Minimax (SET B), zastąpiony przez `eml_kernels.wgsl`
- Linia `#[cfg(feature = "gguf")] pub mod loader;` z `lib.rs`
- Feature `gguf = []` z `Cargo.toml`
- Binary `real_tinyllama` z `Cargo.toml` (używał usuniętego loadera)
- Powielone `[[bin]]` + `[[bench]]` dla tych samych plików w `Cargo.toml`

### Naprawiamy unused imports / dead code:
| Plik | Problem | Akcja |
|------|---------|-------|
| `src/tslp/ganardi.rs:14-16` | TslpRhs, LeafKind, GrammarNodeId, konst, var, HashMap | Usunąć |
| `src/tslp/parallel_prefix.rs:46` | asis_preprocess_weights | Usunąć |
| `src/round_trip.rs:171` | crate::ast::* | Usunąć |
| `src/tslp/grammar.rs:130` | crate::ast::* | Usunąć |
| `src/tslp/rake_compress.rs:134` | crate::ast::* | Usunąć |
| `src/asis.rs:95` | zmienna `y` | Prefix `_y` |
| `src/fusions.rs:27` | `swiglu_fused` pub(crate) → pub | Zmienić na `pub` |
| `src/bin/softmax_verify.rs:4` | `max_eml` unused | Usunąć funkcję |
| `benches/tinyllama_ops.rs:11` | `N_LAYERS` | Usunąć stałą |
| `benches/compress_bench.rs:13` | `std::time::Instant` | Usunąć import |

### ganardi.rs — OOM placeholder:
Zostawić kod ale dodać jasny doc-comment:
```rust
/// # UWAGA: TO JEST PLACEHOLDER — NIE PRAWDZIWY ALGORYTM GANARDIEGO
/// 
/// Obecna implementacja:
///   balance_grammar() → rebuild_tree() → O(N) pamięci → OOM na dużych modelach
///
/// Prawidłowa implementacja SCD (Symmetric Centroid Decomposition):
///   1. Obliczyć π-miarę dla każdej produkcji gramatyki (bez rozwijania drzewa)
///   2. Znaleźć heavy paths bezpośrednio w TslpGrammar.productions
///   3. Zamienić produkcje na drzewa turniejowe lokalnie
///   4. NIE wywoływać rebuild_tree() — operować na DAG gramatyki
///
/// Literatura: Ganardi, Jeż, Lohrey (JACM 2021), Section 4.
```

---

## Faza 2: Python — ekstrakcja wag (poza benchmarkiem)

### Setup venv:
```bash
cd /my_data/engine
python3 -m venv .venv
source .venv/bin/activate
pip install gguf numpy
```

### Skrypt: `scripts/extract_weights.py`

Wczytuje z `models/tinyllama-f16.gguf`:
- `blk.0.attn_q.weight` — kształt [2048, 2048], f16 → f32
- `blk.0.attn_norm.weight` — [2048], f32

Eksportuje `models/layer0_weights.json`:
```json
{
  "k": 2048,
  "d_k": 64,
  "w_q_head0": [[...64 wiersze × 2048 kolumny...]],
  "rms_norm": [...2048...],
  "input_sample": [...2048..., wartości w (1.5, 4.5), seed=42]
}
```

**Dlaczego input_sample ∈ (1.5, 4.5)?**
`mul_cf(x, w)` liczy `ln(ln(x+BIAS))` gdzie BIAS=4.0.
x+4 musi być > 1, co jest zawsze spełnione. Ale dla czystości testu
używamy wartości > 1 bez biasu w próbce.

---

## Faza 3: Rust — `src/bin/eml_benchmark.rs`

```
cargo run --bin eml_benchmark --release -- models/layer0_weights.json
```

### Ścieżka ALU (baseline):
```
dla i w 0..d_k:
    result_alu[i] = Σ_j  input[j] * W_Q[i][j]
```
Zwykłe `f32` operacje, SIMD-friendly.

### Ścieżka EML (symboliczna):
```
dla i w 0..d_k:
    vars = [var("x0"), ..., var("x2047")]
    tree = build_dot_product_eml(vars, W_Q[i])
    tree_opt = rewrite(tree)
    result_eml[i] = try_evaluate(tree_opt, {x_j: input[j]})
```

### Oczekiwany output:
```
=== TinyLlama Q-projection benchmark (layer 0, head 0) ===
Input dim (K): 2048
Output rows:     64  (1 head × d_k)

--- ALU (tradycyjny f32 matmul) ---
  Czas:        0.15 ms
  result[0]:   0.013421
  result[1]:  -0.002145

--- EML (symboliczny + TRS + evaluate) ---
  build_tree:  87.4 ms  (64 drzew × ~1370 węzłów)
  rewrite:     21.3 ms  (TRS do fixpoint)
  evaluate:     6.1 ms  (try_evaluate × 64)
  TOTAL:      114.8 ms
  result[0]:   0.013421
  result[1]:  -0.002145

Parity: max_diff = 3.1e-5  OK (< BF16 eps 0.0078125)
EML węzłów per wiersz: 1370  (CF+ASIS K=2048: 14×2048-9=28663 → po TRS)
EML overhead vs ALU: ~765×  (oczekiwane — drzewo EML oceniane na CPU)

Uwaga: overhead jest celowy. Na GPU (WGSL/Vulkan) każdy węzeł drzewa
= 1 FMA cycle, budowa offline. Różnica spada do <3× vs cuBLAS.
```

---

## Kolejność wykonania

1. `cargo fix --lib -p eml-trs --allow-dirty` (auto-fix importów)
2. Ręczne poprawki (ganardi.rs, fusions.rs, softmax_verify.rs, itp.)
3. `cargo test` → wszystkie zielone
4. Stworzyć `.venv` i `scripts/extract_weights.py`
5. `python3 scripts/extract_weights.py` → `models/layer0_weights.json`
6. `src/bin/eml_benchmark.rs` + wpis w `Cargo.toml`
7. `cargo run --bin eml_benchmark --release`
8. Wyniki

---

## Pliki

| Akcja | Plik |
|-------|------|
| DELETE | `src/loader/gguf.rs` |
| DELETE | `src/loader/mod.rs` |
| DELETE | `src/backends/wgsl.rs` |
| MODIFY | `src/lib.rs` |
| MODIFY | `Cargo.toml` |
| MODIFY | `src/tslp/ganardi.rs` |
| MODIFY | `src/tslp/grammar.rs` |
| MODIFY | `src/tslp/rake_compress.rs` |
| MODIFY | `src/tslp/parallel_prefix.rs` |
| MODIFY | `src/round_trip.rs` |
| MODIFY | `src/asis.rs` |
| MODIFY | `src/fusions.rs` |
| MODIFY | `benches/tinyllama_ops.rs` |
| MODIFY | `benches/compress_bench.rs` |
| MODIFY | `src/bin/softmax_verify.rs` |
| NEW | `scripts/extract_weights.py` |
| NEW | `src/bin/eml_benchmark.rs` |
