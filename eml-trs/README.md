# eml-trs

Term Rewriting System dla operatora EML (Exp-Minus-Log).

Implementacja kompilatora algebraicznego który redukuje sieci neuronowe
do minimalnej liczby operacji EML przez:
- TRS (Term Rewriting System) z regułami algebraicznymi
- Constant Folding wag (mnożenie: 17→5 węzłów)
- ASIS (Algorytm Subtraktywnego Iloczynu Skalarnego, -22.2%)
- DAG (Directed Acyclic Graph, eliminacja duplikatów)

## Podstawa matematyczna

Operator EML: eml(x,y) = exp(x) - ln(y)
Paper: arXiv:2603.21852v2 (Odrzywołek, JU Kraków, 2026)

## Uruchomienie

```bash
cargo test
cargo run --bin tinyllama_costs
cargo run --bin trs_bench
```

## Wyniki (TinyLlama 1.1B)

| Operacja | Naiwny | ASIS | ASIS+CF | Redukcja |
|:---------|-------:|-----:|--------:|:--------:|
| Dot product K=4096 | 147,437 | 114,677 | 57,327 | 61.1% |
| Softmax n=2048 | 71,647 | — | 57,327 (log) | 20.0% |
| RoPE (1 token, per para) | 68 | — | 50 (CF) | 26.4% |
| SwiGLU (per wymiar) | 68 | — | 32 (Fuzja) | 52.9% |
| Cała warstwa (B węzłów) | ~13.1 B | — | ~4.8 B | 63.1% |

Kluczowe odkrycia:
1. **Softmax** = 0.84% kosztu Attention. MatMul dominuje.
2. **Dolna granica Attention**: Ω(n²d) węzłów dla TinyLlamy wynosi 268 milionów węzłów.
3. **Optymalizacja**: Połączenie DAG, ASIS, Constant Folding i Fuzji daje redukcję całej warstwy o 63.1%.
