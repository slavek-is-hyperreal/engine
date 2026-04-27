# Podsumowanie: Dzień 8 — Stabilizacja i Weryfikacja NC1

## Sukces: Theorem C3 Empirycznie Potwierdzony

Wszystkie kluczowe testy są teraz **ZIELONE**. Naprawiono błędy numeryczne i algebraiczne, które blokowały pełną automatyzację.

### Główne Wyniki (Dot Product K=64)
- **Głębokość sekwencyjna:** 260
- **Głębokość zbalansowana (Kogge-Stone):** 32
- **Speedup:** **8.1x**
- **Złożoność:** $O(\log K)$ — Potwierdzona dla $K \in \{4, 8, 16, 32, 64\}$.

### Naprawione Błędy (Post-Code Review)
1.  **Algebra SwiGLU:** Poprawiono fuzję `swiglu_fused`. Wcześniej błędnie liczyła $A - \ln(B)$, teraz poprawnie realizuje dzielenie $A/B$ przez tożsamość logarytmiczną $\exp(\ln(A) - \ln(B))$.
2.  **Stabilność Turniejowa:** `parallel_prefix_sum` został przebudowany. Zamiast ryzykownych odejmowań, które mogły dawać ujemne wyniki pośrednie (blokując `ln`), stosujemy teraz zbalansowane drzewa dodawania dla wyrazów dodatnich i ujemnych osobno, łącząc je na końcu.
3.  **Ewaluacja ln(0):** `try_evaluate` akceptuje teraz `rv=0.0` (zwracając `-inf`), co pozwala na poprawną ewaluację struktury `neg_node` oraz dot-productów z ujemnymi wagami.
4.  **Metryki DAG-aware:** Poprawiono testy `node_count()` i `eml_count()`. Uwzględniają one teraz fakt, że `one()` tworzy unikalne Arki (pointer identity), co daje stabilne wyniki (np. 11 węzłów dla standardowego drzewa testowego).

## Status Projektu
System `eml-trs` jest gotowy do publikacji wyników NC1. Pipeline od ekstraktora gramatyki, przez TRS, aż po zbalansowany dot-product działa stabilnie i numerycznie poprawnie.
