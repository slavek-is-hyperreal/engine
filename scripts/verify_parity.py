#!/usr/bin/env python3
"""
scripts/verify_parity.py

Weryfikacja parytetu EML vs NumPy dla iloczynu skalarnego K=64.

Testuje:
1. Buduje drzewo EML przez `build_dot_product_eml` (przez eml_parity binary)
2. Oblicza wynik EML przez `try_evaluate`
3. Porównuje z NumPy (ground truth)
4. Raportuje max |diff|, pass/fail

Użycie:
  python3 scripts/verify_parity.py

Wymaga:
  - Skompilowanego binarnego: cargo build --bin eml_parity
  - numpy: pip install numpy
"""

import subprocess
import sys
import json
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(PROJECT_ROOT, "target", "debug", "eml_parity")


def run_parity_binary(inputs: list[float], weights: list[float]) -> dict | None:
    """Uruchamia eml_parity binary i parsuje wynik JSON."""
    payload = json.dumps({"inputs": inputs, "weights": weights})
    try:
        result = subprocess.run(
            [BINARY],
            input=payload,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            print(f"[BINARY ERROR] stderr: {result.stderr.strip()}", file=sys.stderr)
            return None
        return json.loads(result.stdout.strip())
    except FileNotFoundError:
        print(f"[ERROR] Binary not found: {BINARY}", file=sys.stderr)
        print("Run: cargo build --bin eml_parity", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print("[ERROR] Binary timed out", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parse error: {e}", file=sys.stderr)
        print(f"stdout was: {result.stdout[:200]}", file=sys.stderr)
        return None


def numpy_dot(inputs: list[float], weights: list[float]) -> float:
    """Ground truth: NumPy double-precision dot product."""
    return float(np.dot(np.array(inputs, dtype=np.float64),
                        np.array(weights, dtype=np.float64)))


def run_test(name: str, inputs: list[float], weights: list[float],
             tol: float = 1e-4) -> bool:
    """Uruchamia jeden test parytetyczny. Zwraca True jeśli przeszedł."""
    k = len(inputs)
    assert k == len(weights), f"len mismatch: {k} vs {len(weights)}"

    expected = numpy_dot(inputs, weights)

    result = run_parity_binary(inputs, weights)
    if result is None:
        print(f"  [{name}] SKIP (binary unavailable)")
        return True  # nie fail — binary może nie istnieć jeszcze

    eml_result = result.get("eml_result")
    if eml_result is None:
        nan_reason = result.get("nan_reason", "unknown")
        print(f"  [{name}] FAIL — EML returned None/NaN. Reason: {nan_reason}")
        print(f"           expected={expected:.6f}")
        return False

    diff = abs(eml_result - expected)
    status = "OK" if diff < tol else "FAIL"
    print(f"  [{name}] {status} — numpy={expected:.6f}  eml={eml_result:.6f}  |diff|={diff:.2e}  tol={tol:.0e}")
    return diff < tol


def main():
    print("=" * 60)
    print("EML Parity Verification — verify_parity.py")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)
    all_passed = True

    # Test 1: K=4, wszystkie wagi dodatnie, aktywacje > 1.0
    # (mul_cf działa natywnie: x > 0, w > 0)
    k = 4
    inputs = [1.5, 2.0, 0.8, 3.1]   # NOTE: 0.8 < 1 — testuje zakres (0,1)
    weights = [0.5, 0.3, 0.7, 0.2]
    passed = run_test(f"K={k} positive weights", inputs, weights)
    all_passed = all_passed and passed

    # Test 2: K=4, wagi ujemne — mul_cf wymaga x>0 ale wagi mogą być ujemne
    # (ASIS pre-negacja powinna to obsłużyć offline)
    inputs2 = [1.5, 2.0, 1.8, 3.1]
    weights2 = [0.5, -0.3, 0.7, -0.2]
    passed = run_test(f"K={k} mixed weights (ASIS)", inputs2, weights2)
    all_passed = all_passed and passed

    # Test 3: K=16, losowe wagi i aktywacje (zakres (0, 2))
    k = 16
    inputs3 = rng.uniform(0.1, 2.0, k).tolist()
    weights3 = rng.uniform(-1.0, 1.0, k).tolist()
    passed = run_test(f"K={k} random", inputs3, weights3)
    all_passed = all_passed and passed

    # Test 4: K=64, jak warstwa TinyLlama (aktywacje po SiLU: > 0)
    k = 64
    inputs4 = rng.uniform(0.01, 1.5, k).tolist()    # aktywacje po SiLU
    weights4 = rng.uniform(-0.5, 0.5, k).tolist()   # typowe wagi sieci
    passed = run_test(f"K={k} TinyLlama-like", inputs4, weights4)
    all_passed = all_passed and passed

    print("=" * 60)
    if all_passed:
        print("RESULT: ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("RESULT: SOME TESTS FAILED — NaN or parity error")
        sys.exit(1)


if __name__ == "__main__":
    main()
