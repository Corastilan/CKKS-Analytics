# Homomorphic Encryption with CKKS on the Iris Dataset

A privacy-preserving analytics pipeline built on Microsoft SEAL's CKKS scheme, demonstrating how aggregate computations can be performed on encrypted data without ever exposing raw values.

---

This simulates a two-party scenario where Alice holds sensitive data and wants Carol to run aggregate queries on it, with Carol never seeing the plaintext. Using the CKKS (Cheon-Kim-Kim-Song) homomorphic encryption scheme, all computation happens on ciphertexts.

The pipeline has four steps:

1. Dataset loading: Load the Iris dataset and compute plaintext ground truths for later verification.
2. Query definitions: Define homomorphic sum, average, and dot-product query functions.
3. CKKS setup: Configure encryption parameters and generate all necessary keys.
4. Performance benchmarking: Compare CKKS encrypted computation against NumPy plaintext baselines across varying dataset sizes and polynomial degrees, producing a six-panel figure and results CSV.

---

## Security Model

| Party | Role | Data Access |
|---|---|---|
| Alice | Data owner | Holds plaintext data and secret key |
| Carol | Compute server | Receives only ciphertexts; never sees raw values |

Alice encrypts her data and sends ciphertexts to Carol. Carol runs all queries homomorphically and returns encrypted results. Only Alice can decrypt.

---

## Project Structure

```
.
├── ckks.py                  # Steps 1–3: dataset loading, CKKS setup, encrypted queries
├── plaintext_vs_ckks.py     # Step 4: performance benchmark and visualization
├── HE_with_CKKS.md          # Implementation documentation (Steps 1–3)
├── performanceanalysis.md   # Implementation documentation (Step 4)
├── results.csv              # Benchmark output (generated on run)
└── performance.pdf          # Six-panel performance figure (generated on run)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `seal` | Microsoft SEAL Python bindings (CKKS scheme) |
| `numpy` | Plaintext baseline operations and array handling |
| `pandas` | DataFrame handling, results collection, CSV export |
| `sklearn` | Iris dataset source |
| `matplotlib` | Performance visualization (Step 4 only) |

---

## Encryption Parameters

| Parameter | Value (deg 8192) | Value (deg 16384) |
|---|---|---|
| Scheme | CKKS | CKKS |
| Poly modulus degree | 8192 | 16384 |
| Coefficient modulus bits | `[60, 40, 40, 60]` | `[60, 40, 40, 40, 60]` |
| Scale | 2⁴⁰ | 2⁴⁰ |
| Slot count | 4096 | 8192 |

> The parameters for `ckks.py` are fixed and must not be changed. They are calibrated for the depth of operations performed (one `multiply_plain` + one `rescale`).

---

## Key Generation

Four keys are generated from a single `KeyGenerator` and reused across all queries:

| Key | Purpose |
|---|---|
| `secret_key` | Decryption (Alice only) |
| `public_key` | Encryption |
| `relin_keys` | Relinearization after ciphertext multiplication |
| `galois_keys` | Ciphertext rotation during `he_sum` |

---

## Core Query Functions

### `he_sum(cipher, n) → float`
Computes the homomorphic sum of the first `n` slots via a rotate-and-add tree reduction in ⌈log₂(n)⌉ steps:

```
result = cipher
step = 1
while step < n:
    result += rotate(result, step)
    step *= 2
return slot_0(decrypt(result))
```

### `he_average(cipher, n) → float`
Divides the homomorphic sum by `n` in plaintext after decryption.

### `he_dot_with_mask(cipher, mask) → float`
Multiplies the ciphertext element-wise by a plaintext mask, then rescales and sums. This is used for approximate min/max estimation.

> True homomorphic min/max requires a polynomial approximation of the comparison function and is not implemented here. Plaintext min/max values are reported as reference only.

---

## Running the Project

Steps 1–3 (encrypted queries on all four Iris features):
```bash
python ckks.py
```

Step 4 (performance benchmark across dataset sizes and polynomial degrees):
```bash
python plaintext_vs_ckks.py
```

This produces `results.csv` and `performance.pdf`.

---

## Benchmark Parameters

| Parameter | Values |
|---|---|
| Dataset sizes | 10, 25, 50, 75, 100, 150 rows |
| Polynomial degrees | 8192, 16384 |
| Trials per measurement | 3 |
| Source column | Sepal length (Iris column 0) |

> When `n_rows > 150`, the Iris sepal-length column is tiled to reach the requested size, keeping the benchmark self-contained.

---

## Expected Accuracy

CKKS is an approximate scheme in which errors are a function of the scale and coefficient modulus configuration:

| Metric | Expected Error |
|---|---|
| Sum error | < 1e-3 |
| Average error | < 1e-5 |

These errors are negligible for statistical analytics workloads.

---

## Performance Overview

CKKS computation is substantially slower than plaintext NumPy for all tested sizes. The dominant cost shifts with dataset size:

| Regime | Dominant cost |
|---|---|
| Small n (≤ 25 rows) | Encryption |
| Larger n (≥ 75 rows) | HE sum (rotate-and-add steps grow as ⌈log₂(n)⌉) |

Doubling the polynomial degree from 8192 to 16384 increases total latency by a roughly constant multiplicative factor across all dataset sizes.

---

## Output Files

### `results.csv`
One row per `(poly_degree, n_rows)` combination with columns for key generation time, plaintext time, encryption time, HE sum time, overhead factor, true and HE averages/sums, and approximation errors.

### `performance.pdf`
A 2×3 matplotlib figure with the following panels:

| Panel | Description |
|---|---|
| HE total latency | Log-scale plot of `total_he_ms` vs. `n_rows` for both polynomial degrees |
| Plaintext vs. HE components | Log-scale breakdown of plaintext, encryption, HE sum, and total (deg 8192) |
| HE overhead vs. plaintext | Grouped bar chart of overhead in units of ×1,000 |
| CKKS approximation error | Scatter plot of average error vs. `n_rows` with 1e-3 threshold line |
| Time breakdown | Stacked area chart of encryption vs. HE sum time (deg 8192) |
| Cost of doubling poly degree | Line plot of `total_he_ms(16384) / total_he_ms(8192)` ratio per dataset size |

---

## Limitations

- Single column: Only sepal length is benchmarked. Extending to all four Iris features would multiply encryption time by four but leave HE sum time unchanged per column.
- No BFV comparison: An integer-domain BFV vs. CKKS comparison is not included and would require a separate context using `scheme_type.bfv`.
- Galois key size: `create_galois_keys()` generates keys for all rotation steps, which is memory-intensive at degree 16384. Production deployments should specify only the steps needed for a given `n`.
- Trial count: `N_TRIALS = 3` keeps runtime short. Increase this for tighter confidence intervals on mean latency.
