# Performance Analysis: CKKS Homomorphic Encryption vs. Plaintext

> **Note:** This is Step 4 of the homomorphic encryption pipeline. It benchmarks encrypted vs. plaintext computation across varying dataset sizes and polynomial degrees, producing a six-panel matplotlib figure (`performance.pdf`) and a raw results CSV (`results.csv`).

## Overview

This script extends Steps 1–3 (dataset loading, query functions, and CKKS setup) into a systematic performance benchmark. It measures the cost Alice pays for data confidentiality by comparing NumPy plaintext operations against full CKKS encrypted computation across two polynomial modulus degrees and six dataset sizes.

The benchmark has three phases:

1. **Context and key generation**: Build one SEAL context and one key set per polynomial degree; key generation cost is recorded but amortized across all queries.
2. **Benchmark loop**: For each `(poly_degree, n_rows)` combination, time plaintext sum/average, HE encryption, and HE rotate-and-add sum independently across `N_TRIALS` trials.
3. **Visualization**: Render six matplotlib subplots summarizing latency, overhead, approximation error, time breakdown, and the cost of doubling the polynomial degree.

---

## Dependencies

| Package | Purpose |
|---|---|
| `seal` | Microsoft SEAL Python bindings (CKKS scheme) |
| `numpy` | Plaintext baseline operations and statistical aggregation |
| `pandas` | Results collection and CSV export |
| `sklearn` | Iris dataset source |
| `matplotlib` | Six-panel performance figure |

---

## Experimental Parameters

| Parameter | Values |
|---|---|
| Dataset sizes (`DATASET_SIZES`) | 10, 25, 50, 75, 100, 150 rows |
| Polynomial degrees (`POLY_DEGREES`) | 8192, 16384 |
| Scale (`SCALE`) | 2⁴⁰ |
| Trials per measurement (`N_TRIALS`) | 3 |
| Source column | Sepal length (column 0 of the Iris dataset) |

> **Note on tiling**: When `n_rows > 150`, the 150-row Iris sepal-length column is tiled (repeated and truncated) to reach the requested size. This keeps the benchmark self-contained without requiring an external dataset.

---

## CKKS Context Configuration

Two contexts are built, one per polynomial degree. The coefficient modulus bit-widths differ slightly to accommodate the larger noise budget required at degree 16384.

| Poly degree | Coeff modulus bits | Slot count |
|---|---|---|
| 8192 | `[60, 40, 40, 60]` | 4096 |
| 16384 | `[60, 40, 40, 40, 60]` | 8192 |

Both contexts use `scheme_type.ckks` and a fixed scale of 2⁴⁰.

---

## Key Generation

One key set is generated per polynomial degree and reused across all dataset sizes. Key generation time (`keygen_ms`) is recorded in the results but is not included in the per-query overhead calculation.

| Key | Used for |
|---|---|
| `secret_key` | Decryption (Alice only) |
| `public_key` | Encryption |
| `relin_keys` | Relinearization after ciphertext multiplication |
| `galois_keys` | Ciphertext rotation during `he_sum` |

---

## Core Helper Functions

### `build_context(poly_degree) → SEALContext`

Constructs a `SEALContext` with `scheme_type.ckks` and the appropriate coefficient modulus for the given polynomial degree.

### `build_keys(context) → (sk, pk, rlk, galk)`

Instantiates a `KeyGenerator` and returns all four keys required for the benchmark.

### `encrypt_col(values, encoder, encryptor, slot_count) → Ciphertext`

Encodes and encrypts a NumPy array as a single batched CKKS ciphertext.

- Pads the input to `slot_count` with zeros.
- Encodes using `CKKSEncoder` at the configured scale.
- Returns an encrypted `Ciphertext` object.

### `decrypt_col(ct, encoder, decryptor, n) → np.ndarray`

Decrypts and decodes a ciphertext, returning only the first `n` slots as a NumPy array.

### `he_sum(ct, n, evaluator, galois_keys, encoder, decryptor) → float`

Computes the homomorphic sum of the first `n` slots via a rotate-and-add tree reduction (⌈log₂(n)⌉ steps). Decrypts the final ciphertext and returns the value from slot 0.

```
result = ciphertext
step = 1
while step < n:
    result += rotate(result, step)
    step *= 2
return slot_0(decrypt(result))
```

### `timed(fn, trials=N_TRIALS) → (result, mean_ms, std_ms)`

Runs `fn()` exactly `trials` times using `time.perf_counter`, returning the last result alongside the mean and standard deviation of wall-clock latency in milliseconds.

---

## Benchmark Loop

For each `(poly_degree, n_rows)` pair the script:

1. Builds or reuses the data array (tile if `n_rows > 150`).
2. **Plaintext baseline**: times `np.sum` + `np.mean` via `timed`.
3. **HE encrypt**: times `encrypt_col` via `timed`.
4. **HE sum**: times `he_sum` via `timed`; derives `he_avg = he_s / n_rows` in plaintext.
5. Computes absolute errors (`avg_err`, `sum_err`) and overhead factor (`total_he_ms / pt_ms`).
6. Appends one row to `records`.

---

## Output

### CSV: `results.csv`

One row per `(poly_degree, n_rows)` combination. Columns:

| Column | Description |
|---|---|
| `poly_degree` | Polynomial modulus degree (8192 or 16384) |
| `n_rows` | Dataset size in rows |
| `keygen_ms` | Key generation time in ms (amortized, not in overhead) |
| `pt_ms` | Plaintext sum+average time in ms |
| `enc_ms` | HE encryption time in ms |
| `sum_ms` | HE rotate-and-add sum time in ms |
| `total_he_ms` | `enc_ms + sum_ms` |
| `overhead_x` | `total_he_ms / pt_ms` (raw ratio) |
| `true_avg` / `he_avg` | Plaintext and HE averages |
| `avg_err` | `|he_avg − true_avg|` |
| `true_sum` / `he_sum` | Plaintext and HE sums |
| `sum_err` | `|he_sum − true_sum|` |

### Figure: `performance.pdf`

A 2×3 matplotlib grid with the following panels:

| Panel | Title | Description |
|---|---|---|
| (0,0) | HE total latency | Log-scale line plot of `total_he_ms` vs. `n_rows` for both degrees |
| (0,1) | Plaintext vs. HE components (deg 8192) | Log-scale lines for plaintext, encryption, HE sum, and total |
| (0,2) | HE overhead vs. plaintext | Grouped bar chart of overhead in units of ×1,000 |
| (1,0) | CKKS approximation error | Scatter plot of `avg_err` vs. `n_rows` on log scale; 1e-3 threshold line |
| (1,1) | Time breakdown: encrypt vs. sum (deg 8192) | Stacked area chart of `enc_ms` and `sum_ms` |
| (1,2) | Cost of doubling poly degree | Line + fill of `total_he_ms(16384) / total_he_ms(8192)` ratio per dataset size |

---

## Expected Results

### Latency

CKKS computation is substantially slower than plaintext NumPy for all tested sizes, with overhead driven primarily by encryption rather than the rotate-and-add sum at small `n`.

| Regime | Dominant cost |
|---|---|
| Small `n` (≤ 25) | Encryption |
| Larger `n` (≥ 75) | HE sum (rotate-and-add steps grow as ⌈log₂(n)⌉) |

Doubling the polynomial degree from 8192 to 16384 increases total latency by a roughly constant factor visible in panel (1,2).

### Approximation Error

CKKS is an **approximate** scheme; errors depend on scale and coefficient modulus configuration.

| Metric | Expected error |
|---|---|
| Sum error (`sum_err`) | < 1e-3 |
| Average error (`avg_err`) | < 1e-5 |

These errors are negligible for statistical analytics workloads. The 1e-3 reference line in panel (1,0) makes this boundary explicit.

---

## Security Model

| Party | Role | Data access |
|---|---|---|
| **Alice** | Data owner | Holds plaintext data and secret key |
| **Carol** | Compute server | Receives only ciphertexts; never sees raw values |

The benchmark simulates Alice's encryption cost and Carol's computation cost separately via `enc_ms` and `sum_ms`. Decryption (Alice's side) is included inside `he_sum` for correctness checking but is not reported as a separate overhead column.

---

## Limitations

- **Single column**: Only sepal length (column 0) is benchmarked. Extending to all four Iris features would multiply encryption time by four but leave HE sum time unchanged per column.
- **No BFV comparison**: Step 4b (integer-domain BFV vs. CKKS with scaled floats) is not included in this script. That comparison requires a separate context using `scheme_type.bfv` and integer-scaled input values.
- **Galois key size**: `create_galois_keys()` generates keys for all rotation steps, which is memory-intensive at degree 16384. Production deployments should specify only the steps needed for a given `n`.
- **Trials and variance**: `N_TRIALS = 3` keeps runtime short; increase this for tighter confidence intervals on the mean latency figures.
