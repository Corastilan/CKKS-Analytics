# Homomorphic Encryption with CKKS on the Iris Dataset


## Overview

 `ckks.py `demonstrates privacy-preserving analytics using the CKKS (Cheon-Kim-Kim-Song) homomorphic encryption scheme via Microsoft SEAL's Python bindings. It simulates a scenario where Alice holds sensitive data and wants Carol to perform aggregate computations on it without Carol ever seeing the raw values.

The script currently has three phases:

1. Plaintext baseline: load the Iris dataset and compute ground-truth statistics for comparison.
2. CKKS setup: configure encryption parameters and generate the necessary keys.
3. Encrypted queries: encrypt each feature column and run sum/average queries entirely on ciphertexts.

---

## Dependencies

| Package | Purpose |
|---|---|
| `seal` | Microsoft SEAL Python bindings (CKKS scheme) |
| `numpy` | Numerical array operations |
| `pandas` | DataFrame handling and output formatting |
| `sklearn` | Loading the Iris dataset |

---

## Dataset

The script uses the Iris dataset from `sklearn.datasets`, which contains 150 rows and 4 numeric features:

- sepal length (cm)
- sepal width (cm)
- petal length (cm)
- petal width (cm)

Plaintext ground truths (mean, min, max, sum) are printed at startup for later error verification.

---

## CKKS Encryption Parameters

| Parameter | Value |
|---|---|
| Scheme | CKKS |
| Poly modulus degree | 8192 |
| Scale | 2⁴⁰ |
| Coefficient modulus bits | `[60, 40, 40, 60]` |
| Slot count | 4096 (half of poly modulus degree) |

> Note: These parameters are fixed and must not be changed, as they are calibrated for the depth of operations performed (one `multiply_plain` + one `rescale`).

---

## Key Generation

Four keys are generated from a single `KeyGenerator`:

| Key | Used For |
|---|---|
| `secret_key` | Decryption (Alice only) |
| `public_key` | Encryption |
| `relin_keys` | Relinearization after ciphertext multiplication |
| `galois_keys` | Ciphertext rotation (required for `he_sum`) |

---

## Core Helper Functions

### `encrypt_column(values, name) → Ciphertext`

Encodes and encrypts a NumPy array as a single batched CKKS ciphertext.

- Pads the input to `slot_count` with zeros.
- Encodes using `CKKSEncoder` at the configured scale.
- Returns an encrypted `Ciphertext` object.

### `decrypt_column(cipher, n) → np.ndarray`

Decrypts and decodes a ciphertext, returning only the first `n` slots.

---

## Query Functions

### `he_sum(cipher, n) → float`

Computes the homomorphic sum of the first `n` slots using a rotate-and-add tree reduction (⌈log₂(n)⌉ steps).

```
result = cipher
step = 1
while step < n:
    result += rotate(result, step)
    step *= 2
return slot_0(result)
```

Reads the accumulated sum from slot 0 of the final ciphertext.

### `he_average(cipher, n) → float`

Divides the homomorphic sum by `n` in plaintext after decryption.

```
he_average = he_sum(cipher, n) / n
```

### `he_dot_with_mask(cipher, mask) → float`

Multiplies the ciphertext element-wise by a plaintext mask vector, then rescales and decrypts to compute a weighted dot product. Used for approximate min/max estimation.

> Note: True homomorphic min/max requires polynomial approximation and is not implemented in this script. Plaintext min/max values are reported for reference only.

---

## Execution Flow

For each of the 4 Iris feature columns, the script:

1. Extracts the column as a float array.
2. Encrypts the array into a single ciphertext and records the time.
3. Runs `he_sum` on the ciphertext and records the time and error vs. plaintext.
4. Runs `he_average` on the ciphertext and records the time and error vs. plaintext.
5. Appends all results to a summary table.

---

## Output

### Per-feature console output

```
Feature: sepal length (cm)
  Encrypt time      : X.XX ms
  HE Sum            : XXX.XXXX  (true: XXX.XXXX, err: X.XXe-XX)
  HE Sum time       : X.XX ms
  HE Average        : X.XXXX   (true: X.XXXX,   err: X.XXe-XX)
  HE Average time   : X.XX ms
  [Plaintext] Min   : X.XXXX  Max: X.XXXX
```

### Summary table columns

| Column | Description |
|---|---|
| `Feature` | Iris feature name |
| `True Sum` / `HE Sum` | Plaintext and encrypted sums |
| `Sum Err` | Absolute error between the two |
| `True Avg` / `HE Avg` | Plaintext and encrypted averages |
| `Avg Err` | Absolute error between the two |
| `True Min` / `True Max` | Plaintext min and max (reference only) |
| `Enc ms` | Encryption time in milliseconds |
| `Sum ms` | Homomorphic sum query time in milliseconds |
| `Avg ms` | Homomorphic average query time in milliseconds |

---

## Expected Accuracy

CKKS is an approximate scheme by design. Errors are a function of the scale and coefficient modulus configuration.

| Metric | Expected Error |
|---|---|
| Sum error | < 1e-3 |
| Average error | < 1e-5 |

These errors are negligible for statistical analytics workloads.

---

## Security Model

| Party | Role | Data Access |
|---|---|---|
| Alice | Data owner | Holds plaintext data and secret key |
| Carol | Compute server | Receives only ciphertexts; never sees raw values |

All computations (`he_sum`, `he_average`) are performed on ciphertexts on Carol's side. Alice sends encrypted data and receives encrypted results, which only she can decrypt.

---

## Limitations

- Min/Max cannot be computed exactly with CKKS without a polynomial approximation of the comparison function (not implemented here).
- Performance scales with `poly_modulus_degree`; larger degrees improve security but increase latency.
- The Galois key set generated by `create_galois_keys()` covers all rotation steps needed for a 150-element rotate-and-add, but generates keys for all possible steps, which is memory-intensive.
