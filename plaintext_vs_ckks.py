"""
Performance Analysis: Plaintext vs. CKKS Homomorphic Encryption
"""

import time
from typing import cast

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from seal import (
    Ciphertext,
    CKKSEncoder,
    CoeffModulus,
    Decryptor,
    EncryptionParameters,
    Encryptor,
    Evaluator,
    KeyGenerator,
    SEALContext,
    scheme_type,
)
from sklearn.datasets import load_iris
from sklearn.utils import Bunch

iris = cast(Bunch, load_iris())
df_full = pd.DataFrame(iris.data, columns=iris.feature_names)
# Use sepal length as the representative column for benchmarking
SOURCE_COL = df_full.values[:, 0].astype(float)

# parameters
DATASET_SIZES = [10, 25, 50, 75, 100, 150]
POLY_DEGREES = [8192, 16384]
SCALE = 2.0**40
N_TRIALS = 3  # more trials can be run


def build_context(poly_degree: int) -> SEALContext:
    parms = EncryptionParameters(scheme_type.ckks)  # type: ignore
    parms.set_poly_modulus_degree(poly_degree)
    if poly_degree == 8192:
        parms.set_coeff_modulus(CoeffModulus.Create(poly_degree, [60, 40, 40, 60]))
    else:
        parms.set_coeff_modulus(CoeffModulus.Create(poly_degree, [60, 40, 40, 40, 60]))
    return SEALContext(parms)


def build_keys(context):
    kg = KeyGenerator(context)
    sk = kg.secret_key()
    pk = kg.create_public_key()
    rlk = kg.create_relin_keys()
    galk = kg.create_galois_keys()
    return sk, pk, rlk, galk


def encrypt_col(values, encoder, encryptor, slot_count):
    padded = list(values) + [0.0] * (slot_count - len(values))
    plain = encoder.encode(padded, SCALE)
    return encryptor.encrypt(plain)


def decrypt_col(ct, encoder, decryptor, n):
    plain = decryptor.decrypt(ct)
    result = encoder.decode(plain)
    return np.array(result[:n])


def he_sum(ct, n, evaluator, galois_keys, encoder, decryptor):
    result = Ciphertext(ct)
    step = 1
    while step < n:
        rotated = evaluator.rotate_vector(result, step, galois_keys)
        evaluator.add_inplace(result, rotated)
        step *= 2
    vals = decrypt_col(result, encoder, decryptor, 1)
    return float(vals[0])


def timed(fn, trials=N_TRIALS):
    """Run fn() `trials` times; return (last_result, mean_ms, std_ms)."""
    times: list[float] = []
    result = None
    for _ in range(trials):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000)
    assert result is not None, "Function under test returned None"
    return result, float(np.mean(times)), float(np.std(times))


# Benchmark loop

records = []

for poly_degree in POLY_DEGREES:
    print(f"\n{'=' * 60}\nPoly modulus degree: {poly_degree}\n{'=' * 60}")

    context = build_context(poly_degree)
    t0 = time.perf_counter()
    sk, pk, rlk, galk = build_keys(context)
    keygen_ms = (time.perf_counter() - t0) * 1000

    encryptor = Encryptor(context, pk)
    decryptor = Decryptor(context, sk)
    evaluator = Evaluator(context)
    encoder = CKKSEncoder(context)
    slot_count = encoder.slot_count()

    print(f"  Key generation: {keygen_ms:.1f} ms  |  Slot count: {slot_count}")

    for n_rows in DATASET_SIZES:
        # Build data (tile Iris column if n_rows > 150)
        if n_rows <= len(SOURCE_COL):
            data = SOURCE_COL[:n_rows]
        else:
            reps = (n_rows // len(SOURCE_COL)) + 1
            data = np.tile(SOURCE_COL, reps)[:n_rows]

        # Plaintext baseline
        def pt_ops():
            return np.sum(data), np.mean(data)

        (pt_sum, pt_avg), pt_ms, _ = timed(pt_ops)

        # HE: encrypt
        def enc_op():
            return encrypt_col(data, encoder, encryptor, slot_count)

        ct, enc_ms, _ = timed(enc_op)

        # HE: sum
        def sum_op():
            return he_sum(ct, n_rows, evaluator, galk, encoder, decryptor)

        he_s, sum_ms, _ = timed(sum_op)

        he_avg = he_s / n_rows
        avg_err = abs(he_avg - pt_avg)
        sum_err = abs(he_s - pt_sum)
        total_he_ms = enc_ms + sum_ms
        overhead = total_he_ms / max(pt_ms, 1e-9)

        records.append(
            {
                "poly_degree": poly_degree,
                "n_rows": n_rows,
                "keygen_ms": keygen_ms,
                "pt_ms": pt_ms,
                "enc_ms": enc_ms,
                "sum_ms": sum_ms,
                "total_he_ms": total_he_ms,
                "overhead_x": overhead,
                "true_avg": pt_avg,
                "he_avg": he_avg,
                "avg_err": avg_err,
                "true_sum": pt_sum,
                "he_sum": he_s,
                "sum_err": sum_err,
            }
        )

        print(
            f"  n={n_rows:4d} | PT={pt_ms * 1000:.1f} µs | "
            f"Enc={enc_ms:.1f} ms | Sum={sum_ms:.1f} ms | "
            f"Total={total_he_ms:.1f} ms | Overhead={overhead:.0f}× | "
            f"AvgErr={avg_err:.2e}"
        )

df = pd.DataFrame(records)
df.to_csv("results.csv", index=False)
print("\n[Saved] results.csv")


# Pull out series for plotting
def series(deg, col):
    return df[df.poly_degree == deg].sort_values("n_rows")[col].tolist()  # type: ignore


enc8 = series(8192, "enc_ms")
sum8 = series(8192, "sum_ms")
total8 = series(8192, "total_he_ms")
pt8 = series(8192, "pt_ms")
err8 = series(8192, "avg_err")
over8 = series(8192, "overhead_x")

enc16 = series(16384, "enc_ms")
sum16 = series(16384, "sum_ms")
total16 = series(16384, "total_he_ms")
err16 = series(16384, "avg_err")
over16 = series(16384, "overhead_x")

ratio = [total16[i] / total8[i] for i in range(len(DATASET_SIZES))]

# Matplotlib
C8 = "#378ADD"  # deg 8192
C16 = "#D85A30"  # deg 16384
CPT = "#3B6D11"  # plaintext
CENC = "#888780"  # encrypt component
CSUM = "#7F77DD"  # sum component
CAMB = "#BA7517"  # ratio

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)

fig = plt.figure(figsize=(13, 10))
fig.suptitle(
    "CKKS Homomorphic Encryption: Performance Analysis\n"
    "Plaintext (NumPy) vs. Encrypted Computation on Iris Dataset",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# HE total latency vs dataset size
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(DATASET_SIZES, total8, "o-", color=C8, lw=2, ms=5, label="deg 8192")
ax1.plot(DATASET_SIZES, total16, "s--", color=C16, lw=2, ms=5, label="deg 16384")
ax1.set_yscale("log")
ax1.set_xlabel("Dataset size (rows)")
ax1.set_ylabel("Latency (ms, log scale)")
ax1.set_title("HE total latency\n(encrypt + HE sum)")
ax1.legend()
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f} ms"))
ax1.set_xticks(DATASET_SIZES)

# Plaintext vs HE components (deg 8192)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(DATASET_SIZES, pt8, "^-", color=CPT, lw=2, ms=5, label="Plaintext")
ax2.plot(DATASET_SIZES, enc8, "o-", color=CENC, lw=2, ms=5, label="HE encrypt")
ax2.plot(
    DATASET_SIZES, sum8, "s--", color=CSUM, lw=2, ms=5, label="HE sum (rotate-add)"
)
ax2.plot(DATASET_SIZES, total8, "D-", color=C8, lw=2, ms=5, label="HE total")
ax2.set_yscale("log")
ax2.set_xlabel("Dataset size (rows)")
ax2.set_ylabel("Time (ms, log scale)")
ax2.set_title("Plaintext vs. HE components\n(poly deg 8192)")
ax2.legend(fontsize=8)
ax2.yaxis.set_major_formatter(
    FuncFormatter(lambda v, _: f"{v:.4f}" if v < 0.01 else f"{v:.0f}")
)
ax2.set_xticks(DATASET_SIZES)

# Overhead factor (grouped bars)
ax3 = fig.add_subplot(gs[0, 2])
x = np.arange(len(DATASET_SIZES))
w = 0.35
over8k = [o / 1000 for o in over8]
over16k = [o / 1000 for o in over16]
bars8 = ax3.bar(
    x - w / 2, over8k, w, color=C8, alpha=0.85, label="deg 8192", edgecolor=C8
)
bars16 = ax3.bar(
    x + w / 2, over16k, w, color=C16, alpha=0.85, label="deg 16384", edgecolor=C16
)
ax3.set_xlabel("Dataset size (rows)")
ax3.set_ylabel("Overhead (×1,000)")
ax3.set_title("HE overhead vs. plaintext\n(×1,000)")
ax3.set_xticks(x)
ax3.set_xticklabels([str(s) for s in DATASET_SIZES])
ax3.legend()
ax3.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}k×"))
for bar in list(bars8) + list(bars16):
    h = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        h + max(over8k) * 0.01,
        f"{h:.0f}k",
        ha="center",
        va="bottom",
        fontsize=7,
    )

# CKKS approximation error
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(DATASET_SIZES, err8, color=C8, s=60, zorder=3, label="deg 8192", marker="o")
ax4.scatter(
    DATASET_SIZES, err16, color=C16, s=60, zorder=3, label="deg 16384", marker="s"
)
ax4.axhline(1e-3, color="gray", lw=1, ls=":", label="1e-3 threshold")
ax4.set_yscale("log")
ax4.set_xlabel("Dataset size (rows)")
ax4.set_ylabel("|HE avg − true avg|")
ax4.set_title("CKKS approximation error\n(average query)")
ax4.legend(fontsize=8)
ax4.set_xticks(DATASET_SIZES)
ax4.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}"))

# Stacked area: encrypt vs sum time (deg 8192)
ax5 = fig.add_subplot(gs[1, 1])
ax5.stackplot(
    DATASET_SIZES,
    enc8,
    sum8,
    labels=["Encryption", "HE sum (rotate-add)"],
    colors=[CENC, CSUM],
    alpha=0.85,
)
ax5.set_xlabel("Dataset size (rows)")
ax5.set_ylabel("Latency (ms)")
ax5.set_title("Time breakdown: encrypt vs. sum\n(poly deg 8192)")
ax5.legend(loc="upper left", fontsize=8)
ax5.set_xticks(DATASET_SIZES)

# Latency ratio deg 16384 / deg 8192
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(DATASET_SIZES, ratio, "D-", color=CAMB, lw=2, ms=6)
ax6.axhline(1.0, color="gray", lw=1, ls=":")
ax6.fill_between(DATASET_SIZES, 1.0, ratio, alpha=0.15, color=CAMB)
ax6.set_xlabel("Dataset size (rows)")
ax6.set_ylabel("Latency ratio (16384 / 8192)")
ax6.set_title("Cost of doubling poly degree\n(16384 vs. 8192)")
ax6.set_xticks(DATASET_SIZES)
ax6.set_ylim(0, max(ratio) * 1.3)
for xi, yi in zip(DATASET_SIZES, ratio):
    ax6.text(
        xi,
        yi + max(ratio) * 0.04,
        f"{yi:.1f}×",
        ha="center",
        fontsize=8,
        color="#633806",
    )

plt.savefig("performance.pdf", bbox_inches="tight")

print("performance.pdf saved to files")
