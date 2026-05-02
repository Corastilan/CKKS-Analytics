import tenseal as ts
import numpy as np
import time
import sys
from sklearn.datasets import load_iris

def create_context(n_elements):
    # Sets up CKKS parameters.
    # For 10,000 points, we need a degree of at least 32768 (16,384 slots)
    poly_mod = 8192 if n_elements <= 4096 else 32768
    
    # Adjusting coefficient modulus for security/depth at higher degrees
    coeff_bits = [60, 40, 40, 60] if poly_mod == 8192 else [60, 40, 40, 40, 40, 60]
    
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_mod,
        coeff_mod_bit_sizes=coeff_bits
    )
    context.global_scale = 2 ** 40
    context.generate_relin_keys()
    context.generate_galois_keys()
    return context

# Data Generation
def get_data(n_elements):
    # Loads Iris for 150, or generates synthetic data for larger sets.
    if n_elements <= 150:
        iris = load_iris()
        return iris.data[:n_elements, 0]
    else:
        # Generate synthetic data based on Iris-like distribution (Sepal Length)
        return np.random.normal(5.8, 0.8, n_elements)

# Carol, the cloud server
def carol_compute_average(encrypted_vector, n_elements):
    # Performs sum and scalar multiplication on encrypted data.
    return encrypted_vector.sum() * (1.0 / n_elements)

def run_experiment(n_elements):
    print(f"\n==== Experiment: {n_elements} Data Points ====")
    
    # Setup
    context = create_context(n_elements)
    data = get_data(n_elements)

    # Alice encryption
    t_start = time.perf_counter()
    enc_vector = ts.ckks_vector(context, data)
    enc_time = (time.perf_counter() - t_start) * 1000

    # Carol (Server) computation
    t_start = time.perf_counter()
    enc_avg = carol_compute_average(enc_vector, n_elements)
    he_compute_time = (time.perf_counter() - t_start) * 1000

    # Alice Decryption
    t_start = time.perf_counter()
    he_avg = enc_avg.decrypt()[0]
    dec_time = (time.perf_counter() - t_start) * 1000

    # Plaintext comparison
    t_start = time.perf_counter()
    plain_avg = np.mean(data)
    plain_time = (time.perf_counter() - t_start) * 1000

    # Print results
    print(f"HE Calculated Average:    {he_avg:.6f}")
    print(f"Plaintext Average:       {plain_avg:.6f}")
    print(f"Accuracy Difference:     {abs(he_avg - plain_avg):.2e}")

    print(f"\nPERFORMANCE:")
    print(f"Encryption Time:         {enc_time:.2f} ms")
    print(f"HE Compute Time:         {he_compute_time:.2f} ms")
    print(f"Plaintext Compute:       {plain_time:.4f} ms")
    
    # Memory Insight
    serialized_size = sys.getsizeof(enc_vector.serialize()) / 1024
    print(f"Encrypted Data Size:     {serialized_size:.2f} KB")
    
    return {
        "size": n_elements,
        "he_time": he_compute_time,
        "plain_time": plain_time,
        "overhead": he_compute_time / plain_time if plain_time > 0 else 0
    }

def main():
    sizes = [150, 1000, 10000]
    results = []
    for size in sizes:
        results.append(run_experiment(size))
    
    print("\n" + "="*75)
    print(f"{'Data Points':<15} | {'HE Time (ms)':<15} | {'Plain Time (ms)':<15} | {'Overhead Factor'}")
    print("-" * 70)
    for res in results:
        print(f"{res['size']:<15} | {res['he_time']:<15.2f} | {res['plain_time']:<15.4f} | {res['overhead']:.1f}x slower")
    print("="*75)

if __name__ == "__main__":
    main()