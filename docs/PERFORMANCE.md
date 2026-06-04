# Performance notes

A curated log of performance work: what was measured, what changed, and *why*.
Git holds the exact diffs; this file holds the reasoning and the numbers that
diffs can't show. Reproduce the measurements with
[`../benchmarks/profile_current.py`](../benchmarks/profile_current.py).

All timings below are single-threaded, on an 8-core development laptop
(NumPy + OpenBLAS), on one `integrate_floquet_current_on_fermi_disk` call
(pointwise polar path, Dirac model) at `n_k_points = 41`, `A = 1.5e-9`.

> **Cold vs. warm — read this first.** Two regimes have very different costs and
> it's easy to conflate them:
>
> - **Cold** = first computation of a grid/amplitude (cache miss). Dominated by
>   building and diagonalizing the Floquet Hamiltonian at every k-point.
> - **Warm** = the same states requested again (cache hit) — e.g. a second
>   observable on the same grid+amplitude, or a repeated call.
>
> An earlier draft reported a single-calc "90×" that was actually a *warm-cache*
> measurement. The honest figures are below.

---

## Headline numbers

| Regime | Time (n_k=41) | Bound by |
| --- | --- | --- |
| Cold, original | ~2.9 s | Floquet-matrix construction + velocity finite difference |
| Cold, + harmonic vectorization | ~0.9 s | per-k construction + diagonalization |
| Cold, + batched construction | **~0.63 s** | `eigh` (~27%) + reconstruct/glue |
| Warm (cached repeat) | **~0.055 s** | cache reads + the `einsum` expectation value |

The optimizations split cleanly: the **analytic velocity operator** made the
*warm* path ~90× faster (5.0 s → 0.055 s); **vectorizing the Fourier harmonics**
took the *cold* path 2.9 s → 0.9 s; and **batched-over-k construction** took it
0.9 s → 0.63 s (~1.46×). What's left is `eigh` (LAPACK-bound, not worth
batching — see microbenchmark) plus reconstruct/glue.

---

## Profiles

### Cold bottleneck (current code)
Building the Floquet Hamiltonian dominates: `compute_fourier_harmonics` samples
`H_t` across the time grid at every k-point. After vectorization (below) the
cold cost is ~0.9 s, with `eigh` now a visible minority and the rest split
between residual construction, state selection, and reconstruction.

### Why `eigh` was never the target
Profiling proved diagonalization is *not* the bottleneck — the Floquet matrices
are small (46×46). `eigh` is ~9% of the cold cost and ~0% of the warm cost.
GPU acceleration or batched `eigh` would optimize something that costs little;
the wins were all in **reducing Python-level Hamiltonian construction**.

---

## Changes and their effect

### 1. Memoize the polarization axis (`utils/drive_fields.py`)
`_normalized_polarization_axis` recomputed `np.linalg.norm` of a **constant**
polarization vector on every drive-field evaluation. Wrapped in an `lru_cache`
keyed on a hashable tuple (NumPy arrays are unhashable, so the wrapper converts
first). Output unchanged; the repeated norms collapse to one.

### 2. Analytic velocity operator + vectorized `H_t`
(`builtin_models/*.py`, `core/driven_bloch_hamiltonian.py`,
`calculators/floquet_velocity_calculator.py`)

- **Analytic Dirac velocity operator:** for `H_t(t,k) = H_static(k − qA(t)/ℏ)`
  the velocity operator `(1/ℏ) ∂H_t/∂k = v_f σ_axis` is **constant** —
  independent of time, momentum, and drive. The finite-difference loop (which
  re-evaluated `H_t` ~410k times per current calc) is replaced by this exact
  closed form. This is what makes the warm path ~90× faster.
- **Vectorized `H_t`:** all built-in `H_static`/`H_t` broadcast over a leading
  time axis, and the velocity calculator's per-time loop became a single array
  call when `supports_vectorized_time = True` (legacy loop kept as fallback).
- **Resolution order in the velocity calculator:** analytic operator → vectorized
  finite difference → per-time loop.

Models without an analytic form (graphene, rotating-frame) use the vectorized
finite difference; their `∂H/∂k` genuinely varies with `t` and `k`.

### 3. Vectorize the Fourier harmonics (`builders/floquet_builder.py`)
`compute_fourier_harmonics` sampled `H_t` at scalar times in a double loop
(~100k `H_t` calls per cold current calc, ~73% of cold time). Now it evaluates
`H_t` on the whole time grid in one call and contracts against the Fourier
phases with an `einsum`, falling back to the loop for non-array-aware `Ht`.
Cold single calc: **~2.9 s → ~0.9 s**.

### Correctness
- Vectorized `H_t` vs scalar loop (all built-in models): **bit-identical**.
- Vectorized Fourier harmonics vs scalar loop: max abs diff **~5×10⁻³⁷**.
- Analytic velocity operator: exactly `v_f σ`.
- Analytic vs finite-difference current: max rel diff **~2×10⁻¹⁴** (analytic is
  the exact value the finite difference approximated).
- Production scan matches the pre-change baseline at `rtol = 1e-6`.

---

## Floquet-state cache

The `FloquetStateCache` stores eigensystems, selected states, and reconstructed
states keyed by (rounded) momentum and time.

- **Cold pass:** each k-point is visited once → all misses. Caching is roughly
  free here (cold on ≈ cold off ≈ ~0.9 s); the key-rounding overhead is
  negligible.
- **Repeated access:** a 53× speedup (warm: ~3.0 s → ~0.055 s) whenever the same
  states are requested again — multiple observables sharing a grid+amplitude,
  repeated calls, or adaptive refinement.

`FloquetTransportManager(..., use_cache=False)` is the default, tuned for the
single-observable cold sweep where each momentum is visited once (so the cache
is ~13% pure overhead). Pass `use_cache=True` for workflows that revisit the
same momenta — multiple observables sharing a grid, repeated calls, or adaptive
refinement — where the 53× warm speedup applies.

---

## Sweep-level parallelism (orthogonal axis)

`current vs A` sweeps are embarrassingly parallel: each amplitude rebuilds its
own model/manager. `utils/parallel.py` provides `parallel_map`, used by the
sweep scripts (`plot_current_vs_amplitude.py`,
`validate_cartesian_vs_polar_integration.py`,
`check_curvature_integration.py`).

### Design rules learned here
- **Pin BLAS to one thread** when process-parallelizing the sweep, or N worker
  processes × M BLAS threads oversubscribe the cores. The guard
  (`OMP_NUM_THREADS=1`, etc.) must be set **before NumPy is imported**, so it
  lives inline at the top of each entry-point script — it cannot move into a
  shared helper, because importing `floquet_toolkit` already pulls in NumPy.
- **Parallelism, BLAS-pinning, and vectorization are orthogonal and compose.**
  Vectorization removes Python-call overhead and runs single-threaded, so it
  still helps each worker even with BLAS pinned. Parallelism divides the sweep
  across cores; vectorization shrinks the per-amplitude work; they multiply.
- The single-calc speedups are **per-amplitude**; they multiply with the
  ~core-count sweep parallelism for full `current vs A` wall-clock time.

---

## Batching microbenchmarks (k-loop refactor scoping)

Before committing to batching the per-k-point loop, the two heavy pieces were
microbenchmarked in isolation (1681 matrices, 46×46, complex128, BLAS=1):

| Piece | Per-k | Batched | Speedup | Bound by |
| --- | --- | --- | --- | --- |
| `eigh` | 462 ms | 422 ms | **1.10×** | LAPACK work |
| Floquet construction | 299 ms | 36 ms | **8.22×** | Python overhead |

Conclusion: batching `eigh` is pointless (and confirms GPU would not help —
the kernel is already LAPACK-bound), but batching the **construction** is a
real ~8× on its ~30% share. See `bench_batched_eigh.py` and
`bench_batched_construction.py`.

### Implemented: batched-over-k construction

Construction batching is now shipped (the eigh / GPU paths were *not* pursued):

- `FloquetBuilder.compute_floquet_hamiltonians_batched` / `compute_fourier_harmonics_batched`
  build all grid matrices in one vectorized `(k, t)` evaluation + block-Toeplitz
  band-fill. The per-k `compute_floquet_hamiltonian` is kept unchanged.
- `FloquetStateProvider.diagonalize_floquet_hamiltonian_batched` does one batched
  `eigh` over the stack.
- `FloquetCurrentCalculator._integrate_local_current` batch-primes all grid
  eigensystems before the per-point loop (`_prime_batched_eigensystems`), so grid
  integration uses the batched construction while **single-momentum callers and
  the adaptive/tracked paths keep the per-k path**. Gated on
  `supports_vectorized_time`, with a per-k fallback otherwise.

Verified bit-identical: batched vs per-k construction and eigenvalues agree to
**0.0**; the production scan matches the pre-change baseline at `rtol=1e-6`;
full test suite green. Measured cold single calc (n_k=41): **915 ms → 628 ms
(~1.46×)** — in line with the projection.

## Reproducing

```bash
python3 -m benchmarks.profile_current            # cold breakdown for one current calc
python3 -m benchmarks.profile_current 51 2.0e-9  # custom n_k_points, amplitude
```
