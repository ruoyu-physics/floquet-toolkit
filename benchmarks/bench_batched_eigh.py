"""Microbenchmark: looped vs. batched ``eigh`` on the Floquet workload shape.

The cold ``current vs A`` sweep spends ~27% of its time diagonalizing one small
Floquet matrix per k-point in a Python loop. Batching the k-loop would instead
stack them into a single ``(n_k, dim, dim)`` array and call ``np.linalg.eigh``
once. This script measures the realistic ceiling of that change for the
diagonalization step alone — *before* committing to the cross-module refactor.

Workload shape (defaults): ``n_k = 1681`` Hermitian ``complex128`` matrices of
size ``dim = (2*n_trunc+1)*2 = 46``.

BLAS is pinned to one thread to match the per-worker reality of the parallel
amplitude sweep (and because 46x46 is below OpenBLAS's multithreading
threshold anyway).

Usage (from the repository root)::

    python3 -m benchmarks.bench_batched_eigh            # defaults
    python3 -m benchmarks.bench_batched_eigh 1681 46    # n_k, dim
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import sys
import time

import numpy as np

N_K = 1681
DIM = 46          # (2 * n_trunc + 1) * 2 for n_trunc = 11
N_REPEATS = 5


def _random_hermitian_stack(n_k: int, dim: int, seed: int = 0) -> np.ndarray:
    """Return ``n_k`` random Hermitian complex128 matrices of size ``dim``."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_k, dim, dim)) + 1j * rng.standard_normal((n_k, dim, dim))
    return a + np.conj(np.transpose(a, (0, 2, 1)))  # Hermitize


def _best_time(fn, repeats: int) -> float:
    """Return the best (min) wall time of ``fn`` over ``repeats`` runs."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    n_k = int(sys.argv[1]) if len(sys.argv) > 1 else N_K
    dim = int(sys.argv[2]) if len(sys.argv) > 2 else DIM
    mats = _random_hermitian_stack(n_k, dim)

    # Warm up (first call pays one-time setup).
    np.linalg.eigh(mats[0])
    np.linalg.eigh(mats[:2])

    looped = _best_time(lambda: [np.linalg.eigh(m) for m in mats], N_REPEATS)
    batched = _best_time(lambda: np.linalg.eigh(mats), N_REPEATS)

    # Correctness: eigenvalues must agree (eigh sorts ascending).
    w_loop = np.array([np.linalg.eigvalsh(m) for m in mats])
    w_batch = np.linalg.eigvalsh(mats)
    max_diff = float(np.max(np.abs(w_loop - w_batch)))

    print(f"n_k = {n_k}   dim = {dim}x{dim}   complex128   BLAS threads = 1")
    print(f"  looped  eigh : {looped * 1e3:8.1f} ms   ({looped / n_k * 1e6:6.1f} us/matrix)")
    print(f"  batched eigh : {batched * 1e3:8.1f} ms   ({batched / n_k * 1e6:6.1f} us/matrix)")
    print(f"  speedup      : {looped / batched:6.2f}x")
    print(f"  eigenvalue max abs diff: {max_diff:.2e}")
    print()
    print("Note: this bounds ONLY the diagonalization step (~27% of cold sweep")
    print("time). Net sweep speedup from batching is at most this x weighted by")
    print("that 27% share (Amdahl), plus whatever batching the construction adds.")


if __name__ == "__main__":
    main()
