"""Microbenchmark: per-k vs. batched-over-k Floquet-matrix construction.

The cold ``current vs A`` sweep spends ~30% of its time building the Floquet
Hamiltonian one k-point at a time (Fourier harmonics + an inner
``(2*n_trunc+1)**2`` block-assembly loop). This script times the shipped
batched construction (``FloquetBuilder.compute_floquet_hamiltonians_batched``,
used by grid integration) against the per-k path.

It does NOT include ``eigh`` (benchmarked separately in
``bench_batched_eigh.py``). BLAS is pinned to one thread to match the
per-worker reality of the parallel amplitude sweep.

Usage (from the repository root)::

    python3 -m benchmarks.bench_batched_construction          # defaults
    python3 -m benchmarks.bench_batched_construction 1681     # n_k
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
from functools import partial

import numpy as np

from floquet_toolkit import DiracModel, DiracParameters
from floquet_toolkit.builders import FloquetBuilder
from floquet_toolkit.config import FloquetParameters, MEV_TO_J, UnitConvention
from floquet_toolkit.utils import build_circular_drive, fermi_momentum

N_K = 1681
N_REPEATS = 3

SI = UnitConvention.SI_UNITS()
DP = DiracParameters(units=SI, vf=1.0e6, mass=-1.0 * MEV_TO_J, e_fermi=25.0 * MEV_TO_J)
FP = FloquetParameters(n_trunc=11, n_harmonics=3, n_time=61, dk=1.0e4)
OMEGA = 45.0 * MEV_TO_J / SI.hbar


def _best(fn, repeats=N_REPEATS):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def per_k_construction(model, kxs, kys):
    """Current path: build each Floquet matrix in its own FloquetBuilder."""
    out = np.empty((len(kxs), FP.n_blocks * 2, FP.n_blocks * 2), dtype=complex)
    for i, (kx, ky) in enumerate(zip(kxs, kys)):
        builder = FloquetBuilder(partial(model.H_t, kx=kx, ky=ky), model.omega, model.hbar, FP)
        out[i] = builder.compute_floquet_hamiltonian()
    return out


def batched_construction(model, kxs, kys):
    """Shipped library path: all Floquet matrices built in one batched call."""
    builder = FloquetBuilder(model.H_t, model.omega, model.hbar, FP)
    return builder.compute_floquet_hamiltonians_batched(kxs, kys)


def main() -> None:
    n_k = int(sys.argv[1]) if len(sys.argv) > 1 else N_K
    model = DiracModel(DP, build_circular_drive(1.5e-9, units=SI, omega=OMEGA, handedness="right"))
    k_fermi = fermi_momentum(DP)
    rng = np.random.default_rng(0)
    radii = k_fermi * np.sqrt(rng.uniform(0, 1, n_k))
    angles = rng.uniform(0, 2 * np.pi, n_k)
    kxs, kys = radii * np.cos(angles), radii * np.sin(angles)

    # Correctness: batched matrices must equal the per-k builder output.
    ref = per_k_construction(model, kxs[:8], kys[:8])
    bat = batched_construction(model, kxs[:8], kys[:8])
    max_diff = float(np.max(np.abs(ref - bat)))

    per_k = _best(lambda: per_k_construction(model, kxs, kys))
    batched = _best(lambda: batched_construction(model, kxs, kys))

    print(f"n_k = {n_k}   matrix = {FP.n_blocks * 2}x{FP.n_blocks * 2}   BLAS threads = 1")
    print(f"  per-k construction  : {per_k * 1e3:8.1f} ms")
    print(f"  batched construction: {batched * 1e3:8.1f} ms")
    print(f"  speedup             : {per_k / batched:6.2f}x")
    print(f"  max abs matrix diff : {max_diff:.2e}")
    print()
    print("Construction is ~30% of cold sweep time; net sweep gain is this")
    print("speedup weighted by that share (Amdahl), combined with eigh (~3%).")


if __name__ == "__main__":
    main()
