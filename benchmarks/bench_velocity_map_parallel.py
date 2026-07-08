"""Smoke test: serial vs k-point-parallel for the BATCHED velocity/current map.

A *velocity map* computes the local Floquet current ``max_t |j(k,t)|`` at every
point of a single-amplitude k-grid (the workload behind
``scripts/plot_local_floquet_current_map.py``).

The library provides a vectorized
``FloquetVelocityCalculator.compute_floquet_velocity_map`` (one batched ``eigh``
+ batched state selection + batched velocity). This benchmark uses THAT as the
serial baseline — not a per-k Python loop — and then parallelizes it by handing
each worker a chunk of the grid (so batching and process-parallelism compose).
A per-k loop is timed too, purely as a reference for what batching alone buys.

It is a smoke test, not a rigorous benchmark: small grid, one run each.

NOTE: the parallel path uses ``ProcessPoolExecutor``; some sandboxes block
process creation. Run with ``n_jobs=1`` there to verify correctness, and with
``n_jobs=-1`` on a real machine for the actual speedup.

Usage (from the repository root)::

    python3 -m benchmarks.bench_velocity_map_parallel            # n_jobs=-1 (all cores)
    python3 -m benchmarks.bench_velocity_map_parallel 81 -1      # n_k axis, n_jobs
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

from floquet_toolkit import DiracModel, DiracParameters, FloquetLocalManager
from floquet_toolkit.config import FloquetParameters, MEV_TO_J, UnitConvention
from floquet_toolkit.utils import build_circular_drive, fermi_momentum, parallel_map
from floquet_toolkit.utils.parallel import resolve_worker_count

SI_UNITS = UnitConvention.SI_UNITS()
DIRAC_PARAMS = DiracParameters(units=SI_UNITS, vf=1.0e6, mass=-1.0 * MEV_TO_J, e_fermi=25.0 * MEV_TO_J)
FLOQUET_PARAMS = FloquetParameters(n_trunc=11, n_harmonics=3, n_time=61, dk=1.0e4)
OMEGA = 45.0 * MEV_TO_J / SI_UNITS.hbar
AMPLITUDE = 1.5e-9
N_K_AXIS = 25  # points per axis before masking to the Fermi disk


def _build_manager(amplitude: float) -> FloquetLocalManager:
    drive = build_circular_drive(amplitude, units=SI_UNITS, omega=OMEGA, handedness="right")
    return FloquetLocalManager(DiracModel(DIRAC_PARAMS, drive).to_driven_hamiltonian(), FLOQUET_PARAMS)


def _time_grid() -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi / OMEGA, FLOQUET_PARAMS.n_time, endpoint=False)


def _peak_map_batched(manager: FloquetLocalManager, time, kxs: np.ndarray, kys: np.ndarray) -> np.ndarray:
    """max_t |j(k,t)| over a k-grid via the vectorized velocity map."""
    jx_map, jy_map = manager.floquet_velocity_calculator.compute_floquet_velocity_map(
        time, kxs, kys, bands=("conduction",), include_charge=True
    )["conduction"]
    return np.max(np.hypot(jx_map, jy_map), axis=-1)


def _peak_map_per_k_loop(manager: FloquetLocalManager, time, kxs: np.ndarray, kys: np.ndarray) -> np.ndarray:
    """Reference only: the old per-k Python loop (un-batched)."""
    out = np.empty(kxs.shape[0], dtype=float)
    for i in range(kxs.shape[0]):
        jx_t, jy_t = manager.compute_floquet_velocity(
            time, float(kxs[i]), float(kys[i]), band="conduction", include_charge=True
        )
        out[i] = float(np.max(np.hypot(jx_t, jy_t)))
    return out


def velocity_map_serial(amplitude: float, kxs: np.ndarray, kys: np.ndarray) -> np.ndarray:
    """Correct serial baseline: one manager, one batched velocity-map call."""
    return _peak_map_batched(_build_manager(amplitude), _time_grid(), kxs, kys)


def _batched_map_chunk(args):
    """Module-level worker (picklable): rebuild manager once, run the batched map on a chunk."""
    amplitude, kx_chunk, ky_chunk = args
    return _peak_map_batched(_build_manager(amplitude), _time_grid(), np.asarray(kx_chunk), np.asarray(ky_chunk))


def velocity_map_parallel(amplitude: float, kxs: np.ndarray, kys: np.ndarray, n_jobs: int) -> np.ndarray:
    """k-point parallelism over the BATCHED map: split the grid into per-worker chunks."""
    n_workers = resolve_worker_count(n_jobs, kxs.shape[0])
    index_chunks = np.array_split(np.arange(kxs.shape[0]), n_workers)
    items = [(amplitude, kxs[idx], kys[idx]) for idx in index_chunks]
    results = parallel_map(_batched_map_chunk, items, n_jobs=n_jobs)
    return np.concatenate(results)


def main() -> None:
    n_axis = int(sys.argv[1]) if len(sys.argv) > 1 else N_K_AXIS
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else -1

    k_fermi = fermi_momentum(DIRAC_PARAMS)
    axis = np.linspace(-k_fermi, k_fermi, n_axis)
    kx_grid, ky_grid = np.meshgrid(axis, axis, indexing="ij")
    mask = kx_grid**2 + ky_grid**2 <= k_fermi**2
    kxs, kys = kx_grid[mask], ky_grid[mask]
    print(f"velocity map: {kxs.size} k-points (n_axis={n_axis}), n_jobs={n_jobs}")

    manager = _build_manager(AMPLITUDE)
    time_grid = _time_grid()

    # Warm up import/setup.
    _peak_map_batched(manager, time_grid, kxs[:2], kys[:2])

    # Reference: un-batched per-k loop (shows what batching alone buys).
    t0 = time.perf_counter()
    loop_map = _peak_map_per_k_loop(manager, time_grid, kxs, kys)
    t_loop = time.perf_counter() - t0

    # Correct serial baseline: the batched velocity map.
    t0 = time.perf_counter()
    serial = velocity_map_serial(AMPLITUDE, kxs, kys)
    t_serial = time.perf_counter() - t0

    print(f"  per-k loop (reference)   : {t_loop * 1e3:8.0f} ms")
    print(f"  batched map (serial)     : {t_serial * 1e3:8.0f} ms   ({t_loop / t_serial:5.2f}x vs loop)")

    # Parallelize the batched map across k-chunks.
    t0 = time.perf_counter()
    try:
        parallel = velocity_map_parallel(AMPLITUDE, kxs, kys, n_jobs)
        t_parallel = time.perf_counter() - t0
        max_diff = float(np.max(np.abs(serial - parallel)))
        print(f"  batched map (parallel)   : {t_parallel * 1e3:8.0f} ms   ({t_serial / t_parallel:5.2f}x vs batched serial)")
        print(f"  max abs map diff (serial vs parallel): {max_diff:.2e}")
    except (PermissionError, OSError) as exc:
        print(f"  batched map (parallel)   : UNAVAILABLE here ({type(exc).__name__}: {exc})")
        print("  -> process creation is blocked in this environment; run on a real")
        print("     machine with n_jobs=-1 to measure the parallel speedup.")

    # Sanity: batched serial must match the per-k loop.
    print(f"  max abs map diff (loop vs batched serial): {float(np.max(np.abs(loop_map - serial))):.2e}")


if __name__ == "__main__":
    main()
