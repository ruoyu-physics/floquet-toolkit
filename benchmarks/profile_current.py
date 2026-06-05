"""Profile a single Floquet current calculation to locate the bottleneck.

Runs one ``integrate_current`` call on a pointwise polar quadrature (the path
the production scripts use) under ``cProfile`` and reports the heaviest
functions, with eigendecomposition and Hamiltonian/drive-field construction
called out explicitly.

Self-contained: builds the model/manager directly from ``floquet_toolkit`` and
does not depend on anything under ``scripts/``.

Usage (from the repository root)::

    python3 -m benchmarks.profile_current               # defaults below
    python3 -m benchmarks.profile_current 51 1.5e-9     # n_k_points, amplitude
    python3 benchmarks/profile_current.py               # also works directly

Set ``DUMP_PROFILE = True`` to also write a binary ``.prof`` (openable with
``snakeviz`` or ``pstats``) into ``benchmarks/results/``.
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

import cProfile
import io
import pstats
import sys

# Make the repository root importable regardless of how this file is launched.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402  (imported after sys.path / thread setup)

from floquet_toolkit import (  # noqa: E402
    DiracModel,
    DiracParameters,
    FloquetTransportManager,
    KQuadrature,
)
from floquet_toolkit.config import FloquetParameters, MEV_TO_J, UnitConvention  # noqa: E402
from floquet_toolkit.utils import build_circular_drive, fermi_momentum  # noqa: E402

# Mirror the production sweep configuration so the profile is representative.
SI_UNITS = UnitConvention.SI_UNITS()
DIRAC_PARAMS = DiracParameters(units=SI_UNITS, vf=1.0e6, mass=-1.0 * MEV_TO_J, e_fermi=25.0 * MEV_TO_J)
FLOQUET_PARAMS = FloquetParameters(n_trunc=11, n_harmonics=3, n_time=61, dk=1.0e4)
OMEGA = 45.0 * MEV_TO_J / SI_UNITS.hbar

N_K_POINTS = 41
AMPLITUDE = 1.5e-9
TOP_N = 15
DUMP_PROFILE = False
RESULTS_DIR = os.path.join(_REPO_ROOT, "benchmarks", "results")


def current_peak(amplitude: float, n_k_points: int) -> float:
    """Return ``max_t |j(t)|`` for one circular-drive amplitude (cold: fresh manager)."""
    drive = build_circular_drive(amplitude, units=SI_UNITS, omega=OMEGA, handedness="right")
    manager = FloquetTransportManager(DiracModel(DIRAC_PARAMS, drive).to_driven_hamiltonian(), FLOQUET_PARAMS)
    quadrature = KQuadrature.polar(fermi_momentum(DIRAC_PARAMS), n_k_points=n_k_points)
    _, integrated_jx, integrated_jy = manager.integrate_current(
        quadrature,
        kind="floquet",
        include_charge=True,
        state_selection_algorithm="pointwise",
        band_selection_mode="overlap",
    )
    return float(np.max(np.hypot(integrated_jx, integrated_jy)))


def _self_time(stats: pstats.Stats, substrings) -> float:
    """Sum the self-time (tottime) of profiled functions matching any name."""
    total = 0.0
    for key, value in stats.stats.items():
        filename, _lineno, funcname = key
        tottime = value[2]  # value = (call_count, n_calls, tottime, cumtime, callers)
        label = f"{filename}:{funcname}".lower()
        if any(sub in label for sub in substrings):
            total += tottime
    return total


def main() -> None:
    n_k_points = int(sys.argv[1]) if len(sys.argv) > 1 else N_K_POINTS
    amplitude = float(sys.argv[2]) if len(sys.argv) > 2 else AMPLITUDE

    # Warm up once so first-call import/setup costs do not pollute the numbers.
    # A fresh manager is built per call, so the profiled call is cold regardless.
    current_peak(amplitude * 0.5 + 1.0e-10, n_k_points)

    profiler = cProfile.Profile()
    profiler.enable()
    current_peak(amplitude, n_k_points)  # cold (fresh manager)
    profiler.disable()

    stats = pstats.Stats(profiler)
    total = stats.total_tt
    print(f"n_k_points = {n_k_points}   amplitude = {amplitude:.2e}")
    print(f"TOTAL wall time for one current calc: {total:.3f} s\n")

    print("=" * 78)
    print(f"Top {TOP_N} by SELF time (tottime) — where CPU cycles actually go:")
    print("=" * 78)
    buf = io.StringIO()
    pstats.Stats(profiler, stream=buf).sort_stats("tottime").print_stats(TOP_N)
    print(buf.getvalue())

    eigh_t = _self_time(stats, ["eigh", "eigvalsh", "_syevd"])
    build_t = _self_time(stats, ["dirac.py", "graphene", "drive_fields", "floquet_builder", "norm"])
    print("=" * 78)
    print("Category self-time breakdown (approximate, by name match):")
    print("=" * 78)
    print(f"  eigendecomposition (eigh)       : {eigh_t:.3f} s  ({100 * eigh_t / total:.1f}%)")
    print(f"  Hamiltonian / drive-field build : {build_t:.3f} s  ({100 * build_t / total:.1f}%)")

    if DUMP_PROFILE:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, f"current_nk{n_k_points}.prof")
        stats.dump_stats(out_path)
        print(f"\nBinary profile written to {out_path}")


if __name__ == "__main__":
    main()
