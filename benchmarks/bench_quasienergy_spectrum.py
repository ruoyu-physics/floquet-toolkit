"""Benchmark: Floquet quasi-energy spectrum at a few (kx, ky), scanned over
drive polarization (right circular, left circular, linear).

Companion to ``bench_curvature_vs_hfe.py`` and ``bench_velocity_vs_hfe.py``:
same massless Dirac model, same three drives, same high-frequency regime. Here
we simply record the Floquet quasi-energy spectrum at a handful of momenta,
obtained directly from the state provider:

    quasi_energy, _ = state_provider.diagonalize_floquet_hamiltonian(kx, ky)

``diagonalize_floquet_hamiltonian`` returns the full extended-space spectrum
(``n_blocks * dimension`` eigenvalues, replicated across Floquet zones every
``hbar*omega``). We record that entire spectrum at each momentum -- not just the
principal-zone representatives -- so the whole Floquet-zone ladder is captured.
The undriven static band energies ``+/- hbar*vf*|k|`` are printed alongside as a
reference (the central Floquet replica sits near them).

Physics recorded by the scan:

- At ``k = 0`` the static bands are degenerate (gapless Dirac point), but a
  circular drive opens a dynamical gap ``2*|m_eff|`` with
  ``m_eff = vf**2 * e_charge**2 * (AR**2 - AL**2) / (2*hbar*omega)``. Right and
  left circular open the *same* gap (it depends on ``m_eff**2``); linear leaves
  the point gapless.
- Away from ``k = 0`` the quasi-energy tracks the static band energy
  ``hbar*vf*|k|`` (the gap correction is negligible on the drive momentum scale).

k points are placed on the gap momentum scale ``k_gap = |m_eff|/(hbar*vf)`` so
the dynamical gap and its crossover to the linear-in-k regime are both visible.

Self-contained: builds the model/manager directly from ``floquet_toolkit`` and
does not depend on anything under ``scripts/``.

Usage (from the repository root)::

    python3 -m benchmarks.bench_quasienergy_spectrum
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
from datetime import datetime, timezone

# Make the repository root importable regardless of how this file is launched.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402  (imported after sys.path / thread setup)

from floquet_toolkit import (  # noqa: E402
    DiracModel,
    DiracParameters,
    DriveParameters,
    FloquetLocalManager,
    FloquetParameters,
    UnitConvention,
)
from floquet_toolkit.config import MEV_TO_J  # noqa: E402
from floquet_toolkit.utils import build_circular_drive  # noqa: E402

SI_UNITS = UnitConvention.SI_UNITS()

# Gapless Dirac cone: any quasi-energy gap comes entirely from the drive.
VF = 1.0e6
MASS = 0.0
DIRAC_PARAMS = DiracParameters(units=SI_UNITS, vf=VF, mass=MASS, e_fermi=50.0 * MEV_TO_J)

# High frequency: hbar*omega is chosen far above the drive coupling scale
# vf*|e|*A, i.e. deep in the high-frequency-expansion regime.
AMPLITUDE = 3.0e-9
OMEGA = 400.0 * MEV_TO_J / SI_UNITS.hbar

# The three polarizations scanned, all at the same amplitude and frequency
# (see bench_curvature_vs_hfe.py for the identical drive set).
DRIVES = [
    ("right circular", build_circular_drive(AMPLITUDE, units=SI_UNITS, omega=OMEGA, handedness="right")),
    ("left circular", build_circular_drive(AMPLITUDE, units=SI_UNITS, omega=OMEGA, handedness="left")),
    ("linear", DriveParameters(units=SI_UNITS, omega=OMEGA, AL=AMPLITUDE, AR=AMPLITUDE)),
]

# A single-tone drive (any polarization) has Fourier harmonics m = 0, +/-1 only,
# so n_harmonics=1 is exact. n_trunc sets the number of Floquet blocks kept, so
# the extended spectrum has (2*n_trunc+1)*dimension eigenvalues per momentum.
FLOQUET_PARAMS = FloquetParameters(n_trunc=6, n_harmonics=1, n_time=61)

# Quasienergies per row when printing the full extended spectrum.
QUASIENERGIES_PER_LINE = 6

# Written on every run (git-ignored, like benchmarks/results/*.prof) so results
# can be diffed across code changes without re-running the script.
SAVE_RESULTS = True
RESULTS_DIR = os.path.join(_REPO_ROOT, "benchmarks", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "quasienergy_spectrum.txt")

# k points on the gap momentum scale k_gap = |m_eff|/(hbar*vf) of the circular
# drive, spanning k=0 (dynamical gap) out to a few k_gap (linear-in-k regime),
# and including an off-axis point. Linear (m_eff = 0) uses the same ruler.
_M_EFF_CIRCULAR = VF**2 * SI_UNITS.e_charge**2 * AMPLITUDE**2 / (2.0 * SI_UNITS.hbar * OMEGA)
K_GAP = abs(_M_EFF_CIRCULAR) / (SI_UNITS.hbar * VF)
K_POINTS = [
    (0.0, 0.0),
    (1.0 * K_GAP, 0.0),
    (0.0, 1.0 * K_GAP),
    (2.0 * K_GAP, 0.0),
    (2.0 * K_GAP, 2.0 * K_GAP),
]


def _format_spectrum(values_mev) -> list:
    """Wrap a sorted quasienergy array (in meV) into aligned, indented rows."""
    rows = []
    for start in range(0, len(values_mev), QUASIENERGIES_PER_LINE):
        chunk = values_mev[start : start + QUASIENERGIES_PER_LINE]
        rows.append("    " + " ".join(f"{v:13.5f}" for v in chunk))
    return rows


def run_spectrum(label: str, drive: DriveParameters) -> list:
    """Build and return the report lines for one drive's spectrum section."""
    model = DiracModel(DIRAC_PARAMS, drive).to_driven_hamiltonian()
    manager = FloquetLocalManager(model, FLOQUET_PARAMS)
    provider = manager.state_provider
    al, ar = drive.AL, drive.AR
    m_eff = VF**2 * SI_UNITS.e_charge**2 * (ar**2 - al**2) / (2.0 * SI_UNITS.hbar * OMEGA)
    dimension = 2 * FLOQUET_PARAMS.n_trunc + 1  # blocks
    dimension *= provider.dimension  # * bands = extended-space size

    lines = [
        f"### {label} drive",
        f"AL={al:.3e} AR={ar:.3e}  m_eff={m_eff:+.4e} J ({m_eff / MEV_TO_J:+.5f} meV)  "
        f"predicted dynamical gap 2|m_eff| = {2 * abs(m_eff) / MEV_TO_J:.5f} meV",
        f"full extended-space spectrum: {dimension} quasienergies per k "
        f"(sorted, meV); static band energies shown for reference",
    ]

    for kx, ky in K_POINTS:
        static_energy, _ = provider.diagonalize_static_hamiltonian(kx, ky)
        quasi_energy, _ = provider.diagonalize_floquet_hamiltonian(kx, ky)
        static = np.sort(np.asarray(static_energy, dtype=float)) / MEV_TO_J
        quasi = np.sort(np.asarray(quasi_energy, dtype=float)) / MEV_TO_J

        lines.append("")
        lines.append(
            f"k = ({kx:.3e}, {ky:.3e}) 1/m   "
            f"static bands = [{static[0]:+.5f}, {static[-1]:+.5f}] meV"
        )
        lines.extend(_format_spectrum(quasi))
    return lines


def main() -> None:
    lines = []
    lines.append(f"generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(
        f"mass={MASS:.3e} vf={VF:.3e} omega={OMEGA:.6e} "
        f"n_trunc={FLOQUET_PARAMS.n_trunc} n_harmonics={FLOQUET_PARAMS.n_harmonics}"
    )
    lines.append(f"hbar*omega = {SI_UNITS.hbar * OMEGA / MEV_TO_J:.1f} meV")
    lines.append(f"k_gap = |m_eff| / (hbar*vf) = {K_GAP:.3e} 1/m (k points are multiples of this)")

    for label, drive in DRIVES:
        lines.append("")
        lines.extend(run_spectrum(label, drive))

    report = "\n".join(lines)
    print(report)

    if SAVE_RESULTS:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            f.write(report + "\n")
        print(f"\nWrote results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
