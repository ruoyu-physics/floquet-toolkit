"""Benchmark: Floquet Berry curvature vs. two independent order-1 high-frequency
approximations, for a driven massless Dirac model, scanned over drive
polarization (right circular, left circular, linear).

At leading order (``1/omega``) the van Vleck effective Hamiltonian for a
circularly/elliptically driven Dirac cone is just the static Dirac
Hamiltonian with a renormalized, momentum-independent mass:

    m_eff = mass + vf**2 * e_charge**2 * (AR**2 - AL**2) / (2 * hbar * omega)

(this is the standard photo-induced-gap result of Oka & Aoki, and Kitagawa
et al.). The polarization scan exercises the three qualitatively distinct
regimes of this formula at fixed amplitude and frequency:

- right circular (``AR != 0``, ``AL == 0``): ``m_eff > 0``, gap opens.
- left circular (``AL != 0``, ``AR == 0``): ``m_eff < 0``, same gap magnitude
  but the Berry curvature flips sign with drive handedness.
- linear (``AL == AR``): ``m_eff == 0`` exactly, so no gap opens and the
  order-1 Berry curvature vanishes -- a null check on the whole pipeline.

Because the resulting effective Hamiltonian has exactly the static Dirac form,
its Berry curvature is the *same* closed-form expression as
``DiracModel.analytic_static_berry_curvature``, just with ``mass -> m_eff``.

This script compares three independently computed things at several k points:

- "HFE analytic": the closed-form ``m_eff`` formula above, evaluated directly
  -- no matrix construction at all.
- "HFE numerical": ``HFEBuilder``'s order-1 nested-commutator construction
  (``compute_hfe_berry_curvature(order=1)``), which builds the same effective
  Hamiltonian generically from numerically sampled Fourier harmonics. The two
  should agree to numerical precision -- this is a correctness check of the
  hand-derived formula against the toolkit's general-purpose HFE builder.
- "Floquet exact": the numerically-exact, time-averaged Berry curvature of the
  full (untruncated-in-order) Floquet eigenstates, as an independent,
  non-perturbative reference.

Mass is set to zero, so the Dirac cone is gapless in the undriven system --
any Berry curvature is generated dynamically by the drive through
high-frequency photon dressing (and, for linear polarization, none is).

Note on ``dk``: the drive-induced gap is tiny in this deep-high-frequency
regime (``m_eff`` corresponds to a curvature feature on a momentum scale of
only ~1e3-1e4 1/m around k=0), so the Fukui-plaquette finite-difference step
must be much smaller than in a typical (finite-mass) run or it under-resolves
the curvature right at the Dirac point.

Self-contained: builds the model/manager directly from ``floquet_toolkit`` and
does not depend on anything under ``scripts/``.

Usage (from the repository root)::

    python3 -m benchmarks.bench_curvature_vs_hfe
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

# Gapless Dirac cone: any Berry curvature comes entirely from the drive.
VF = 1.0e6
MASS = 0.0
DIRAC_PARAMS = DiracParameters(units=SI_UNITS, vf=VF, mass=MASS, e_fermi=50.0 * MEV_TO_J)

# High frequency: hbar*omega is chosen far above the drive coupling scale
# vf*|e|*A, i.e. deep in the high-frequency-expansion regime.
AMPLITUDE = 2.0e-9
OMEGA = 400.0 * MEV_TO_J / SI_UNITS.hbar

# The three polarizations scanned, all at the same amplitude and frequency.
# Circular drives put the full amplitude in one handedness; the linear drive
# splits it equally (AL == AR), which sets m_eff == 0 exactly (no gap). Using
# AMPLITUDE for each linear component keeps the per-handedness drive coupling --
# and hence the high-frequency ratio -- identical across all three rows.
DRIVES = [
    ("right circular", build_circular_drive(AMPLITUDE, units=SI_UNITS, omega=OMEGA, handedness="right")),
    ("left circular", build_circular_drive(AMPLITUDE, units=SI_UNITS, omega=OMEGA, handedness="left")),
    ("linear", DriveParameters(units=SI_UNITS, omega=OMEGA, AL=AMPLITUDE, AR=AMPLITUDE)),
]

# A single-tone circular drive only has Fourier harmonics m = 0, +/-1, so
# n_harmonics=1 is exact (no truncation error from higher harmonics). dk is
# small compared to the drive-induced gap's momentum scale -- see module
# docstring -- so the Fukui plaquette resolves the curvature at k=0.
FLOQUET_PARAMS = FloquetParameters(n_trunc=6, n_harmonics=1, n_time=61, dk=1.0e2)
BAND = "conduction"
HFE_ORDER = 1

# Written on every run (git-ignored, like benchmarks/results/*.prof) so results
# can be diffed across code changes without re-running the script.
SAVE_RESULTS = True
RESULTS_DIR = os.path.join(_REPO_ROOT, "benchmarks", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "curvature_vs_hfe.txt")

# With mass=0 there is no mass-based momentum scale, so k points are placed
# relative to the drive-induced momentum scale q*A/hbar instead.
K0 = abs(SI_UNITS.e_charge) * AMPLITUDE / SI_UNITS.hbar
K_POINTS = [
    (0.0, 0.0),
    (0.5 * K0, 0.0),
    (1.0 * K0, 0.0),
    (2.0 * K0, 0.0),
    (1.0 * K0, 1.0 * K0),
]


def photo_induced_mass(al: float, ar: float) -> float:
    """Return the order-1 van Vleck photo-induced mass ``m_eff`` (see docstring)."""
    return MASS + VF**2 * SI_UNITS.e_charge**2 * (ar**2 - al**2) / (2.0 * SI_UNITS.hbar * OMEGA)


def analytic_hfe_dirac_curvature(
    kx: float, ky: float, al: float, ar: float, band: str = "conduction"
) -> float:
    """Closed-form order-1 HFE Berry curvature for the driven Dirac model.

    No matrix construction: the order-1 van Vleck correction for this model is
    a pure mass renormalization (see module docstring), so the curvature is
    ``DiracModel.analytic_static_berry_curvature`` with ``mass -> m_eff``. For
    linear polarization ``m_eff == 0`` and the curvature vanishes identically
    (the ``mass=0`` Dirac cone carries no smooth Berry curvature), which also
    avoids the ``0 / 0`` at ``k = 0`` in the finite-mass formula below.
    """
    hbar = SI_UNITS.hbar
    mass_eff = photo_induced_mass(al, ar)
    if mass_eff == 0.0:
        return 0.0
    prefactor = 0.5 * mass_eff * (hbar * VF) ** 2
    denominator = ((hbar * VF) ** 2 * (kx**2 + ky**2) + mass_eff**2) ** 1.5
    if band in ("conduction", 1):
        return -prefactor / denominator
    if band in ("valence", 0):
        return prefactor / denominator
    raise ValueError("band must be 'conduction', 'valence', 0, or 1")


def _high_frequency_ratio(al: float, ar: float) -> float:
    """Return ``hbar*omega / drive_coupling``, the high-frequency diagnostic.

    The coupling uses the larger of the two circular components, so a circular
    drive and the equal-component linear drive share the same ratio.
    """
    drive_coupling = VF * abs(SI_UNITS.e_charge) * max(al, ar)
    return SI_UNITS.hbar * OMEGA / drive_coupling


def _format_error(value, reference) -> str:
    """Right-aligned relative-error cell, or ``n/a`` when the reference is ~0.

    For linear polarization the exact Floquet curvature vanishes, so a relative
    error is undefined; the section summary reports absolute magnitudes instead.
    """
    if reference == 0.0:
        return f"{'n/a':>13}"
    return f"{abs(value - reference) / abs(reference):13.4%}"


def run_comparison(label: str, drive: DriveParameters) -> list:
    """Build and return the report lines for one drive's comparison section."""
    model = DiracModel(DIRAC_PARAMS, drive).to_driven_hamiltonian()
    manager = FloquetLocalManager(model, FLOQUET_PARAMS)
    time_grid = FLOQUET_PARAMS.time_grid(drive.period)
    al, ar = drive.AL, drive.AR
    mass_eff = photo_induced_mass(al, ar)
    gapped = mass_eff != 0.0

    lines = [
        f"### {label} drive",
        f"AL={al:.3e} AR={ar:.3e}  m_eff={mass_eff:+.4e} J "
        f"({mass_eff / MEV_TO_J:+.4f} meV)  "
        f"hbar*omega / drive coupling = {_high_frequency_ratio(al, ar):.1f}",
    ]
    header = (
        f"{'kx [1/m]':>12} {'ky [1/m]':>12} {'HFE analytic':>14} "
        f"{'HFE numerical':>14} {'Floquet exact':>14} "
        f"{'err analytic':>13} {'err numerical':>13}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    errors_analytic = []
    errors_numerical = []
    offpoint_mags = []  # |curvature| at k != 0, where smooth curvature must vanish
    for kx, ky in K_POINTS:
        analytic_curv = analytic_hfe_dirac_curvature(kx, ky, al, ar, band=BAND)
        numerical_curv = manager.compute_hfe_berry_curvature(
            kx, ky, band=BAND, dk=FLOQUET_PARAMS.dk, order=HFE_ORDER
        )
        floquet_curv_t = manager.compute_instantaneous_berry_curvature(
            time=time_grid, kx=kx, ky=ky, band=BAND, dk=FLOQUET_PARAMS.dk
        )
        floquet_avg = float(np.mean(floquet_curv_t))

        if gapped:
            errors_analytic.append(abs(analytic_curv - floquet_avg) / abs(floquet_avg))
            errors_numerical.append(abs(numerical_curv - floquet_avg) / abs(floquet_avg))
        elif (kx, ky) != (0.0, 0.0):
            offpoint_mags.append(max(abs(floquet_avg), abs(numerical_curv)))

        reference = floquet_avg if gapped else 0.0
        lines.append(
            f"{kx:12.3e} {ky:12.3e} {analytic_curv:14.4e} "
            f"{numerical_curv:14.4e} {floquet_avg:14.4e} "
            f"{_format_error(analytic_curv, reference)} "
            f"{_format_error(numerical_curv, reference)}"
        )

    if gapped:
        lines.append(
            f"max relative error vs. exact Floquet: "
            f"analytic HFE {max(errors_analytic):.4%}, "
            f"numerical HFE {max(errors_numerical):.4%}"
        )
    else:
        # Linear drive leaves the cone gapless: k=0 is a band-touching point
        # whose finite-plaquette Berry flux is the sign-ambiguous +/- pi
        # artifact (~ pi/dk**2), while every k != 0 carries no smooth curvature.
        lines.append(
            f"m_eff = 0 (no gap): away from k=0, max |curvature| = "
            f"{max(offpoint_mags):.3e} (HFE and Floquet agree at ~0); "
            f"k=0 is a gapless Dirac point (sign-ambiguous +/-pi/dk**2 flux)."
        )
    return lines


def main() -> None:
    lines = []
    lines.append(f"generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(
        f"mass={MASS:.3e} vf={VF:.3e} omega={OMEGA:.6e} "
        f"n_trunc={FLOQUET_PARAMS.n_trunc} n_harmonics={FLOQUET_PARAMS.n_harmonics} "
        f"dk={FLOQUET_PARAMS.dk:.3e} hfe_order={HFE_ORDER}"
    )
    lines.append(f"hbar*omega = {SI_UNITS.hbar * OMEGA / MEV_TO_J:.1f} meV")
    lines.append("(hbar*omega / drive coupling >> 1: deep in the high-frequency regime)")

    for label, drive in DRIVES:
        lines.append("")
        lines.extend(run_comparison(label, drive))

    report = "\n".join(lines)
    print(report)

    if SAVE_RESULTS:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            f.write(report + "\n")
        print(f"\nWrote results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
