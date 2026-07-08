"""Benchmark: Floquet group velocity vs. two independent order-1 high-frequency
approximations, for a driven massless Dirac model, scanned over drive
polarization (right circular, left circular, linear).

Companion to ``bench_curvature_vs_hfe.py``: same model, drives, and
high-frequency regime, but comparing the band group velocity
``v = (1/hbar) dE/dk`` instead of the Berry curvature. At order 1 the van Vleck
effective Hamiltonian is the static Dirac form with a photo-induced mass

    m_eff = mass + vf**2 * e_charge**2 * (AR**2 - AL**2) / (2 * hbar * omega),

so the conduction-band group velocity has the closed form

    v(k) = hbar * vf**2 * k / sqrt((hbar * vf)**2 * |k|**2 + m_eff**2).

Two features distinguish this from the curvature benchmark:

- Velocity depends on ``m_eff`` only through the band energy (``m_eff**2``), so
  -- unlike the Berry curvature, which flips sign with drive handedness -- the
  group velocity is *identical* for right- and left-circular polarization.
- For linear polarization ``m_eff == 0`` and the cone stays gapless, so
  ``|v| = vf`` exactly (the bare Dirac-cone speed) at every ``k != 0``, whereas
  a circular drive suppresses ``|v|`` below ``vf`` on the gap momentum scale
  ``k_gap = |m_eff| / (hbar * vf)``.

The three quantities compared at each k point are the direct analogues of the
curvature benchmark:

- "HFE analytic": the closed-form ``v(k)`` above with ``m_eff`` -- no matrices.
- "HFE numerical": ``compute_hfe_velocity(order=1)``, the velocity expectation
  in the order-1 effective-Hamiltonian eigenstate.
- "Floquet exact": the time-averaged velocity expectation in the exact Floquet
  mode (``compute_floquet_velocity`` averaged over one period).

k points are placed on the gap momentum scale ``k_gap`` (not the drive scale
used for curvature) because that is where the velocity suppression lives; at
the drive scale the velocity is indistinguishable from ``vf`` for every
polarization.

Self-contained: builds the model/manager directly from ``floquet_toolkit`` and
does not depend on anything under ``scripts/``.

Usage (from the repository root)::

    python3 -m benchmarks.bench_velocity_vs_hfe
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

# Gapless Dirac cone: any velocity suppression comes entirely from the drive.
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
# so n_harmonics=1 is exact. dk sets the finite-difference step for the velocity
# operator of the effective/static Hamiltonians (the exact Floquet path uses the
# model's analytic dH/dk, so it is dk-independent).
FLOQUET_PARAMS = FloquetParameters(n_trunc=6, n_harmonics=1, n_time=61, dk=1.0e2)
BAND = "conduction"
HFE_ORDER = 1

# Written on every run (git-ignored, like benchmarks/results/*.prof) so results
# can be diffed across code changes without re-running the script.
SAVE_RESULTS = True
RESULTS_DIR = os.path.join(_REPO_ROOT, "benchmarks", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "velocity_vs_hfe.txt")

# The velocity suppression lives on the gap momentum scale k_gap = |m_eff|/(hbar*vf),
# so k points are placed as multiples of the circular-drive k_gap. (Linear has
# m_eff = 0; it uses the same ruler purely for a common set of sample points.)
_M_EFF_CIRCULAR = VF**2 * SI_UNITS.e_charge**2 * AMPLITUDE**2 / (2.0 * SI_UNITS.hbar * OMEGA)
K_GAP = abs(_M_EFF_CIRCULAR) / (SI_UNITS.hbar * VF)
K_POINTS = [
    (0.0, 0.0),
    (0.5 * K_GAP, 0.0),
    (1.0 * K_GAP, 0.0),
    (2.0 * K_GAP, 0.0),
    (1.0 * K_GAP, 1.0 * K_GAP),
]

# Velocity magnitudes below this (relative to vf) are treated as ~0, so the
# k=0 band extremum -- where v vanishes and a relative error is undefined --
# reports "n/a" rather than a spurious 0/0.
VELOCITY_FLOOR = 1.0e-6 * VF


def photo_induced_mass(al: float, ar: float) -> float:
    """Return the order-1 van Vleck photo-induced mass ``m_eff`` (see docstring)."""
    return MASS + VF**2 * SI_UNITS.e_charge**2 * (ar**2 - al**2) / (2.0 * SI_UNITS.hbar * OMEGA)


def analytic_hfe_dirac_velocity(kx, ky, al, ar, band="conduction"):
    """Closed-form order-1 HFE group velocity ``(vx, vy)`` for the Dirac model.

    No matrix construction: the order-1 effective Hamiltonian is static Dirac
    with ``mass -> m_eff``, whose conduction-band velocity is
    ``v = hbar * vf**2 * k / E`` with ``E = sqrt((hbar*vf)**2 |k|**2 + m_eff**2)``.
    The valence band carries the opposite sign. At ``k = 0`` the velocity
    vanishes (band extremum), which the formula gives directly.
    """
    hbar = SI_UNITS.hbar
    mass_eff = photo_induced_mass(al, ar)
    energy = np.sqrt((hbar * VF) ** 2 * (kx**2 + ky**2) + mass_eff**2)
    if energy == 0.0:  # k = 0 and m_eff = 0 simultaneously (gapless Dirac point)
        return 0.0, 0.0
    scale = hbar * VF**2 / energy
    if band in ("conduction", 1):
        return scale * kx, scale * ky
    if band in ("valence", 0):
        return -scale * kx, -scale * ky
    raise ValueError("band must be 'conduction', 'valence', 0, or 1")


def _high_frequency_ratio(al: float, ar: float) -> float:
    """Return ``hbar*omega / drive_coupling``, the high-frequency diagnostic.

    The coupling uses the larger of the two circular components, so a circular
    drive and the equal-component linear drive share the same ratio.
    """
    drive_coupling = VF * abs(SI_UNITS.e_charge) * max(al, ar)
    return SI_UNITS.hbar * OMEGA / drive_coupling


def _format_error(rel_error) -> str:
    """Right-aligned relative-error cell, or ``n/a`` when it is undefined."""
    if rel_error is None:
        return f"{'n/a':>13}"
    return f"{rel_error:13.4%}"


def _relative_vector_error(vx, vy, ref_x, ref_y):
    """Relative error of a velocity vector vs. reference, or ``None`` if ref ~0.

    Uses the full vector difference (so a direction mismatch is caught) over the
    reference magnitude; returns ``None`` when the reference speed is below
    ``VELOCITY_FLOOR`` (e.g. the k=0 band extremum), where the ratio is ill-posed.
    """
    ref_speed = float(np.hypot(ref_x, ref_y))
    if ref_speed < VELOCITY_FLOOR:
        return None
    return float(np.hypot(vx - ref_x, vy - ref_y)) / ref_speed


def run_comparison(label: str, drive: DriveParameters) -> list:
    """Build and return the report lines for one drive's comparison section."""
    model = DiracModel(DIRAC_PARAMS, drive).to_driven_hamiltonian()
    manager = FloquetLocalManager(model, FLOQUET_PARAMS)
    time_grid = FLOQUET_PARAMS.time_grid(drive.period)
    al, ar = drive.AL, drive.AR
    mass_eff = photo_induced_mass(al, ar)

    lines = [
        f"### {label} drive",
        f"AL={al:.3e} AR={ar:.3e}  m_eff={mass_eff:+.4e} J "
        f"({mass_eff / MEV_TO_J:+.4f} meV)  "
        f"hbar*omega / drive coupling = {_high_frequency_ratio(al, ar):.1f}",
    ]
    header = (
        f"{'kx [1/m]':>12} {'ky [1/m]':>12} {'|v| analytic':>14} "
        f"{'|v| numerical':>14} {'|v| Floquet':>14} "
        f"{'err analytic':>13} {'err numerical':>13}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    errors_analytic = []
    errors_numerical = []
    for kx, ky in K_POINTS:
        ana_vx, ana_vy = analytic_hfe_dirac_velocity(kx, ky, al, ar, band=BAND)
        num_vx, num_vy = manager.compute_hfe_velocity(
            kx, ky, band=BAND, order=HFE_ORDER, dk=FLOQUET_PARAMS.dk
        )
        flo_vx_t, flo_vy_t = manager.compute_floquet_velocity(
            time=time_grid, kx=kx, ky=ky, band=BAND, dk=FLOQUET_PARAMS.dk
        )
        flo_vx = float(np.mean(flo_vx_t))
        flo_vy = float(np.mean(flo_vy_t))

        if (kx, ky) == (0.0, 0.0):
            # Band extremum for gapped drives (v = 0) and a gapless Dirac point
            # for linear (v singular / direction undefined): the velocity
            # comparison is ill-posed here, so it is reported but not scored.
            err_analytic = err_numerical = None
        else:
            err_analytic = _relative_vector_error(ana_vx, ana_vy, flo_vx, flo_vy)
            err_numerical = _relative_vector_error(num_vx, num_vy, flo_vx, flo_vy)
        if err_analytic is not None:
            errors_analytic.append(err_analytic)
        if err_numerical is not None:
            errors_numerical.append(err_numerical)

        lines.append(
            f"{kx:12.3e} {ky:12.3e} {np.hypot(ana_vx, ana_vy):14.4e} "
            f"{np.hypot(num_vx, num_vy):14.4e} {np.hypot(flo_vx, flo_vy):14.4e} "
            f"{_format_error(err_analytic)} {_format_error(err_numerical)}"
        )

    lines.append(
        f"max relative error vs. exact Floquet (k != 0): "
        f"analytic HFE {max(errors_analytic):.4%}, "
        f"numerical HFE {max(errors_numerical):.4%}"
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
    lines.append(f"k_gap = |m_eff| / (hbar*vf) = {K_GAP:.3e} 1/m (k points are multiples of this)")
    lines.append("(hbar*omega / drive coupling >> 1: deep in the high-frequency regime)")
    lines.append("(k=0 is a band extremum / gapless point: velocity ill-posed, excluded from error stats)")

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
