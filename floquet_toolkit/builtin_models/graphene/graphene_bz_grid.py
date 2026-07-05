"""Adaptive band-structure integration grid for the built-in graphene model.

This builds a curvilinear quadrature mesh over the graphene Brillouin zone whose
cells follow the nearest-neighbor conduction band

    E(k) = t * |gamma(k)|,
    |gamma(k)|^2 = 3 + 2 cos(sqrt3 a kx) + 4 cos(3 a ky / 2) cos(sqrt3 a kx / 2),

i.e. the same dispersion as :class:`~floquet_toolkit.builtin_models.graphene.\
graphene.GrapheneModel` (``t`` = ``hopping``, ``a`` = ``lattice_spacing``).
Unlike the uniform Cartesian / parallelogram meshes in
:mod:`floquet_toolkit.utils.kquadrature`, the cells here are adapted to the
energy landscape: energy contours (rings) and gradient flowlines (spokes) bound
each cell, with the van Hove separatrix ``E = t`` always a cell boundary. The
result integrates band-resolved quantities far more accurately at fixed cell
count, and is returned ready to use as a :class:`KQuadrature`.

Cell centroids and areas are computed analytically: each cell's boundary
polygon is assembled from the stored spoke polylines plus their band-edge
crossings, and the exact shoelace area/centroid integrals are evaluated --
resolution-independent and tiling the zone to machine precision.

Frame convention
----------------
The geometry construction (flowline tracing, separatrix bisection, sector
tiling) is carried out in an internal *reduced* frame with ``a = 1`` and a fixed
reference hopping ``_T_REDUCED`` -- the frame the algorithm was developed and
numerically tuned in (the RK4 step control and saddle/turning-point thresholds
in :func:`_trace` depend on that scale). The reduced frame is graphene's
dispersion with ``kx`` and ``ky`` interchanged relative to the codebase
convention set by :class:`GrapheneParameters`' neighbor vectors. Every public
output is mapped back to the physical codebase frame by the exact relation

    k_phys = (k_reduced_y, k_reduced_x) / a,

so reduced momenta are swapped and scaled by ``1 / a``, areas by ``1 / a**2``,
and reduced (``t = _T_REDUCED``) energies by ``hopping / _T_REDUCED``. One checks
that the swapped, rescaled reduced reciprocal vectors equal the model's, so the
mapped cells tile the physical Brillouin zone exactly.

Public entry points
-------------------
- :func:`graphene_bz_quadrature` -> :class:`KQuadrature` (the integration object).
- :func:`build_graphene_bz_grid` -> dict with cell centroids, areas, energies,
  patch/sector/band labels, spoke polylines and diagnostics (for plotting /
  reuse), all in the physical codebase frame.
"""

from __future__ import annotations

import math

import numpy as np

from ...utils.geometry import polygon_area_centroid
from ...utils.kquadrature import KQuadrature
from .graphene import GrapheneParameters

# ----------------------------------------------------- reduced-frame model ----
# Internal a = 1 frame with a fixed reference hopping. Geometry is independent of
# the hopping value (it only sets the overall energy scale); _T_REDUCED is the
# scale the tracer thresholds in _trace were tuned against and must not change.
_T_REDUCED = 2.7
_S3 = math.sqrt(3.0)
_B1 = (2 * math.pi / 3) * np.array([1.0, _S3])
_B2 = (2 * math.pi / 3) * np.array([1.0, -_S3])
_BINV = np.linalg.inv(np.column_stack([_B1, _B2]))
_CORNERS = [np.zeros(2), _B1.copy(), _B2.copy(), _B1 + _B2]
_K = (2 / 3) * _B1 + (1 / 3) * _B2
_KP = (1 / 3) * _B1 + (2 / 3) * _B2
_MA = 0.5 * _B1
_MB = _B1 + 0.5 * _B2
_MC = 0.5 * (_B1 + _B2)
_MD = 0.5 * _B2
_ME = _B2 + 0.5 * _B1
_SEGMENTS = [(_MA, _MB), (_MB, _MC), (_MC, _MA),      # K-pocket triangle edges
             (_MD, _ME), (_ME, _MC), (_MC, _MD)]      # K'-pocket triangle edges
_BZ_AREA = abs(_B1[0] * _B2[1] - _B1[1] * _B2[0])     # = 8*sqrt(3)*pi^2/9


def _E_scalar(kx, ky):
    g = 3 + 2 * math.cos(_S3 * ky) + 4 * math.cos(1.5 * kx) * math.cos(0.5 * _S3 * ky)
    return _T_REDUCED * math.sqrt(g if g > 0 else 0.0)


def _E_vec(kx, ky):
    g = 3 + 2 * np.cos(_S3 * ky) + 4 * np.cos(1.5 * kx) * np.cos(0.5 * _S3 * ky)
    return _T_REDUCED * np.sqrt(np.clip(g, 0.0, None))


def _grad_scalar(kx, ky):
    g = 3 + 2 * math.cos(_S3 * ky) + 4 * math.cos(1.5 * kx) * math.cos(0.5 * _S3 * ky)
    if g < 1e-30:
        g = 1e-30
    gx = -6.0 * math.sin(1.5 * kx) * math.cos(0.5 * _S3 * ky)
    gy = -2 * _S3 * math.sin(_S3 * ky) - 2 * _S3 * math.cos(1.5 * kx) * math.sin(0.5 * _S3 * ky)
    c = _T_REDUCED / (2.0 * math.sqrt(g))
    return c * gx, c * gy


def _inside(x, y, tol=0.02):
    al = _BINV[0, 0] * x + _BINV[0, 1] * y
    be = _BINV[1, 0] * x + _BINV[1, 1] * y
    return (-tol <= al <= 1 + tol) and (-tol <= be <= 1 + tol)


# ------------------------------------------------------ flowline tracing ----
def _trace(x, y, sign, ds=0.004, maxsteps=120000, stop_at_seam=False):
    """Arclength RK4 along sign*gradE/|gradE| (adaptive near saddles).

    Ascent (+1) stops on capture at a Gamma corner; descent (-1) near E=0.
    With stop_at_seam=True the trace halts on the E = _T_REDUCED crossing
    instead, bisection-refined onto the level set (used by the truncated
    families).
    """
    pts = [(x, y)]
    side0 = 1.0 if _E_scalar(x, y) > _T_REDUCED else -1.0
    for i in range(maxsteps):
        gx, gy = _grad_scalar(x, y)
        n = math.hypot(gx, gy)
        E = _E_scalar(x, y)
        if sign > 0 and n < 0.02 and E > 2.5 * _T_REDUCED:
            break
        if sign < 0 and E < 0.04:
            break
        h = ds * min(1.0, max(0.01, n / 2.0))

        def d(qx, qy):
            ax, ay = _grad_scalar(qx, qy)
            m = math.hypot(ax, ay)
            return (sign * ax / m, sign * ay / m) if m > 1e-15 else (0.0, 0.0)

        k1 = d(x, y)
        k2 = d(x + 0.5 * h * k1[0], y + 0.5 * h * k1[1])
        k3 = d(x + 0.5 * h * k2[0], y + 0.5 * h * k2[1])
        k4 = d(x + h * k3[0], y + h * k3[1])
        px, py = x, y
        x += h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
        y += h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
        if stop_at_seam and (_E_scalar(x, y) - _T_REDUCED) * side0 < 0.0:
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if (_E_scalar(px + mid * (x - px), py + mid * (y - py)) - _T_REDUCED) * side0 > 0:
                    lo = mid
                else:
                    hi = mid
            x = px + 0.5 * (lo + hi) * (x - px)
            y = py + 0.5 * (lo + hi) * (y - py)
            break
        if i % 2 == 0:
            pts.append((x, y))
        if not _inside(x, y):
            break
    pts.append((x, y))
    return np.array(pts)


# ------------------------------------------- seam calibration (alpha mix) ----
def _seam_labels(l):
    """Seed at fraction l along representative segment MA -> MC.

    Returns (u, v): angular labels at Gamma and at K, each in [0, 1].
    """
    s = _MA + l * (_MC - _MA)
    eG = _trace(s[0], s[1], +1)[-1]
    eK = _trace(s[0], s[1], -1)[-1]
    thG = math.degrees(math.atan2(eG[1], eG[0])) % 360             # about Gamma0
    thK = math.degrees(math.atan2(eK[1] - _K[1], eK[0] - _K[0])) % 360  # about K
    return (thG - 30.0) / 30.0, (210.0 - thK) / 60.0


def _solve_seam_positions(n_per_segment, alpha, tol=0.004, verbose=True):
    """Cell-centered equidistribution of w = (1-alpha) u + alpha v over the
    segment. Returns the sorted list of n fractions l in (0, 1)."""
    if n_per_segment % 2:
        raise ValueError('n_per_segment must be even (the midpoint flowline '
                         'is the K-Gamma skeleton spoke, included separately).')
    scan_l = [0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20,
              0.25, 0.30, 0.35, 0.40, 0.45, 0.499]
    scan = []
    for l in scan_l:
        u, v = _seam_labels(l)
        scan.append((l, (1 - alpha) * u + alpha * v))
    halves = []
    for j in range(n_per_segment // 2):
        wt = (2 * j + 1) / n_per_segment
        lo = hi = None
        for a, b in zip(scan[:-1], scan[1:]):
            if a[1] >= wt >= b[1]:
                lo, hi = a[0], b[0]
                break
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            u, v = _seam_labels(mid)
            w = (1 - alpha) * u + alpha * v
            if abs(w - wt) < tol:
                break
            if w > wt:
                lo = mid
            else:
                hi = mid
        halves.append(mid)
        if verbose:
            print(f'  seam seed: w = {wt:.3f} -> l = {mid:.4f} '
                  f'(theta_G = {30 * u:6.2f} deg, theta_K = {60 * v:6.2f} deg)')
    return sorted(halves + [1.0 - l for l in halves])


# --------------------------------------------------------- spoke assembly ----
def _ang(p, pole):
    return math.degrees(math.atan2(p[1] - pole[1], p[0] - pole[0])) % 360.0


def _ref_point(poly, pole, r=0.18):
    for p in poly:
        if np.linalg.norm(p - pole) > r:
            return p
    return poly[-1]


def _build_spokes(lset, verbose=True):
    """Returns pocket_spokes = {0: [...], 1: [...]} and gamma_spokes =
    {corner_idx: [...]}; each spoke dict: poly (pole -> seam), seampt, angle."""
    pockets = {0: [], 1: []}        # 0 = K, 1 = K'
    gammas = {0: [], 1: [], 2: [], 3: []}
    poles = [_K, _KP]

    def add_pocket(ip, poly, seampt):
        pole = poles[ip]
        pockets[ip].append({'poly': poly, 'seampt': np.asarray(seampt),
                            'angle': _ang(_ref_point(poly, pole), pole)})

    def add_gamma(ic, poly, seampt):
        c = _CORNERS[ic]
        gammas[ic].append({'poly': poly, 'seampt': np.asarray(seampt),
                           'angle': _ang(_ref_point(poly, c), c)})

    # generic seam-seeded spokes
    for P, Q in _SEGMENTS:
        for l in lset:
            s = P + l * (Q - P)
            dn = _trace(s[0], s[1], -1)
            up = _trace(s[0], s[1], +1)
            ip = 0 if np.linalg.norm(dn[-1] - _K) < np.linalg.norm(dn[-1] - _KP) else 1
            ic = int(np.argmin([np.linalg.norm(up[-1] - c) for c in _CORNERS]))
            add_pocket(ip, np.vstack([poles[ip], dn[::-1], s]), s)
            add_gamma(ic, np.vstack([_CORNERS[ic], up[::-1], s]), s)

    # skeleton: K-Gamma (split at the seam crossing = facing-segment midpoint)
    for ip, pole, trips in [(0, _K, [(0, _MA, _MC), (1, _MA, _MB), (3, _MB, _MC)]),
                            (1, _KP, [(0, _MD, _MC), (2, _MD, _ME), (3, _ME, _MC)])]:
        for ic, Pm, Qm in trips:
            mid = 0.5 * (Pm + Qm)
            add_pocket(ip, np.vstack([pole, mid]), mid)
            add_gamma(ic, np.vstack([_CORNERS[ic], mid]), mid)

    # skeleton: K-M (pocket side only; seam point = the M vertex)
    for ip, pole, Ms in [(0, _K, [_MA, _MB, _MC]), (1, _KP, [_MD, _ME, _MC])]:
        for M in Ms:
            add_pocket(ip, np.vstack([pole, M]), M)

    # skeleton: Gamma-M rays per corner (zone-edge halves + interior ky=0)
    for ic, Ms in [(0, [_MA, _MD, _MC]), (1, [_MA, _MB]),
                   (2, [_MD, _ME]),      (3, [_MB, _ME, _MC])]:
        for M in Ms:
            add_gamma(ic, np.vstack([_CORNERS[ic], M]), M)

    for ip in pockets:
        pockets[ip].sort(key=lambda s: s['angle'])
    wedge_center = {0: 0.0, 1: 270.0, 2: 90.0, 3: 180.0}
    for ic in gammas:
        c0 = wedge_center[ic]
        for s in gammas[ic]:
            s['rel'] = (s['angle'] - c0 + 180.0) % 360.0 - 180.0
        gammas[ic].sort(key=lambda s: s['rel'])
    if verbose:
        n_pocket = [len(pockets[i]) for i in (0, 1)]
        n_gamma = [len(gammas[i]) for i in range(4)]
        print(f'  spokes: {n_pocket[0]} around K, {n_pocket[1]} around K\', '
              f'{n_gamma} at the four Gamma corners')
    return pockets, gammas


def _build_spokes_uniform(n_K, n_gamma, r0=0.35, verbose=True):
    """Per-patch uniform spoke families, truncated on the separatrix E = t.

    n_K equally spaced spokes around each Dirac point (offset 30 deg) and
    n_gamma around the assembled Gamma point (offset 0 deg). With
    n_K % 6 == 0 and n_gamma % 12 == 0 the straight symmetry skeleton
    (K-M, K-Gamma, Gamma-M, Gamma-K rays) consists of family members; these
    are inserted analytically, and the generic members are traced with a
    bisection stop on E = t. Returns the same structures as _build_spokes."""
    if n_K % 6:
        raise ValueError('n_K must be a multiple of 6 (family offset 30 deg) '
                         'so the K-M and K-Gamma rays are family members.')
    if n_gamma % 12:
        raise ValueError('n_gamma must be a multiple of 12 (family offset '
                         '0 deg) so the Gamma-M and Gamma-K rays are family '
                         'members.')
    pockets = {0: [], 1: []}
    gammas = {0: [], 1: [], 2: [], 3: []}
    poles = [_K, _KP]

    # skeleton: exact straight members
    kg_trips = {0: [(0, _MA, _MC), (1, _MA, _MB), (3, _MB, _MC)],
                1: [(0, _MD, _MC), (2, _MD, _ME), (3, _ME, _MC)]}
    km = {0: [_MA, _MB, _MC], 1: [_MD, _ME, _MC]}
    gm = {0: [_MA, _MD, _MC], 1: [_MA, _MB], 2: [_MD, _ME], 3: [_MB, _ME, _MC]}
    for ip in (0, 1):
        pole = poles[ip]
        for M in km[ip]:
            pockets[ip].append({'poly': np.vstack([pole, M]),
                                'seampt': np.asarray(M),
                                'angle': _ang(M, pole)})
        for ic, Pm, Qm in kg_trips[ip]:
            mid = 0.5 * (Pm + Qm)
            pockets[ip].append({'poly': np.vstack([pole, mid]),
                                'seampt': mid, 'angle': _ang(mid, pole)})
            gammas[ic].append({'poly': np.vstack([_CORNERS[ic], mid]),
                               'seampt': mid, 'angle': _ang(mid, _CORNERS[ic])})
    for ic in range(4):
        for M in gm[ic]:
            gammas[ic].append({'poly': np.vstack([_CORNERS[ic], M]),
                               'seampt': np.asarray(M),
                               'angle': _ang(M, _CORNERS[ic])})

    # generic members, truncated on the seam
    for ip in (0, 1):
        pole = poles[ip]
        for j in range(n_K):
            th = (30.0 + 360.0 * j / n_K) % 360.0
            if abs((th % 60.0) - 30.0) < 1e-9:
                continue                                         # skeleton
            thr = math.radians(th)
            lx = pole[0] + r0 * math.cos(thr)
            ly = pole[1] + r0 * math.sin(thr)
            up = _trace(lx, ly, +1, stop_at_seam=True)
            dn = _trace(lx, ly, -1)
            pockets[ip].append({'poly': np.vstack([pole, dn[::-1], up]),
                                'seampt': up[-1], 'angle': th})
    for ic in range(4):
        c = _CORNERS[ic]
        for j in range(n_gamma):
            phi = (360.0 * j / n_gamma) % 360.0
            m30 = phi % 30.0
            if m30 < 1e-9 or 30.0 - m30 < 1e-9:
                continue                                         # skeleton
            ph = math.radians(phi)
            lx = c[0] + r0 * math.cos(ph)
            ly = c[1] + r0 * math.sin(ph)
            if not _inside(lx, ly, tol=-1e-9):
                continue                                         # other wedge
            dn = _trace(lx, ly, -1, stop_at_seam=True)
            up = _trace(lx, ly, +1)
            gammas[ic].append({'poly': np.vstack([c, up[::-1], dn]),
                               'seampt': dn[-1], 'angle': phi})

    for ip in pockets:
        pockets[ip].sort(key=lambda s: s['angle'])
    wedge_center = {0: 0.0, 1: 270.0, 2: 90.0, 3: 180.0}
    for ic in gammas:
        c0 = wedge_center[ic]
        for s in gammas[ic]:
            s['rel'] = (s['angle'] - c0 + 180.0) % 360.0 - 180.0
        gammas[ic].sort(key=lambda s: s['rel'])
    if verbose:
        n_pocket = [len(pockets[i]) for i in (0, 1)]
        n_gam = [len(gammas[i]) for i in range(4)]
        print(f'  spokes (per-patch uniform, truncated): {n_pocket[0]} around K, '
              f'{n_pocket[1]} around K\', {n_gam} at the four Gamma corners')
    return pockets, gammas


# ------------------------------------------------- shoelace cell assembly ----
def _spoke_profile(poly, min_points=64):
    """Return ``(pts, E)`` for one spoke, oriented by increasing band energy.

    Short polylines (the analytic 2-point skeleton spokes) are first resampled
    by arclength: the band energy is *not* linear along a straight segment, so
    energy interpolation needs intermediate vertices. Energies are then made
    non-decreasing with a running maximum, which removes the tiny tracer jitter
    near the endpoints without moving any vertex.
    """
    pts = np.asarray(poly, dtype=float)
    if pts.shape[0] < min_points:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] > 0.0:
            u = np.linspace(0.0, s[-1], min_points)
            pts = np.column_stack([np.interp(u, s, pts[:, 0]),
                                   np.interp(u, s, pts[:, 1])])
    E = _E_vec(pts[:, 0], pts[:, 1])
    if E[0] > E[-1]:
        pts = pts[::-1]
        E = E[::-1]
    return pts, np.maximum.accumulate(E)


def _edge_crossings(pts, E, edges):
    """Locate the band-edge crossings along one spoke.

    Returns ``(cross, lo_idx, hi_idx)``: ``cross[i]`` is the point where the
    spoke crosses energy ``edges[i]`` (linear interpolation in energy along the
    polyline, clipped to the spoke's range); the slice
    ``pts[lo_idx[i]:hi_idx[i+1]]`` gives the spoke vertices strictly between
    ``edges[i]`` and ``edges[i+1]``. The first/last crossings are snapped to the
    exact polyline endpoints (pole / seam point / Gamma corner), so adjacent
    patches share identical seam vertices and the Brillouin zone tiles exactly.
    """
    e = np.clip(np.asarray(edges, dtype=float), E[0], E[-1])
    cross = np.column_stack([np.interp(e, E, pts[:, 0]),
                             np.interp(e, E, pts[:, 1])])
    cross[0] = pts[0]
    cross[-1] = pts[-1]
    lo_idx = np.searchsorted(E, e, side='right')
    hi_idx = np.searchsorted(E, e, side='left')
    return cross, lo_idx, hi_idx


def _patch_cells_shoelace(spokes, pc, edges, wrap, cells):
    """Append exact (centroid, area) entries for every cell of one patch.

    Each cell boundary is assembled by reference from shared geometry -- the two
    bounding spokes' stored polylines and the per-spoke crossing table -- so the
    edge shared by two neighboring cells is traversed with identical vertices in
    opposite directions and the patch tiles exactly. The loop runs crossing(lo)
    -> up spoke A -> crossing(hi) -> across -> down spoke B -> close; area and
    centroid follow from the shoelace formulas.
    """
    profiles = [_spoke_profile(s['poly']) for s in spokes]
    tables = [_edge_crossings(pts, E, edges) for pts, E in profiles]
    n = len(spokes)
    n_sectors = n if wrap else n - 1
    n_bands = len(edges) - 1
    for i in range(n_sectors):
        a, b = i, (i + 1) % n
        pts_a, _ = profiles[a]
        pts_b, _ = profiles[b]
        cross_a, lo_a, hi_a = tables[a]
        cross_b, lo_b, hi_b = tables[b]
        for bi in range(n_bands):
            loop = np.vstack([
                cross_a[bi][None, :],
                pts_a[lo_a[bi]:hi_a[bi + 1]],
                cross_a[bi + 1][None, :],
                cross_b[bi + 1][None, :],
                pts_b[lo_b[bi]:hi_b[bi + 1]][::-1],
                cross_b[bi][None, :],
            ])
            area, centroid = polygon_area_centroid(loop)
            if area <= 0.0:
                continue
            cells[(pc, i, int(bi))] = (centroid, area)


def _cells_shoelace(pockets, gammas, pocket_edges, gamma_edges):
    """Exact cell centroids/areas for all six patches via the shoelace formula."""
    cells = {}
    for ip in (0, 1):
        _patch_cells_shoelace(pockets[ip], ip, pocket_edges, wrap=True, cells=cells)
    for ic in range(4):
        _patch_cells_shoelace(gammas[ic], ic + 2, gamma_edges, wrap=False,
                              cells=cells)
    return cells


# --------------------------------------------------- reduced-frame builder ----
def _build_reduced_grid(spoke_mode, n_per_segment, alpha, pocket_levels,
                        gamma_levels, n_K, n_gamma, verbose):
    """Build the grid in the internal reduced (a = 1, t = _T_REDUCED) frame.

    ``pocket_levels`` / ``gamma_levels`` are either an integer count of equally
    spaced interior contours or an explicit array of interior level energies in
    reduced (t = _T_REDUCED) units. Cell centroids and areas are the exact
    shoelace polygon integrals assembled from the spoke geometry. Returns
    reduced-frame arrays; the public wrappers map them to the physical codebase
    frame."""
    if spoke_mode == 'seam-blended':
        if verbose:
            print('calibrating seam seeds ...')
        lset = _solve_seam_positions(n_per_segment, alpha, verbose=verbose)
        if verbose:
            print('tracing spokes ...')
        pockets, gammas = _build_spokes(lset, verbose=verbose)
    elif spoke_mode == 'per-patch-uniform':
        if verbose:
            print('tracing per-patch uniform (truncated) spokes ...')
        pockets, gammas = _build_spokes_uniform(n_K, n_gamma, verbose=verbose)
    else:
        raise ValueError("spoke_mode must be 'seam-blended' or "
                         "'per-patch-uniform'")

    t = _T_REDUCED
    if np.isscalar(pocket_levels):
        pocket_edges = np.concatenate([[0.0], t * np.arange(1, pocket_levels + 1)
                                       / (pocket_levels + 1), [t]])
    else:
        pocket_edges = np.concatenate([[0.0], np.asarray(pocket_levels), [t]])
    if np.isscalar(gamma_levels):
        gamma_edges = np.concatenate([[t], t + 2 * t * np.arange(1, gamma_levels + 1)
                                      / (gamma_levels + 1), [3 * t]])
    else:
        gamma_edges = np.concatenate([[t], np.asarray(gamma_levels), [3 * t]])

    if verbose:
        print('shoelace cell assembly ...')
    cells = _cells_shoelace(pockets, gammas, pocket_edges, gamma_edges)
    ids = sorted(cells)
    kpts = np.array([cells[c][0] for c in ids])
    areas = np.array([cells[c][1] for c in ids])

    pole_lab = np.array([min(c[0], 2) for c in ids])      # 2 = any Gamma corner
    sector_lab = np.array([c[1] for c in ids])
    band_lab = np.array([c[2] for c in ids])
    Ec = _E_vec(kpts[:, 0], kpts[:, 1])

    spokes_out = ([s['poly'] for ip in (0, 1) for s in pockets[ip]]
                  + [s['poly'] for ic in range(4) for s in gammas[ic]])
    diagnostics = {'bz_area': _BZ_AREA, 'area_sum': float(areas.sum()),
                   'area_error': float(areas.sum() / _BZ_AREA - 1.0),
                   'n_cells': len(ids)}
    return {'k_points': kpts, 'areas': areas, 'E_centroid': Ec,
            'pole': pole_lab, 'sector': sector_lab, 'band': band_lab,
            'spokes': spokes_out,
            'pocket_edges': pocket_edges, 'gamma_edges': gamma_edges,
            'diagnostics': diagnostics}


# --------------------------------------------------- physical-frame output ----
def _resolve_scales(model_params, lattice_spacing, hopping):
    """Return ``(lattice_spacing, hopping)`` from a parameter bundle / overrides.

    ``hopping`` defaults to ``2 hbar v_F / (3 a)`` -- the same relation
    :class:`GrapheneModel` uses -- read from ``model_params``."""
    if model_params is None and lattice_spacing is None and hopping is None:
        model_params = GrapheneParameters()
    if model_params is not None:
        if lattice_spacing is None:
            lattice_spacing = model_params.lattice_spacing
        if hopping is None:
            hopping = (2.0 * model_params.units.hbar * model_params.vf
                       / (3.0 * model_params.lattice_spacing))
    if lattice_spacing is None or hopping is None:
        raise ValueError(
            "provide a model_params bundle, or both lattice_spacing and hopping."
        )
    if lattice_spacing <= 0.0:
        raise ValueError("lattice_spacing must be positive.")
    return float(lattice_spacing), float(hopping)


def _to_physical_xy(arr, lattice_spacing):
    """Map reduced (a = 1) momenta to the physical codebase frame.

    ``k_phys = (k_reduced_y, k_reduced_x) / a`` (swap kx<->ky, scale by 1/a)."""
    arr = np.asarray(arr, dtype=float)
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 1] / lattice_spacing
    out[..., 1] = arr[..., 0] / lattice_spacing
    return out


def _to_internal_levels(levels, energy_scale):
    """Convert physical interior level energies to reduced (t=_T_REDUCED) units.

    Integer counts pass through unchanged (they are unit-free)."""
    if levels is None or np.isscalar(levels):
        return levels
    return np.asarray(levels, dtype=float) / energy_scale


def build_graphene_bz_grid(
    model_params: GrapheneParameters | None = None,
    *,
    lattice_spacing: float | None = None,
    hopping: float | None = None,
    spoke_mode: str = 'per-patch-uniform',
    n_per_segment: int = 6,
    alpha: float = 0.5,
    pocket_levels=3,
    gamma_levels=3,
    n_K: int = 24,
    n_gamma: int = 48,
    verbose: bool = False,
) -> dict:
    """Build the adaptive graphene Brillouin-zone integration grid.

    The cells follow the nearest-neighbor conduction band of the built-in
    graphene model, with the van Hove separatrix ``E = hopping`` as a cell
    boundary. All outputs are in the physical codebase frame (momenta in
    ``1 / length``, areas in ``1 / length**2``, energies in the convention's
    energy unit, e.g. joules for the SI preset).

    Args:
        model_params: Graphene parameter bundle supplying ``lattice_spacing``
            and -- via ``2 hbar v_F / (3 a)`` -- the ``hopping``. Defaults to
            ``GrapheneParameters()`` when nothing else is given. The drive is
            irrelevant: the grid is a property of the static band.
        lattice_spacing, hopping: Explicit overrides for ``a`` and ``t``; useful
            for a dimensionless grid without a parameter bundle. Either both or
            neither must be supplied when ``model_params`` is ``None``.
        spoke_mode: ``'seam-blended'`` (one continuous spoke family seeded on the
            separatrix, placed by the ``alpha``-blended angular label, plus the
            straight skeleton; controlled by ``n_per_segment``, ``alpha``) or
            ``'per-patch-uniform'`` (two independent truncated families, equally
            spaced around K/K' with ``n_K`` spokes -- a multiple of 6 -- and
            around Gamma with ``n_gamma`` -- a multiple of 12). The default
            ``(n_K, n_gamma) = (24, 48)`` reproduces the default seam-blended
            spoke counts for like-for-like comparison.
        n_per_segment: Even number of generic spokes seeded per separatrix
            segment (seam-blended mode).
        alpha: Seam blend in ``[0, 1]`` balancing the angular voids at the K and
            Gamma poles (seam-blended mode); ``0.5`` balances them.
        pocket_levels: Integer count of equally spaced interior energy contours
            below the separatrix, or an explicit array of interior level
            energies (same energy unit as the model).
        gamma_levels: As ``pocket_levels`` but for the contours between the
            separatrix and the band top ``3 * hopping``.
        n_K, n_gamma: Spoke counts for ``'per-patch-uniform'`` mode.
        verbose: Print calibration / assembly diagnostics.

    Returns:
        Dict with ``Nc`` cells:
            ``k_points`` (Nc, 2)  cell centroids in the physical frame.
            ``areas``     (Nc,)   cell areas; sum to the Brillouin-zone area.
            ``E_centroid``(Nc,)   band energy at each centroid.
            ``pole``      (Nc,)   0 = K pocket, 1 = K' pocket, 2 = Gamma region.
            ``sector``    (Nc,)   angular sector index within the patch.
            ``band``      (Nc,)   energy band index within the patch.
            ``spokes``    list of (M, 2) physical-frame polylines (for plotting).
            ``ring_levels`` dict: ``'pocket'`` / ``'gamma'`` interior level
                energies and ``'seam' = hopping``.
            ``meta``      dict: parameters, scales and area / assignment
                diagnostics.
    """
    lattice_spacing, hopping = _resolve_scales(model_params, lattice_spacing, hopping)
    energy_scale = hopping / _T_REDUCED

    reduced = _build_reduced_grid(
        spoke_mode=spoke_mode,
        n_per_segment=n_per_segment,
        alpha=alpha,
        pocket_levels=_to_internal_levels(pocket_levels, energy_scale),
        gamma_levels=_to_internal_levels(gamma_levels, energy_scale),
        n_K=n_K,
        n_gamma=n_gamma,
        verbose=verbose,
    )

    k_points = _to_physical_xy(reduced['k_points'], lattice_spacing)
    areas = reduced['areas'] / lattice_spacing ** 2
    E_centroid = reduced['E_centroid'] * energy_scale
    spokes = [_to_physical_xy(poly, lattice_spacing) for poly in reduced['spokes']]

    diag = reduced['diagnostics']
    meta = {
        'spoke_mode': spoke_mode, 'n_per_segment': n_per_segment, 'alpha': alpha,
        'n_K': n_K, 'n_gamma': n_gamma,
        'lattice_spacing': lattice_spacing, 'hopping': hopping,
        'bz_area': diag['bz_area'] / lattice_spacing ** 2,
        'area_sum': diag['area_sum'] / lattice_spacing ** 2,
        'area_error': diag['area_error'],
        'n_cells': diag['n_cells'],
    }
    if verbose:
        print(f'  {meta["n_cells"]} cells;  '
              f'sum(areas)/BZ_area - 1 = {meta["area_error"]:+.2e}')

    return {
        'k_points': k_points,
        'areas': areas,
        'E_centroid': E_centroid,
        'pole': reduced['pole'],
        'sector': reduced['sector'],
        'band': reduced['band'],
        'spokes': spokes,
        'ring_levels': {'pocket': reduced['pocket_edges'][1:-1] * energy_scale,
                        'gamma': reduced['gamma_edges'][1:-1] * energy_scale,
                        'seam': hopping},
        'meta': meta,
    }


def graphene_bz_quadrature(
    model_params: GrapheneParameters | None = None,
    **kwargs,
) -> KQuadrature:
    """Build the adaptive graphene BZ grid as a ready-to-use :class:`KQuadrature`.

    Thin wrapper over :func:`build_graphene_bz_grid` (same arguments) that bundles
    the cell centroids and areas into the toolkit's integration primitive: the
    flat ``(Nc,)`` grid carries each cell's centroid as a sample point and its
    area as the integration weight, so ``quad.integrate(value_map)`` with
    ``value_map`` evaluated at ``(quad.kx_grid, quad.ky_grid)`` returns the
    Brillouin-zone integral ``sum(area * value)``.
    """
    grid = build_graphene_bz_grid(model_params, **kwargs)
    k_points = grid['k_points']
    return KQuadrature(
        kx_grid=np.ascontiguousarray(k_points[:, 0]),
        ky_grid=np.ascontiguousarray(k_points[:, 1]),
        weights=np.ascontiguousarray(grid['areas']),
    )
