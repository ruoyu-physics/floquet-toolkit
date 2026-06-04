# Architecture

A map of how `floquet_toolkit` is organized and how data flows through it.
The goal of this document is that you can change one layer without having to
hold the whole codebase in your head: each layer has a single job and talks to
its neighbors through a small, stable interface.

For *why* particular optimizations were made (and their timings), see
[`PERFORMANCE.md`](PERFORMANCE.md). This file is about *structure*.

---

## The layers at a glance

```
   model physics            DrivenBlochHamiltonian  ←  builtin_models/  +  config
   (what H(k,t) is)         core/                       (Dirac, graphene, …)
        │
        ▼
   user-facing API          managers/                FloquetManager
   (what you call)                                    ├─ FloquetLocalManager      (local-in-k)
        │                                             └─ FloquetTransportManager  (integrated)
        ▼
   observables              calculators/             velocity · current · curvature ·
   (the algorithms)                                  berry-phase · spectrum · perturbation
        │
        ▼
   Floquet states           calculators/states/      Provider (select) · Tracker (connect) ·
   (build / select / reuse)                          Cache (reuse)
        │
        ▼
   numerical primitives     builders/  +  utils/     FloquetBuilder · HFEBuilder ·
   (matrices, k-grids)                               kspace · drive_fields · geometry
```

Dependencies point **downward only**: a calculator may use providers and
builders; a provider never imports a calculator; `utils/` and `core/` depend on
nothing else in the package. If you ever find yourself wanting an upward import,
that's the signal a responsibility is in the wrong layer.

---

## The layers in detail

### `core/` — what a driven system *is*
[`DrivenBlochHamiltonian`](../floquet_toolkit/core/driven_bloch_hamiltonian.py)
is the one object every calculator is built around. It is a plain container for
the callables and metadata that define a 2D driven Bloch problem:

- `H_t(t, kx, ky)` — the time-periodic Hamiltonian (the only required piece).
- `H_static(kx, ky)` — the static/time-averaged Hamiltonian, used as the
  reference for band selection.
- `omega`, `units` (an `UnitConvention`), `dimension`.
- **Optional fast-path flags** the layers below opt into:
  - `supports_vectorized_time` — `H_t` broadcasts over a `(k, t)` grid, enabling
    batched construction. **This single flag is what unlocks the batched
    velocity map and batched Floquet-matrix construction.**
  - `analytic_velocity_operator(time, kx, ky, axis)` — a closed form for
    $(1/\hbar)\,\partial H/\partial k$, avoiding finite differences.

### `config.py` — numerical knobs and units
Frozen dataclasses: `UnitConvention` (SI vs. unitless), `DriveParameters`
(drive amplitudes/polarization), and `FloquetParameters` (the truncation:
`n_trunc`, `n_harmonics`, `n_time`; note `n_blocks = 2*n_trunc + 1`).

### `builtin_models/` — ready-made physics
`DiracModel`, `GrapheneModel`, `RotatingFrameDiracModel` (and `driven_*`
factory functions). Each `to_driven_hamiltonian()` produces a
`DrivenBlochHamiltonian` with the flags above set appropriately. To add a model,
this is the only place you write physics — everything downstream is generic.

### `managers/` — the user-facing facade
This is the API surface. You normally start here.
- [`FloquetLocalManager`](../floquet_toolkit/managers/floquet_local_manager.py)
  — quantities *at* a momentum: velocities, curvatures, spectra, state
  selection. Owns its own `FloquetStateProvider` and the calculators.
- [`FloquetTransportManager`](../floquet_toolkit/managers/floquet_transport_manager.py)
  — quantities *integrated over* k-space: Floquet/adiabatic currents on a Fermi
  disk, Berry curvature integrals. Owns the cache (`use_cache`) shared across
  its calculators.
- [`FloquetManager`](../floquet_toolkit/managers/floquet_manager.py) — a thin
  compatibility wrapper exposing `.local` and `.transport`, delegating unknown
  attributes to whichever sub-manager defines them.

Managers hold no algorithms; they wire models + params + cache into calculators
and forward calls.

### `calculators/` — the algorithms
One class per family of observable. Each is constructed from a
`DrivenBlochHamiltonian`, a `FloquetParameters`, and an optional cache.
- [`FloquetVelocityCalculator`](../floquet_toolkit/calculators/floquet_velocity_calculator.py)
  — local velocity/current expectations from several state sources (exact
  Floquet, tracked, perturbative, static, adiabatic, HFE). Both the single-`k`
  routes and the **batched velocity maps** live here (see the data-flow example
  below).
- [`FloquetCurrentCalculator`](../floquet_toolkit/calculators/floquet_current_calculator.py)
  — integrates a *local velocity map* over a circular momentum region. It
  generates the k-grid, asks the velocity calculator for the map, then applies
  the integration rule. It does not compute velocities itself.
- `FloquetCurvatureCalculator`, `FloquetBerryPhaseCalculator`,
  `FloquetSpectrumCalculator`, `FloquetPerturbationCalculator` — the analogous
  algorithms for Berry curvature, Berry phases, quasienergy spectra, and
  perturbative corrections.

### `calculators/states/` — building, selecting, and reusing Floquet states
The trickiest part of the codebase, deliberately split by responsibility:
- [`FloquetStateProvider`](../floquet_toolkit/calculators/states/floquet_state_provider.py)
  — **selection**. Diagonalizes the Floquet Hamiltonian (per-`k` or batched via
  `diagonalize_floquet_hamiltonian_batched`) and picks the eigenstate connected
  to a target static band by maximal overlap. `select_floquet_state` is the
  single-`k` entry point; `select_floquet_state_from_eigensystem` does the
  selection given an *already-diagonalized* eigensystem (so batched callers
  don't re-diagonalize). Also reconstructs the time-domain state
  $\psi(t) = \sum_m v_m e^{-im\omega t}$.
- [`FloquetStateTracker`](../floquet_toolkit/calculators/states/floquet_state_tracker.py)
  — **connection**. Selection is per-point and can hop branches; the tracker
  propagates one branch consistently across a grid via BFS from a seed. This is
  inherently *sequential* (each point references its neighbor), so it is not
  batchable — but the velocity evaluation on its output is.
- [`FloquetStateCache`](../floquet_toolkit/calculators/states/floquet_state_cache.py)
  — **reuse**. Memoizes eigensystems, selected states, reconstructed states, and
  tracked grids, keyed by momentum/grid/time. Off by default (`use_cache=False`);
  worth turning on when the same momenta are revisited.

### `builders/` — the Floquet/HFE matrices
- [`FloquetBuilder`](../floquet_toolkit/builders/floquet_builder.py) — Fourier
  harmonics `H_m` and the truncated extended-space Floquet matrix. Has both a
  single-`k` path and `*_batched` paths that evaluate `H_t` on a whole `(k, t)`
  grid in one call (used when `supports_vectorized_time`).
- [`HFEBuilder`](../floquet_toolkit/builders/hfe_builder.py) — high-frequency
  effective Hamiltonians, consuming a `FloquetBuilder`.

### `utils/` — leaf helpers (no package dependencies)
`kspace` (grid creation + Cartesian/polar integration rules), `drive_fields`
(vector-potential / E-field components), `geometry` (signed loop area),
`parallel` (worker-count resolution + `parallel_map`).

---

## Worked example: a Cartesian Floquet current with pointwise selection

This is the path the recent batching work reshaped, and it shows the layering in
action. Calling
`transport.integrate_floquet_current_on_fermi_disk(..., state_selection_algorithm="pointwise", grid_type="cartesian")`:

1. **Manager** forwards to `FloquetCurrentCalculator`.
2. **Current calculator** builds the k-grid with `_build_integration_grid`: a
   regular Cartesian grid over the square enclosing the disk, plus a boolean
   `mask` (from `utils/kspace.create_circular_mask`) marking which points fall
   inside the radius.
3. It hands the full grid to the **velocity calculator's**
   `compute_floquet_velocity_map`, which selects and evaluates in batch:
   - **Selection (per point, independent).** When the model is batchable, it
     diagonalizes the Floquet Hamiltonian at every momentum in one call
     (`provider.diagonalize_floquet_hamiltonian_batched`), then for each point
     picks the eigenstate with maximal overlap onto the target static band
     (`provider.select_floquet_state_from_eigensystem`). No tracker — each
     momentum is decided on its own.
   - **Evaluation (batched).** It forwards the selected states to
     `compute_floquet_velocity_map_from_states`, which in one pass reconstructs
     every $\psi(k, t)$, builds the velocity operators $(v_x, v_y)$ over the
     whole $(k, t)$ grid, and evaluates $\langle\psi|v|\psi\rangle$.
4. The current calculator integrates the resulting `(n_kx, n_ky, n_time)` maps
   with `_integrate_current_map` → `utils.integrate_cartesian_grid`, which
   applies the **mask** (out-of-disk points contribute zero) and the uniform
   area weight. The map is computed on every grid point; the mask — not a
   pre-filter — is what restricts the integral to the disk.

```
integrate_floquet_current_on_fermi_disk(pointwise, cartesian)   (manager → current calc)
  ├─ _build_integration_grid                     (kx_grid, ky_grid, mask)
  └─ velocity.compute_floquet_velocity_map
        ├─ provider.diagonalize_floquet_hamiltonian_batched      ← batched
        ├─ provider.select_floquet_state_from_eigensystem (×k)   ← per point, independent
        └─ compute_floquet_velocity_map_from_states              ← batched
              reconstruct ψ(k,t) · build (v_x,v_y) · ⟨ψ|v|ψ⟩
  └─ _integrate_current_map                       (→ utils.integrate_cartesian_grid, masked)
```

Other paths differ only in how the *grid* and *selection* are produced, then
reuse the same evaluator and integrator:
- **`grid_type="polar"`** swaps the grid + integration rule (polar Jacobian,
  no mask) but keeps the same selection and velocity-map calls.
- **`state_selection_algorithm="tracked"`** replaces the independent per-point
  selection with `FloquetStateTracker`, which propagates one branch across the
  grid via BFS from a seed (inherently *sequential*), then feeds its state grid
  straight into `compute_floquet_velocity_map_from_states`.
- **`grid_type="adaptive_cartesian"`** stays on a per-point path (it refines the
  grid on the fly, so there is no fixed grid to batch).
- The **adiabatic** current reuses the generic per-point
  `_integrate_local_current` because its local quantity isn't a Floquet velocity.

The key design line: **the velocity calculator produces a local map; the current
calculator only integrates it.** Selection ("which state") is kept separate from
evaluation ("velocity of that state"), which is what lets the pointwise and
tracked paths share one batched evaluator.

---

## Floquet conventions (signs & normalization)

These are the sign/normalization choices baked into
[`FloquetBuilder`](../floquet_toolkit/builders/floquet_builder.py) and matched
everywhere downstream (state reconstruction, quasienergy folding). They're worth
pinning down because a wrong sign or a missing `1/T` leaves results *plausible
but subtly off* — and because the conventions are coupled across files, so they
can't be inferred from any one of them. Pay attention to these when comparing
against a paper or another code.

- **Fourier harmonics.** $H_m = \frac{1}{T} \int_0^T H(t)\, e^{+im\omega t}\, dt$,
  evaluated numerically as a uniform average over `n_time` time samples. The
  time-domain Hamiltonian reconstructs with the opposite sign:
  $H(t) = \sum_m H_m e^{-im\omega t}$. Harmonics are stored with index $m$ at
  array position `m + n_harmonics` (so $m = 0$ is the middle), shape
  `(2·n_harmonics + 1, N, N)`.
- **Floquet (extended-space) matrix.** Block $(m, n)$ is
  $H_{m-n} - m\hbar\omega\,\delta_{mn}$, with sidebands $m, n$ truncated to
  `[−n_trunc, n_trunc]` — an `(n_blocks·N) × (n_blocks·N)` matrix where
  `n_blocks = 2·n_trunc + 1`. The off-diagonal coupling uses $H_{m-n}$ only when
  $|m - n| \le$ `n_harmonics`, else zero. So **`n_harmonics` (how many Fourier
  components exist) is distinct from `n_trunc` (how many sidebands the matrix
  keeps).**
- **State reconstruction matches the Hamiltonian.** A Floquet eigenvector's
  sideband blocks $v_m$ rebuild the time-domain state as
  $\psi(t) = \sum_m v_m e^{-im\omega t}$ — the same sign convention as $H(t)$,
  which is why the reconstruction and harmonic conventions must always move
  together.
- **Quasienergy zone.** Quasienergies are defined modulo $\hbar\omega$; folding
  (when requested) maps them into the principal zone
  $[-\hbar\omega/2,\ \hbar\omega/2)$.

---

## Conventions worth knowing

- **`H_t(t, kx, ky)` broadcasts.** Built-in models add matrix axes with
  `[..., None, None]`, so `H_t` accepts scalar or array `(t, k)` and returns
  `batch_shape + (dim, dim)`. The batched construction relies on calling it as
  `H_t(t[None, :], kx[:, None], ky[:, None]) → (n_k, n_time, dim, dim)`.
- **Caching is opt-in and value-preserving.** A cached run must return the same
  numbers as an uncached one (guarded by a test). Cache on for repeated visits,
  off for single cold passes.
- **Optimizations must preserve results.** Batched and single-`k` paths agree to
  floating-point rounding (~1e-16 relative). When adding a fast path, keep the
  reference path and assert equivalence rather than trusting it by eye.
- **Add physics in one place.** New models go in `builtin_models/` and expose
  themselves through `DrivenBlochHamiltonian` flags; the calculators stay
  generic.
