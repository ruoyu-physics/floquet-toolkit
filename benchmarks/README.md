# Benchmarks & profiling

Diagnostic harnesses for measuring performance — *not* part of the scientific
pipeline. Use these to decide **where** to optimize before changing code
("measure, don't guess"), and to confirm a change actually helped.

## Layout

| Path | Purpose | Tracked in git? |
| --- | --- | --- |
| `profile_current.py` | Profile one Floquet current calculation; reports the heaviest functions and an eigh-vs-construction breakdown. | yes |
| `bench_curvature_vs_hfe.py` | Accuracy check (not timing): cross-validates a hand-derived closed-form order-1 HFE formula against `HFEBuilder`'s numerical construction and the exact time-averaged Floquet result, for a high-frequency-driven massless Dirac model. | yes |
| `results/` | Generated artifacts (`.prof` dumps, timing logs). | no — git-ignored |

Curated findings and the optimization history live in
[`../docs/PERFORMANCE.md`](../docs/PERFORMANCE.md). Raw output belongs in
`results/`; the *narrative* belongs in that doc.

## Running

From the repository root:

```bash
python3 -m benchmarks.profile_current            # defaults (n_k_points=41, A=1.5e-9)
python3 -m benchmarks.profile_current 51 2.0e-9  # custom n_k_points and amplitude
```

The profiler forces serial execution (`PARALLEL_JOBS = 1`) so the numbers reflect
a single calculation rather than a process pool. To save a binary profile for
inspection with [`snakeviz`](https://jiffyclub.github.io/snakeviz/) or `pstats`,
set `DUMP_PROFILE = True` in `profile_current.py` (output lands in `results/`).

## Interpreting the output

- **`tottime` (self time)** — time spent *inside* a function itself; this is where
  CPU cycles actually burn.
- **`cumtime` (cumulative)** — time in a function plus everything it calls; useful
  for finding the top of an expensive call tree.

If `eigh` dominates, batched/vectorized linear algebra (or a GPU) may help. If
Hamiltonian construction dominates (as it did here), the win is vectorization or
an analytic operator — not faster diagonalization.
