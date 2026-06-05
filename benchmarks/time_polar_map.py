"""Time the polar velocity map (no plotting) at chosen core counts.

Must be run as a FILE, not via ``-c`` or a heredoc: multiprocessing 'spawn'
(macOS default) re-imports the main module in each worker, which fails when the
main module is stdin/-c (there is no file to import). A real file plus the
``if __name__ == "__main__"`` guard avoids that.

Usage (from the repository root)::

    python3 -m benchmarks.time_polar_map            # PARALLEL_JOBS 1 then -1
    python3 -m benchmarks.time_polar_map 1 2 4 6 8  # specific core counts to sweep
"""

import os
import sys
import time

# Make the repository root importable whether launched as `python3 -m
# benchmarks.time_polar_map` or `python3 benchmarks/time_polar_map.py`. This
# also lets spawn workers re-import `scripts` (they re-run this module).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import scripts.plot_local_floquet_current_map as m


def main() -> None:
    jobs_list = [int(arg) for arg in sys.argv[1:]] or [1, -1]
    for jobs in jobs_list:
        m.PARALLEL_JOBS = jobs
        t0 = time.perf_counter()
        m.polar_local_floquet_current_map()
        print(f"PARALLEL_JOBS={jobs:>3}: {(time.perf_counter() - t0) * 1e3:.0f} ms")


if __name__ == "__main__":
    main()
