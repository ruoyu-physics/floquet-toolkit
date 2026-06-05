"""Process-based parallel map for independent sweep items (e.g. drive amplitudes).

This helper is intentionally free of any NumPy/Floquet imports so it stays
lightweight and import-cheap.

Note: it deliberately does NOT configure BLAS/LAPACK threading. Those backends
read their thread-count environment variables at import time, so the
single-thread pinning that should pair with process parallelism (to avoid
oversubscribing cores) must be set at the very top of each entry-point script,
*before* NumPy — or anything that imports it — is imported. Importing this
module cannot do it for you, because importing the wider ``floquet_toolkit``
package already pulls in NumPy. See the sweep scripts for the standard snippet.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor


def resolve_worker_count(n_jobs: int, n_items: int) -> int:
    """Return the effective worker count, capped at the number of items.

    ``n_jobs < 0`` requests every available core; ``n_jobs == 1`` (or a single
    item) forces serial execution.
    """
    if n_jobs < 0:
        n_jobs = os.cpu_count() or 1
    return max(1, min(n_jobs, n_items))


def parallel_map(func, items, n_jobs: int = -1) -> list:
    """Map ``func`` over ``items`` across processes, preserving input order.

    Each item is treated as an independent unit of work, so this is suited to
    sweeps where every element rebuilds its own model/manager and shares no
    state with the others. Falls back to a plain serial loop when only one
    worker or one item is requested, which avoids process-pool overhead for
    tiny sweeps and keeps debugging and profiling simple.

    ``func`` and every element of ``items`` must be picklable: define worker
    functions at module level (use ``functools.partial`` to bind extra
    arguments), since the work runs in processes spawned from the entry-point
    module.
    """
    items = list(items)
    workers = resolve_worker_count(n_jobs, len(items))
    if workers <= 1:
        return [func(item) for item in items]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # ``executor.map`` preserves input order, so results line up with
        # ``items`` exactly like a serial list comprehension would.
        return list(executor.map(func, items))


def _contiguous_chunks(n_items: int, n_chunks: int) -> list:
    """Return contiguous ``(start, stop)`` index ranges splitting ``n_items``.

    Produces at most ``n_chunks`` balanced, order-preserving pieces (empty
    chunks are dropped).
    """
    if n_items <= 0:
        return []
    base, extra = divmod(n_items, n_chunks)
    ranges = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < extra else 0)
        if size == 0:
            break
        ranges.append((start, start + size))
        start += size
    return ranges


def parallel_chunk_map(func, items, n_jobs: int = -1) -> list:
    """Map ``func`` over contiguous CHUNKS of ``items`` across processes.

    Unlike :func:`parallel_map` (one call per item), each worker receives a
    whole contiguous chunk and ``func(chunk)`` is called once per chunk, so
    expensive per-worker setup (rebuilding a model, one batched diagonalization)
    is amortized across many items. ``func(chunk)`` must return a sequence
    aligned with its chunk; the per-chunk results are concatenated back into a
    single list in input order.

    This is the primitive for **k-point parallelization** of a single-amplitude
    grid: split the momenta into one chunk per core and compute each chunk —
    ideally with the vectorized/batched routines — in its own process. (For a
    multi-amplitude sweep, parallelize amplitudes with :func:`parallel_map`
    instead; do not nest the two.)

    Falls back to a single in-process ``func(items)`` call when only one worker
    or chunk is needed. ``func`` and the items must be picklable (define ``func``
    at module level).
    """
    items = list(items)
    workers = resolve_worker_count(n_jobs, len(items))
    if workers <= 1:
        return list(func(items))
    chunks = [items[start:stop] for start, stop in _contiguous_chunks(len(items), workers)]
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        flat = []
        for chunk_result in executor.map(func, chunks):
            flat.extend(chunk_result)
        return flat
