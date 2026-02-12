"""
Batch binding layer for parallel ppxf execution across many bins.

Replaces the per-module joblib + memmap boilerplate with a unified,
memory-efficient parallel executor optimized for CANFAR Science Platform
nodes (typically 16 CPUs) processing up to millions of spectra.

Key optimizations over the existing joblib approach
---------------------------------------------------
1. **In-memory shared arrays** via ``multiprocessing.shared_memory``
   (POSIX shared memory backed by ``/dev/shm`` on Linux).  Eliminates
   the joblib dump/load memmap-to-disk overhead entirely — data stays
   in RAM at all times.
2. **Dynamic load balancing** via ``imap_unordered`` with adaptive
   mini-chunks.  Avoids the straggler problem of static chunking where
   one slow chunk blocks overall progress.
3. **Wave processing** for memory-bounded execution of millions of
   spectra.  Configurable wave size keeps peak RSS within CANFAR limits.
4. **Adaptive chunk sizing** - automatically tunes chunk granularity to
   balance IPC overhead against scheduling flexibility.
5. **Unified API** - one call replaces ~70 lines of duplicated
   boilerplate in each wrapper module.

Start-method strategy
---------------------
We default to **spawn + SharedMemory** on all platforms.  This avoids
deadlocks that arise when ``fork()`` is called after libraries like
h5py, numpy BLAS, or Python's logging module have created internal
threads/locks.  SharedMemory segments live in ``/dev/shm`` (RAM-backed
tmpfs on Linux), so there is no disk I/O penalty.

To opt into raw ``fork`` COW mode (faster pool startup, but risks
deadlocks if any threads are active), set the environment variable
``NGIST_USE_FORK=1``.

Usage
-----
Each ppxf wrapper module defines a module-level worker function::

    def _my_bin_worker(bin_idx, shared, params):
        return run_ppxf(
            shared['templates'],
            shared['bin_data'][:, bin_idx].copy(),
            shared['noise'][:, bin_idx].copy(),
            params['velscale'],
            ...
        )

Then invokes the executor::

    executor = BatchExecutor(ncpu=16, wave_size=50000)
    results = executor.run(
        worker_fn=_my_bin_worker,
        shared_arrays={'templates': T, 'bin_data': D, 'noise': N},
        params={'velscale': vs, ...},
        bin_indices=np.arange(nbins),
    )
"""

import logging
import multiprocessing as mp
import multiprocessing.shared_memory
import numpy as np
import os
import sys
import time

from tqdm import tqdm

logger = logging.getLogger("ngist.batch")


# ============================================================================
# Module-level worker state (populated by pool initializer, read by workers)
# ============================================================================
_worker_shared = {}
_worker_fn = None
_worker_params = {}


def _init_worker_fork(shared_dict, worker_fn, params):
    """Pool initializer for fork mode.

    On Linux the *shared_dict* arrays already live in the process
    address space via copy-on-write page sharing — no pickling needed.
    """
    global _worker_shared, _worker_fn, _worker_params
    _worker_shared = shared_dict
    _worker_fn = worker_fn
    _worker_params = params


def _init_worker_shm(shm_meta, worker_fn, params):
    """Pool initializer for spawn mode.

    Reconstructs numpy arrays from named ``SharedMemory`` segments
    (backed by /dev/shm on Linux — pure RAM, no disk I/O) so every
    worker sees the same physical pages.
    """
    global _worker_shared, _worker_fn, _worker_params
    _worker_fn = worker_fn
    _worker_params = params
    _worker_shared = {}
    for key, (shm_name, shape, dtype_str) in shm_meta.items():
        shm = mp.shared_memory.SharedMemory(name=shm_name)
        arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        _worker_shared[key] = arr
        # prevent garbage-collection of the SharedMemory handle
        _worker_shared[f"__shm_{key}"] = shm


def _dispatch_chunk(chunk):
    """Process a mini-chunk of bin indices, returning (bin_idx, result) pairs.

    Using mini-chunks (rather than single bins) amortises the IPC
    serialisation cost while still allowing dynamic load balancing.
    """
    out = []
    for bin_idx in chunk:
        try:
            result = _worker_fn(bin_idx, _worker_shared, _worker_params)
            out.append((bin_idx, result))
        except Exception as e:
            logger.warning("ppxf failed on bin %d: %s", bin_idx, e)
            out.append((bin_idx, None))
    return out


# ============================================================================
# BatchExecutor
# ============================================================================
class BatchExecutor:
    """Parallel batch executor for ppxf fitting.

    Parameters
    ----------
    ncpu : int
        Number of worker processes (default 4).
    wave_size : int
        Maximum bins loaded into memory at once.  ``0`` (default) means
        process everything in a single pass.  Set to e.g. ``50000`` when
        scaling beyond ~100 K bins to keep peak RSS bounded.
    chunk_target : int
        Target number of mini-chunks **per CPU**.  Higher values give
        finer-grained dynamic scheduling at the cost of more IPC
        messages.  ``8`` is a good default for typical ppxf runtimes
        (0.1–5 s per bin).
    """

    def __init__(self, ncpu=4, wave_size=0, chunk_target=8):
        self.ncpu = max(1, ncpu)
        self.wave_size = wave_size
        self.chunk_target = chunk_target
        self._can_fork = self._check_fork_available()
        self._shm_handles = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_fork_available():
        """Return True only if the user explicitly opted into fork mode.

        Raw ``fork()`` after threads have been created (by numpy BLAS,
        h5py, Python logging, etc.) can deadlock.  We therefore default
        to the safe ``spawn + SharedMemory`` path on **all** platforms.

        SharedMemory segments live in ``/dev/shm`` on Linux (RAM-backed
        tmpfs), so there is zero disk I/O penalty compared to fork COW.

        Set ``NGIST_USE_FORK=1`` to force fork mode if you are certain
        no thread-unsafe libraries have been loaded.
        """
        if os.environ.get("NGIST_USE_FORK", "0") != "1":
            return False
        if sys.platform == "darwin":
            return False
        try:
            mp.get_context("fork")
            return True
        except ValueError:
            return False

    def _compute_chunks(self, bin_indices):
        """Split *bin_indices* into adaptive mini-chunks.

        The chunk size is chosen so that there are roughly
        ``ncpu * chunk_target`` chunks in total — enough for dynamic
        scheduling, but large enough that IPC overhead is negligible.
        """
        nbins = len(bin_indices)
        target_chunks = self.ncpu * self.chunk_target
        chunk_size = max(1, nbins // target_chunks)
        # Clamp to [10, 500] for a good IPC-vs-scheduling trade-off
        chunk_size = max(min(chunk_size, 500), min(10, nbins))
        chunks = [
            bin_indices[i : i + chunk_size].tolist()
            for i in range(0, nbins, chunk_size)
        ]
        return chunks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        worker_fn,
        shared_arrays,
        params,
        bin_indices,
        desc="ppxf",
        fail_value=None,
    ):
        """Execute *worker_fn* for each bin in parallel.

        Parameters
        ----------
        worker_fn : callable
            **Must be a module-level function** (picklable).  Signature::

                worker_fn(bin_idx: int, shared: dict, params: dict) -> tuple

            *shared* contains large read-only arrays (templates, spectra,
            noise …).  *params* holds small scalars / arrays.

        shared_arrays : dict[str, np.ndarray]
            Large read-only arrays shared across all workers.

        params : dict
            Small parameters broadcast to every worker (scalars, config
            values, small arrays like ``goodPixels``).  Pickled once per
            worker process via the pool initializer.

        bin_indices : array-like of int
            Which bins to process.

        desc : str
            ``tqdm`` progress-bar description.

        fail_value : any
            Sentinel inserted for bins whose *worker_fn* raised.

        Returns
        -------
        results : list
            Ordered list of worker return values, one per bin index.
        """
        bin_indices = np.asarray(bin_indices, dtype=int)
        nbins = len(bin_indices)
        if nbins == 0:
            return []

        # Wave processing — split the full index range into waves
        if self.wave_size > 0 and nbins > self.wave_size:
            return self._run_waves(
                worker_fn, shared_arrays, params, bin_indices,
                desc, fail_value,
            )

        return self._run_pool(
            worker_fn, shared_arrays, params, bin_indices,
            desc, fail_value,
        )

    # ------------------------------------------------------------------
    # Wave-level orchestration
    # ------------------------------------------------------------------
    def _run_waves(self, worker_fn, shared_arrays, params, bin_indices,
                   desc, fail_value):
        nbins = len(bin_indices)
        n_waves = (nbins + self.wave_size - 1) // self.wave_size
        logger.info(
            "Wave processing: %d bins in %d waves of up to %d",
            nbins, n_waves, self.wave_size,
        )
        all_results = []
        for w in range(n_waves):
            s = w * self.wave_size
            e = min(s + self.wave_size, nbins)
            wave_desc = f"{desc} [wave {w + 1}/{n_waves}]"
            wave_results = self._run_pool(
                worker_fn, shared_arrays, params, bin_indices[s:e],
                wave_desc, fail_value,
            )
            all_results.extend(wave_results)
        return all_results

    # ------------------------------------------------------------------
    # Pool-level execution
    # ------------------------------------------------------------------
    def _run_pool(self, worker_fn, shared_arrays, params, bin_indices,
                  desc, fail_value):
        nbins = len(bin_indices)
        results = [fail_value] * nbins
        idx_map = {int(bi): i for i, bi in enumerate(bin_indices)}
        chunks = self._compute_chunks(bin_indices)

        t0 = time.time()
        try:
            if self._can_fork:
                self._run_fork(
                    worker_fn, shared_arrays, params, chunks,
                    results, idx_map, desc,
                )
            else:
                self._run_shm(
                    worker_fn, shared_arrays, params, chunks,
                    results, idx_map, desc,
                )
        except Exception:
            self._cleanup_shm()
            raise

        elapsed = time.time() - t0
        logger.info(
            "%s: %d bins in %.1fs using %d cores (%.1f bins/s)",
            desc, nbins, elapsed, self.ncpu, nbins / max(elapsed, 0.001),
        )
        return results

    # ------------------------------------------------------------------
    # Fork mode — opt-in via NGIST_USE_FORK=1 (risk of deadlock)
    # ------------------------------------------------------------------
    def _run_fork(self, worker_fn, shared_arrays, params, chunks,
                  results, idx_map, desc):
        ctx = mp.get_context("fork")
        with ctx.Pool(
            self.ncpu,
            initializer=_init_worker_fork,
            initargs=(shared_arrays, worker_fn, params),
        ) as pool:
            for chunk_results in tqdm(
                pool.imap_unordered(_dispatch_chunk, chunks),
                total=len(chunks),
                desc=desc,
                ascii=" #",
                unit="chunk",
            ):
                for bin_idx, result in chunk_results:
                    local_i = idx_map[bin_idx]
                    if result is not None:
                        results[local_i] = result

    # ------------------------------------------------------------------
    # SharedMemory mode (default on all platforms — safe, no disk I/O)
    # ------------------------------------------------------------------
    def _run_shm(self, worker_fn, shared_arrays, params, chunks,
                 results, idx_map, desc):
        shm_meta = {}
        try:
            for key, arr in shared_arrays.items():
                arr = np.ascontiguousarray(arr)
                shm = mp.shared_memory.SharedMemory(
                    create=True, size=max(arr.nbytes, 1),
                )
                np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
                shm_meta[key] = (shm.name, arr.shape, arr.dtype.str)
                self._shm_handles.append(shm)

            ctx = mp.get_context("spawn")
            with ctx.Pool(
                self.ncpu,
                initializer=_init_worker_shm,
                initargs=(shm_meta, worker_fn, params),
            ) as pool:
                for chunk_results in tqdm(
                    pool.imap_unordered(_dispatch_chunk, chunks),
                    total=len(chunks),
                    desc=desc,
                    ascii=" #",
                    unit="chunk",
                ):
                    for bin_idx, result in chunk_results:
                        local_i = idx_map[bin_idx]
                        if result is not None:
                            results[local_i] = result
        finally:
            self._cleanup_shm()

    def _cleanup_shm(self):
        """Release all SharedMemory resources."""
        for shm in self._shm_handles:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        self._shm_handles.clear()
