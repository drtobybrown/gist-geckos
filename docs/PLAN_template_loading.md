# Plan: Faster pPXF Startup (Template Loading & Consistency)

**General principle:** The faster option is the default (e.g. adaptive grid, parallelized template loading when added). If `NCPU` is not set in the config, the pipeline uses the maximum available in the environment (e.g. `os.cpu_count()`).

**Goals**
- Reduce time-to-first-pPXF on production cubes (template preparation and data load).
- Keep results **bitwise or numerically consistent** with current behaviour.
- Keep **ADAPTIVE_GRID as the default** in every module that uses templates (KIN, CONT, SFH, GAS).

**Out of scope**
- Changing pPXF fitting logic or adaptive/reduced grid behaviour.
- Changing default config values for ADAPTIVE_GRID/REDUCED_GRID (they stay as-is; adaptive remains default where already set).

---

## 1. Defaults: Adaptive grid in every module

**Current state**
- **KIN, CONT, SFH, GAS:** All have `ADAPTIVE_GRID : [4, 2, 1]` set in the workflow/tutorial MasterConfig (default = adaptive in every module).

**Target**
- **Config default:** Keep `ADAPTIVE_GRID : [4, 2, 1]` in all four modules in the reference MasterConfig; comments should say “Default” so it’s clear adaptive is the default.
- **Code:** No change to logic; all four modules already respect ADAPTIVE_GRID when present. No “fallback to full grid” unless the user comments out or removes the key.
- **Check:** After any change, run one full pipeline (e.g. NGC0000) with default config and confirm KIN, CONT, SFH, GAS all log use of adaptive (or reduced) grid where applicable.

**Verification**
- Grep/config check: KIN, CONT, SFH, GAS sections all contain `ADAPTIVE_GRID : [4, 2, 1]` in the default config file(s).
- Single pipeline run with default config; logs show adaptive/reduced for each module.

---

## 2. Template cache (main startup win)

**Idea**
- Prepare the stellar template library once per “configuration fingerprint” and write a cache file (e.g. HDF5 or FITS).
- On later runs (same config, same library), **load from cache** instead of opening hundreds of FITS files and re-running convolution + log-rebin.

**Cache key (fingerprint)**
Must uniquely describe the prepared template array and metadata. Suggested inputs:
- `TEMPLATE_DIR` + `LIBRARY` (resolved path or canonical form).
- `lmin`, `lmax`, `velscale`.
- LSF: paths or content hashes for **data LSF** and **template LSF** (e.g. `LSF_DATA`, `LSF_TEMP` / `lsf_MUSE-WFM`, `lsf_MILES`), and the module’s LSF usage (e.g. `getLSF(config, module_used)`).
- `TEMPLATE_SET` (e.g. `miles`).
- `sortInGrid` (True for SFH, False for KIN/CONT/GAS).
- `NORM_TEMP` (LIGHT/MASS) if it affects the returned arrays.

Optional: include a hash of the list of template filenames (or mtimes) so that adding/removing/updating a FITS file invalidates the cache.

**Cache location**
- **Preferred:** Scratch when available: `$SCRATCH_DIR/.ngist_template_cache` or `/scratch/.ngist_template_cache` (fast local storage). Uses `SCRATCH_DIR` env var or `/scratch` if writable.
- **Fallback:** Under `TEMPLATE_DIR/.prepared` so one cache is shared across runs that use the same library and LSF.

**Cache format**
- HDF5: one file per fingerprint (e.g. `sha256(fingerprint).h5`) with datasets `templates`, `logLam`, `lamRange`, and attributes or a small JSON for `nAges`, `nMetal`, `nAlpha`, `ncomb`, `logAge_grid`, `metal_grid`, `alpha_grid` if needed by callers.
- Ensure `prepareTemplates_Module` return signature is unchanged so KIN/CONT/SFH/GAS need no changes beyond calling a wrapper that “load or compute then return”.

**Implementation sketch**
- New helper (e.g. in `prepareTemplates/`): `prepareTemplates_Module_cached(config, lmin, lmax, velscale, LSF_Data, LSF_Templates, module_used, sortInGrid)`.
  - Build fingerprint from the inputs above (and optionally file list/mtime).
  - If cache file exists and is valid: load from cache, return same 11-tuple as `prepareTemplates_Module`.
  - Else: call existing `prepareTemplates_Module` (or call the backend `prepareSpectralTemplateLibrary` directly), then write cache, then return.
- **Call sites:** Replace `_prepareTemplates.prepareTemplates_Module(...)` with `_prepareTemplates.prepareTemplates_Module_cached(...)` in KIN, CONT, SFH, GAS (and any other modules that use it). Alternatively, add a config flag `CACHE_TEMPLATES: True` and only use cache when set, so behaviour is opt-in until verified.
- **Invalidation:** Cache valid only if fingerprint matches and cache file exists. No TTL; invalidate by deleting the cache file or changing config/library/LSF.

**Verification (consistency)**
- **Regression test:** For a fixed small setup (e.g. NGC0000, one bin or combined spectrum), run twice:
  - Run 1: no cache (delete cache if present) → get kinematics (and optionally CONT/SFH/GAS outputs).
  - Run 2: with cache (second run) → same outputs.
- Compare: `_kin.fits` (and optionally CONT/SFH/GAS) arrays **element-wise** (or with `np.allclose` for floats). Require match within floating-point tolerance (or bitwise if no randomness).
- Add a test (e.g. in `tests/`) that: (1) runs template prep with cache disabled/missing, (2) runs again with cache enabled, (3) asserts the two returned template arrays and metadata are the same (allclose or equal).
- Optional: run existing `test_adaptive_kin.py` and any full-pipeline test with cache on and confirm no change in results.

---

## 3. Parallel template loading (optional, smaller win)

**Idea**
- In `miles.py` (and similarly in other backends if needed), load and process template FITS files in **parallel** (e.g. `concurrent.futures.ProcessPoolExecutor` or `ThreadPoolExecutor` over I/O-bound work). Each worker: open one FITS, read, convolve, log_rebin, return one column (or a chunk).
- Main process assembles the full template matrix and grid metadata.

**Constraints**
- Must produce **identical** template array and metadata to the sequential version (same order, same normalisation).
- Avoid loading the whole library into memory in one go if the number of templates is very large (e.g. process in chunks or use shared memory for the output array).

**Verification**
- Unit test: run `prepareSpectralTemplateLibrary` sequentially and in parallel (same config); assert outputs match (allclose).
- No change to downstream results: same regression as for cache (run pipeline twice, compare outputs).

**Order**
- Implement **after** cache, so we only need to validate parallel vs sequential once; cache can then store the result of either path. **Default:** parallel loading on when implemented (faster option = default).

---

## 4. HDF5 read (optional)

**Idea**
- For the initial read of `_BinSpectra.hdf5` in KIN (and similar in CONT/SFH), ensure the slice `f['SPEC'][idx_lam, :]` is a single contiguous read and that chunking of `SPEC` is friendly to that access pattern. If the file is already written with chunks that match this slice, no code change may be needed beyond documenting it.
- If production cubes use very large bins × pixels, consider memory-mapping or streaming only the combined spectrum for the first run; this is a larger refactor and optional.

**Verification**
- Same as above: run with and without any change; compare kinematics (and other outputs) for consistency.

---

## 5. Implementation order

| Phase | Task | Verification |
|-------|------|--------------|
| **0** | **Defaults** – Set `ADAPTIVE_GRID : [4, 2, 1]` in CONT, SFH, GAS in the default MasterConfig(s). | Done. |
| **1** | **Cache key + write** – Define fingerprint (library, lmin, lmax, velscale, LSF, sortInGrid, NORM_TEMP). Implement cache write after `prepareSpectralTemplateLibrary`. | Done. `template_cache.write_cache`. |
| **2** | **Cache read** – In `prepareTemplates_Module` (or a cached wrapper), check cache first; if hit, load and return same 11-tuple. | Done. `template_cache.read_cache`; default `CACHE_TEMPLATES: True`. |
| **3** | **Integration** – KIN/CONT/SFH/GAS use `prepareTemplates_Module` which uses cache when `CACHE_TEMPLATES` True. | Run pipeline with cache on; second run uses cache. |
| **4** | **Regression test** – `test_template_cache_consistency.py`: roundtrip read/write + uncached vs cached equality. | Done. CI runs tests. |
| **5** | **(Optional) Parallel load** – Parallelize template loading in `miles.py`; keep output identical. | Not yet implemented. |

---

## 6. Config / code checklist (adaptive default)

- [x] **KIN:** `ADAPTIVE_GRID : [4, 2, 1]` set in default config.
- [x] **CONT:** `ADAPTIVE_GRID : [4, 2, 1]` set in default MasterConfig (comment: Default).
- [x] **SFH:** `ADAPTIVE_GRID : [4, 2, 1]` set in default MasterConfig (comment: Default).
- [x] **GAS:** `ADAPTIVE_GRID : [4, 2, 1]` set in default MasterConfig (comment: Default).
- [x] Code: No change to `build_grid_config` or when adaptive is used; “default” = adaptive in every module.

---

## 7. Success criteria

- **Correctness:** For the same input data and config, pipeline outputs (kinematics and, where applicable, CONT/SFH/GAS) are unchanged when using template cache (and, if added, parallel loading). Existing adaptive-grid tests (e.g. `test_adaptive_kin.py`) still pass.
- **Default:** In the reference config(s), KIN, CONT, SFH, and GAS all have `ADAPTIVE_GRID : [4, 2, 1]` as the default.
- **Performance:** On a production-sized run, time from pipeline start to first pPXF call is significantly reduced when cache is used (target: e.g. &gt;50% reduction where template load dominates).
- **CI:** New or updated tests run in CI and ensure template cache (and optional parallel path) do not change results.
