"""
Template cache: persist prepared stellar templates (HDF5) keyed by config fingerprint
so later runs skip loading hundreds of FITS and re-convolving. Default on (faster).
Cache is written to scratch (SCRATCH_DIR or /scratch) when available, else TEMPLATE_DIR/.prepared.
"""
import hashlib
import logging
import os

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


def _build_fingerprint(config, lmin, lmax, velscale, module_used, sortInGrid):
    """Build a string that uniquely identifies this template preparation request."""
    template_dir = os.path.realpath(
        os.path.join(
            config["GENERAL"].get("TEMPLATE_DIR", ""),
            config[module_used].get("LIBRARY", ""),
        )
    )
    config_dir = config["GENERAL"].get("CONFIG_DIR", "")
    lsf_data = config["GENERAL"].get("LSF_DATA", "")
    lsf_temp = config[module_used].get("LSF_TEMP", "")
    template_set = config[module_used].get("TEMPLATE_SET", "miles")
    norm_temp = config[module_used].get("NORM_TEMP", "LIGHT")
    # Include redshift so LSF wavelength scaling is reflected
    redshift = config["GENERAL"].get("REDSHIFT", 0)
    parts = (
        template_dir,
        str(lmin),
        str(lmax),
        str(velscale),
        module_used,
        config_dir,
        lsf_data,
        lsf_temp,
        template_set,
        str(sortInGrid),
        norm_temp,
        str(redshift),
    )
    return "|".join(parts)


def _fingerprint_hash(fingerprint):
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:32]


def get_cache_dir(config):
    """
    Prefer scratch if available (fast local storage); else TEMPLATE_DIR/.prepared.
    Uses SCRATCH_DIR env var or /scratch, with subdir .ngist_template_cache.
    """
    scratch = os.environ.get("SCRATCH_DIR", "/scratch")
    if scratch and os.path.exists(scratch) and os.access(scratch, os.W_OK):
        return os.path.join(scratch, ".ngist_template_cache")
    base = config["GENERAL"].get("TEMPLATE_DIR", "")
    return os.path.join(base, ".prepared")


def get_cache_path(config, fingerprint):
    """Path to the HDF5 cache file for this fingerprint."""
    cache_dir = get_cache_dir(config)
    name = _fingerprint_hash(fingerprint) + ".h5"
    return os.path.join(cache_dir, name)


def read_cache(path):
    """
    Load prepared templates from HDF5. Returns 11-tuple as from prepareTemplates_Module
    or None if read fails / file missing.
    """
    if h5py is None or not os.path.isfile(path):
        return None
    try:
        with h5py.File(path, "r") as f:
            templates = np.asarray(f["templates"])
            logLam = np.asarray(f["logLam"])
            lamRange = np.asarray(f["lamRange"])
            ntemplates = int(f.attrs["ntemplates"])
            ncomb = int(f.attrs["ncomb"])
            nAges = int(f.attrs["nAges"])
            nMetal = int(f.attrs["nMetal"])
            nAlpha = int(f.attrs["nAlpha"])
            sortInGrid = bool(f.attrs.get("sortInGrid", False))
            if sortInGrid and "logAge_grid" in f:
                logAge_grid = np.asarray(f["logAge_grid"])
                metal_grid = np.asarray(f["metal_grid"])
                alpha_grid = np.asarray(f["alpha_grid"])
            else:
                logAge_grid = np.nan
                metal_grid = np.nan
                alpha_grid = np.nan
        lamRange_list = [float(lamRange[0]), float(lamRange[1])]
        return (
            templates,
            lamRange_list,
            logLam,
            ntemplates,
            logAge_grid,
            metal_grid,
            alpha_grid,
            ncomb,
            nAges,
            nMetal,
            nAlpha,
        )
    except Exception as e:
        logging.warning("Template cache read failed (%s): %s", path, e)
        return None


def write_cache(path, result):
    """Write the 11-tuple from prepareSpectralTemplateLibrary to HDF5."""
    if h5py is None:
        return
    (
        templates,
        lamRange_spmod,
        logLam2,
        ntemplates,
        logAge_grid,
        metal_grid,
        alpha_grid,
        ncomb,
        nAges,
        nMetal,
        nAlpha,
    ) = result
    try:
        cache_dir = os.path.dirname(path)
        os.makedirs(cache_dir, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.create_dataset("templates", data=templates, compression="gzip")
            f.create_dataset("logLam", data=logLam2)
            f.create_dataset("lamRange", data=np.asarray(lamRange_spmod))
            f.attrs["ntemplates"] = ntemplates
            f.attrs["ncomb"] = ncomb
            f.attrs["nAges"] = nAges
            f.attrs["nMetal"] = nMetal
            f.attrs["nAlpha"] = nAlpha
            sortInGrid = not (
                np.isscalar(logAge_grid)
                and (logAge_grid is np.nan or (isinstance(logAge_grid, float) and np.isnan(logAge_grid)))
            )
            f.attrs["sortInGrid"] = sortInGrid
            if sortInGrid and hasattr(logAge_grid, "shape"):
                f.create_dataset("logAge_grid", data=logAge_grid, compression="gzip")
                f.create_dataset("metal_grid", data=metal_grid, compression="gzip")
                f.create_dataset("alpha_grid", data=alpha_grid, compression="gzip")
        logging.info("Wrote template cache: %s", path)
    except Exception as e:
        logging.warning("Template cache write failed (%s): %s", path, e)
