import logging
import os

import fitsio
import h5py
import numpy as np
from astropy.io import fits
from ppxf.ppxf_util import log_rebin
from printStatus import printStatus


def get_input_bunit(config):
    """
    Read BUNIT from the input cube FITS if present (any of ext 0, 1, 2).
    Used to propagate flux/data units to HDF5 metadata. Returns None if not found.
    """
    try:
        with fitsio.FITS(config["GENERAL"]["INPUT"]) as f:
            for ext in (0, 1, 2):
                if ext >= len(f):
                    continue
                hdr = f[ext].read_header()
                if "BUNIT" in hdr:
                    return str(hdr["BUNIT"]).strip()
    except Exception:
        pass
    return None


# ====================================================================== #
#  Public entry point                                                      #
# ====================================================================== #

def prepSpectra(config, cube):
    """
    Prepare spectra for analysis modules.

    If *cube* is a ``LazyCube`` (out-of-core mode), spectra are streamed
    from the FITS file in spatial tiles — never loading the full cube.
    Otherwise, the legacy in-memory code path is used.
    """
    if hasattr(cube, "tile_iterator"):
        return _prepSpectra_streaming(config, cube)
    else:
        return _prepSpectra_legacy(config, cube)


# ====================================================================== #
#  Streaming (out-of-core) implementation                                  #
# ====================================================================== #

def _prepSpectra_streaming(config, cube):
    """Process spectra in spatial tiles — O(tile) memory, not O(cube)."""

    # --- Read mask and bin table ---
    maskfile = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_mask.fits"
    )
    with fitsio.FITS(maskfile) as mhdu:
        mask = mhdu[1].read()["MASK"]
    idxUnmasked = np.where(mask == 0)[0]

    tablefile = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_table.fits"
    )
    with fitsio.FITS(tablefile) as thdu:
        data = thdu[1].read()
        binNum = data["BIN_ID"][idxUnmasked]

    # Full spaxel → bin mapping (-1 = masked)
    nspaxels = cube.nspaxels
    spaxel_to_bin = np.full(nspaxels, -1, dtype=np.int64)
    spaxel_to_bin[idxUnmasked] = binNum

    ubins = np.unique(binNum)
    nbins = len(ubins)
    # Map bin IDs to contiguous 0..nbins-1
    bin_remap = np.full(int(ubins.max()) + 1, -1, dtype=np.int64)
    bin_remap[ubins] = np.arange(nbins)

    wave = cube["wave"]
    nwave = len(wave)
    velscale = config["PREPARE_SPECTRA"]["VELSCALE"]
    bunit = cube.get("bunit")
    write_all = config["GAS"]["LEVEL"] == "SPAXEL"

    # --- Probe log_rebin to get output grid ---
    wave_range = np.array([np.amin(wave), np.amax(wave)])
    dummy = np.ones(nwave, dtype=np.float64)
    probe, logLam, _ = log_rebin(wave_range, dummy, velscale=velscale)
    npix_log = len(logLam)

    # --- Allocate accumulators ---
    # Linear binned spectra
    bin_sum_spec_lin = np.zeros((nwave, nbins), dtype=np.float64)
    bin_sum_err_lin = np.zeros((nwave, nbins), dtype=np.float64)
    # Log-rebinned binned spectra
    bin_sum_spec_log = np.zeros((npix_log, nbins), dtype=np.float64)
    bin_sum_err_log = np.zeros((npix_log, nbins), dtype=np.float64)

    logging.info(
        "Streaming prepareSpectra: %d spaxels, %d bins, %d unmasked, "
        "accumulators ~%.0f MB",
        nspaxels, nbins, len(idxUnmasked),
        (bin_sum_spec_lin.nbytes + bin_sum_err_lin.nbytes
         + bin_sum_spec_log.nbytes + bin_sum_err_log.nbytes) / 1e6,
    )

    # --- Open AllSpectra HDF5 for incremental writes if needed ---
    allspec_path = None
    allspec_file = None
    allspec_spec = None
    allspec_espec = None

    if write_all:
        allspec_path = (
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_AllSpectra.hdf5"
        )
        printStatus.running("Preparing: " + config["GENERAL"]["RUN_ID"] + "_AllSpectra.hdf5")
        allspec_file = h5py.File(allspec_path, "w")
        # Chunked for efficient column-block reads later (GAS SPAXEL)
        chunk_cols = min(1024, nspaxels)
        allspec_spec = allspec_file.create_dataset(
            "SPEC", shape=(npix_log, nspaxels), dtype="float64",
            chunks=(npix_log, chunk_cols),
        )
        allspec_espec = allspec_file.create_dataset(
            "ESPEC", shape=(npix_log, nspaxels), dtype="float64",
            chunks=(npix_log, chunk_cols),
        )

    # --- Stream tiles ---
    printStatus.running("Streaming tiles: log-rebinning + spatial binning")
    tiles_processed = 0

    for indices, spec_tile, error_tile in cube.tile_iterator():
        n_tile = len(indices)

        # ---- Linear binning accumulation ----
        tile_bins = spaxel_to_bin[indices]
        valid = tile_bins >= 0
        if valid.any():
            v_bins = bin_remap[tile_bins[valid]]
            v_spec = np.nan_to_num(spec_tile[:, valid], nan=0.0)
            v_err = np.nan_to_num(error_tile[:, valid], nan=0.0)
            # Accumulate per-bin
            np.add.at(bin_sum_spec_lin, (slice(None), v_bins), v_spec)
            np.add.at(bin_sum_err_lin, (slice(None), v_bins), v_err)

        # ---- Log-rebin each spaxel in this tile ----
        log_tile = np.full((npix_log, n_tile), np.nan, dtype=np.float64)
        log_err_tile = np.full((npix_log, n_tile), np.nan, dtype=np.float64)

        for j in range(n_tile):
            try:
                log_s, _, _ = log_rebin(wave_range, spec_tile[:, j], velscale=velscale)
                log_tile[:, j] = log_s
            except Exception:
                log_tile[:, j] = np.nan

            try:
                log_e, _, _ = log_rebin(wave_range, error_tile[:, j], velscale=velscale)
                log_err_tile[:, j] = log_e
            except Exception:
                log_err_tile[:, j] = np.nan

        # ---- Write AllSpectra (all spaxels including masked) ----
        if write_all and allspec_spec is not None:
            allspec_spec[:, indices[0] : indices[-1] + 1] = log_tile
            allspec_espec[:, indices[0] : indices[-1] + 1] = log_err_tile

        # ---- Log-rebinned binning accumulation ----
        if valid.any():
            v_bins = bin_remap[tile_bins[valid]]
            v_log = np.nan_to_num(log_tile[:, valid], nan=0.0)
            v_log_err = np.nan_to_num(log_err_tile[:, valid], nan=0.0)
            np.add.at(bin_sum_spec_log, (slice(None), v_bins), v_log)
            np.add.at(bin_sum_err_log, (slice(None), v_bins), v_log_err)

        # Free tile memory explicitly
        del spec_tile, error_tile, log_tile, log_err_tile

        tiles_processed += 1
        if tiles_processed % 10 == 0:
            logging.info("  Processed %d tiles (%d spaxels so far)",
                         tiles_processed, indices[-1] + 1)

    printStatus.updateDone("Streaming tiles: log-rebinning + spatial binning", progressbar=True)

    # --- Finalise AllSpectra HDF5 ---
    if allspec_file is not None:
        allspec_file.create_dataset("LOGLAM", data=logLam)
        allspec_file.attrs["VELSCALE"] = velscale
        allspec_file.attrs["CRPIX1"] = 1.0
        allspec_file.attrs["CRVAL1"] = logLam[0]
        allspec_file.attrs["CDELT1"] = logLam[1] - logLam[0]
        if bunit is not None:
            allspec_file.attrs["BUNIT"] = bunit
            allspec_spec.attrs["BUNIT"] = bunit
            allspec_espec.attrs["BUNIT"] = bunit
        allspec_file.close()
        printStatus.updateDone(
            "Writing: " + config["GENERAL"]["RUN_ID"] + "_AllSpectra.hdf5"
        )
        logging.info("Wrote: " + allspec_path)

    # --- Finalise bin error (sqrt of summed variance) ---
    bin_err_lin = np.sqrt(bin_sum_err_lin)
    bin_err_log = np.sqrt(bin_sum_err_log)

    # --- Save linear BinSpectra ---
    saveBinSpectra(config, bin_sum_spec_lin, bin_err_lin, velscale, wave, "lin", bunit=bunit)

    # --- Save log BinSpectra ---
    saveBinSpectra(config, bin_sum_spec_log, bin_err_log, velscale, logLam, "log", bunit=bunit)

    logging.info("Streaming prepareSpectra complete.")
    return None


# ====================================================================== #
#  Legacy (in-memory) implementation — preserved for backward compat       #
# ====================================================================== #

def _prepSpectra_legacy(config, cube):
    """Original in-memory implementation for cubes returned as plain dicts."""

    # Read maskfile
    maskfile = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_mask.fits"
    )
    with fitsio.FITS(maskfile) as mhdu:
        mask = mhdu[1].read()["MASK"]
    idxUnmasked = np.where(mask == 0)[0]

    # Read binning pattern
    tablefile = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_table.fits"
    )
    with fitsio.FITS(tablefile) as tablehdu:
        data = tablehdu[1].read()
        binNum = data["BIN_ID"][idxUnmasked]

    # Apply spatial bins to linear spectra
    bin_data, bin_error, bin_flux = applySpatialBins(
        binNum,
        cube["spec"][:, idxUnmasked],
        cube["error"][:, idxUnmasked],
        config["PREPARE_SPECTRA"]["VELSCALE"],
        "lin",
    )
    saveBinSpectra(
        config, bin_data, bin_error,
        config["PREPARE_SPECTRA"]["VELSCALE"],
        cube["wave"], "lin",
        bunit=cube.get("bunit"),
    )

    # Log-rebin spectra
    log_spec, log_error, logLam = log_rebinning(config, cube)

    # Save all log-rebinned spectra only if running in full spaxel mode
    if config["GAS"]["LEVEL"] == "SPAXEL":
        saveAllSpectra(
            config, log_spec, log_error,
            config["PREPARE_SPECTRA"]["VELSCALE"], logLam,
            bunit=cube.get("bunit"),
        )

    # Apply bins to log spectra
    bin_data, bin_error, bin_flux = applySpatialBins(
        binNum,
        log_spec[:, idxUnmasked],
        log_error[:, idxUnmasked],
        config["PREPARE_SPECTRA"]["VELSCALE"],
        "log",
    )
    saveBinSpectra(
        config, bin_data, bin_error,
        config["PREPARE_SPECTRA"]["VELSCALE"],
        logLam, "log",
        bunit=cube.get("bunit"),
    )

    return None


# ====================================================================== #
#  Shared helper functions                                                 #
# ====================================================================== #

def log_rebinning(config, cube):
    """Logarithmically rebin spectra and error spectra (in-memory path)."""
    printStatus.running("Log-rebinning the spectra")
    log_spec, logLam = run_log_rebinning(
        cube["spec"], config["PREPARE_SPECTRA"]["VELSCALE"],
        len(cube["x"]), cube["wave"],
    )
    printStatus.updateDone("Log-rebinning the spectra", progressbar=True)
    logging.info("Log-rebinned the spectra")

    printStatus.running("Log-rebinning the error spectra")
    log_error, _ = run_log_rebinning(
        cube["error"], config["PREPARE_SPECTRA"]["VELSCALE"],
        len(cube["x"]), cube["wave"],
    )
    printStatus.updateDone("Log-rebinning the error spectra", progressbar=True)
    logging.info("Log-rebinned the error spectra")

    return (log_spec, log_error, logLam)


def run_log_rebinning(
    binned_data, velocity_scale, num_bins, wavelength, chunk_size=1000
):
    """Perform log-rebinning on the given binned_data (in-memory path)."""
    wavelength_range = np.array([np.amin(wavelength), np.amax(wavelength)])

    ssp_new, log_lam, _ = log_rebin(
        wavelength_range, binned_data[:, 0], velscale=velocity_scale
    )
    log_binned_data = np.zeros([len(log_lam), num_bins])

    for i in range(0, num_bins, chunk_size):
        for j in range(i, min(i + chunk_size, num_bins)):
            try:
                ssp_new, _, _ = log_rebin(
                    wavelength_range, binned_data[:, j], velscale=velocity_scale
                )
                log_binned_data[:, j] = ssp_new
            except Exception:
                log_binned_data[:, j] = np.zeros(len(log_lam))
                log_binned_data[:, j][:] = np.nan

    return (log_binned_data, log_lam)


def saveAllSpectra(config, log_spec, log_error, velscale, logLam, bunit=None):
    """Save all logarithmically rebinned spectra to file."""
    if bunit is None:
        bunit = get_input_bunit(config)

    outfn_spectra = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_AllSpectra.hdf5"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_AllSpectra.hdf5")

    with h5py.File(outfn_spectra, "w") as f:
        nspaxels = log_spec.shape[1]
        chunk_cols = min(1024, nspaxels)
        spec_dset = f.create_dataset(
            "SPEC", shape=log_spec.shape, dtype=log_spec.dtype,
            chunks=(log_spec.shape[0], chunk_cols),
        )
        espec_dset = f.create_dataset(
            "ESPEC", shape=log_error.shape, dtype=log_error.dtype,
            chunks=(log_error.shape[0], chunk_cols),
        )

        chunk_size = 1000
        for i in range(0, len(log_spec), chunk_size):
            spec_dset[i : i + chunk_size] = log_spec[i : i + chunk_size]
            espec_dset[i : i + chunk_size] = log_error[i : i + chunk_size]

        f.create_dataset("LOGLAM", data=logLam)
        f.attrs["VELSCALE"] = velscale
        f.attrs["CRPIX1"] = 1.0
        f.attrs["CRVAL1"] = logLam[0]
        f.attrs["CDELT1"] = logLam[1] - logLam[0]
        if bunit is not None:
            f.attrs["BUNIT"] = bunit
            spec_dset.attrs["BUNIT"] = bunit
            espec_dset.attrs["BUNIT"] = bunit

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_AllSpectra.hdf5"
    )
    logging.info("Wrote: " + outfn_spectra)


def saveBinSpectra(config, log_spec, log_error, velscale, logLam, flag, bunit=None):
    """Save spatially binned spectra and error spectra to disk."""
    if bunit is None:
        bunit = get_input_bunit(config)

    outfile = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])

    if flag == "log":
        outfn_spectra = outfile + "_BinSpectra.hdf5"
        printStatus.running(
            "Writing: " + config["GENERAL"]["RUN_ID"] + "_BinSpectra.hdf5"
        )
    elif flag == "lin":
        outfn_spectra = outfile + "_BinSpectra_linear.hdf5"
        printStatus.running(
            "Writing: " + config["GENERAL"]["RUN_ID"] + "_BinSpectra_linear.hdf5"
        )

    with h5py.File(outfn_spectra, "w") as f:
        nbins = log_spec.shape[1]
        chunk_cols = min(1024, nbins)
        spec_dset = f.create_dataset(
            "SPEC", shape=log_spec.shape, dtype=log_spec.dtype,
            chunks=(log_spec.shape[0], chunk_cols),
        )
        espec_dset = f.create_dataset(
            "ESPEC", shape=log_error.shape, dtype=log_error.dtype,
            chunks=(log_error.shape[0], chunk_cols),
        )

        chunk_size = 1000
        for i in range(0, len(log_spec), chunk_size):
            spec_dset[i : i + chunk_size] = log_spec[i : i + chunk_size]
            espec_dset[i : i + chunk_size] = log_error[i : i + chunk_size]

        f.create_dataset("LOGLAM", data=logLam)
        f.attrs["VELSCALE"] = velscale
        f.attrs["CRPIX1"] = 1.0
        f.attrs["CRVAL1"] = logLam[0]
        f.attrs["CDELT1"] = logLam[1] - logLam[0]
        if bunit is not None:
            f.attrs["BUNIT"] = bunit
            spec_dset.attrs["BUNIT"] = bunit
            espec_dset.attrs["BUNIT"] = bunit

    if flag == "log":
        printStatus.updateDone(
            "Writing: " + config["GENERAL"]["RUN_ID"] + "_BinSpectra.hdf5"
        )
    elif flag == "lin":
        printStatus.updateDone(
            "Writing: " + config["GENERAL"]["RUN_ID"] + "_BinSpectra_linear.hdf5"
        )
    logging.info("Wrote: " + outfn_spectra)


def applySpatialBins(binNum, spec, espec, velscale, flag):
    """Apply the spatial binning scheme to the spectra (in-memory path)."""
    printStatus.running("Applying the spatial bins to " + flag + "-data")
    bin_data, bin_error, bin_flux = spatialBinning(binNum, spec, espec)
    printStatus.updateDone(
        "Applying the spatial bins to " + flag + "-data", progressbar=True
    )
    logging.info("Applied spatial bins to " + flag + "-data")
    return (bin_data, bin_error, bin_flux)


def spatialBinning(binNum, spec, error):
    """Spectra belonging to the same spatial bin are added (vectorized per bin)."""
    ubins = np.unique(binNum)
    nbins = len(ubins)
    npix = spec.shape[0]
    bin_idx = np.searchsorted(ubins, binNum)
    bin_data = np.zeros([npix, nbins])
    bin_error = np.zeros([npix, nbins])
    bin_flux = np.zeros(nbins)

    for i in range(nbins):
        k = bin_idx == i
        av_spec = np.nansum(spec[:, k], axis=1)
        av_err_spec = np.sqrt(np.nansum(error[:, k], axis=1))
        bin_data[:, i] = np.ravel(av_spec)
        bin_error[:, i] = np.ravel(av_err_spec)
        bin_flux[i] = np.mean(av_spec)
        printStatus.progressBar(i + 1, nbins, barLength=50)

    return (bin_data, bin_error, bin_flux)
