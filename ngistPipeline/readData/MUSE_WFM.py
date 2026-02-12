import logging
import os

import extinction
import fitsio
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from ngistPipeline.auxiliary.lazy_cube import LazyCube
from ngistPipeline.readData import der_snr as der_snr
from printStatus import printStatus


# ======================================
# Helper routine from PHANGS DAP
# ======================================
def reshape_extintion_curve(extinction_curve, cube):
    extra_dims = cube.ndim - extinction_curve.ndim
    new_shape = extinction_curve.shape + (1,) * extra_dims
    reshaped_extinction_curve = extinction_curve.reshape(new_shape)
    return reshaped_extinction_curve


# ======================================
# Routine to load MUSE-cubes (out-of-core)
# ======================================
def readCube(config):
    """Read a MUSE-WFM cube and return a memory-efficient ``LazyCube``.

    Only small per-spaxel statistics (x, y, signal, noise, snr) are kept in
    RAM.  The full spectral arrays are never loaded; downstream modules
    access them on demand via ``cube.tile_iterator()``.

    The streaming pass also computes the defunct-spaxel mask (all-NaN or
    non-positive median), eliminating the need for ``spatialMasking`` to
    read full spectra.
    """
    loggingBlanks = (len(os.path.splitext(os.path.basename(__file__))[0]) + 33) * " "

    # Read MUSE-cube header
    printStatus.running("Reading the MUSE-WFM cube (out-of-core)")
    logging.info("Reading the MUSE-WFM cube: " + config["GENERAL"]["INPUT"])

    with fitsio.FITS(config["GENERAL"]["INPUT"]) as fits_obj:
        if len(fits_obj) == 1:
            ihdu = 0
            printStatus.running("data in first HDU")
        else:
            ihdu = 1

        hdr = fits_obj[ihdu].read_header()
        s = (hdr["NAXIS3"], hdr["NAXIS2"], hdr["NAXIS1"])  # (nwave, ny, nx)
        ny, nx = s[1], s[2]
        nspaxels = ny * nx

        # Build astropy WCS header
        hdr_dict = {k: hdr[k] for k in hdr.keys()}
        astro_hdr = fits.Header(hdr_dict)
        wcshdr = WCS(astro_hdr).to_header()

        # Wavelength axis and trim
        if "CD3_3" not in hdr:
            cdelt2 = hdr["CDELT3"]
            cdelt3 = hdr["CDELT3"]
        else:
            cdelt2 = hdr["CD2_2"]
            cdelt3 = hdr["CD3_3"]

        wave_full = hdr["CRVAL3"] + (np.arange(s[0])) * cdelt3
        wave_full = wave_full / (1 + config["GENERAL"]["REDSHIFT"])
        lmin = config["READ_DATA"]["LMIN_TOT"]
        lmax = config["READ_DATA"]["LMAX_TOT"]
        idx = np.where(np.logical_and(wave_full >= lmin, wave_full <= lmax))[0]

        if len(idx) == 0:
            logging.warning("No wavelength pixels in [LMIN_TOT, LMAX_TOT]")
            # Return an empty LazyCube
            wave = np.array([], dtype=np.float64)
            x = np.zeros(nspaxels, dtype=np.float64)
            y = np.zeros(nspaxels, dtype=np.float64)
            return LazyCube(
                fits_path=config["GENERAL"]["INPUT"],
                data_ext=ihdu, error_ext=None,
                wave_start=0, wave_end=0,
                shape_yx=(ny, nx),
                x=x, y=y, wave=wave,
                signal=np.zeros(nspaxels), noise=np.zeros(nspaxels),
                snr=np.zeros(nspaxels), pixelsize=cdelt2 * 3600.0,
                wcshdr=wcshdr, bunit=None,
                defunct_mask=np.ones(nspaxels, dtype=bool),
            )

        wave_start = int(idx[0])
        wave_end = int(idx[-1] + 1)
        wave = wave_full[idx]
        nwave = len(wave)
        has_error = len(fits_obj) >= 3

    # -----------------------------------------------------------------
    # Galactic extinction correction curve (computed once, applied per tile)
    # -----------------------------------------------------------------
    extinction_curve = None
    if config["READ_DATA"]["EBmV"] is not None:
        Rv = 3.1
        Av = Rv * config["READ_DATA"]["EBmV"]
        ones = np.ones_like(wave)
        extinction_curve = extinction.apply(extinction.ccm89(wave, Av, Rv), ones)
        # extinction_curve is the *multiplicative* correction: corrected = spec / curve

    # -----------------------------------------------------------------
    # Spatial coordinates (small — always in memory)
    # -----------------------------------------------------------------
    origin = [
        float(config["READ_DATA"]["ORIGIN"].split(",")[0].strip()),
        float(config["READ_DATA"]["ORIGIN"].split(",")[1].strip()),
    ]
    xaxis = (np.arange(nx) - origin[0]) * cdelt2 * 3600.0
    yaxis = (np.arange(ny) - origin[1]) * cdelt2 * 3600.0
    xgrid, ygrid = np.meshgrid(xaxis, yaxis)
    x = xgrid.ravel()
    y = ygrid.ravel()
    pixelsize = cdelt2 * 3600.0

    logging.info(
        "Extracting spatial information:\n"
        + loggingBlanks
        + "* Spatial coordinates are centred to "
        + str(origin)
        + "\n"
        + loggingBlanks
        + "* Spatial pixelsize is "
        + str(pixelsize)
    )
    logging.info(
        "Shortening spectra to the wavelength range from "
        + str(config["READ_DATA"]["LMIN_TOT"])
        + "A to "
        + str(config["READ_DATA"]["LMAX_TOT"])
        + "A."
    )

    # -----------------------------------------------------------------
    # SNR wavelength sub-range (indices within the trimmed wave array)
    # -----------------------------------------------------------------
    idx_snr = np.where(
        np.logical_and(
            wave >= config["READ_DATA"]["LMIN_SNR"],
            wave <= config["READ_DATA"]["LMAX_SNR"],
        )
    )[0]

    # -----------------------------------------------------------------
    # Streaming pass: compute per-spaxel stats + defunct mask
    # -----------------------------------------------------------------
    signal = np.empty(nspaxels, dtype=np.float64)
    noise = np.empty(nspaxels, dtype=np.float64)
    snr = np.empty(nspaxels, dtype=np.float64)
    defunct_mask = np.zeros(nspaxels, dtype=bool)
    der_snr_noise = None  # only populated when no error extension

    if not has_error:
        der_snr_noise = np.zeros(nspaxels, dtype=np.float64)

    # Tile size: ~10 000 spaxels per tile
    rows_per_tile = max(1, 10_000 // nx)

    ext_div = None
    if extinction_curve is not None:
        ext_div = extinction_curve.reshape(-1, 1)

    logging.info(
        "Streaming cube in tiles of %d rows (%d spaxels) to compute stats",
        rows_per_tile, rows_per_tile * nx,
    )

    with fitsio.FITS(config["GENERAL"]["INPUT"]) as fits_obj:
        for y_start in range(0, ny, rows_per_tile):
            y_end = min(y_start + rows_per_tile, ny)
            n_tile = (y_end - y_start) * nx
            sp_start = y_start * nx
            sp_end = sp_start + n_tile

            # Read flux tile → (nwave, n_tile)
            data_3d = fits_obj[ihdu][wave_start:wave_end, y_start:y_end, :]
            spec_tile = np.asarray(data_3d, dtype=np.float64).reshape(nwave, n_tile)

            # Read or estimate error tile
            if has_error:
                err_3d = fits_obj[2][wave_start:wave_end, y_start:y_end, :]
                error_tile = np.asarray(err_3d, dtype=np.float64).reshape(nwave, n_tile)
            else:
                # Estimate noise per spaxel with DER_SNR
                tile_noise = der_snr.der_snr_2d(spec_tile)
                der_snr_noise[sp_start:sp_end] = tile_noise
                error_tile = np.broadcast_to(
                    tile_noise.reshape(1, -1), spec_tile.shape
                ).copy()

            # Apply extinction correction
            if ext_div is not None:
                np.divide(spec_tile, ext_div, out=spec_tile)
                np.divide(error_tile, ext_div, out=error_tile)

            # ---- Per-spaxel statistics (SNR range) ----
            spec_snr = spec_tile[idx_snr, :]
            err_snr = error_tile[idx_snr, :]

            signal[sp_start:sp_end] = np.nanmedian(spec_snr, axis=0)
            noise[sp_start:sp_end] = np.sqrt(np.nanmedian(err_snr, axis=0))
            snr[sp_start:sp_end] = np.nanmedian(
                spec_snr / np.sqrt(err_snr), axis=0
            )

            # ---- Defunct spaxel detection ----
            all_nan = np.all(np.isnan(spec_tile), axis=0)
            median_nonpos = np.nanmedian(spec_tile, axis=0) <= 0.0
            defunct_mask[sp_start:sp_end] = all_nan | median_nonpos

    logging.info(
        "Computing the signal-to-noise ratio in the wavelength range from "
        + str(config["READ_DATA"]["LMIN_SNR"])
        + "A to "
        + str(config["READ_DATA"]["LMAX_SNR"])
        + "A."
    )

    # BUNIT metadata
    with fitsio.FITS(config["GENERAL"]["INPUT"]) as fits_obj:
        hdr = fits_obj[ihdu].read_header()
    bunit = hdr.get("BUNIT")
    if bunit is not None:
        bunit = str(bunit).strip()

    # -----------------------------------------------------------------
    # Build the LazyCube
    # -----------------------------------------------------------------
    cube = LazyCube(
        fits_path=config["GENERAL"]["INPUT"],
        data_ext=ihdu,
        error_ext=2 if has_error else None,
        wave_start=wave_start,
        wave_end=wave_end,
        shape_yx=(ny, nx),
        extinction_curve=extinction_curve,
        der_snr_noise=der_snr_noise,
        # small arrays:
        x=x,
        y=y,
        wave=wave,
        signal=signal,
        noise=noise,
        snr=snr,
        pixelsize=pixelsize,
        wcshdr=wcshdr,
        bunit=bunit,
        defunct_mask=defunct_mask,
    )

    # DEBUG mode: restrict to one central row
    if config["READ_DATA"]["DEBUG"] == True:
        logging.info(
            "DEBUG mode is activated. Only one line of spaxels is used."
        )
        cube.apply_debug(ny, nx)

    printStatus.updateDone(
        "Done reading " + str(cube.nspaxels) + " spectra from the MUSE-WFM cube (out-of-core)"
    )
    logging.info(
        "Finished reading the MUSE cube! %d spaxels (out-of-core, peak RSS ~%.0f MB)",
        cube.nspaxels,
        (signal.nbytes + noise.nbytes + snr.nbytes + defunct_mask.nbytes
         + x.nbytes + y.nbytes + wave.nbytes) / 1e6,
    )

    return cube
