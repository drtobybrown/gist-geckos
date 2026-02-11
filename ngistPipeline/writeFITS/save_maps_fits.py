#!/usr/bin/env python

import logging
import optparse
import os
import warnings

import datetime
import extinction
import fitsio
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import CubicSpline
from astropy import units as u
from astropy.wcs import WCS
from printStatus import printStatus

from ngistPipeline.utils.wcs_utils import (diagonal_wcs_to_cdelt,
                                          strip_wcs_from_header)

warnings.filterwarnings("ignore")


def _load_input_spectra_trimmed(config):
    """
    Load only the wavelength-trimmed spectra from the input cube (no variance,
    no SNR). Used by saveContLineCube to avoid a full readCube() and reduce peak RAM.
    Returns (spec_2d, wave_1d) in rest frame, trimmed to LMIN_TOT..LMAX_TOT.
    """
    with fitsio.FITS(config["GENERAL"]["INPUT"]) as fits_obj:
        ihdu = 0 if len(fits_obj) == 1 else 1
        hdr = fits_obj[ihdu].read_header()
        s = (hdr["NAXIS3"], hdr["NAXIS2"], hdr["NAXIS1"])
        if "CD3_3" not in hdr:
            cdelt3 = hdr["CDELT3"]
        else:
            cdelt3 = hdr["CD3_3"]
        wave_full = hdr["CRVAL3"] + (np.arange(s[0])) * cdelt3
        wave_full = wave_full / (1 + config["GENERAL"]["REDSHIFT"])
        lmin = config["READ_DATA"]["LMIN_TOT"]
        lmax = config["READ_DATA"]["LMAX_TOT"]
        idx = np.where(np.logical_and(wave_full >= lmin, wave_full <= lmax))[0]

        if len(idx) > 0:
            start = idx[0]
            end = idx[-1] + 1
            data_slice = fits_obj[ihdu][start:end, :, :]
            data_slice = np.asarray(data_slice, dtype=np.float64)
            spec = np.reshape(data_slice, [len(idx), s[1] * s[2]])
        else:
            spec = np.zeros((0, s[1] * s[2]))

    wave = wave_full[idx]
    if config["READ_DATA"]["EBmV"] is not None:
        Rv = 3.1
        Av = Rv * config["READ_DATA"]["EBmV"]
        ones = np.ones_like(wave)
        extinction_curve = extinction.apply(extinction.ccm89(wave, Av, Rv), ones)
        spec = spec / extinction_curve.reshape(-1, 1)
    return spec, wave


def write_fits_cube(data, header, filename, overwrite=False,
                    include_origin_notes=True):
    """
    Write a FITS cube with a WCS to a filename using fitsio
    """

    # Ensure header is in a format fitsio accepts (dict or FITSHDR)
    # If it's astropy Header, convert to dict-like
    if isinstance(header, fits.Header):
        # converting to dict might lose comments/history if not careful, but for WCS it's mostly keywords
        header_dict = {k: v for k, v in header.items()}
    else:
        header_dict = header # Assuming it's already dict-like or FITSHDR

    # Add history
    if include_origin_notes:
         now = datetime.datetime.strftime(datetime.datetime.now(),
                                        "%Y/%m/%d-%H:%M")
         # We can't easily add history to a dict, but fitsio.write takes a header object.
         # For now we skip explicit history addition or rely on what's in header already.
         pass

    if os.path.exists(filename) and not overwrite:
         # fitsio clobber=True handles overwrite. If False and exists, it might append or error depending on mode.
         # fitsio.write default mode is 'rw'.
         raise OSError(f"File {filename} exists and overwrite=False")

    fitsio.write(filename, data, header=header_dict, clobber=overwrite)

def savefitsmaps(module_id, method_id, outdir=""):
    
    """
    savefitsmaps _summary_

    Parameters
    ----------
    module_id : _type_
        _description_
    outdir : str, optional
        _description_, by default ""
    """

    runname = outdir
    rootname = outdir.rstrip("/").split("/")[-1]

    # Read bintable using fitsio
    table_path = os.path.join(outdir, rootname) + "_table.fits"
    with fitsio.FITS(table_path) as table_fits:
        data = table_fits[1].read()

        idx_inside = np.where(data["BIN_ID"] >= 0)[0]
        X = np.array(data["X"]) * -1
        Y = np.array(data["Y"])
        FLUX = np.array(data["FLUX"])
        XBIN = np.array(data["XBIN"])
        YBIN = np.array(data["YBIN"])
        binNum_long = np.array(data["BIN_ID"])
        ubins = np.unique(np.abs(np.array(data["BIN_ID"])))

        hdr0 = table_fits[0].read_header()
        pixelsize = hdr0["PIXSIZE"]

        oldwcshdr = table_fits[2].read_header() # fitsio header

        SNR = np.array(data["SNR"])
        SNRBIN = np.array(data["SNRBIN"])

    # update WCS
    # Convert fitsio header to dict, then to astropy Header
    astro_oldwcshdr = fits.Header({k: oldwcshdr[k] for k in oldwcshdr})
    wcs = WCS(astro_oldwcshdr).celestial
    newwcshdr = strip_wcs_from_header(astro_oldwcshdr)
    newwcshdr.update(diagonal_wcs_to_cdelt(wcs).to_header())

    # Convert newwcshdr (astropy Header) back to dict for fitsio
    newwcshdr_dict = {k: v for k, v in newwcshdr.items()}

    # Check spatial coordinates
    if len(np.where(np.logical_or(X == 0.0, np.isnan(X) == True))[0]) == len(X):
        print(
            "All X-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!"
        )
    if len(np.where(np.logical_or(Y == 0.0, np.isnan(Y) == True))[0]) == len(Y):
        print(
            "All Y-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!\n"
        )

    # Read Results
    if module_id == "SPATIAL_BINNING":
        # Most table results already read from table_hdu above
        #define names
        names = ["BINID","FLUX","SNR","SNRBIN","XBIN","YBIN"]

        result = np.zeros((len(binNum_long), len(names)))
        result[:,0] = binNum_long
        result[:,1] = FLUX
        result[:,2] = SNR
        result[:,3] = SNRBIN
        result[:,4] = XBIN # Units are arcseconds, and (0,0) is the centre spaxel
        result[:,5] = YBIN

    elif module_id == "KIN":
        # read results
        with fitsio.FITS(os.path.join(outdir, rootname) + "_kin.fits") as hdu:
            data_kin = hdu[1].read()
            names = list(data_kin.dtype.names)
            result = np.zeros((len(ubins), len(names)))
            for i, name in enumerate(names):
                result[:, i] = np.array(data_kin[name])

    elif module_id == "SFH":
        # Read results
        with fitsio.FITS(os.path.join(outdir, rootname) + "_sfh.fits") as sfh_hdu:
            data_sfh = sfh_hdu[1].read()
            names = list(data_sfh.dtype.names)
            result = np.zeros((len(ubins), len(names)))
            for i, name in enumerate(names):
                result[:, i] = np.array(data_sfh[name])
    
    elif module_id == "UMOD":
        if method_id == "twocomp_ppxf":
            # read results
            print(outdir, rootname)
            with fitsio.FITS(os.path.join(outdir, rootname) + "_twocomp_kin.fits") as hdu:
                data_umod = hdu[1].read()
                names = list(data_umod.dtype.names)
                result = np.zeros((len(ubins), len(names)))
                for i, name in enumerate(names):
                    result[:, i] = np.array(data_umod[name])
        else:
            printStatus.warning(
            "UMOD Method not recognised for saving maps"
            )

    if (module_id == 'KIN') | (module_id == "UMOD") | (module_id == "SFH"):
        # Convert results to long version
        result_long = np.zeros((len(binNum_long), result.shape[1]))
        result_long[:, :] = np.nan
        for i in range(len(ubins)):
            idx = np.where(ubins[i] == np.abs(binNum_long))[0]
            result_long[idx, :] = result[i, :]
        result = result_long

    # result[:, 0] = result[:, 0] - np.nanmedian(result[:, 0]) [median subtraction on products]

    ####### Adding the ability to output maps as fits files
    out_filename = os.path.join(outdir, rootname) + "_" + module_id + "_maps.fits"
    if os.path.exists(out_filename):
        os.remove(out_filename)

    with fitsio.FITS(out_filename, 'rw') as out_fits:
        # Write empty primary first to match astropy structure
        out_fits.write(None)

        for iterate in range(0, len(names)):
            # Prepare main plot
            val = result[:, iterate]

            # Create image in pixels
            xmin = np.min(X)
            xmax = np.max(X)
            ymin = np.min(Y)
            ymax = np.max(Y)
            npixels_x = int(np.round((xmax - xmin) / pixelsize) + 1)
            npixels_y = int(np.round((ymax - ymin) / pixelsize) + 1)
            i = np.array(np.round((X - xmin) / pixelsize), dtype=np.int32)
            j = np.array(np.round((Y - ymin) / pixelsize), dtype=np.int32)
            image = np.full((npixels_x, npixels_y), np.nan)
            # Reverse the i index to each row of the image
            # because ra increases West-East (right-left in image plane)
            image[i[::-1][idx_inside], j[idx_inside]] = val[idx_inside]
            # Transpose x and y because numpy uses arr[row, col] and FITS uses
            # im[ra, dec] = arr[col, row]
            image = image.T

            # append fits image
            out_fits.write(image, header=newwcshdr_dict, extname=names[iterate])


def savefitsmaps_GASmodule(module_id="GAS", outdir="", LEVEL="", AoNThreshold=4):
    """
    savefitsmaps_GASmodule _summary_

    Parameters
    ----------
    module_id : str, optional
        _description_, by default "GAS"
    outdir : str, optional
        _description_, by default ""
    LEVEL : str, optional
        _description_, by default ""
    AoNThreshold : int, optional
        _description_, by default 4
    """

    runname = outdir
    rootname = outdir.rstrip("/").split("/")[-1]

    # Construct a mask for defunct spaxels
    with fitsio.FITS(os.path.join(outdir, rootname) + "_mask.fits") as mhdu:
        mask = mhdu[1].read()["MASK_DEFUNCT"]
    maskedSpaxel = np.array(mask, dtype=bool)

    # Read bintable using fitsio
    table_path = os.path.join(outdir, rootname) + "_table.fits"
    with fitsio.FITS(table_path) as table_fits:
        data = table_fits[1].read()

        idx_inside = np.where(data["BIN_ID"] >= 0)[0]
        X = np.array(data["X"]) * -1
        Y = np.array(data["Y"])
        FLUX = np.array(data["FLUX"])
        binNum_long = np.array(data["BIN_ID"])
        ubins = np.unique(np.abs(binNum_long))

        hdr0 = table_fits[0].read_header()
        pixelsize = hdr0["PIXSIZE"]

        oldwcshdr = table_fits[2].read_header() # fitsio header

    # update WCS
    # Convert fitsio header to dict, then to astropy Header
    astro_oldwcshdr = fits.Header({k: oldwcshdr[k] for k in oldwcshdr})
    wcs = WCS(astro_oldwcshdr).celestial
    newwcshdr = strip_wcs_from_header(astro_oldwcshdr)
    newwcshdr.update(diagonal_wcs_to_cdelt(wcs).to_header())

    # Convert newwcshdr (astropy Header) back to dict for fitsio
    newwcshdr_dict = {k: v for k, v in newwcshdr.items()}

    maskedSpaxel = maskedSpaxel[idx_inside]

    # Check spatial coordinates
    if len(np.where(np.logical_or(X == 0.0, np.isnan(X) == True))[0]) == len(X):
        print(
            "All X-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!"
        )
    if len(np.where(np.logical_or(Y == 0.0, np.isnan(Y) == True))[0]) == len(Y):
        print(
            "All Y-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!\n"
        )

    if LEVEL == "SPAXEL":
        with fitsio.FITS(os.path.join(outdir, rootname) + "_gas_SPAXEL.fits") as gas_hdu:
             results = gas_hdu[1].read()
    elif LEVEL == "BIN":
        with fitsio.FITS(os.path.join(outdir, rootname) + "_gas_BIN.fits") as gas_hdu:
             results = gas_hdu[1].read()
    elif LEVEL == None:
        print("LEVEL keyword not set!")
        return

    # Convert results to long version
    if LEVEL == "BIN":
        _, idxConvert = np.unique(np.abs(binNum_long), return_inverse=True)
        results = results[idxConvert]

    out_filename = os.path.join(outdir, rootname) + "_" + module_id + "_" + LEVEL + "_maps.fits"
    if os.path.exists(out_filename):
        os.remove(out_filename)

    with fitsio.FITS(out_filename, 'rw') as out_fits:
        # Write empty primary first
        out_fits.write(None)

        # Iterate over all lines
        for line in results.dtype.names:
            if line[-3:] == "AON":
                continue
            if line in ["EBmV_0", "EBmV_1"]:
                continue

            data = results[line]

            # we don't need to mask bin IDs
            if np.issubdtype(data.dtype, np.integer):
                data = data.astype(float) # Convert to float for NaN
                data[np.where(data == -1)[0]] = np.nan

            # Create image in pixels
            xmin = np.min(X)
            xmax = np.max(X)
            ymin = np.min(Y)
            ymax = np.max(Y)
            npixels_x = int(np.round((xmax - xmin) / pixelsize) + 1)
            npixels_y = int(np.round((ymax - ymin) / pixelsize) + 1)
            col = np.array(np.round((X - xmin) / pixelsize), dtype=np.int32)
            row = np.array(np.round((Y - ymin) / pixelsize), dtype=np.int32)
            image = np.full((npixels_x, npixels_y), np.nan)

            # reverse the index to flip vertically
            image[col[::-1][idx_inside][~maskedSpaxel], row[idx_inside][~maskedSpaxel]] = data[idx_inside][~maskedSpaxel]

            # Transpose x and y
            image = image.T

            # append fits image
            out_fits.write(image, header=newwcshdr_dict, extname=line)


def savefitsmaps_LSmodule(module_id="LS", outdir="", RESOLUTION=""):
    """
    savefitsmaps_LSmodule _summary_

    Parameters
    ----------
    module_id : str, optional
        _description_, by default "LS"
    outdir : str, optional
        _description_, by default ""
    RESOLUTION : str, optional
        _description_, by default ""
    """
    runname = outdir
    rootname = outdir.rstrip("/").split("/")[-1]

    # Read bintable using fitsio
    table_path = os.path.join(outdir, rootname) + "_table.fits"
    with fitsio.FITS(table_path) as table_fits:
        data = table_fits[1].read()

        idx_inside = np.where(data["BIN_ID"] >= 0)[0]
        X = np.array(data["X"]) * -1
        Y = np.array(data["Y"])
        FLUX = np.array(data["FLUX"])
        binNum_long = np.array(data["BIN_ID"])
        ubins = np.unique(np.abs(np.array(data["BIN_ID"])))

        hdr0 = table_fits[0].read_header()
        pixelsize = hdr0["PIXSIZE"]

        oldwcshdr = table_fits[2].read_header() # fitsio header

    # update WCS
    # Convert fitsio header to dict, then to astropy Header
    astro_oldwcshdr = fits.Header({k: oldwcshdr[k] for k in oldwcshdr})
    wcs = WCS(astro_oldwcshdr).celestial
    newwcshdr = strip_wcs_from_header(astro_oldwcshdr)
    newwcshdr.update(diagonal_wcs_to_cdelt(wcs).to_header())

    # Convert newwcshdr (astropy Header) back to dict for fitsio
    newwcshdr_dict = {k: v for k, v in newwcshdr.items()}

    # Check spatial coordinates
    if len(np.where(np.logical_or(X == 0.0, np.isnan(X) == True))[0]) == len(X):
        print(
            "All X-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!"
        )
    if len(np.where(np.logical_or(Y == 0.0, np.isnan(Y) == True))[0]) == len(Y):
        print(
            "All Y-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!\n"
        )

    # Read results
    if RESOLUTION == "ORIGINAL":
        kin_path = os.path.join(outdir, rootname) + "_ls_OrigRes.fits"
    else:
        kin_path = os.path.join(outdir, rootname) + "_ls_AdapRes.fits"

    with fitsio.FITS(kin_path) as hdu:
        data_ls = hdu[1].read()
        names = list(data_ls.dtype.names)
        result = np.zeros((len(ubins), len(names)))
        for i, name in enumerate(names):
            result[:, i] = np.array(data_ls[name])

    # Convert results to long version
    result_long = np.zeros((len(binNum_long), result.shape[1]))
    result_long[:, :] = np.nan
    for i in range(len(ubins)):
        idx = np.where(ubins[i] == np.abs(binNum_long))[0]
        result_long[idx, :] = result[i, :]
    result = result_long

    # result[:, 0] = result[:, 0] - np.nanmedian(result[:, 0]) [median subtraction on products]

    ####### Adding the ability to output maps as fits files
    out_filename = os.path.join(outdir, rootname) + "_" + module_id + "_" + RESOLUTION + "_maps.fits"
    if os.path.exists(out_filename):
        os.remove(out_filename)

    with fitsio.FITS(out_filename, 'rw') as out_fits:
        # Write empty primary first
        out_fits.write(None)

        for iterate in range(0, len(names)):
            # Prepare main plot
            val = result[:, iterate]

            # Create image in pixels
            xmin = np.min(X)
            xmax = np.max(X)
            ymin = np.min(Y)
            ymax = np.max(Y)
            npixels_x = int(np.round((xmax - xmin) / pixelsize) + 1)
            npixels_y = int(np.round((ymax - ymin) / pixelsize) + 1)
            i = np.array(np.round((X - xmin) / pixelsize), dtype=np.int32)
            j = np.array(np.round((Y - ymin) / pixelsize), dtype=np.int32)
            image = np.full((npixels_x, npixels_y), np.nan)

            # Reverse the i index to each row of the image
            # because ra increases West-East (right-left in image plane)
            image[i[::-1][idx_inside], j[idx_inside]] = val[idx_inside]

            # Transpose x and y to reorient the image correctly
            # im[ra, dec] = arr[col, row]
            image = image.T

            # append fits image
            out_fits.write(image, header=newwcshdr_dict, extname=names[iterate])


def saveContLineCube(config):
    """
    saveContLineCubes _summary_

    Write continuum-only and line-only cubes to FITS files.

    Parameters
    ----------
    config : str, optional
        gistPipeline config
    """

    # read cube header using fitsio
    with fitsio.FITS(config["GENERAL"]["INPUT"]) as f:
        cubehdr = f[0].read_header()
        if "NAXIS1" not in cubehdr and len(f) > 1:
            cubehdr = f[1].read_header()
        elif "NAXIS1" not in cubehdr and len(f) > 2:
            cubehdr = f[2].read_header()

        # Propagate BUNIT and CUNIT3
        for ext in (0, 1, 2):
            if ext >= len(f):
                continue
            h = f[ext].read_header()
            if "BUNIT" in h and "BUNIT" not in cubehdr:
                cubehdr["BUNIT"] = h["BUNIT"]
            if "CUNIT3" in h and "CUNIT3" not in cubehdr:
                cubehdr["CUNIT3"] = h["CUNIT3"]

    NX = cubehdr["NAXIS1"]
    NY = cubehdr["NAXIS2"]

    # Load only wavelength-trimmed spectra
    spectra_all, linLam_full = _load_input_spectra_trimmed(config)
    idx_lam = np.where(
        np.logical_and(linLam_full > config["CONT"]["LMIN"], linLam_full < config["CONT"]["LMAX"])
    )[0]
    spectra_all = spectra_all[idx_lam, :]
    linLam = linLam_full[idx_lam]

    # Get PPXF best-fit continuum and logLam from CONT module
    cont_path = os.path.join(
        config["GENERAL"]["OUTPUT"],
        config["GENERAL"]["RUN_ID"] + "_kin-bestfit-cont.fits",
    )
    printStatus.running("Opening: -kin-bestfit-cont.fits")
    with fitsio.FITS(cont_path) as cont_fits:
        ppxf_bestfit = cont_fits[1].read()["BESTFIT"]
        logLam = cont_fits[2].read()["LOGLAM"]

    # table HDU
    table_path = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_table.fits"
    with fitsio.FITS(table_path) as f:
        data = f[1].read()
        spaxID = data["ID"]
        binID = data["BIN_ID"]

    ubins = np.unique(np.abs(binID[binID >= 0]))
    if len(ubins) == 0:
        ubins = np.unique(np.abs(binID))
    # Map BIN_ID -> row index in ppxf_bestfit (same order as CONT module / BinSpectra)
    bin_id_to_idx = {int(b): i for i, b in enumerate(ubins)}

    idx_snr = np.where(
        np.logical_and(
            linLam >= config["READ_DATA"]["LMIN_SNR"],
            linLam <= config["READ_DATA"]["LMAX_SNR"],
        )
    )[0]

    # Per-bin scaling (not per-spaxel) so continuum maps match KIN/GAS: constant within
    # each bin. Per-spaxel scaling (obsSignal(s)/fitSignal(bin)) caused shell-like
    # artifacts: fitSignal is constant per bin, so the scaling jumped at bin boundaries
    # and created concentric structure not present in KIN or other GAS maps.
    n_bins = ppxf_bestfit.shape[0]
    if len(ubins) > n_bins:
        logging.warning(
            "saveContLineCube: table has %d unique bins but _kin-bestfit-cont has %d rows; "
            "some bins may get zero continuum.",
            len(ubins),
            n_bins,
        )
    fitSignal_per_bin = np.zeros(n_bins)
    obsSignal_median_per_bin = np.zeros(n_bins)
    fitSpec_lin_per_bin = np.zeros((n_bins, len(linLam)))
    for i, b in enumerate(ubins):
        if i >= n_bins:
            break
        fitSpec = np.asarray(ppxf_bestfit[i, :])
        # Skip bins where the CONT fit failed (all zeros or contains non-finite values)
        if not np.all(np.isfinite(fitSpec)) or np.all(fitSpec == 0):
            fitSpec_lin_per_bin[i, :] = np.nan
            continue
        fitSpec_func = CubicSpline(np.exp(logLam), fitSpec, extrapolate=False)
        fitSpec_lin = fitSpec_func(linLam)
        fitSignal_per_bin[i] = np.nanmedian(fitSpec_lin[idx_snr])
        fitSpec_lin_per_bin[i, :] = fitSpec_lin
        spax_in_bin = np.where(np.abs(binID) == b)[0]
        obs_signals = [
            np.nanmedian(spectra_all[:, s][idx_snr]) for s in spax_in_bin
        ]
        obsSignal_median_per_bin[i] = np.nanmedian(obs_signals)

    scale_per_bin = np.where(
        fitSignal_per_bin > 0,
        obsSignal_median_per_bin / fitSignal_per_bin,
        1.0,
    )

    # Build and write one cube at a time to reduce peak RAM (avoid holding all three)
    def _fit_spec_lin_for_spaxel(binID_spax, obsSpec_lin):
        if binID_spax < 0:
            bin_idx = bin_id_to_idx.get(int(np.abs(binID_spax)), -1)
        else:
            bin_idx = bin_id_to_idx.get(int(binID_spax), -1)
        if bin_idx < 0:
            return np.zeros(len(obsSpec_lin))
        return fitSpec_lin_per_bin[bin_idx, :] * scale_per_bin[bin_idx]

    contCube = np.full([len(linLam), NY * NX], np.nan)
    for s in spaxID:
        fitSpec_lin = _fit_spec_lin_for_spaxel(binID[s], spectra_all[:, s])
        contCube[:, s] = fitSpec_lin
    # spectral axes in observed wavelength frame
    # (cube is de-redshifted during read in by MUSE_WFM.py)
    cubehdr["NAXIS3"] = len(linLam)
    cubehdr["CRVAL3"] = linLam[0] * (1 + config["GENERAL"]["REDSHIFT"]) #
    cubehdr["CRPIX3"] = 1
    #cubehdr["CTYPE3"] = "AWAV"
    #cubehdr["CUNIT3"] = "angstrom"

    # set the WCS keywords to CDELT standard format
    #cdi_j_wcs = WCS(cubehdr)
    #newcubehdr = strip_wcs_from_header(cubehdr)  # remove all WCS keys from header
    #newcubehdr.update(
    #    diagonal_wcs_to_cdelt(cdi_j_wcs).to_header()
    #)  # replace with CDELT standard keys

    # as cube is de-redshifted during read in by MUSE_WFM.py
    cubehdr["CD3_3"] = np.abs(np.diff(linLam * (1 + config["GENERAL"]["REDSHIFT"])))[0]

    # Save CONT cube first, then free it before building LINE/ORIG (reduces peak RAM)
    outfits_cont = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_CONTcube.fits"
    )
    write_fits_cube(
        data=np.float32(contCube.reshape((len(linLam), NY, NX))),
        header=cubehdr,
        filename=outfits_cont,
        overwrite=True
    )
    del contCube

    # Build and write LINE cube, then ORIG (one at a time to limit peak memory)
    lineCube = np.full([len(linLam), NY * NX], np.nan)
    for s in spaxID:
        obsSpec_lin = spectra_all[:, s]
        fitSpec_lin = _fit_spec_lin_for_spaxel(binID[s], obsSpec_lin)
        lineCube[:, s] = obsSpec_lin - fitSpec_lin
    outfits_line = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_LINEcube.fits"
    )
    write_fits_cube(
        data=np.float32(lineCube.reshape((len(linLam), NY, NX))),
        header=cubehdr,
        filename=outfits_line,
        overwrite=True
    )
    del lineCube

    origCube = np.full([len(linLam), NY * NX], np.nan)
    for s in spaxID:
        origCube[:, s] = spectra_all[:, s]
    outfits_orig = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_ORIGcube.fits"
    )
    write_fits_cube(
        data=np.float32(origCube.reshape((len(linLam), NY, NX))),
        header=cubehdr,
        filename=outfits_orig,
        overwrite=True
    )
