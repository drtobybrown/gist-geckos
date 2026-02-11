import logging
import os

import fitsio
import numpy as np
from astropy.io import fits
from printStatus import printStatus


def generate_spatial_mask(config, cube):
    """
    Generates a spatial mask for the input cube based on defunct spaxels, signal-to-noise ratio threshold,
    and an additional mask file provided in the configuration.

    Parameters:
    config (dict): Configuration settings for the spatial masking.
    cube (dict): Input cube containing 'snr', 'signal', and other necessary data.

    Returns:
    None
    """
    # Mask defunct spaxels
    masked_defunct = mask_defunct_spaxels(cube)

    # Apply signal-to-noise ratio threshold
    masked_snr = apply_snr_threshold(cube["snr"], cube["signal"], config["SPATIAL_MASKING"]["MIN_SNR"])

    # Apply additional mask file
    masked_mask = apply_mask_file(config, cube)

    # Combine all masks
    combined_mask = np.logical_or.reduce((masked_defunct, masked_snr, masked_mask))

    # Save the combined mask
    save_mask(combined_mask, masked_defunct, masked_snr, masked_mask, config)


def generateSpatialMask(config, cube):
    """
    Default implementation of the spatialMasking module.

    This function masks defunct spaxels, rejects spaxels with a signal-to-noise ration below a given threshold, and
    masks spaxels according to a provided mask file. Finally, all masks are combined and saved.
    """

    # Mask defunct spaxels
    maskedDefunct = maskDefunctSpaxels(cube)

    # Mask spaxels with SNR below threshold
    maskedSNR = applySNRThreshold(
        cube["snr"], cube["signal"], config["SPATIAL_MASKING"]["MIN_SNR"]
    )

    # Mask spaxels according to spatial mask file
    maskedMask = applyMaskFile(config, cube)

    # Create combined mask
    combinedMaskIdx = np.where(
        np.logical_or.reduce(
            (maskedDefunct == True, maskedSNR == True, maskedMask == True)
        )
    )[0]
    combinedMask = np.zeros(len(cube["snr"]), dtype=bool)
    combinedMask[combinedMaskIdx] = True
    logging.info(
        "Combined mask: " + str(len(combinedMaskIdx)) + " spaxels are rejected."
    )

    # Save mask to file
    saveMask(combinedMask, maskedDefunct, maskedSNR, maskedMask, config)

    # Return
    return None


def maskDefunctSpaxels(cube):
    """
    Mask defunct spaxels: those with all-NaN spectra or non-positive median flux.
    Spaxels with only some NaNs (e.g. bad pixels) but valid median are kept so
    that real data with occasional bad pixels is not over-masked.
    """
    spec = cube["spec"]

    # Defunct = entirely NaN spectrum OR median flux <= 0 (reject zero/negative)
    all_nan = np.all(np.isnan(spec), axis=0)
    median_nonpositive = np.nanmedian(spec, axis=0) <= 0.0
    idx_bad = np.where(np.logical_or(all_nan, median_nonpositive))[0]
    idx_good = np.where(~np.logical_or(all_nan, median_nonpositive))[0]

    logging.info(
        "Masking defunct spaxels: " + str(len(idx_bad)) + " spaxels are rejected."
    )

    masked = np.ones(len(cube["snr"]), dtype=bool)
    masked[idx_good] = False

    return masked

def applySNRThreshold(snr, signal, min_snr, threshold_method="isophote"):
    """
    Mask those spaxels that are above the isophote level with a mean
    signal-to-noise ratio of MIN_SNR.
    """
    if threshold_method == "isophote":
        idx_snr = np.where(np.abs(snr - min_snr) < 2.0)[0]
        meanmin_signal = np.mean(signal[idx_snr])
        idx_inside = np.where(signal >= meanmin_signal)[0]
        idx_outside = np.where(signal < meanmin_signal)[0]

    if threshold_method == "actual":
        idx_inside = np.where(snr >= min_snr)[0]
        idx_outside = np.where(snr < min_snr)[0]

    if len(idx_inside) == 0 and len(idx_outside) == 0:
        idx_inside = np.arange(len(snr))
        idx_outside = np.array([], dtype=np.int64)

    logging.info(
        "Masking low signal-to-noise spaxels: "
        + str(len(idx_outside))
        + " spaxels are rejected."
    )

    masked = np.zeros(len(snr), dtype=bool)
    masked[idx_inside] = False
    masked[idx_outside] = True

    return masked


def applyMaskFile(config, cube):
    """
    Select those spaxels that are unmasked in the input masking file.
    """

    if (
        config["SPATIAL_MASKING"]["MASK"] == False
        or config["SPATIAL_MASKING"]["MASK"] == None
    ):
        logging.info("No maskfile specified.")
        idxGood = np.arange(len(cube["snr"]))
        idxBad = np.array([], dtype=np.int64)

    else:
        maskfile = os.path.join(
            os.path.dirname(config["GENERAL"]["INPUT"]),
            config["SPATIAL_MASKING"]["MASK"],
        )

        if os.path.isfile(maskfile) == True:
            with fitsio.FITS(maskfile) as hdu:
                if len(hdu) == 1:
                    mask = hdu[0].read()
                else:
                    mask = hdu[1].read()
            s = np.shape(mask)
            mask = np.reshape(mask, s[0] * s[1])

            idxGood = np.where(mask == 0)[0]
            idxBad = np.where(mask == 1)[0]

            logging.info(
                "Masking spaxels according to maskfile: "
                + str(len(idxBad))
                + " spaxels are rejected."
            )

        elif os.path.isfile(maskfile) == False:
            logging.info("No maskfile found at " + maskfile)
            idxGood = np.arange(len(cube["snr"]))
            idxBad = np.array([], dtype=np.int64)

    masked = np.zeros(len(cube["snr"]), dtype=bool)
    masked[idxGood] = False
    masked[idxBad] = True

    return masked


def saveMask(combinedMask, maskedDefunct, maskedSNR, maskedMask, config):
    """Save the mask to disk."""
    outfits = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_mask.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_mask.fits")

    if os.path.exists(outfits):
        os.remove(outfits)

    # Create numpy structured array for table
    n = len(combinedMask)
    dt = [
        ("MASK", np.int32),
        ("MASK_DEFUNCT", np.int32),
        ("MASK_SNR", np.int32),
        ("MASK_FILE", np.int32)
    ]
    data = np.zeros(n, dtype=dt)
    data["MASK"] = np.array(combinedMask, dtype=np.int32)
    data["MASK_DEFUNCT"] = np.array(maskedDefunct, dtype=np.int32)
    data["MASK_SNR"] = np.array(maskedSNR, dtype=np.int32)
    data["MASK_FILE"] = np.array(maskedMask, dtype=np.int32)

    with fitsio.FITS(outfits, 'rw') as f:
         # Primary
         f.write(None)
         # Table
         header = {"EXTNAME": "MASKFILE"}
         f.write(data, header=header)

    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_mask.fits")
    logging.info("Wrote mask file: " + outfits)

    return None
