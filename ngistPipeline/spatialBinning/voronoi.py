import functools
import logging
import os

import numpy as np
import scipy.spatial.distance as dist
from astropy.io import fits
from printStatus import printStatus
from scipy.spatial import cKDTree
from vorbin.voronoi_2d_binning import voronoi_2d_binning

"""
PURPOSE:
  This file contains a collection of functions necessary to Voronoi-bin the data.
  The Voronoi-binning makes use of the algorithm from Cappellari & Copin 2003
  (ui.adsabs.harvard.edu/?#abs/2003MNRAS.342..345C).
"""


def sn_func(index, signal=None, noise=None, covar_vor=0.00):
    """
    This function is passed to the Voronoi binning routine of Cappellari &
    Copin 2003 (ui.adsabs.harvard.edu/?#abs/2003MNRAS.342..345C) and used to
    estimate the noise in the bin from the noise in the spaxels. This
    implementation is identical to the default one, but accounts for spatial
    correlations in the noise by applying an empirical equation (see e.g.
    Garcia-Benito et al. 2015;
    ui.adsabs.harvard.edu/?#abs/2015A&A...576A.135G) together with the
    parameter defined in the Config-file.

    NaN safety: ``np.nansum`` is used so that any non-finite scalars that slip
    through upstream sanitisation do not contaminate the bin S/N. If the
    aggregate is non-finite (e.g. all members were NaN) we return ``-inf`` so
    the caller (vorbin) treats the bin as far below ``target_sn`` and never
    seeds binning from such a candidate.
    """

    # Add the noise in the spaxels to obtain the noise in the bin
    s_sum = np.nansum(signal[index])
    n_sq_sum = np.nansum(noise[index] ** 2)
    if not np.isfinite(s_sum) or not np.isfinite(n_sq_sum) or n_sq_sum <= 0:
        return -np.inf
    sn = s_sum / np.sqrt(n_sq_sum)

    # Account for spatial correlations in the noise by applying an empirical
    # equation (see e.g. Garcia-Benito et al. 2015;
    # ui.adsabs.harvard.edu/?#abs/2015A&A...576A.135G)
    sn /= 1 + covar_vor * np.log10(index.size)

    return sn


def generateSpatialBins(config, cube):
    """
    This function applies the Voronoi-binning algorithm of Cappellari & Copin
    2003 (ui.adsabs.harvard.edu/?#abs/2003MNRAS.342..345C) to the data. It can
    be accounted for spatial correlations in the noise (see function sn_func()).
    A BIN_ID is assigned to every spaxel. Spaxels which were masked are excluded
    from the Voronoi-binning, but are assigned a negative BIN_ID, with the
    absolute value of the BIN_ID corresponding to the nearest Voronoi-bin that
    satisfies the minimum SNR threshold.  All results are saved in a dedicated
    table to provide easy means of matching spaxels and bins.
    """
    # Pass a function for the SNR calculation to the Voronoi-binning algorithm,
    # in order to account for spatial correlations in the noise
    sn_func_covariances = functools.partial(
        sn_func, covar_vor=config["SPATIAL_BINNING"]["COVARIANCE"]
    )

    # Generate the Voronoi bins
    printStatus.running("Defining the Voronoi bins")
    logging.info("Defining the Voronoi bins")

    # Read maskfile
    maskfile = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_mask.fits"
    )
    mask = fits.open(maskfile, memmap=True)[1].data.MASK
    idxUnmasked = np.where(mask == 0)[0]
    idxMasked = np.where(mask == 1)[0]

    # Defensive guard: drop any spaxels with non-finite scalar signal or noise
    # that may have slipped past the spatial mask. vorbin is path-dependent and
    # NaN inputs can silently bias seeding (np.argmax) and accretion decisions.
    finite_scalars = (
        np.isfinite(cube["signal"][idxUnmasked])
        & np.isfinite(cube["noise"][idxUnmasked])
        & (cube["noise"][idxUnmasked] > 0)
    )
    n_dropped = int(np.sum(~finite_scalars))
    if n_dropped > 0:
        dropped = idxUnmasked[~finite_scalars]
        idxUnmasked = idxUnmasked[finite_scalars]
        # Treat dropped spaxels as masked so their nearest-bin assignment is
        # still recorded with a negative BIN_ID downstream.
        idxMasked = np.unique(np.concatenate([idxMasked, dropped]))
        logging.warning(
            "Dropped %d spaxels with non-finite scalar signal/noise from Voronoi inputs",
            n_dropped,
        )

    if len(idxUnmasked) == 0:
        printStatus.updateFailed("Defining the Voronoi bins")
        print(
            "No unmasked spaxels remaining. Voronoi binning cannot proceed. "
            "Check spatial masking (defunct/SNR/maskfile) or input data."
        )
        logging.error(
            "Defining the Voronoi bins failed: no unmasked spaxels. "
            "All spaxels were rejected by the spatial mask (defunct, SNR, or maskfile)."
        )
        return "SKIP"

    try:
        # Do the Voronoi binning
        binNum, xNode, yNode, xBar, yBar, sn, nPixels, _ = voronoi_2d_binning(
            cube["x"][idxUnmasked],
            cube["y"][idxUnmasked],
            cube["signal"][idxUnmasked],
            cube["noise"][idxUnmasked],
            config["SPATIAL_BINNING"]["TARGET_SNR"],
            plot=False,
            quiet=True,
            pixelsize=cube["pixelsize"],
            sn_func=sn_func_covariances,
        )

        printStatus.updateDone("Defining the Voronoi bins")
        print("             " + str(np.max(binNum) + 1) + " voronoi bins generated!")
        logging.info(str(np.max(binNum) + 1) + " Voronoi bins generated!")

    # Handle common exceptions
    except ValueError as e:
        # Sufficient SNR and no binning needed
        if str(e) == "All pixels have enough S/N and binning is not needed":
            printStatus.updateWarning("Defining the Voronoi bins")
            print(
                "             "
                + "The Voronoi-binning routine of Cappellari & Copin (2003) returned the following error:"
            )
            print("             " + str(e))
            printStatus.warning("Analysis will continue without Voronoi-binning!")
            print(
                "             "
                + str(len(idxUnmasked))
                + " spaxels will be treated as Voronoi-bins."
            )

            logging.warning(
                "Defining the Voronoi bins failed. The Voronoi-binning routine of Cappellari & Copin "
                + "(2003) returned the following error: \n"
                + str(e)
            )
            logging.info(
                "Analysis will continue without Voronoi-binning! "
                + str(len(idxUnmasked))
                + " spaxels will be treated as Voronoi-bins."
            )

            binNum, xNode, yNode, sn, nPixels = noBinning(
                cube["x"], cube["y"], cube["snr"], idxUnmasked
            )

        # Any uncaught exceptions, causing the galaxy to be skipped
        else:
            printStatus.updateFailed("Defining the Voronoi bins")
            print(
                "The Voronoi-binning routine of Cappellari & Copin (2003) returned the following error: \n"
                + str(e)
            )
            logging.error(
                "Defining the Voronoi bins failed. The Voronoi-binning routine of Cappellari & Copin "
                + "(2003) returned the following error: \n"
                + str(e)
            )
            return "SKIP"

    # Find the nearest Voronoi bin for the pixels outside the Voronoi region
    binNum_outside = find_nearest_voronoibin(
        cube["x"], cube["y"], idxMasked, xNode, yNode
    )

    # Generate extended binNum-list:
    #   Positive binNum (including zero) indicate the Voronoi bin of the spaxel (for unmasked spaxels)
    #   Negative binNum indicate the nearest Voronoi bin of the spaxel (for masked spaxels)
    ubins = np.unique(binNum)
    nbins = len(ubins)
    binNum_long = np.zeros(len(cube["x"]))
    binNum_long[:] = np.nan
    binNum_long[idxUnmasked] = binNum
    binNum_long[idxMasked] = -1 * binNum_outside

    # Save bintable: data for *ALL* spectra inside and outside of the Voronoi region!
    save_table(
        config,
        cube["x"],
        cube["y"],
        cube["signal"],
        cube["snr"],
        binNum_long,
        ubins,
        xNode,
        yNode,
        sn,
        nPixels,
        cube["pixelsize"],
        cube["wcshdr"],
    )

    return None


def noBinning(x, y, snr, idx_inside):
    """
    In case no Voronoi-binning is required/possible, treat spaxels in the input
    data as Voronoi bins, in order to continue the analysis.
    """
    binNum = np.arange(0, len(idx_inside))
    xNode = x[idx_inside]
    yNode = y[idx_inside]
    sn = snr[idx_inside]
    nPixels = np.ones(len(idx_inside))

    return (binNum, xNode, yNode, sn, nPixels)


def find_nearest_voronoibin(x, y, idx_outside, xNode, yNode):
    """
    Find the nearest Voronoi-bin for each spaxel that does not satisfy the minimum SNR threshold.
    
    Args:
    - x (array): x-coordinates of all spaxels
    - y (array): y-coordinates of all spaxels
    - idx_outside (array): indices of spaxels which do not satisfy the minimum SNR threshold
    - xNode (array): x-coordinates of the Voronoi bins
    - yNode (array): y-coordinates of the Voronoi bins
    
    Returns:
    - closest (array): array of indices representing the nearest Voronoi-bin for each spaxel
    """
    # Create an array of pixel coordinates
    pix_coords = np.column_stack((x[idx_outside], y[idx_outside]))
    # Create an array of bin coordinates
    bin_coords = np.column_stack((xNode, yNode))

    # Build a KDTree from the bin coordinates
    tree = cKDTree(bin_coords)
    # Query the nearest bin for each pixel
    closest = tree.query(pix_coords, k=1, workers=-1)[1]

    return closest

def save_table(
    config,
    x,
    y,
    signal,
    snr,
    binNum_new,
    ubins,
    xNode,
    yNode,
    sn,
    nPixels,
    pixelsize,
    wcshdr,
):
    """
    Save all relevant information about the Voronoi binning to disk. In
    particular, this allows to later match spaxels and their corresponding bins.

    Args:
        config (dict): Configuration settings.
        x (ndarray): X-coordinates of the spaxels.
        y (ndarray): Y-coordinates of the spaxels.
        signal (ndarray): Flux values of the spaxels.
        snr (ndarray): Signal-to-noise ratio values of the spaxels.
        binNum_new (ndarray): Array of bin IDs for each spaxel.
        ubins (ndarray): Unique bin IDs.
        xNode (ndarray): X-coordinates of the bin nodes.
        yNode (ndarray): Y-coordinates of the bin nodes.
        sn (ndarray): Signal-to-noise ratio values of the bins.
        nPixels (ndarray): Number of spaxels in each bin.
        pixelsize (float): Size of each pixel.
        wcshdr (Header): WCS header information.

    Returns:
        None
    """
    outfits_table = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_table.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_table.fits")
    xshape = x.shape
    yshape = y.shape
    printStatus.running("size of the input array=" + str(xshape) + str(yshape))
    # Expand data to spaxel level
    xNode_new = np.zeros(len(x))
    yNode_new = np.zeros(len(x))
    sn_new = np.zeros(len(x))
    nPixels_new = np.zeros(len(x))
    for i in range(len(ubins)):
        idx = np.where(ubins[i] == np.abs(binNum_new))[0]
        xNode_new[idx] = xNode[i]
        yNode_new[idx] = yNode[i]
        sn_new[idx] = sn[i]
        nPixels_new[idx] = nPixels[i]

    # Primary HDU
    priHDU = fits.PrimaryHDU()
    # Table HDU with output data
    cols = []
    cols.append(fits.Column(name="ID", format="J", array=np.arange(len(x))))
    cols.append(fits.Column(name="BIN_ID", format="J", array=binNum_new))
    cols.append(fits.Column(name="X", format="D", array=x))
    cols.append(fits.Column(name="Y", format="D", array=y))
    cols.append(fits.Column(name="FLUX", format="D", array=signal))
    cols.append(fits.Column(name="SNR", format="D", array=snr))
    cols.append(fits.Column(name="XBIN", format="D", array=xNode_new))
    cols.append(fits.Column(name="YBIN", format="D", array=yNode_new))
    cols.append(fits.Column(name="SNRBIN", format="D", array=sn_new))
    cols.append(fits.Column(name="NSPAX", format="J", array=nPixels_new))

    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    tbhdu.name = "TABLE"

    # create empty imageHDU with wcs header info
    imghdu = fits.ImageHDU(data=None, header=wcshdr)

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, tbhdu, imghdu])
    HDUList.writeto(outfits_table, overwrite=True)
    fits.setval(outfits_table, "PIXSIZE", value=pixelsize)

    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_table.fits")
    logging.info("Wrote Voronoi table: " + outfits_table)
