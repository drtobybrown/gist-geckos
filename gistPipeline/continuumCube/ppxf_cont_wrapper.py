import code
import logging
import os
import time

import numpy as np
from astropy.io import fits
from astropy.stats import biweight_location
from multiprocess import Process, Queue
from ppxf.ppxf import ppxf
from printStatus import printStatus

from gistPipeline.auxiliary import _auxiliary
from gistPipeline.prepareTemplates import _prepareTemplates

from concurrent.futures import ThreadPoolExecutor, as_completed

# PHYSICAL CONSTANTS
C = 299792.458  # km/s


"""
PURPOSE:
  This module creates a continuum and line-only cube.
  Basically, it acts as an interface between pipeline and the pPXF routine from
  Cappellari & Emsellem 2004 (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
  ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C).
"""


def robust_sigma(y, zero=False):
    """
    Biweight estimate of the scale (standard deviation).
    Implements the approach described in
    "Understanding Robust and Exploratory Data Analysis"
    Hoaglin, Mosteller, Tukey ed., 1983, Chapter 12B, pg. 417

    """
    y = np.ravel(y)
    d = y if zero else y - np.median(y)

    mad = np.median(np.abs(d))
    u2 = (d / (9.0 * mad)) ** 2  # c = 9
    good = u2 < 1.0
    u1 = 1.0 - u2[good]
    num = y.size * ((d[good] * u1**2) ** 2).sum()
    den = (u1 * (1.0 - 5.0 * u2[good])).sum()
    sigma = np.sqrt(num / (den * (den - 1.0)))  # see note in above reference

    return sigma


def workerPPXF(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process.
    """
    for (
        templates,
        bin_data,
        noise,
        velscale,
        start,
        goodPixels_ppxf,
        nmoments,
        mdeg,
        reddening,
        doclean,
        logLam,
        offset,
        velscale_ratio,
        nsims,
        nbins,
        i,
        optimal_template_in,
    ) in iter(inQueue.get, "STOP"):
        (
            sol,
            ppxf_reddening,
            bestfit,
            optimal_template,
            mc_results,
            formal_error,
            spectral_mask,
        ) = run_ppxf(
            templates,
            bin_data,
            noise,
            velscale,
            start,
            goodPixels_ppxf,
            nmoments,
            mdeg,
            reddening,
            doclean,
            logLam,
            offset,
            velscale_ratio,
            nsims,
            nbins,
            i,
            optimal_template_in,
        )

        outQueue.put(
            (
                i,
                sol,
                ppxf_reddening,
                bestfit,
                optimal_template,
                mc_results,
                formal_error,
                spectral_mask,
            )
        )


def run_ppxf(
    templates,
    log_bin_data,
    log_bin_error,
    velscale,
    start,
    goodPixels,
    nmoments,
    mdeg,
    reddening,
    doclean,
    logLam,
    offset,
    velscale_ratio,
    nsims,
    nbins,
    i,
    optimal_template_in,
):
    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    stellar kinematics.
    """
    printStatus.progressBar(i, nbins, barLength=50)

    try:
        # Call PPXF for first time to get optimal template
        if len(optimal_template_in) == 1:
            print("Running pPXF for the first time")
            pp = ppxf(
                templates,
                log_bin_data,
                log_bin_error,
                velscale,
                start,
                goodpixels=goodPixels,
                plot=True,
                quiet=True,
                moments=nmoments,
                degree=-1,
                mdegree=mdeg,
                reddening=reddening,
                lam=np.exp(logLam),
                velscale_ratio=velscale_ratio,
                vsyst=offset,
            )
        else:
            # First Call PPXF - do fit and estimate noise
            # use fake noise for first iteration
            fake_noise = np.full_like(log_bin_data, 1.0)

            pp_step1 = ppxf(
                optimal_template_in,
                log_bin_data,
                fake_noise,
                velscale,
                start,
                goodpixels=goodPixels,
                plot=False,
                quiet=True,
                moments=nmoments,
                degree=-1,
                mdegree=mdeg,
                reddening=reddening,
                lam=np.exp(logLam),
                velscale_ratio=velscale_ratio,
                vsyst=offset,
            )

            # Find a proper estimate of the noise
            noise_orig = biweight_location(log_bin_error[goodPixels])
            noise_est = robust_sigma(
                pp_step1.galaxy[goodPixels] - pp_step1.bestfit[goodPixels]
            )

            # Calculate the new noise, and the sigma of the distribution.
            noise_new = log_bin_error * (noise_est / noise_orig)
            noise_new_std = robust_sigma(noise_new)

            # A temporary fix for the noise issue where a single high S/N spaxel causes clipping of the entire spectrum
            noise_new[np.where(noise_new <= noise_est - noise_new_std)] = noise_est

            ################ 2 ##################
            # Second Call PPXF - use best-fitting template, determine outliers
            # only do this if doclean is set
            if doclean == True:
                pp_step2 = ppxf(
                    optimal_template_in,
                    log_bin_data,
                    noise_new,
                    velscale,
                    start,
                    goodpixels=goodPixels,
                    plot=False,
                    quiet=True,
                    moments=nmoments,
                    degree=-1,
                    mdegree=mdeg,
                    reddening=reddening,
                    lam=np.exp(logLam),
                    velscale_ratio=velscale_ratio,
                    vsyst=offset,
                    clean=True,
                )

                # update goodpixels
                goodPixels = pp_step2.goodpixels

                # repeat noise scaling # Find a proper estimate of the noise
                noise_orig = biweight_location(log_bin_error[goodPixels])
                noise_est = robust_sigma(
                    pp_step1.galaxy[goodPixels] - pp_step2.bestfit[goodPixels]
                )

                # Calculate the new noise, and the sigma of the distribution.
                noise_new = log_bin_error * (noise_est / noise_orig)
                noise_new_std = robust_sigma(noise_new)

            # A temporary fix for the noise issue where a single high S/N spaxel
            # causes clipping of the entire spectrum
            noise_new[np.where(noise_new <= noise_est - noise_new_std)] = noise_est

            ################ 3 ##################
            # Third Call PPXF - use all templates, get best-fit
            pp = ppxf(
                templates,
                log_bin_data,
                noise_new,
                velscale,
                start,
                goodpixels=goodPixels,
                plot=False,
                quiet=True,
                moments=nmoments,
                degree=-1,
                mdegree=mdeg,
                reddening=reddening,
                lam=np.exp(logLam),
                velscale_ratio=velscale_ratio,
                vsyst=offset,
            )

        # update goodpixels again
        goodPixels = pp.goodpixels

        # make spectral mask
        spectral_mask = np.full_like(log_bin_data, 0.0)
        spectral_mask[goodPixels] = 1.0

        # Make the unconvolved optimal stellar template
        normalized_weights = pp.weights / np.sum(pp.weights)
        optimal_template = np.zeros(templates.shape[0])
        for j in range(0, templates.shape[1]):
            optimal_template = (
                optimal_template + templates[:, j] * normalized_weights[j]
            )

        # Correct the formal errors assuming that the fit is good
        formal_error = pp.error * np.sqrt(pp.chi2)

        # Do MC-Simulations
        sol_MC = np.zeros((nsims, nmoments))
        mc_results = np.zeros(nmoments)
        for o in range(0, nsims):
            # Add noise to bestfit:
            #   - Draw random numbers from normal distribution with mean of 0 and sigma of 1 (np.random.normal(0,1,npix)
            #   - standard deviation( (galaxy spectrum - bestfit)[goodpix] )
            noisy_bestfit = pp.bestfit + np.random.normal(
                0, 1, len(log_bin_data)
            ) * np.std(log_bin_data[goodPixels] - pp.bestfit[goodPixels])

            mc = ppxf(
                templates,
                noisy_bestfit,
                log_bin_error,
                velscale,
                start,
                goodpixels=goodPixels,
                plot=False,
                quiet=True,
                moments=nmoments,
                degree=-1,
                mdegree=mdeg,
                velscale_ratio=velscale_ratio,
                vsyst=offset,
                bias=0.0,
            )
            sol_MC[o, :] = mc.sol[:]

        if nsims != 0:
            mc_results = np.nanstd(sol_MC, axis=0)

        return (
            pp.sol[:],
            pp.reddening,
            pp.bestfit,
            optimal_template,
            mc_results,
            formal_error,
            spectral_mask,
        )

    except:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


def save_ppxf(
    config,
    ppxf_result,
    ppxf_reddening,
    mc_results,
    formal_error,
    ppxf_bestfit,
    logLam,
    goodPixels,
    optimal_template,
    logLam_template,
    npix,
    spectral_mask,
    optimal_template_comb,
    bin_data,
):
    """Saves all results to disk."""
    # SAVE BESTFIT
    outfits_ppxf = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_kin-bestfit-cont.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_kin-bestfit-cont.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with PPXF bestfit
    cols = []
    cols.append(fits.Column(name="BESTFIT", format=str(npix) + "D", array=ppxf_bestfit))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "BESTFIT"

    # Table HDU with PPXF logLam
    cols = []
    cols.append(fits.Column(name="LOGLAM", format="D", array=logLam))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = "LOGLAM"

    # Table HDU with PPXF goodpixels
    cols = []
    cols.append(fits.Column(name="GOODPIX", format="J", array=goodPixels))
    goodpixHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    goodpixHDU.name = "GOODPIX"

    # Table HDU with PPXF goodpixels
    cols = []
    cols.append(fits.Column(name="SPEC", format=str(npix) + "D", array=bin_data.T))
    specHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    specHDU.name = "SPEC"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["KIN"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["KIN"])
    logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["KIN"])
    goodpixHDU = _auxiliary.saveConfigToHeader(goodpixHDU, config["KIN"])
    specHDU = _auxiliary.saveConfigToHeader(specHDU, config["KIN"])

    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, goodpixHDU, specHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_kin-bestfit-cont.fits"
    )
    logging.info("Wrote: " + outfits_ppxf)

    

def createContinuumCube(config):
    """
    Perform the measurement of stellar kinematics, using the pPXF routine. This
    function basically read all necessary input data, hands it to pPXF, and
    saves the outputs following the GIST conventions.
    """
    # Read data from file
    hdu = fits.open(
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_BinSpectra.fits"
    )
    bin_data = np.array(hdu[1].data.SPEC.T)
    bin_err = np.array(hdu[1].data.ESPEC.T)
    logLam = np.array(hdu[2].data.LOGLAM)
    idx_lam = np.where(
        np.logical_and(
            np.exp(logLam) > config["KIN"]["LMIN"],
            np.exp(logLam) < config["KIN"]["LMAX"],
        )
    )[0]
    bin_data = bin_data[idx_lam, :]
    bin_err = bin_err[idx_lam, :]
    logLam = logLam[idx_lam]
    npix = bin_data.shape[0]
    nbins = bin_data.shape[1]
    ubins = np.arange(0, nbins)
    velscale = hdu[0].header["VELSCALE"]

    # Read LSF information

    LSF_Data, LSF_Templates = _auxiliary.getLSF(config, "CONT")  # added input of module

    # Prepare templates
    velscale_ratio = 2
    logging.info("Using full spectral library for PPXF")
    (
        templates,
        lamRange_spmod,
        logLam_template,
        ntemplates,
    ) = _prepareTemplates.prepareTemplates_Module(
        config,
        config["CONT"]["LMIN"],
        config["CONT"]["LMAX"],
        velscale / velscale_ratio,
        LSF_Data,
        LSF_Templates,
        "CONT",
    )[
        :4
    ]
    templates = templates.reshape((templates.shape[0], ntemplates))

    # Last preparatory steps
    offset = (logLam_template[0] - logLam[0]) * C
    # noise  = np.ones((npix,nbins))
    noise = bin_err  # is actual noise, not variance
    nsims = config["CONT"]["MC_PPXF"]

    # Initial guesses
    start = np.zeros((nbins, 2))
    if (
        os.path.isfile(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin-guess.fits"
        )
        == True
    ):
        printStatus.done(
            "Using V and SIGMA from '"
            + config["GENERAL"]["RUN_ID"]
            + "_kin-guess.fits' as initial guesses"
        )
        logging.info(
            "Using V and SIGMA from '"
            + config["GENERAL"]["RUN_ID"]
            + "_kin-guess.fits' as initial guesses"
        )
        guess = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin-guess.fits"
        )[1].data
        start[:, 0] = guess.V
        start[:, 1] = guess.SIGMA
    else:
        # Use the same initial guess for all bins, as stated in MasterConfig
        printStatus.done(
            "Using V and SIGMA from the MasterConfig file as initial guesses"
        )
        logging.info("Using V and SIGMA from the MasterConfig file as initial guesses")
        start[:, 0] = 0.0
        start[:, 1] = config["CONT"]["SIGMA"]

    # Define goodpixels
    goodPixels_ppxf = _auxiliary.spectralMasking(
        config, config["CONT"]["SPEC_MASK"], logLam
    )

    # Array to store results of ppxf
    ppxf_result = np.zeros((nbins, 6))
    ppxf_reddening = np.zeros(nbins)
    ppxf_bestfit = np.zeros((nbins, npix))
    optimal_template = np.zeros((nbins, templates.shape[0]))
    mc_results = np.zeros((nbins, 6))
    formal_error = np.zeros((nbins, 6))
    spectral_mask = np.zeros((nbins, bin_data.shape[0]))

    # ====================
    # Run PPXF once on combined mean spectrum to get a single optimal template
    comb_spec = np.nanmean(bin_data[:, :], axis=1)
    comb_espec = np.nanmean(bin_err[:, :], axis=1)
    optimal_template_init = [0]

    (
        tmp_ppxf_result,
        tmp_ppxf_reddening,
        tmp_ppxf_bestfit,
        optimal_template_out,
        tmp_mc_results,
        tmp_formal_error,
        tmp_spectral_mask,
    ) = run_ppxf(
        templates,
        comb_spec,
        comb_espec,
        velscale,
        start[0, :],
        goodPixels_ppxf,
        config["CONT"]["MOM"],
        config["CONT"]["MDEG"],
        config["CONT"]["REDDENING"],
        config["CONT"]["DOCLEAN"],
        logLam,
        offset,
        velscale_ratio,
        nsims,
        nbins,
        0,
        optimal_template_init,
    )
    # now define the optimal template that we'll use throughout
    optimal_template_comb = optimal_template_out

    # ====================
    # Run PPXF
    start_time = time.time()
    if config["GENERAL"]["PARALLEL"] == True:
        printStatus.running("Running PPXF in parallel mode")
        logging.info("Running PPXF in parallel mode")

        # Define run_ppxf_wrapper function to be run in parallel
        def worker(i):
            return run_ppxf(
                templates,
                bin_data[:, i],
                noise[:, i],
                velscale,
                start[i, :],
                goodPixels_ppxf,
                config["CONT"]["MOM"],
                config["CONT"]["MDEG"],
                config["CONT"]["REDDENING"],
                config["CONT"]["DOCLEAN"],
                logLam,
                offset,
                velscale_ratio,
                nsims,
                nbins,
                i,
                optimal_template_comb,
            )

        # Create a pool of threads
        with ThreadPoolExecutor(max_workers=min(32, config["GENERAL"]["NCPU"]+4)) as executor:

            # Use a list comprehension to create a list of Future objects
            futures = [executor.submit(worker, i) for i in range(nbins)]

            # Iterate over the futures as they complete
            for future in as_completed(futures):
                # Get the result from the future
                result = future.result()

                # Get the index of the future in the list
                i = futures.index(future)

                # Assign the results to the arrays
                (
                    ppxf_result[i, : config["CONT"]["MOM"]],
                    ppxf_reddening[i],
                    ppxf_bestfit[i, :],
                    optimal_template[i, :],
                    mc_results[i, : config["CONT"]["MOM"]],
                    formal_error[i, : config["CONT"]["MOM"]],
                    spectral_mask[i, :],
                ) = result

        printStatus.updateDone("Running PPXF in parallel mode", progressbar=True)

    elif config["GENERAL"]["PARALLEL"] == False:
        printStatus.running("Running PPXF in serial mode")
        logging.info("Running PPXF in serial mode")
        for i in range(0, nbins):
            # for i in range(1, 2):
            (
                ppxf_result[i, : config["CONT"]["MOM"]],
                ppxf_reddening[i],
                ppxf_bestfit[i, :],
                optimal_template[i, :],
                mc_results[i, : config["CONT"]["MOM"]],
                formal_error[i, : config["CONT"]["MOM"]],
                spectral_mask[i, :],
            ) = run_ppxf(
                templates,
                bin_data[:, i],
                noise[:, i],
                velscale,
                start[i, :],
                goodPixels_ppxf,
                config["CONT"]["MOM"],
                config["CONT"]["MDEG"],
                config["CONT"]["REDDENING"],
                config["CONT"]["DOCLEAN"],
                logLam,
                offset,
                velscale_ratio,
                nsims,
                nbins,
                i,
                optimal_template_comb,
            )
        printStatus.updateDone("Running PPXF in serial mode", progressbar=True)

    print(
        "             Running PPXF on %s spectra took %.2fs using %i cores"
        % (nbins, time.time() - start_time, config["GENERAL"]["NCPU"])
    )
    logging.info(
        "Running PPXF on %s spectra took %.2fs using %i cores"
        % (nbins, time.time() - start_time, config["GENERAL"]["NCPU"])
    )

    # Check for exceptions which occurred during the analysis
    idx_error = np.where(np.isnan(ppxf_result[:, 0]) == True)[0]
    if len(idx_error) != 0:
        printStatus.warning(
            "There was a problem in the analysis of the spectra with the following BINID's: "
        )
        print("             " + str(idx_error))
        logging.warning(
            "There was a problem in the analysis of the spectra with the following BINID's: "
            + str(idx_error)
        )
    else:
        print("             " + "There were no problems in the analysis.")
        logging.info("There were no problems in the analysis.")
    print("")

    # Save stellar kinematics to file
    save_ppxf(
        config,
        ppxf_result,
        ppxf_reddening,
        mc_results,
        formal_error,
        ppxf_bestfit,
        logLam,
        goodPixels_ppxf,
        optimal_template,
        logLam_template,
        npix,
        spectral_mask,
        optimal_template_comb,
        bin_data,
    )

    # Return

    return None