import logging
import os
import time

import h5py
import numpy as np
from astropy.io import fits
from astropy.stats import biweight_location
from ppxf.ppxf import ppxf
from printStatus import printStatus

from ngistPipeline.auxiliary import _auxiliary
from ngistPipeline.auxiliary.batch_ppxf import BatchExecutor
from ngistPipeline.prepareTemplates import _prepareTemplates

robust_sigma = _auxiliary.robust_sigma

# PHYSICAL CONSTANTS
C = 299792.458  # km/s


"""
PURPOSE:
  This module creates a continuum and line-only cube.
  Basically, it acts as an interface between pipeline and the pPXF routine from
  Cappellari & Emsellem 2004 (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
  ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C).
"""


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
    # printStatus.progressBar(i, nbins, barLength=50)

    try:
        # Require valid goodpixels and normalisation for PPXF
        if goodPixels is None or len(goodPixels) < 10:
            raise ValueError("goodPixels empty or too few pixels for PPXF")
        median_log_bin_data = np.nanmedian(log_bin_data)
        if not np.isfinite(median_log_bin_data) or median_log_bin_data <= 0:
            raise ValueError(
                "spectrum median is not finite and positive (got %s)" % median_log_bin_data
            )

        # normalise galaxy spectra and noise
        log_bin_error = log_bin_error / median_log_bin_data
        log_bin_data = log_bin_data / median_log_bin_data

        # Remove goodPixels where data or error is NaN, non-finite, or non-positive
        # (e.g. from NaN variance channels in MUSE cubes)
        valid = (
            np.isfinite(log_bin_data[goodPixels])
            & np.isfinite(log_bin_error[goodPixels])
            & (log_bin_error[goodPixels] > 0)
        )
        goodPixels = goodPixels[valid]
        if len(goodPixels) < 10:
            raise ValueError(
                "Too few valid goodPixels after removing NaN/non-positive noise (%d remain)"
                % len(goodPixels)
            )

        # Replace any remaining NaN in the full arrays with safe values so pPXF
        # doesn't choke on non-goodPixel entries it may still inspect.
        nan_data = ~np.isfinite(log_bin_data)
        nan_err = ~np.isfinite(log_bin_error) | (log_bin_error <= 0)
        log_bin_data[nan_data] = 0.0
        log_bin_error[nan_err] = 1e10  # large error effectively down-weights these pixels

        # Call PPXF for first time to get optimal template
        if len(optimal_template_in) == 1:
            printStatus.running("Running pPXF for the first time")
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
        optimal_template = templates @ normalized_weights

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
        
        # add normalisation factor back in main results
        pp.bestfit = pp.bestfit * median_log_bin_data
        if pp.reddening is not None:
            pp.reddening = pp.reddening * median_log_bin_data

        return (
            pp.sol[:],
            pp.reddening,
            pp.bestfit,
            optimal_template,
            mc_results,
            formal_error,
            spectral_mask,
        )

    except Exception:
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
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["CONT"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["CONT"])
    logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["CONT"])
    goodpixHDU = _auxiliary.saveConfigToHeader(goodpixHDU, config["CONT"])
    specHDU = _auxiliary.saveConfigToHeader(specHDU, config["CONT"])

    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, goodpixHDU, specHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_kin-bestfit-cont.fits"
    )
    logging.info("Wrote: " + outfits_ppxf)

    

def _cont_bin_worker(bin_idx, shared, params):
    """Module-level worker for BatchExecutor: continuum fit for one bin."""
    return run_ppxf(
        shared["templates"],
        shared["bin_data"][:, bin_idx].copy(),
        shared["noise"][:, bin_idx].copy(),
        params["velscale"],
        shared["start"][bin_idx, :].copy(),
        params["goodPixels_ppxf"].copy(),
        params["nmoments"],
        params["mdeg"],
        params["reddening"],
        params["doclean"],
        params["logLam"],
        params["offset"],
        params["velscale_ratio"],
        params["nsims"],
        params["nbins"],
        bin_idx,
        params["optimal_template_comb"],
    )


def createContinuumCube(config):
    """
    Perform the measurement of stellar kinematics, using the pPXF routine. This
    function basically read all necessary input data, hands it to pPXF, and
    saves the outputs following the GIST conventions.
    """
    # Read data from file
    infile = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_BinSpectra.hdf5"
    printStatus.running("Reading: " + config["GENERAL"]["RUN_ID"] + "_BinSpectra.hdf5")
    
    # Open the HDF5 file
    with h5py.File(infile, 'r') as f:
        
        # Read the data from the file
        logLam = f['LOGLAM'][:]
        idx_lam = np.where(
        np.logical_and(
            np.exp(logLam) > config["CONT"]["LMIN"],
            np.exp(logLam) < config["CONT"]["LMAX"],
        )
        )[0]

        bin_data = f['SPEC'][:][idx_lam, :]
        bin_err = f['ESPEC'][:][idx_lam, :]
        velscale = f.attrs['VELSCALE']
    logLam = logLam[idx_lam]
    npix = bin_data.shape[0]
    nbins = bin_data.shape[1]
    ubins = np.arange(0, nbins)

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
        with fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin-guess.fits",
            memmap=True,
        ) as guess_hdu:
            guess = guess_hdu[1].data
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

        shared_arrays = {
            "templates": templates,
            "bin_data": bin_data,
            "noise": noise,
            "start": start,
        }
        params = {
            "velscale": velscale,
            "goodPixels_ppxf": goodPixels_ppxf,
            "nmoments": config["CONT"]["MOM"],
            "mdeg": config["CONT"]["MDEG"],
            "reddening": config["CONT"]["REDDENING"],
            "doclean": config["CONT"]["DOCLEAN"],
            "logLam": logLam,
            "offset": offset,
            "velscale_ratio": velscale_ratio,
            "nsims": nsims,
            "nbins": nbins,
            "optimal_template_comb": optimal_template_comb,
        }
        fail_value = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

        wave_size = config["GENERAL"].get("WAVE_SIZE", 0)
        executor = BatchExecutor(
            ncpu=config["GENERAL"]["NCPU"],
            wave_size=wave_size,
            scratch_dir=config["GENERAL"]["OUTPUT"],
        )
        ppxf_tmp = executor.run(
            worker_fn=_cont_bin_worker,
            shared_arrays=shared_arrays,
            params=params,
            bin_indices=np.arange(nbins),
            desc="CONT ppxf",
            fail_value=fail_value,
        )

        for i in range(nbins):
            ppxf_result[i, : config["CONT"]["MOM"]] = ppxf_tmp[i][0]
            ppxf_reddening[i] = ppxf_tmp[i][1]
            ppxf_bestfit[i, :] = ppxf_tmp[i][2]
            optimal_template[i, :] = ppxf_tmp[i][3]
            mc_results[i, : config["CONT"]["MOM"]] = ppxf_tmp[i][4]
            formal_error[i, : config["CONT"]["MOM"]] = ppxf_tmp[i][5]
            spectral_mask[i, :] = ppxf_tmp[i][6]

        printStatus.updateDone("Running PPXF in parallel mode", progressbar=False)

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
        printStatus.updateDone("Running PPXF in serial mode", progressbar=False)

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