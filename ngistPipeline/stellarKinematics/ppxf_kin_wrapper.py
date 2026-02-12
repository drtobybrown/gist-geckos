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
  This module executes the analysis of stellar kinematics in the pipeline.
  Basically, it acts as an interface between pipeline and the pPXF routine from
  Cappellari & Emsellem 2004 (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
  ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C).

ADAPTIVE COARSE-TO-FINE TEMPLATE GRID (ADAPTIVE_GRID):
  When KIN.ADAPTIVE_GRID is set (e.g. [4, 2, 1] or [4, 2, 1, 2, 1, 0]):
  1. First pass: run pPXF on a sparse (age, metal, alpha) grid using the given steps.
  2. Find the best template from the returned weights (argmax).
  3. Second pass: run pPXF on a fine subgrid (a box around that best template).
  4. Return the second-pass result so kinematics are full-resolution in that region.
  This reduces cost when the full grid is large (e.g. 50x15x3) while keeping the
  same best-fit behaviour. Config takes precedence over REDUCED_GRID when both are set.
"""

def run_ppxf(
    templates,
    log_bin_data,
    log_bin_error,
    velscale,
    start,
    bias,
    goodPixels,
    nmoments,
    adeg,
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
        # If combined run failed, caller may pass scalar np.nan; treat as first run (full templates)
        try:
            use_first_run = len(optimal_template_in) == 1
        except (TypeError, AttributeError):
            use_first_run = True

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

        #calculate the snr before the fit (may be used for bias)
        snr_prefit = np.nanmedian(log_bin_data[goodPixels]/log_bin_error[goodPixels])

        # Call PPXF for first time to get optimal template
        if use_first_run:
            if i == 0:
                printStatus.running("Running pPXF for the first time")
            pp = ppxf(
                templates,
                log_bin_data,
                log_bin_error,
                velscale,
                start,
                goodpixels=goodPixels,
                plot=False,
                quiet=True,
                moments=nmoments,
                degree=adeg,
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
                degree=adeg,
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
                    degree=adeg,
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
                    pp_step2.galaxy[goodPixels] - pp_step2.bestfit[goodPixels]
                )

                # Calculate the new noise, and the sigma of the distribution.
                noise_new = log_bin_error * (noise_est / noise_orig)
                noise_new_std = robust_sigma(noise_new)

            # A fix for the noise issue where a single high S/N spaxel
            # causes clipping of the entire spectrum
            noise_new[np.where(noise_new <= noise_est - noise_new_std)] = noise_est

            ################ 3 ##################
            # Third Call PPXF - use all templates, get best-fit

            if bias == 'muse_snr_prefit':
                bias = 0.01584469*snr_prefit**0.54639427 - 0.01687899
            elif bias == 'muse':
                # recalculate the snr
                snr_step2 = np.nanmedian(log_bin_data[goodPixels]/noise_new[goodPixels])
                bias = 0.01584469*snr_step2**0.54639427 - 0.01687899
            else:
                bias = bias

            pp = ppxf(
                templates,
                log_bin_data,
                noise_new,
                velscale,
                start,
                bias=bias,
                goodpixels=goodPixels,
                plot=False,
                quiet=True,
                moments=nmoments,
                degree=adeg,
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

        # Calculate the true S/N from the residual the long version
        #noise_est_final = robust_sigma(pp.galaxy[goodPixels] - pp.bestfit[goodPixels])
        #noise_orig = biweight_location(log_bin_error[goodPixels])
        #noise_final = log_bin_error * (noise_est_final / noise_orig)
        #noise_final_std = robust_sigma(noise_final)
        #noise_final[np.where(noise_final <= noise_est_final - noise_final_std)] = noise_est_final
        #snr_postfit = np.nanmedian(pp.galaxy[goodPixels]/noise_final[goodPixels])
        
        # Calculate the true S/N from the residual the short version
        noise_est = robust_sigma(pp.galaxy[goodPixels] - pp.bestfit[goodPixels])
        snr_postfit = np.nanmedian(pp.galaxy[goodPixels]/noise_est)

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
                degree=adeg,
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

        # Weights (length = n_templates) for adaptive coarse-to-fine grid refinement
        weights_out = np.asarray(pp.weights, dtype=np.float64)

        return (
            pp.sol[:],
            pp.reddening,
            pp.bestfit,
            optimal_template,
            mc_results,
            formal_error,
            spectral_mask,
            snr_postfit,
            weights_out,
        )

    except Exception as e:
        if i == 0:
            logging.warning(
                "PPXF failed on combined spectrum (or first bin): %s", e, exc_info=True
            )
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None)


def _build_coarse_idx(nAges, nMetal, nAlpha, step_age, step_metal, step_alpha):
    """Linear indices for a coarse (age, metal, alpha) grid. Order: age fastest, then metal, then alpha."""
    idx_list = []
    for i in range(nAlpha):
        for k in range(nMetal):
            for j in range(nAges):
                if (j % step_age == 0) and (k % step_metal == 0) and (i % step_alpha == 0):
                    t = j + nAges * k + nAges * nMetal * i
                    idx_list.append(t)
    return np.array(idx_list, dtype=np.intp)


def _build_fine_window_idx(nAges, nMetal, nAlpha, j0, k0, i0, radius_age, radius_metal, radius_alpha):
    """Linear indices for a (age, metal, alpha) box around (j0, k0, i0)."""
    j_lo = max(0, j0 - radius_age)
    j_hi = min(nAges, j0 + radius_age + 1)
    k_lo = max(0, k0 - radius_metal)
    k_hi = min(nMetal, k0 + radius_metal + 1)
    i_lo = max(0, i0 - radius_alpha)
    i_hi = min(nAlpha, i0 + radius_alpha + 1)
    idx_list = []
    for i in range(i_lo, i_hi):
        for k in range(k_lo, k_hi):
            for j in range(j_lo, j_hi):
                t = j + nAges * k + nAges * nMetal * i
                idx_list.append(t)
    return np.array(idx_list, dtype=np.intp)


def build_grid_config(module_config, nAges, nMetal, nAlpha, log_prefix=""):
    """
    Build reduced_idx and/or adaptive_grid_config from a module config (KIN, CONT, SFH, etc.).
    Returns (reduced_idx, adaptive_grid_config). Either can be None.
    ADAPTIVE_GRID takes precedence over REDUCED_GRID when both are set.
    """
    reduced_idx = None
    adaptive_grid_config = None
    valid = (
        np.isscalar(nAges) and np.isscalar(nMetal) and np.isscalar(nAlpha)
        and nAges > 0 and nMetal > 0 and nAlpha > 0
    )
    if not valid:
        return reduced_idx, adaptive_grid_config
    if module_config.get("ADAPTIVE_GRID"):
        ag = module_config["ADAPTIVE_GRID"]
        if isinstance(ag, (list, tuple)) and len(ag) >= 3:
            coarse_step = [max(1, int(ag[0])), max(1, int(ag[1])), max(1, int(ag[2]))]
            fine_radius = [2, 1, 0]
            if len(ag) >= 6:
                fine_radius = [int(ag[3]), int(ag[4]), int(ag[5])]
            adaptive_grid_config = {
                "nAges": nAges,
                "nMetal": nMetal,
                "nAlpha": nAlpha,
                "coarse_step": coarse_step,
                "fine_radius": fine_radius,
            }
            logging.info(
                "%sadaptive grid: coarse steps [%d,%d,%d], fine radius [%d,%d,%d]",
                log_prefix, coarse_step[0], coarse_step[1], coarse_step[2],
                fine_radius[0], fine_radius[1], fine_radius[2],
            )
    elif module_config.get("REDUCED_GRID"):
        steps = module_config["REDUCED_GRID"]
        if isinstance(steps, (list, tuple)) and len(steps) >= 3:
            sa, sm, salpha = max(1, int(steps[0])), max(1, int(steps[1])), max(1, int(steps[2]))
            idx_list = []
            for i in range(nAlpha):
                for k in range(nMetal):
                    for j in range(nAges):
                        if (j % sa == 0) and (k % sm == 0) and (i % salpha == 0):
                            idx_list.append(j + nAges * k + nAges * nMetal * i)
            reduced_idx = np.array(idx_list, dtype=np.intp)
            logging.info(
                "%sreduced grid: %d templates (steps age=%d metal=%d alpha=%d)",
                log_prefix, len(reduced_idx), sa, sm, salpha,
            )
    return reduced_idx, adaptive_grid_config


def _run_ppxf_adaptive(
    templates_full,
    nAges,
    nMetal,
    nAlpha,
    log_bin_data,
    log_bin_error,
    velscale,
    start,
    bias,
    goodPixels,
    nmoments,
    adeg,
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
    coarse_step,
    fine_radius,
):
    """
    Coarse-to-fine template grid: run pPXF on a sparse grid, find best (age, metal, alpha),
    then re-run on a fine subgrid around it. Returns the same 8-tuple as run_ppxf (no weights).
    """
    sa, sm, salpha = max(1, int(coarse_step[0])), max(1, int(coarse_step[1])), max(1, int(coarse_step[2]))
    coarse_idx = _build_coarse_idx(nAges, nMetal, nAlpha, sa, sm, salpha)
    templates_coarse = templates_full[:, coarse_idx]

    out = run_ppxf(
        templates_coarse,
        log_bin_data,
        log_bin_error,
        velscale,
        start,
        bias,
        goodPixels,
        nmoments,
        adeg,
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
    (
        coarse_sol,
        coarse_reddening,
        coarse_bestfit,
        coarse_optimal,
        coarse_mc,
        coarse_formal,
        coarse_mask,
        coarse_snr,
        weights_coarse,
    ) = out

    if weights_coarse is None or not np.any(np.isfinite(weights_coarse)):
        return out[:8] + (None,)
    if not np.all(np.isfinite(coarse_sol)):
        return out[:8] + (None,)

    best_local = np.argmax(weights_coarse)
    best_full_t = int(coarse_idx[best_local])
    j0 = best_full_t % nAges
    k0 = (best_full_t // nAges) % nMetal
    i0 = best_full_t // (nAges * nMetal)

    ra, rm, ral = int(fine_radius[0]), int(fine_radius[1]), int(fine_radius[2])
    fine_idx = _build_fine_window_idx(nAges, nMetal, nAlpha, j0, k0, i0, ra, rm, ral)

    if len(fine_idx) <= len(coarse_idx):
        return out[:8] + (None,)

    templates_fine = templates_full[:, fine_idx]
    n_start = max(2, min(nmoments, len(coarse_sol)))
    start_fine = np.zeros(max(2, nmoments))
    start_fine[:n_start] = np.asarray(coarse_sol[:n_start], dtype=np.float64)
    opt_in_fine = [np.asarray(coarse_optimal, dtype=np.float64)]

    out_fine = run_ppxf(
        templates_fine,
        log_bin_data,
        log_bin_error,
        velscale,
        start_fine,
        bias,
        goodPixels,
        nmoments,
        adeg,
        mdeg,
        reddening,
        doclean,
        logLam,
        offset,
        velscale_ratio,
        nsims,
        nbins,
        i,
        opt_in_fine,
    )
    return out_fine[:8] + (None,)


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
    snr_postfit,
):
    """Saves all results to disk."""
    # ========================
    # SAVE RESULTS
    outfits_ppxf = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_kin.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_kin.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with PPXF output data
    cols = []
    cols.append(fits.Column(name="V", format="D", array=ppxf_result[:, 0]))
    cols.append(fits.Column(name="SIGMA", format="D", array=ppxf_result[:, 1]))
    if np.any(ppxf_result[:, 2]) != 0:
        cols.append(fits.Column(name="H3", format="D", array=ppxf_result[:, 2]))
    if np.any(ppxf_result[:, 3]) != 0:
        cols.append(fits.Column(name="H4", format="D", array=ppxf_result[:, 3]))
    if np.any(ppxf_result[:, 4]) != 0:
        cols.append(fits.Column(name="H5", format="D", array=ppxf_result[:, 4]))
    if np.any(ppxf_result[:, 5]) != 0:
        cols.append(fits.Column(name="H6", format="D", array=ppxf_result[:, 5]))

    if np.any(mc_results[:, 0]) != 0:
        cols.append(fits.Column(name="ERR_V", format="D", array=mc_results[:, 0]))
    if np.any(mc_results[:, 1]) != 0:
        cols.append(fits.Column(name="ERR_SIGMA", format="D", array=mc_results[:, 1]))
    if np.any(mc_results[:, 2]) != 0:
        cols.append(fits.Column(name="ERR_H3", format="D", array=mc_results[:, 2]))
    if np.any(mc_results[:, 3]) != 0:
        cols.append(fits.Column(name="ERR_H4", format="D", array=mc_results[:, 3]))
    if np.any(mc_results[:, 4]) != 0:
        cols.append(fits.Column(name="ERR_H5", format="D", array=mc_results[:, 4]))
    if np.any(mc_results[:, 5]) != 0:
        cols.append(fits.Column(name="ERR_H6", format="D", array=mc_results[:, 5]))

    cols.append(fits.Column(name="FORM_ERR_V", format="D", array=formal_error[:, 0]))
    cols.append(
        fits.Column(name="FORM_ERR_SIGMA", format="D", array=formal_error[:, 1])
    )
    if np.any(formal_error[:, 2]) != 0:
        cols.append(
            fits.Column(name="FORM_ERR_H3", format="D", array=formal_error[:, 2])
        )
    if np.any(formal_error[:, 3]) != 0:
        cols.append(
            fits.Column(name="FORM_ERR_H4", format="D", array=formal_error[:, 3])
        )
    if np.any(formal_error[:, 4]) != 0:
        cols.append(
            fits.Column(name="FORM_ERR_H5", format="D", array=formal_error[:, 4])
        )
    if np.any(formal_error[:, 5]) != 0:
        cols.append(
            fits.Column(name="FORM_ERR_H6", format="D", array=formal_error[:, 5])
        )

    # Add reddening if parameter is used
    if np.any(np.isnan(ppxf_reddening)) != True:
        cols.append(fits.Column(name="REDDENING", format="D", array=ppxf_reddening[:]))

    # Add True SNR calculated from residual
    cols.append(fits.Column(name="SNR_POSTFIT", format="D", array=snr_postfit[:]))

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "KIN_DATA"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["KIN"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["KIN"])
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)

    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_kin.fits")
    logging.info("Wrote: " + outfits_ppxf)

    # ========================
    # SAVE BESTFIT
    outfits_ppxf = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_kin-bestfit.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_kin-bestfit.fits")

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

    # Table HDU with ??? --> unclear what this is?
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
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_kin-bestfit.fits"
    )
    logging.info("Wrote: " + outfits_ppxf)

    # ============================
    # SAVE OPTIMAL TEMPLATE RESULT
    outfits = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_kin-optimalTemplates.fits"
    )
    printStatus.running(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_kin-optimalTemplates.fits"
    )

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with optimal templates
    cols = []
    cols.append(
        fits.Column(
            name="OPTIMAL_TEMPLATES",
            format=str(optimal_template.shape[1]) + "D",
            array=optimal_template,
        )
    )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "OPTIMAL_TEMPLATES"

    # Extension 2: Table HDU with logLam_templates
    cols = []
    cols.append(fits.Column(name="LOGLAM_TEMPLATE", format="D", array=logLam_template))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = "LOGLAM_TEMPLATE"

    # Extension 2: Table HDU with logLam_templates
    # Ensure 1D array (combined PPXF can fail and return scalar np.nan)
    opt_comb = optimal_template_comb
    if np.isscalar(opt_comb) or (isinstance(opt_comb, np.ndarray) and opt_comb.ndim == 0):
        opt_comb = np.full(optimal_template.shape[1], np.nan, dtype=float)
    else:
        opt_comb = np.atleast_1d(opt_comb).astype(float)
    cols = []
    cols.append(
        fits.Column(
            name="OPTIMAL_TEMPLATE_ALL", format="D", array=opt_comb
        )
    )
    combHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    combHDU.name = "OPTIMAL_TEMPLATE_ALL"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["KIN"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["KIN"])
    logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["KIN"])
    combHDU = _auxiliary.saveConfigToHeader(combHDU, config["KIN"])
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, combHDU])
    HDUList.writeto(outfits, overwrite=True)

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_kin-optimalTemplates.fits"
    )
    logging.info("Wrote: " + outfits)

    # ============================
    # SAVE SPECTRAL MASK RESULT
    outfits = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_kin-SpectralMask.fits"
    )
    printStatus.running(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_kin-SpectralMask.fits"
    )

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with optimal templates
    cols = []
    cols.append(
        fits.Column(
            name="SPECTRAL_MASK",
            format=str(spectral_mask.shape[1]) + "D",
            array=spectral_mask,
        )
    )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "SPECTRAL_MASK"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["KIN"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["KIN"])
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits, overwrite=True)

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_kin-SpectralMask.fits"
    )
    logging.info("Wrote: " + outfits)

def _kin_bin_worker(bin_idx, shared, params):
    """Module-level worker for BatchExecutor: fits one bin's stellar kinematics.

    Parameters
    ----------
    bin_idx : int
        Index into the shared ``bin_data`` / ``noise`` / ``start`` arrays.
    shared : dict
        Large arrays shared across workers (``templates``, ``bin_data``,
        ``noise``, ``start``).
    params : dict
        Scalar and small-array parameters shared across all bins.

    Returns
    -------
    tuple
        Same 8-element tuple as :func:`run_ppxf`.
    """
    ad = params.get("adaptive_grid_config")
    if ad is not None:
        return _run_ppxf_adaptive(
            shared["templates"],
            ad["nAges"],
            ad["nMetal"],
            ad["nAlpha"],
            shared["bin_data"][:, bin_idx].copy(),
            shared["noise"][:, bin_idx].copy(),
            params["velscale"],
            shared["start"][bin_idx, :].copy(),
            params["bias"],
            params["goodPixels_ppxf"].copy(),
            params["nmoments"],
            params["adeg"],
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
            ad["coarse_step"],
            ad["fine_radius"],
        )

    templates_use = (
        shared["templates"][:, params["reduced_idx"]]
        if params.get("reduced_idx") is not None
        else shared["templates"]
    )
    return run_ppxf(
        templates_use,
        shared["bin_data"][:, bin_idx].copy(),
        shared["noise"][:, bin_idx].copy(),
        params["velscale"],
        shared["start"][bin_idx, :].copy(),
        params["bias"],
        params["goodPixels_ppxf"].copy(),
        params["nmoments"],
        params["adeg"],
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


def extractStellarKinematics(config):
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
            np.exp(logLam) > config["KIN"]["LMIN"],
            np.exp(logLam) < config["KIN"]["LMAX"],
        )
        )[0]

        bin_data = f['SPEC'][idx_lam, :]
        bin_err = f['ESPEC'][idx_lam, :]
        velscale = f.attrs['VELSCALE']
    
    logLam = logLam[idx_lam]
    npix = bin_data.shape[0]
    nbins = bin_data.shape[1]
    ubins = np.arange(0, nbins)

    # Define bias value (even if moments == 2, because keyword needs to be passed on)
    if config["KIN"]["BIAS"] == 'Auto': # 'Auto' setting: bias=None
        bias = None
    elif config["KIN"]["BIAS"] != 'Auto':
        bias = config["KIN"]["BIAS"]

    # Test if bias is either a None or a float
    if (bias != None) & (bias != 'muse') & (bias != 'muse_snr_prefit') & \
        (isinstance(bias, int) == False) & (isinstance(bias, float) == False):
        printStatus.warning("Wrong Bias keyword, setting to None")
        bias = None


    # Read LSF information

    LSF_Data, LSF_Templates = _auxiliary.getLSF(config, "KIN")  # added input of module

    # Prepare templates
    velscale_ratio = 2
    full_template_result = _prepareTemplates.prepareTemplates_Module(
        config,
        config["KIN"]["LMIN"],
        config["KIN"]["LMAX"],
        velscale / velscale_ratio,
        LSF_Data,
        LSF_Templates,
        "KIN",
    )
    (
        templates,
        lamRange_spmod,
        logLam_template,
        ntemplates,
    ) = full_template_result[:4]
    templates = templates.reshape((templates.shape[0], ntemplates))

    # Optional reduced or adaptive (coarse-to-fine) age-metal-alpha grid
    reduced_idx = None
    adaptive_grid_config = None
    nAges, nMetal, nAlpha = None, None, None
    if len(full_template_result) >= 11:
        nAges, nMetal, nAlpha = full_template_result[8], full_template_result[9], full_template_result[10]
        reduced_idx, adaptive_grid_config = build_grid_config(
            config["KIN"], nAges, nMetal, nAlpha, log_prefix="KIN "
        )
    if reduced_idx is None and adaptive_grid_config is None:
        logging.info("Using full spectral library for PPXF")

    # Last preparatory steps
    offset = (logLam_template[0] - logLam[0]) * C
    # noise  = np.ones((npix,nbins))
    noise = bin_err  # is actual noise, not variance
    nsims = config["KIN"]["MC_PPXF"]

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
        start[:, 1] = config["KIN"]["SIGMA"]

    # Define goodpixels
    goodPixels_ppxf = _auxiliary.spectralMasking(
        config, config["KIN"]["SPEC_MASK"], logLam
    )

    # Array to store results of ppxf
    ppxf_result = np.zeros((nbins, 6))
    ppxf_reddening = np.zeros(nbins)
    ppxf_bestfit = np.zeros((nbins, npix))
    optimal_template = np.zeros((nbins, templates.shape[0]))
    mc_results = np.zeros((nbins, 6))
    formal_error = np.zeros((nbins, 6))
    spectral_mask = np.zeros((nbins, bin_data.shape[0]))
    snr_postfit = np.zeros(nbins)

    # ====================
    # Run PPXF once on combined mean spectrum to get a single optimal template
    comb_spec = np.nanmean(bin_data[:, :], axis=1)
    comb_espec = np.nanmean(bin_err[:, :], axis=1)
    optimal_template_init = [0]

    if adaptive_grid_config is not None:
        (
            _,
            _,
            _,
            optimal_template_out,
            _,
            _,
            _,
            _,
        ) = _run_ppxf_adaptive(
            templates,
            adaptive_grid_config["nAges"],
            adaptive_grid_config["nMetal"],
            adaptive_grid_config["nAlpha"],
            comb_spec,
            comb_espec,
            velscale,
            start[0, :],
            bias,
            goodPixels_ppxf,
            config["KIN"]["MOM"],
            config["KIN"]["ADEG"],
            config["KIN"]["MDEG"],
            config["KIN"]["REDDENING"],
            config["KIN"]["DOCLEAN"],
            logLam,
            offset,
            velscale_ratio,
            nsims,
            nbins,
            0,
            optimal_template_init,
            adaptive_grid_config["coarse_step"],
            adaptive_grid_config["fine_radius"],
        )
    else:
        templates_comb = templates[:, reduced_idx] if reduced_idx is not None else templates
        (
            tmp_ppxf_result,
            tmp_ppxf_reddening,
            tmp_ppxf_bestfit,
            optimal_template_out,
            tmp_mc_results,
            tmp_formal_error,
            tmp_spectral_mask,
            tmp_snr_postfit,
            _,
        ) = run_ppxf(
            templates_comb,
            comb_spec,
            comb_espec,
            velscale,
            start[0, :],
            bias,
            goodPixels_ppxf,
            config["KIN"]["MOM"],
            config["KIN"]["ADEG"],
            config["KIN"]["MDEG"],
            config["KIN"]["REDDENING"],
            config["KIN"]["DOCLEAN"],
            logLam,
            offset,
            velscale_ratio,
            nsims,
            nbins,
            0,
            optimal_template_init,
        )
    # If combined run failed, use sentinel so per-bin runs use full template library
    _valid = (
        optimal_template_out is not None
        and not np.isscalar(optimal_template_out)
        and hasattr(optimal_template_out, "__len__")
        and len(optimal_template_out) > 1
    )
    if _valid and isinstance(optimal_template_out, np.ndarray):
        _valid = (
            optimal_template_out.ndim >= 1
            and np.any(np.isfinite(optimal_template_out))
        )
    if not _valid:
        optimal_template_comb = [0]
        logging.info(
            "Combined-spectrum PPXF failed or invalid; per-bin runs will use full template library"
        )
    else:
        optimal_template_comb = optimal_template_out

    # ====================
    # Run PPXF
    start_time = time.time()
    if config["GENERAL"]["PARALLEL"] == True:
        printStatus.running("Running PPXF in parallel mode")
        logging.info("Running PPXF in parallel mode")

        # Shared arrays — large, read-only data accessed by every worker.
        # On Linux (CANFAR) these are shared via fork COW at zero cost.
        shared_arrays = {
            "templates": templates,
            "bin_data": bin_data,
            "noise": noise,
            "start": start,
        }

        # Scalar / small-array parameters broadcast once per worker.
        params = {
            "velscale": velscale,
            "bias": bias,
            "goodPixels_ppxf": goodPixels_ppxf,
            "nmoments": config["KIN"]["MOM"],
            "adeg": config["KIN"]["ADEG"],
            "mdeg": config["KIN"]["MDEG"],
            "reddening": config["KIN"]["REDDENING"],
            "doclean": config["KIN"]["DOCLEAN"],
            "logLam": logLam,
            "offset": offset,
            "velscale_ratio": velscale_ratio,
            "nsims": nsims,
            "nbins": nbins,
            "optimal_template_comb": optimal_template_comb,
            "reduced_idx": reduced_idx,
            "adaptive_grid_config": adaptive_grid_config,
        }

        # Sentinel for failed bins (must match run_ppxf failure return)
        fail_value = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None)

        wave_size = config["GENERAL"].get("WAVE_SIZE", 0)
        executor = BatchExecutor(
            ncpu=config["GENERAL"]["NCPU"],
            wave_size=wave_size,
            scratch_dir=config["GENERAL"]["OUTPUT"],
        )
        ppxf_tmp = executor.run(
            worker_fn=_kin_bin_worker,
            shared_arrays=shared_arrays,
            params=params,
            bin_indices=np.arange(nbins),
            desc="KIN ppxf",
            fail_value=fail_value,
        )

        # Unpack results
        for i in range(nbins):
            ppxf_result[i, : config["KIN"]["MOM"]] = ppxf_tmp[i][0]
            ppxf_reddening[i] = ppxf_tmp[i][1]
            ppxf_bestfit[i, :] = ppxf_tmp[i][2]
            optimal_template[i, :] = ppxf_tmp[i][3]
            mc_results[i, : config["KIN"]["MOM"]] = ppxf_tmp[i][4]
            formal_error[i, : config["KIN"]["MOM"]] = ppxf_tmp[i][5]
            spectral_mask[i, :] = ppxf_tmp[i][6]
            snr_postfit[i] = ppxf_tmp[i][7]

        printStatus.updateDone("Running PPXF in parallel mode", progressbar=False)

    elif config["GENERAL"]["PARALLEL"] == False:
        printStatus.running("Running PPXF in serial mode")
        logging.info("Running PPXF in serial mode")
        if adaptive_grid_config is not None:
            for i in range(0, nbins):
                out_serial = _run_ppxf_adaptive(
                    templates,
                    adaptive_grid_config["nAges"],
                    adaptive_grid_config["nMetal"],
                    adaptive_grid_config["nAlpha"],
                    bin_data[:, i],
                    noise[:, i],
                    velscale,
                    start[i, :],
                    bias,
                    goodPixels_ppxf,
                    config["KIN"]["MOM"],
                    config["KIN"]["ADEG"],
                    config["KIN"]["MDEG"],
                    config["KIN"]["REDDENING"],
                    config["KIN"]["DOCLEAN"],
                    logLam,
                    offset,
                    velscale_ratio,
                    nsims,
                    nbins,
                    i,
                    optimal_template_comb,
                    adaptive_grid_config["coarse_step"],
                    adaptive_grid_config["fine_radius"],
                )
                ppxf_result[i, : config["KIN"]["MOM"]] = out_serial[0]
                ppxf_reddening[i] = out_serial[1]
                ppxf_bestfit[i, :] = out_serial[2]
                optimal_template[i, :] = out_serial[3]
                mc_results[i, : config["KIN"]["MOM"]] = out_serial[4]
                formal_error[i, : config["KIN"]["MOM"]] = out_serial[5]
                spectral_mask[i, :] = out_serial[6]
                snr_postfit[i] = out_serial[7]
        else:
            templates_serial = templates[:, reduced_idx] if reduced_idx is not None else templates
            for i in range(0, nbins):
                (
                    ppxf_result[i, : config["KIN"]["MOM"]],
                    ppxf_reddening[i],
                    ppxf_bestfit[i, :],
                    optimal_template[i, :],
                    mc_results[i, : config["KIN"]["MOM"]],
                    formal_error[i, : config["KIN"]["MOM"]],
                    spectral_mask[i, :],
                    snr_postfit[i],
                    _,
                ) = run_ppxf(
                    templates_serial,
                    bin_data[:, i],
                    noise[:, i],
                    velscale,
                    start[i, :],
                    bias,
                    goodPixels_ppxf,
                    config["KIN"]["MOM"],
                    config["KIN"]["ADEG"],
                    config["KIN"]["MDEG"],
                    config["KIN"]["REDDENING"],
                    config["KIN"]["DOCLEAN"],
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
        snr_postfit,
    )

    # Return

    return None
