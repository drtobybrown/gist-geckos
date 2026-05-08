"""
Shared NaN-handling helpers for pPXF wrappers (KIN, CONT, GAS, SFH).

All wrappers normalise the spectrum, filter goodPixels for finite/positive
data and noise, replace remaining non-finite values with safe sentinels, and
raise a clear ``ValueError`` if there are too few usable pixels for pPXF.

These helpers centralise that logic so the modules behave identically.
"""

import numpy as np

# Minimum number of usable pixels we are willing to feed to pPXF.
# Matches the existing per-wrapper checks in the pipeline.
MIN_GOODPIXELS = 10

# Sentinel error value used to down-weight NaN/non-positive entries in the
# noise array; chosen large relative to any realistic noise scale so pPXF
# effectively ignores those pixels.
NOISE_SENTINEL = 1e10


def validate_normalisation(log_bin_data):
    """
    Compute the median used to normalise the spectrum, raising if it is not
    finite and strictly positive.

    Returns
    -------
    float
        ``np.nanmedian(log_bin_data)`` when valid.

    Raises
    ------
    ValueError
        If the median is non-finite or non-positive.
    """
    median = np.nanmedian(log_bin_data)
    if not np.isfinite(median) or median <= 0:
        raise ValueError(
            "spectrum median is not finite and positive (got %s)" % median
        )
    return median


def clean_goodpixels(log_bin_data, log_bin_error, goodPixels, min_keep=MIN_GOODPIXELS):
    """
    Remove indices from ``goodPixels`` where the data or noise is non-finite
    or where the noise is non-positive.

    Raises
    ------
    ValueError
        If ``goodPixels`` is empty/too short on input or after filtering.
    """
    if goodPixels is None or len(goodPixels) < min_keep:
        raise ValueError("goodPixels empty or too few pixels for PPXF")

    valid = (
        np.isfinite(log_bin_data[goodPixels])
        & np.isfinite(log_bin_error[goodPixels])
        & (log_bin_error[goodPixels] > 0)
    )
    goodPixels = goodPixels[valid]
    if len(goodPixels) < min_keep:
        raise ValueError(
            "Too few valid goodPixels after removing NaN/non-positive noise (%d remain)"
            % len(goodPixels)
        )
    return goodPixels


def sanitize_arrays(log_bin_data, log_bin_error, noise_sentinel=NOISE_SENTINEL):
    """
    Replace any remaining non-finite/non-positive entries in ``log_bin_data``
    and ``log_bin_error`` with safe sentinels so pPXF does not fail when it
    inspects pixels outside ``goodPixels``.

    Modifies the inputs in place and returns them for convenience.
    """
    nan_data = ~np.isfinite(log_bin_data)
    nan_err = ~np.isfinite(log_bin_error) | (log_bin_error <= 0)
    log_bin_data[nan_data] = 0.0
    log_bin_error[nan_err] = noise_sentinel
    return log_bin_data, log_bin_error


def prepare_for_ppxf(
    log_bin_data,
    log_bin_error,
    goodPixels,
    normalise=True,
    min_keep=MIN_GOODPIXELS,
    noise_sentinel=NOISE_SENTINEL,
):
    """
    Convenience wrapper that runs the full sanitisation contract used by the
    pPXF wrappers:

    1. Validate the normalisation median (if ``normalise`` is True).
    2. Optionally normalise data/noise by that median in place.
    3. Filter ``goodPixels`` to finite + positive-noise entries.
    4. Replace residual non-finite entries in the full arrays with sentinels.

    Returns
    -------
    log_bin_data, log_bin_error, goodPixels, median
        Sanitised arrays (in place), filtered ``goodPixels``, and the median
        used for normalisation (``1.0`` when ``normalise=False``).
    """
    if normalise:
        median = validate_normalisation(log_bin_data)
        log_bin_error = log_bin_error / median
        log_bin_data = log_bin_data / median
    else:
        median = 1.0

    goodPixels = clean_goodpixels(
        log_bin_data, log_bin_error, goodPixels, min_keep=min_keep
    )
    log_bin_data, log_bin_error = sanitize_arrays(
        log_bin_data, log_bin_error, noise_sentinel=noise_sentinel
    )
    return log_bin_data, log_bin_error, goodPixels, median
