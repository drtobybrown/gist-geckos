"""
Tests for NaN handling across binning, coadds, spatial masking, and pPXF
wrappers. See PRODUCTS.md Section 12 for the policy under test.
"""
import numpy as np
import pytest

from ngistPipeline.auxiliary._ppxf_sanitize import (
    MIN_GOODPIXELS,
    NOISE_SENTINEL,
    clean_goodpixels,
    prepare_for_ppxf,
    sanitize_arrays,
    validate_normalisation,
)
from ngistPipeline.prepareSpectra.default import spatialBinning as coadd_spatial_bins
from ngistPipeline.spatialBinning.voronoi import sn_func
from ngistPipeline.spatialMasking.default import (
    DEFUNCT_MAX_NAN_FRAC,
    applySNRThreshold,
    maskDefunctSpaxels,
)


# -----------------------------
# Voronoi sn_func NaN safety
# -----------------------------

def test_sn_func_clean_inputs_match_legacy():
    rng = np.random.default_rng(0)
    signal = rng.uniform(1, 5, size=20)
    noise = rng.uniform(0.1, 1.0, size=20)
    index = np.arange(10)
    expected = np.sum(signal[index]) / np.sqrt(np.sum(noise[index] ** 2))
    assert np.isclose(sn_func(index, signal=signal, noise=noise, covar_vor=0.0), expected)


def test_sn_func_handles_partial_nans():
    signal = np.array([1.0, np.nan, 2.0, 3.0, 4.0])
    noise = np.array([0.5, 0.5, np.nan, 0.5, 0.5])
    index = np.arange(5)
    result = sn_func(index, signal=signal, noise=noise, covar_vor=0.0)
    # nansum drops NaN entries: signal_sum = 10.0, noise_sq_sum = 0.25*4 = 1.0
    expected = 10.0 / np.sqrt(1.0)
    assert np.isfinite(result)
    assert np.isclose(result, expected)


def test_sn_func_returns_neg_inf_for_all_nan():
    signal = np.full(5, np.nan)
    noise = np.full(5, np.nan)
    result = sn_func(np.arange(5), signal=signal, noise=noise, covar_vor=0.0)
    assert result == -np.inf


# -----------------------------
# Spatial masking with non-finite scalars
# -----------------------------

def _make_cube(nx=4, ny=3, npix=10, signal=None, noise=None, snr=None, spec=None):
    n_spaxels = nx * ny
    if spec is None:
        spec = np.ones((npix, n_spaxels))
    if signal is None:
        signal = np.ones(n_spaxels)
    if noise is None:
        noise = np.ones(n_spaxels)
    if snr is None:
        snr = signal / noise
    return {"spec": spec, "signal": signal, "noise": noise, "snr": snr}


def test_maskDefunctSpaxels_rejects_nonfinite_scalars():
    cube = _make_cube()
    # spaxel 0: NaN signal; spaxel 1: NaN noise; spaxel 2: NaN snr
    cube["signal"][0] = np.nan
    cube["noise"][1] = np.nan
    cube["snr"][2] = np.nan
    masked = maskDefunctSpaxels(cube)
    assert masked[0]
    assert masked[1]
    assert masked[2]
    assert not masked[3]


def test_maskDefunctSpaxels_rejects_any_nan_at_default():
    npix = 200
    n_spaxels = 4
    spec = np.ones((npix, n_spaxels))
    spec[0, 0] = np.nan
    spec[:3, 1] = np.nan
    cube = _make_cube(nx=2, ny=2, npix=npix, spec=spec)
    assert DEFUNCT_MAX_NAN_FRAC == 0.0
    masked = maskDefunctSpaxels(cube)
    assert masked[0]
    assert masked[1]
    assert not masked[2]
    assert not masked[3]


def test_maskDefunctSpaxels_allows_partial_nan_when_threshold_set():
    npix = 200
    spec = np.ones((npix, 2))
    spec[0, 0] = np.nan
    cube = _make_cube(nx=2, ny=1, npix=npix, spec=spec)
    masked = maskDefunctSpaxels(cube, max_nan_frac=0.01)
    assert not masked[0]


def test_applySNRThreshold_actual_rejects_nan_snr():
    snr = np.array([10.0, np.nan, 0.5, 5.0])
    signal = np.array([1.0, 1.0, 1.0, 1.0])
    masked = applySNRThreshold(snr, signal, min_snr=1.0, threshold_method="actual")
    # Inside set: snr >= 1 and finite -> indices 0 and 3
    assert not masked[0]
    assert masked[1]  # NaN snr explicitly rejected
    assert masked[2]
    assert not masked[3]


# -----------------------------
# pPXF sanitisation helper
# -----------------------------

def test_validate_normalisation_rejects_nonfinite_or_nonpositive():
    with pytest.raises(ValueError):
        validate_normalisation(np.array([np.nan, np.nan]))
    with pytest.raises(ValueError):
        validate_normalisation(np.array([-1.0, -2.0]))
    assert np.isclose(validate_normalisation(np.array([1.0, 2.0, 3.0])), 2.0)


def test_clean_goodpixels_filters_and_raises():
    n = 50
    data = np.ones(n)
    error = np.ones(n)
    data[5] = np.nan
    error[6] = np.nan
    error[7] = -1.0
    goodPixels = np.arange(n)
    cleaned = clean_goodpixels(data, error, goodPixels)
    assert 5 not in cleaned
    assert 6 not in cleaned
    assert 7 not in cleaned
    assert len(cleaned) == n - 3

    # Force too few survivors
    error[:n - 5] = np.nan
    with pytest.raises(ValueError):
        clean_goodpixels(data, error, np.arange(n))


def test_sanitize_arrays_replaces_nonfinite():
    data = np.array([1.0, np.nan, np.inf, -1.0])
    error = np.array([0.5, np.nan, -0.1, 1.0])
    sanitize_arrays(data, error)
    assert np.all(np.isfinite(data))
    assert np.all(np.isfinite(error))
    assert error[1] == NOISE_SENTINEL
    assert error[2] == NOISE_SENTINEL


def test_prepare_for_ppxf_full_contract():
    n = 30
    data = np.ones(n) * 2.0
    error = np.ones(n) * 0.5
    data[3] = np.nan
    error[4] = np.nan
    goodPixels = np.arange(n)

    data, error, gp, median = prepare_for_ppxf(data, error, goodPixels)

    assert np.isclose(median, 2.0)
    assert 3 not in gp
    assert 4 not in gp
    assert len(gp) >= MIN_GOODPIXELS
    assert np.all(np.isfinite(data))
    assert np.all(np.isfinite(error))


def test_prepare_for_ppxf_no_normalise():
    n = 20
    data = np.ones(n) * 5.0
    error = np.ones(n) * 0.1
    goodPixels = np.arange(n)
    data_out, error_out, gp, median = prepare_for_ppxf(
        data.copy(), error.copy(), goodPixels, normalise=False
    )
    assert median == 1.0
    assert np.allclose(data_out, 5.0)


# -----------------------------
# Spectral coadd: NaN variance handling
# -----------------------------

def test_coadd_partial_nan_variance_does_not_zero_error():
    npix = 5
    nspax = 4
    binNum = np.array([0, 0, 1, 1])
    spec = np.ones((npix, nspax))
    error = np.ones((npix, nspax))  # variance per channel
    # Channel 2 of spaxels in bin 0 is NaN in variance
    error[2, 0] = np.nan
    error[2, 1] = np.nan
    bin_data, bin_error, bin_flux = coadd_spatial_bins(binNum, spec, error)

    # Bin 0 channel 2: every spaxel had NaN variance -> sentinel applied
    assert bin_error[2, 0] == np.sqrt(1e20)
    # Bin 0 other channels: combined variance is sum of two unit variances -> sqrt(2)
    assert np.isclose(bin_error[0, 0], np.sqrt(2.0))
    # Bin 1 unchanged: no NaN variance
    assert np.allclose(bin_error[:, 1], np.sqrt(2.0))


def test_coadd_clean_inputs_unchanged():
    """Coadd with no NaNs should be identical to nansum/sqrt(sum) on clean data."""
    npix = 4
    nspax = 6
    rng = np.random.default_rng(1)
    binNum = np.array([0, 0, 0, 1, 1, 2])
    spec = rng.uniform(1, 10, size=(npix, nspax))
    var = rng.uniform(0.1, 0.5, size=(npix, nspax))
    bin_data, bin_error, _ = coadd_spatial_bins(binNum, spec, var)

    # Reference: pure nansum implementation (matches when no NaNs present)
    expected_data_bin0 = np.nansum(spec[:, :3], axis=1)
    expected_err_bin0 = np.sqrt(np.nansum(var[:, :3], axis=1))
    assert np.allclose(bin_data[:, 0], expected_data_bin0)
    assert np.allclose(bin_error[:, 0], expected_err_bin0)
