#!/usr/bin/env python
"""
test_ooc_regression.py — Verify out-of-core (streaming) pipeline produces
numerically identical results to the legacy in-memory path.

Creates a synthetic FITS cube, runs both the streaming (LazyCube) and
in-memory code paths for readData + prepareSpectra, then compares outputs.

Run from repo root:
    python tests/test_ooc_regression.py
"""

import os
import sys
import tempfile
import shutil

import numpy as np

# Ensure project root is on path
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

import fitsio
import h5py
from ppxf.ppxf_util import log_rebin

from ngistPipeline.auxiliary.lazy_cube import LazyCube


# ====================================================================== #
#  Synthetic test cube                                                     #
# ====================================================================== #

NY, NX = 10, 12  # small spatial grid (120 spaxels)
NWAVE_FULL = 200  # wavelength pixels
CRVAL3 = 4700.0
CDELT3 = 1.25
REDSHIFT = 0.01
LMIN_TOT, LMAX_TOT = 4750.0, 4940.0
LMIN_SNR, LMAX_SNR = 4750.0, 4940.0
VELSCALE = 70.0


def _make_synthetic_cube(path):
    """Create a small synthetic MUSE-like FITS cube at *path*."""
    np.random.seed(42)

    wave_obs = CRVAL3 + np.arange(NWAVE_FULL) * CDELT3
    wave_rest = wave_obs / (1 + REDSHIFT)
    idx = np.where((wave_rest >= LMIN_TOT) & (wave_rest <= LMAX_TOT))[0]

    # Flux: Gaussian emission line + continuum + noise
    flux_3d = np.empty((NWAVE_FULL, NY, NX), dtype=np.float32)
    var_3d = np.empty((NWAVE_FULL, NY, NX), dtype=np.float32)

    for iy in range(NY):
        for ix in range(NX):
            continuum = 100.0 + 10.0 * np.sin(2 * np.pi * wave_obs / 500)
            noise_level = 5.0 + 0.5 * (iy + ix)
            flux_3d[:, iy, ix] = (
                continuum + np.random.normal(0, noise_level, NWAVE_FULL)
            ).astype(np.float32)
            var_3d[:, iy, ix] = (noise_level ** 2 * np.ones(NWAVE_FULL)).astype(
                np.float32
            )

    # Make 2 spaxels defunct: one all-NaN, one zero flux
    flux_3d[:, 0, 0] = np.nan
    var_3d[:, 0, 0] = np.nan
    flux_3d[:, 0, 1] = 0.0
    var_3d[:, 0, 1] = 0.0

    # Write 3-extension FITS (primary, DATA, STAT)
    with fitsio.FITS(path, "rw") as f:
        # Primary (empty)
        f.write(np.zeros(1, dtype=np.float32))
        # DATA extension
        hdr = {
            "NAXIS": 3,
            "NAXIS1": NX,
            "NAXIS2": NY,
            "NAXIS3": NWAVE_FULL,
            "CRVAL3": CRVAL3,
            "CDELT3": CDELT3,
            "CD2_2": 0.2 / 3600.0,
            "CD3_3": CDELT3,
            "CRPIX1": 1,
            "CRPIX2": 1,
            "CRPIX3": 1,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CTYPE3": "AWAV",
        }
        f.write(flux_3d, header=hdr)
        # STAT extension (variance)
        f.write(var_3d)


# ====================================================================== #
#  In-memory reference computation                                         #
# ====================================================================== #

def _ref_in_memory(fits_path):
    """Load cube in-memory and compute stats + binned spectra (ground truth)."""
    with fitsio.FITS(fits_path) as f:
        hdr = f[1].read_header()
        s = (hdr["NAXIS3"], hdr["NAXIS2"], hdr["NAXIS1"])
        ny, nx = s[1], s[2]
        nspaxels = ny * nx

        cdelt3 = hdr["CD3_3"]
        cdelt2 = hdr["CD2_2"]

        wave_full = hdr["CRVAL3"] + np.arange(s[0]) * cdelt3
        wave_full = wave_full / (1 + REDSHIFT)
        idx = np.where((wave_full >= LMIN_TOT) & (wave_full <= LMAX_TOT))[0]
        start, end = int(idx[0]), int(idx[-1] + 1)

        data_3d = f[1][start:end, :, :]
        spec = np.asarray(data_3d, dtype=np.float64).reshape(len(idx), nspaxels)
        err_3d = f[2][start:end, :, :]
        espec = np.asarray(err_3d, dtype=np.float64).reshape(len(idx), nspaxels)

    wave = wave_full[idx]

    # Coordinates
    origin = [NX // 2, NY // 2]
    xaxis = (np.arange(nx) - origin[0]) * cdelt2 * 3600.0
    yaxis = (np.arange(ny) - origin[1]) * cdelt2 * 3600.0
    xg, yg = np.meshgrid(xaxis, yaxis)
    x, y = xg.ravel(), yg.ravel()

    # SNR stats
    idx_snr = np.where((wave >= LMIN_SNR) & (wave <= LMAX_SNR))[0]
    signal = np.nanmedian(spec[idx_snr, :], axis=0)
    noise = np.sqrt(np.nanmedian(espec[idx_snr, :], axis=0))
    snr = np.nanmedian(spec[idx_snr, :] / np.sqrt(espec[idx_snr, :]), axis=0)

    # Defunct mask
    all_nan = np.all(np.isnan(spec), axis=0)
    median_nonpos = np.nanmedian(spec, axis=0) <= 0.0
    defunct_mask = all_nan | median_nonpos

    # ---- Mock spatial mask: defunct + snr < 5 ----
    mask = np.zeros(nspaxels, dtype=np.int32)
    mask[defunct_mask] = 1
    mask[snr < 5.0] = 1
    idxUnmasked = np.where(mask == 0)[0]
    # Bin assignment: every 4 unmasked spaxels → one bin
    binNum = np.arange(len(idxUnmasked)) // 4

    # ---- Log-rebin all spaxels ----
    wave_range = np.array([wave.min(), wave.max()])
    probe, logLam, _ = log_rebin(wave_range, np.ones(len(wave)), velscale=VELSCALE)
    npix_log = len(logLam)

    log_spec = np.full((npix_log, nspaxels), np.nan)
    log_espec = np.full((npix_log, nspaxels), np.nan)
    for j in range(nspaxels):
        try:
            log_spec[:, j], _, _ = log_rebin(wave_range, spec[:, j], velscale=VELSCALE)
        except Exception:
            pass
        try:
            log_espec[:, j], _, _ = log_rebin(wave_range, espec[:, j], velscale=VELSCALE)
        except Exception:
            pass

    # ---- Spatial binning (linear) ----
    ubins = np.unique(binNum)
    nbins = len(ubins)
    bin_spec_lin = np.zeros((len(wave), nbins))
    bin_err_lin = np.zeros((len(wave), nbins))
    for i, b in enumerate(ubins):
        k = binNum == b
        bin_spec_lin[:, i] = np.nansum(spec[:, idxUnmasked[k]], axis=1)
        bin_err_lin[:, i] = np.sqrt(np.nansum(espec[:, idxUnmasked[k]], axis=1))

    # ---- Spatial binning (log) ----
    bin_spec_log = np.zeros((npix_log, nbins))
    bin_err_log = np.zeros((npix_log, nbins))
    for i, b in enumerate(ubins):
        k = binNum == b
        bin_spec_log[:, i] = np.nansum(
            np.nan_to_num(log_spec[:, idxUnmasked[k]], nan=0.0), axis=1
        )
        bin_err_log[:, i] = np.sqrt(
            np.nansum(
                np.nan_to_num(log_espec[:, idxUnmasked[k]], nan=0.0), axis=1
            )
        )

    return {
        "x": x, "y": y, "wave": wave,
        "signal": signal, "noise": noise, "snr": snr,
        "defunct_mask": defunct_mask,
        "mask": mask,
        "idxUnmasked": idxUnmasked,
        "binNum": binNum,
        "logLam": logLam,
        "log_spec": log_spec,
        "log_espec": log_espec,
        "bin_spec_lin": bin_spec_lin,
        "bin_err_lin": bin_err_lin,
        "bin_spec_log": bin_spec_log,
        "bin_err_log": bin_err_log,
    }


# ====================================================================== #
#  Streaming (LazyCube) computation                                        #
# ====================================================================== #

def _streaming_path(fits_path, ref):
    """Run the streaming LazyCube path and return comparable results."""
    with fitsio.FITS(fits_path) as f:
        hdr = f[1].read_header()
        s = (hdr["NAXIS3"], hdr["NAXIS2"], hdr["NAXIS1"])
        ny, nx = s[1], s[2]
        nspaxels = ny * nx

        cdelt3 = hdr["CD3_3"]
        cdelt2 = hdr["CD2_2"]

        wave_full = hdr["CRVAL3"] + np.arange(s[0]) * cdelt3
        wave_full /= (1 + REDSHIFT)
        idx = np.where((wave_full >= LMIN_TOT) & (wave_full <= LMAX_TOT))[0]
        wave_start = int(idx[0])
        wave_end = int(idx[-1] + 1)

    wave = wave_full[idx]
    nwave = len(wave)

    # Coordinates
    origin = [NX // 2, NY // 2]
    xaxis = (np.arange(nx) - origin[0]) * cdelt2 * 3600.0
    yaxis = (np.arange(ny) - origin[1]) * cdelt2 * 3600.0
    xg, yg = np.meshgrid(xaxis, yaxis)
    x, y = xg.ravel(), yg.ravel()

    # ---- Streaming stats pass (mimics readData/MUSE_WFM.py) ----
    idx_snr = np.where((wave >= LMIN_SNR) & (wave <= LMAX_SNR))[0]
    signal = np.empty(nspaxels)
    noise = np.empty(nspaxels)
    snr = np.empty(nspaxels)
    defunct_mask = np.zeros(nspaxels, dtype=bool)

    rows_per_tile = max(1, 10_000 // nx)
    with fitsio.FITS(fits_path) as f:
        for y_start in range(0, ny, rows_per_tile):
            y_end = min(y_start + rows_per_tile, ny)
            n_tile = (y_end - y_start) * nx
            sp_start = y_start * nx
            sp_end = sp_start + n_tile

            data_3d = f[1][wave_start:wave_end, y_start:y_end, :]
            spec_tile = np.asarray(data_3d, dtype=np.float64).reshape(nwave, n_tile)
            err_3d = f[2][wave_start:wave_end, y_start:y_end, :]
            error_tile = np.asarray(err_3d, dtype=np.float64).reshape(nwave, n_tile)

            signal[sp_start:sp_end] = np.nanmedian(spec_tile[idx_snr, :], axis=0)
            noise[sp_start:sp_end] = np.sqrt(
                np.nanmedian(error_tile[idx_snr, :], axis=0)
            )
            snr[sp_start:sp_end] = np.nanmedian(
                spec_tile[idx_snr, :] / np.sqrt(error_tile[idx_snr, :]), axis=0
            )

            all_nan = np.all(np.isnan(spec_tile), axis=0)
            median_nonpos = np.nanmedian(spec_tile, axis=0) <= 0.0
            defunct_mask[sp_start:sp_end] = all_nan | median_nonpos

    # ---- Build LazyCube ----
    cube = LazyCube(
        fits_path=fits_path,
        data_ext=1,
        error_ext=2,
        wave_start=wave_start,
        wave_end=wave_end,
        shape_yx=(ny, nx),
        x=x, y=y, wave=wave,
        signal=signal, noise=noise, snr=snr,
        pixelsize=cdelt2 * 3600.0,
        wcshdr=None,
        bunit=None,
        defunct_mask=defunct_mask,
    )

    # ---- Streaming prepareSpectra (matches _prepSpectra_streaming logic) ----
    mask = ref["mask"]
    idxUnmasked = ref["idxUnmasked"]
    binNum = ref["binNum"]

    spaxel_to_bin = np.full(nspaxels, -1, dtype=np.int64)
    spaxel_to_bin[idxUnmasked] = binNum

    ubins = np.unique(binNum)
    nbins = len(ubins)
    bin_remap = np.full(int(ubins.max()) + 1, -1, dtype=np.int64)
    bin_remap[ubins] = np.arange(nbins)

    wave_range = np.array([wave.min(), wave.max()])
    probe, logLam, _ = log_rebin(wave_range, np.ones(nwave), velscale=VELSCALE)
    npix_log = len(logLam)

    # Accumulators
    bin_sum_spec_lin = np.zeros((nwave, nbins), dtype=np.float64)
    bin_sum_err_lin = np.zeros((nwave, nbins), dtype=np.float64)
    bin_sum_spec_log = np.zeros((npix_log, nbins), dtype=np.float64)
    bin_sum_err_log = np.zeros((npix_log, nbins), dtype=np.float64)

    log_spec_all = np.full((npix_log, nspaxels), np.nan, dtype=np.float64)
    log_espec_all = np.full((npix_log, nspaxels), np.nan, dtype=np.float64)

    for indices, spec_tile, error_tile in cube.tile_iterator():
        n_tile = len(indices)

        # Linear binning accumulation
        tile_bins = spaxel_to_bin[indices]
        valid = tile_bins >= 0
        if valid.any():
            v_bins = bin_remap[tile_bins[valid]]
            v_spec = np.nan_to_num(spec_tile[:, valid], nan=0.0)
            v_err = np.nan_to_num(error_tile[:, valid], nan=0.0)
            np.add.at(bin_sum_spec_lin, (slice(None), v_bins), v_spec)
            np.add.at(bin_sum_err_lin, (slice(None), v_bins), v_err)

        # Log-rebin each spaxel
        log_tile = np.full((npix_log, n_tile), np.nan, dtype=np.float64)
        log_err_tile = np.full((npix_log, n_tile), np.nan, dtype=np.float64)
        for j in range(n_tile):
            try:
                log_tile[:, j], _, _ = log_rebin(
                    wave_range, spec_tile[:, j], velscale=VELSCALE
                )
            except Exception:
                pass
            try:
                log_err_tile[:, j], _, _ = log_rebin(
                    wave_range, error_tile[:, j], velscale=VELSCALE
                )
            except Exception:
                pass

        log_spec_all[:, indices[0] : indices[-1] + 1] = log_tile
        log_espec_all[:, indices[0] : indices[-1] + 1] = log_err_tile

        # Log binning accumulation
        if valid.any():
            v_bins = bin_remap[tile_bins[valid]]
            v_log = np.nan_to_num(log_tile[:, valid], nan=0.0)
            v_log_err = np.nan_to_num(log_err_tile[:, valid], nan=0.0)
            np.add.at(bin_sum_spec_log, (slice(None), v_bins), v_log)
            np.add.at(bin_sum_err_log, (slice(None), v_bins), v_log_err)

    bin_err_lin = np.sqrt(bin_sum_err_lin)
    bin_err_log = np.sqrt(bin_sum_err_log)

    return {
        "x": x, "y": y, "wave": wave,
        "signal": signal, "noise": noise, "snr": snr,
        "defunct_mask": defunct_mask,
        "logLam": logLam,
        "log_spec": log_spec_all,
        "log_espec": log_espec_all,
        "bin_spec_lin": bin_sum_spec_lin,
        "bin_err_lin": bin_err_lin,
        "bin_spec_log": bin_sum_spec_log,
        "bin_err_log": bin_err_log,
    }


# ====================================================================== #
#  Test functions                                                          #
# ====================================================================== #

def _assert_close(name, a, b, atol=1e-10, rtol=1e-10):
    """Assert two arrays are close; print diagnostics on failure."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"{name}: shape mismatch {a.shape} vs {b.shape}"

    # Ignore NaN positions (both should be NaN in same places)
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    assert np.array_equal(nan_a, nan_b), (
        f"{name}: NaN positions differ "
        f"(ref has {nan_a.sum()}, streaming has {nan_b.sum()})"
    )

    valid = ~nan_a
    if not valid.any():
        return  # all NaN, nothing to compare

    diff = np.abs(a[valid] - b[valid])
    max_diff = diff.max()
    mean_diff = diff.mean()
    denom = np.maximum(np.abs(a[valid]), np.abs(b[valid]))
    denom = np.where(denom == 0, 1.0, denom)
    max_rel = (diff / denom).max()

    ok = np.allclose(a[valid], b[valid], atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(
        f"  {status} {name:30s}  max_abs={max_diff:.2e}  "
        f"mean_abs={mean_diff:.2e}  max_rel={max_rel:.2e}"
    )
    assert ok, f"{name} exceeds tolerance (atol={atol}, rtol={rtol})"


def run_tests():
    """Main test driver."""
    tmpdir = tempfile.mkdtemp(prefix="ooc_test_")
    fits_path = os.path.join(tmpdir, "test_cube.fits")

    try:
        print("=" * 70)
        print("Out-of-Core Regression Test")
        print("=" * 70)

        # 1) Create synthetic cube
        print("\n[1] Creating synthetic FITS cube ...")
        _make_synthetic_cube(fits_path)
        print(f"    Cube: {NY}x{NX} spatial, {NWAVE_FULL} wave channels")

        # 2) In-memory reference
        print("\n[2] Computing in-memory reference ...")
        ref = _ref_in_memory(fits_path)
        print(f"    {len(ref['idxUnmasked'])} unmasked, "
              f"{len(np.unique(ref['binNum']))} bins, "
              f"{ref['defunct_mask'].sum()} defunct")

        # 3) Streaming (LazyCube) path
        print("\n[3] Computing streaming (LazyCube) path ...")
        ooc = _streaming_path(fits_path, ref)

        # 4) Compare per-spaxel statistics
        print("\n[4] Comparing per-spaxel statistics:")
        _assert_close("signal", ref["signal"], ooc["signal"])
        _assert_close("noise", ref["noise"], ooc["noise"])
        _assert_close("snr", ref["snr"], ooc["snr"])
        assert np.array_equal(ref["defunct_mask"], ooc["defunct_mask"]), \
            "defunct_mask mismatch!"
        print(f"  PASS {'defunct_mask':30s}  exact match")

        # 5) Compare log-rebinned all spectra
        print("\n[5] Comparing log-rebinned AllSpectra:")
        _assert_close("log_spec", ref["log_spec"], ooc["log_spec"])
        _assert_close("log_espec", ref["log_espec"], ooc["log_espec"])
        _assert_close("logLam", ref["logLam"], ooc["logLam"])

        # 6) Compare binned spectra (linear)
        # Use slightly larger tolerance since accumulation order may differ
        print("\n[6] Comparing BinSpectra (linear):")
        _assert_close("bin_spec_lin", ref["bin_spec_lin"], ooc["bin_spec_lin"],
                       atol=1e-8, rtol=1e-8)
        _assert_close("bin_err_lin", ref["bin_err_lin"], ooc["bin_err_lin"],
                       atol=1e-8, rtol=1e-8)

        # 7) Compare binned spectra (log)
        print("\n[7] Comparing BinSpectra (log):")
        _assert_close("bin_spec_log", ref["bin_spec_log"], ooc["bin_spec_log"],
                       atol=1e-8, rtol=1e-8)
        _assert_close("bin_err_log", ref["bin_err_log"], ooc["bin_err_log"],
                       atol=1e-8, rtol=1e-8)

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70)

    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    run_tests()
