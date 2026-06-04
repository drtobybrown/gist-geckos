"""Rest-frame wavelength masks for cube NaN auditing (Angstrom)."""

from __future__ import annotations

import numpy as np

SNR_DEFAULT_LO = 4750.0
SNR_DEFAULT_HI = 7100.0
NAD_GAP_LO = 5860.0
NAD_GAP_HI = 5900.0
LASER_LGS_LO = 5770.0
LASER_LGS_HI = 6050.0


def build_wave_grid(crval3: float, cdelt3: float, naxis3: int) -> np.ndarray:
    return crval3 + np.arange(naxis3, dtype=np.float64) * cdelt3


def mask_indices(wave_rest: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.where((wave_rest >= lo) & (wave_rest <= hi))[0]


def detect_edge_blank_channels(
    frac_nan_per_channel: np.ndarray,
    wave_rest: np.ndarray,
    *,
    blank_threshold: float = 1.0,
) -> dict:
    """
    Find leading/trailing channels that are all-NaN (blank padding).

    Uses per-channel fraction of NaN spaxels in the audit sample; edge pads
    are 100% NaN on every spaxel. Returns suggested rest-frame bounds where
    any channel is not fully blank.
    """
    frac = np.asarray(frac_nan_per_channel, dtype=np.float64)
    wave = np.asarray(wave_rest, dtype=np.float64)
    n = len(frac)
    if n == 0:
        return {
            "n_leading_blank": 0,
            "n_trailing_blank": 0,
            "n_blank_total": 0,
            "has_edge_blank": False,
            "lambda_rest_first_valid": np.nan,
            "lambda_rest_last_valid": np.nan,
            "suggested_lmin_tot": np.nan,
            "suggested_lmax_tot": np.nan,
            "valid_channel_mask": np.array([], dtype=bool),
        }

    is_blank = frac >= blank_threshold - 1e-9
    n_lead = 0
    while n_lead < n and is_blank[n_lead]:
        n_lead += 1
    n_trail = 0
    while n_trail < n - n_lead and is_blank[n - 1 - n_trail]:
        n_trail += 1

    valid = np.ones(n, dtype=bool)
    valid[:n_lead] = False
    if n_trail:
        valid[n - n_trail :] = False

    if valid.any():
        idx = np.where(valid)[0]
        lam_lo = float(wave[idx[0]])
        lam_hi = float(wave[idx[-1]])
    else:
        lam_lo = lam_hi = np.nan

    return {
        "n_leading_blank": n_lead,
        "n_trailing_blank": n_trail,
        "n_blank_total": int(is_blank.sum()),
        "has_edge_blank": (n_lead + n_trail) > 0,
        "lambda_rest_first_valid": lam_lo,
        "lambda_rest_last_valid": lam_hi,
        "suggested_lmin_tot": lam_lo,
        "suggested_lmax_tot": lam_hi,
        "valid_channel_mask": valid,
    }


def trim_indices_excluding_edge_blanks(
    trim_global: np.ndarray,
    valid_channel_mask: np.ndarray,
) -> np.ndarray:
    """Keep trim indices whose full-cube channel is not an edge blank pad."""
    return np.array(
        [i for i in trim_global if valid_channel_mask[i]],
        dtype=np.int64,
    )


def compile_masks(
    wave_rest: np.ndarray,
    lmin_tot: float,
    lmax_tot: float,
    lmin_snr: float = SNR_DEFAULT_LO,
    lmax_snr: float = SNR_DEFAULT_HI,
) -> dict:
    trim = mask_indices(wave_rest, lmin_tot, lmax_tot)
    w = wave_rest[trim]
    snr = mask_indices(w, lmin_snr, lmax_snr)
    nad = mask_indices(w, NAD_GAP_LO, NAD_GAP_HI)
    laser = mask_indices(w, LASER_LGS_LO, LASER_LGS_HI)
    nad_set = set(nad.tolist())
    laser_set = set(laser.tolist())
    snr_no_nad = np.array([i for i in snr if i not in nad_set], dtype=np.int64)
    snr_no_laser = np.array([i for i in snr if i not in laser_set], dtype=np.int64)
    return {
        "trim": trim,
        "trim_wave": w,
        "snr": snr,
        "nad": nad,
        "laser": laser,
        "snr_no_nad": snr_no_nad,
        "snr_no_laser": snr_no_laser,
        "all_local": np.arange(len(w), dtype=np.int64),
    }
