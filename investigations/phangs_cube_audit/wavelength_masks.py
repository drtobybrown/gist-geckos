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
