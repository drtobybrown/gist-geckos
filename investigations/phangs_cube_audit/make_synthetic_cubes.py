#!/usr/bin/env python3
"""Write small synthetic MUSE-like cubes for local pilot validation (fitsio)."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitsio
import numpy as np

# Rest-frame grid similar to trimmed MUSE WFM (2 Å steps)
CRVAL3 = 4798.0
CDELT3 = 2.0
NWAVE = 1101  # ~4800–7000 Å
NY, NX = 32, 32


def wave_rest() -> np.ndarray:
    return CRVAL3 + np.arange(NWAVE) * CDELT3


def write_cube(path: Path, nad_nan: bool, laser_nan: bool) -> None:
    w = wave_rest()
    data = np.ones((NWAVE, NY, NX), dtype=np.float32) * 10.0
    stat = np.ones((NWAVE, NY, NX), dtype=np.float32) * 0.5

    nad = (w >= 5860) & (w <= 5900)
    laser = (w >= 5770) & (w <= 6050)
    if nad_nan:
        data[nad, :, :] = np.nan
        stat[nad, :, :] = np.nan
    if laser_nan:
        data[laser & ~nad, :, :] = np.nan
        stat[laser & ~nad, :, :] = np.nan

    hdr = {
        "NAXIS": 3,
        "NAXIS1": NX,
        "NAXIS2": NY,
        "NAXIS3": NWAVE,
        "CRVAL3": CRVAL3,
        "CDELT3": CDELT3,
        "CTYPE3": "AWAV",
        "BUNIT": "10**(-20) erg s(-1) cm(-2) Angstrom(-1)",
        "EXTNAME": "DATA",
    }
    stat_hdr = {**hdr, "EXTNAME": "STAT"}

    path.parent.mkdir(parents=True, exist_ok=True)
    fitsio.write(path, data, header=hdr, clobber=True)
    with fitsio.FITS(str(path), "rw") as f:
        f.write(stat, header=stat_hdr, extname="STAT")
    print(f"Wrote {path} (nad_nan={nad_nan}, laser_nan={laser_nan})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    phangs = args.out_dir / "PHANGS" / "SYNTH_PHANGS_like_cube.fits"
    mauve = args.out_dir / "MAUVE" / "SYNTH_MAUVE_clean_cube.fits"
    write_cube(phangs, nad_nan=True, laser_nan=True)
    write_cube(mauve, nad_nan=False, laser_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
