#!/usr/bin/env python3
"""Write small synthetic MUSE-like cubes for local pilot validation (fitsio)."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitsio
import numpy as np

# Rest-frame grid similar to trimmed MUSE WFM (2 Å steps)
CDELT3 = 2.0
NY, NX = 32, 32
# PHANGS-like: native axis wider than science range; blank NaN pads at ends
PHANGS_CRVAL3 = 4700.0
PHANGS_NWAVE = 1201  # ~4700–7100 Å native
PHANGS_EDGE_BLANK = 50  # channels at each end (all spaxels NaN), cf. NGC1087
MAUVE_CRVAL3 = 4798.0
MAUVE_NWAVE = 1101  # ~4800–7000 Å, no edge pads


def wave_rest(crval3: float, nwave: int) -> np.ndarray:
    return crval3 + np.arange(nwave) * CDELT3


def write_cube(
    path: Path,
    crval3: float,
    nwave: int,
    nad_nan: bool,
    laser_nan: bool,
    edge_blank: int = 0,
) -> None:
    w = wave_rest(crval3, nwave)
    data = np.ones((nwave, NY, NX), dtype=np.float32) * 10.0
    stat = np.ones((nwave, NY, NX), dtype=np.float32) * 0.5

    if edge_blank > 0:
        eb = min(edge_blank, nwave // 2)
        data[:eb, :, :] = np.nan
        stat[:eb, :, :] = np.nan
        data[-eb:, :, :] = np.nan
        stat[-eb:, :, :] = np.nan
        # Some PHANGS configs still include blank channels inside LMIN/LMAX_TOT
        in_trim_blue = (w >= 4800) & (w <= 4830)
        in_trim_red = (w >= 6970) & (w <= 7000)
        data[in_trim_blue | in_trim_red, :, :] = np.nan
        stat[in_trim_blue | in_trim_red, :, :] = np.nan

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
        "NAXIS3": nwave,
        "CRVAL3": crval3,
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
    print(
        f"Wrote {path} (nad_nan={nad_nan}, laser_nan={laser_nan}, edge_blank={edge_blank})"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    phangs = args.out_dir / "PHANGS" / "SYNTH_PHANGS_like_cube.fits"
    mauve = args.out_dir / "MAUVE" / "SYNTH_MAUVE_clean_cube.fits"
    write_cube(
        phangs,
        PHANGS_CRVAL3,
        PHANGS_NWAVE,
        nad_nan=True,
        laser_nan=True,
        edge_blank=PHANGS_EDGE_BLANK,
    )
    write_cube(mauve, MAUVE_CRVAL3, MAUVE_NWAVE, nad_nan=False, laser_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
