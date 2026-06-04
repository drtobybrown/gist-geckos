"""Low-memory FITS cube access via fitsio."""

from __future__ import annotations

from pathlib import Path

import fitsio
import numpy as np


def list_hdus(path: Path) -> list[dict]:
    with fitsio.FITS(str(path)) as f:
        out = []
        for i, hdu in enumerate(f):
            hdr = hdu.read_header()
            out.append({
                "ext": i,
                "name": hdr.get("EXTNAME", ""),
                "naxis": hdr.get("NAXIS", 0),
                "shape": tuple(hdr[i] for i in range(1, int(hdr.get("NAXIS", 0)) + 1) if hdr.get(f"NAXIS{i}")),
            })
        return out


def find_data_hdu(path: Path) -> tuple[int, int | None]:
    """Return (flux_ext, stat_ext or None). Prefer ext 1=data, 2=stat MUSE layout."""
    hdus = list_hdus(path)
    flux_ext = None
    stat_ext = None
    for h in hdus:
        if h["naxis"] != 3:
            continue
        name = (h["name"] or "").upper()
        if name in ("DATA", "FLUX", "") and flux_ext is None:
            flux_ext = h["ext"]
        elif name in ("STAT", "ERR", "VARIANCE", "IVAR") and stat_ext is None:
            stat_ext = h["ext"]
    if flux_ext is None:
        for h in hdus:
            if h["naxis"] == 3:
                flux_ext = h["ext"]
                break
    if stat_ext is None and len(hdus) > 2:
        for h in hdus:
            if h["naxis"] == 3 and h["ext"] != flux_ext:
                stat_ext = h["ext"]
                break
    if flux_ext is None:
        raise ValueError(f"No 3D HDU in {path}")
    return flux_ext, stat_ext


def read_header_wcs(path: Path, ext: int) -> dict:
    with fitsio.FITS(str(path)) as f:
        hdr = f[ext].read_header()
    crval3 = float(hdr["CRVAL3"])
    cdelt3 = float(hdr.get("CDELT3", hdr.get("CD3_3", 1.0)))
    n3 = int(hdr["NAXIS3"])
    crval2 = float(hdr.get("CRVAL2", 0))
    crval1 = float(hdr.get("CRVAL1", 0))
    return {
        "crval3": crval3,
        "cdelt3": cdelt3,
        "naxis3": n3,
        "naxis2": int(hdr["NAXIS2"]),
        "naxis1": int(hdr["NAXIS1"]),
        "bunit": str(hdr.get("BUNIT", "")),
        "extname": str(hdr.get("EXTNAME", "")),
    }


def read_plane(path: Path, ext: int, iz: int) -> np.ndarray:
    """Read one wavelength plane (ny, nx) as float64."""
    with fitsio.FITS(str(path)) as f:
        slab = f[ext][iz, :, :]
    return np.asarray(slab, dtype=np.float64)
