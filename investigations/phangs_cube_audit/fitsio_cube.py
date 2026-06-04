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
            naxis = int(hdr.get("NAXIS", 0))
            shape = tuple(int(hdr[f"NAXIS{j}"]) for j in range(1, naxis + 1))
            out.append({
                "ext": i,
                "name": str(hdr.get("EXTNAME", "")),
                "naxis": naxis,
                "shape": shape,
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
    if stat_ext is None:
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
    return {
        "crval3": float(hdr["CRVAL3"]),
        "cdelt3": float(hdr.get("CDELT3", hdr.get("CD3_3", 1.0))),
        "naxis3": int(hdr["NAXIS3"]),
        "naxis2": int(hdr["NAXIS2"]),
        "naxis1": int(hdr["NAXIS1"]),
        "bunit": str(hdr.get("BUNIT", "")),
        "extname": str(hdr.get("EXTNAME", "")),
        "ctype3": str(hdr.get("CTYPE3", "")),
    }


def read_plane(path: Path, ext: int, iz: int) -> np.ndarray:
    """Read one wavelength plane (ny, nx) as float64."""
    with fitsio.FITS(str(path)) as f:
        slab = f[ext][iz, :, :]
    return np.asarray(slab, dtype=np.float64)


def estimate_spaxel_chunk_bytes(
    n_trim: int,
    chunk_spaxels: int,
    has_stat: bool,
) -> int:
    """Bytes for one spaxel chunk slab (spec [, stat])."""
    per = 8 * n_trim * chunk_spaxels
    return per * (2 if has_stat else 1)


def check_ram_budget(
    n_trim: int,
    chunk_spaxels: int,
    has_stat: bool,
    max_gb: float = 25.0,
) -> None:
    need = estimate_spaxel_chunk_bytes(n_trim, chunk_spaxels, has_stat)
    if need > max_gb * 1e9:
        raise MemoryError(
            f"Spaxel chunk would need {need / 1e9:.2f} GB > {max_gb} GB; "
            f"reduce SPAXEL_CHUNK (n_trim={n_trim}, chunk={chunk_spaxels})"
        )
