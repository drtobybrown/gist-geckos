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


def flat_to_xy(flat_idx: np.ndarray, nx: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert C-order flat spaxel index to (iy, ix)."""
    iy = flat_idx // nx
    ix = flat_idx % nx
    return iy, ix


def list_3d_hdus(path: Path) -> list[dict]:
    return [h for h in list_hdus(path) if h["naxis"] == 3]


def sample_spaxel_indices(n_spax: int, n_sample: int, seed: int) -> np.ndarray:
    n_sample = min(n_sample, n_spax)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_spax, size=n_sample, replace=False))


def read_spectra_in_hdu(
    path: Path,
    ext: int,
    flat_idx: np.ndarray,
    nx: int,
    nwave: int,
) -> np.ndarray:
    """
    Read full wavelength axis for selected spaxels from one 3D HDU.

    Returns (nwave, n_sample). One fitsio column read per spaxel ([:, iy, ix]).
    """
    ns = len(flat_idx)
    cube = np.empty((nwave, ns), dtype=np.float64)
    iy, ix = flat_to_xy(flat_idx, nx)
    with fitsio.FITS(str(path)) as f:
        hdu = f[ext]
        for k in range(ns):
            cube[:, k] = np.asarray(
                hdu[:, int(iy[k]), int(ix[k])], dtype=np.float64
            ).reshape(-1)
    return cube


def read_spectra_for_spaxels(
    path: Path,
    flux_ext: int,
    stat_ext: int | None,
    flat_idx: np.ndarray,
    nx: int,
    nwave: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read flux and optional stat HDUs (delegates to read_spectra_in_hdu)."""
    flux_cube = read_spectra_in_hdu(path, flux_ext, flat_idx, nx, nwave)
    stat_cube = (
        read_spectra_in_hdu(path, stat_ext, flat_idx, nx, nwave)
        if stat_ext is not None
        else None
    )
    return flux_cube, stat_cube
