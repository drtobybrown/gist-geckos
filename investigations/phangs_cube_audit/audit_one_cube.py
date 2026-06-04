#!/usr/bin/env python3
"""Fitsio audit of one MUSE cube: channel NaNs, spaxel nan_frac, defunct simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fitsio_cube import (
    check_ram_budget,
    find_data_hdu,
    list_hdus,
    read_header_wcs,
    read_plane,
)
from wavelength_masks import (
    LASER_LGS_HI,
    LASER_LGS_LO,
    NAD_GAP_HI,
    NAD_GAP_LO,
    build_wave_grid,
    compile_masks,
    mask_indices,
)

DEFUNCT_THRESHOLDS = (0.0, 0.01, 0.05)
SPAXEL_CHUNK = 4096
MAX_RAM_GB = 25.0


def _defunct_counts(nf: np.ndarray, thresholds: tuple[float, ...]) -> dict[str, int]:
    return {str(t): int(np.sum(nf > t)) for t in thresholds}


def audit_cube(
    path: Path,
    out_dir: Path,
    redshift: float = 0.0,
    lmin_tot: float = 4800.0,
    lmax_tot: float = 7000.0,
    lmin_snr: float = 4750.0,
    lmax_snr: float = 7100.0,
    hot_frac: float = 0.5,
    spaxel_chunk: int = SPAXEL_CHUNK,
) -> dict:
    path = path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    hdu_list = list_hdus(path)
    flux_ext, stat_ext = find_data_hdu(path)
    meta_flux = read_header_wcs(path, flux_ext)
    n3, ny, nx = meta_flux["naxis3"], meta_flux["naxis2"], meta_flux["naxis1"]
    n_spax = ny * nx

    wave_obs = build_wave_grid(meta_flux["crval3"], meta_flux["cdelt3"], n3)
    wave_rest = wave_obs / (1.0 + redshift)
    masks_full = compile_masks(wave_rest, lmin_tot, lmax_tot, lmin_snr, lmax_snr)
    trim_global = masks_full["trim"]
    w_trim = wave_rest[trim_global]
    n_trim = len(trim_global)
    trim_set = set(trim_global.tolist())

    check_ram_budget(n_trim, spaxel_chunk, stat_ext is not None, MAX_RAM_GB)

    # --- Per-channel NaN fractions (flux + stat) ---
    n_nan_flux = np.zeros(n3, dtype=np.int64)
    n_nan_stat = np.zeros(n3, dtype=np.int64)
    for iz in range(n3):
        plane = read_plane(path, flux_ext, iz)
        n_nan_flux[iz] = int(np.sum(~np.isfinite(plane)))
        if stat_ext is not None:
            stat_plane = read_plane(path, stat_ext, iz)
            n_nan_stat[iz] = int(np.sum(~np.isfinite(stat_plane)))

    channel_rows = []
    for iz in range(n3):
        lam_r = float(wave_rest[iz])
        in_trim = iz in trim_set
        channel_rows.append({
            "iwave": iz,
            "lambda_obs": float(wave_obs[iz]),
            "lambda_rest": lam_r,
            "frac_nan_flux": n_nan_flux[iz] / n_spax,
            "frac_nan_stat": (n_nan_stat[iz] / n_spax) if stat_ext is not None else np.nan,
            "in_trim": in_trim,
            "in_snr_band": (lmin_snr <= lam_r <= lmax_snr) and in_trim,
            "in_nad_gap": (NAD_GAP_LO <= lam_r <= NAD_GAP_HI) and in_trim,
            "in_laser_lgs": (LASER_LGS_LO <= lam_r <= LASER_LGS_HI) and in_trim,
        })
    ch_df = pd.DataFrame(channel_rows)
    ch_df.to_csv(out_dir / f"{stem}_channel_nan.csv", index=False)

    hot = ch_df[
        (ch_df["frac_nan_flux"] >= hot_frac)
        | (ch_df["frac_nan_stat"] >= hot_frac)
    ]
    hot.to_csv(out_dir / f"{stem}_hot_channels.csv", index=False)

    # --- Spaxel nan_frac on trimmed grid (chunked) ---
    snr = mask_indices(w_trim, lmin_snr, lmax_snr)
    nad = mask_indices(w_trim, NAD_GAP_LO, NAD_GAP_HI)
    laser = mask_indices(w_trim, LASER_LGS_LO, LASER_LGS_HI)
    nad_set, laser_set = set(nad.tolist()), set(laser.tolist())
    snr_no_nad = np.array([i for i in snr if i not in nad_set], dtype=np.int64)
    snr_no_laser = np.array([i for i in snr if i not in laser_set], dtype=np.int64)
    all_local = np.arange(n_trim, dtype=np.int64)

    defunct_all = {str(t): 0 for t in DEFUNCT_THRESHOLDS}
    defunct_snr = {str(t): 0 for t in DEFUNCT_THRESHOLDS}
    defunct_no_nad = {str(t): 0 for t in DEFUNCT_THRESHOLDS}
    defunct_no_laser = {str(t): 0 for t in DEFUNCT_THRESHOLDS}

    hist_bins = np.linspace(0, 0.05, 26)
    hist_all = np.zeros(len(hist_bins) - 1, dtype=np.int64)
    hist_snr = np.zeros(len(hist_bins) - 1, dtype=np.int64)
    hist_no_nad = np.zeros(len(hist_bins) - 1, dtype=np.int64)
    hist_no_laser = np.zeros(len(hist_bins) - 1, dtype=np.int64)
    n_finite_snr = 0

    for i0 in range(0, n_spax, spaxel_chunk):
        i1 = min(i0 + spaxel_chunk, n_spax)
        nc = i1 - i0
        spec_chunk = np.empty((n_trim, nc), dtype=np.float64)
        stat_chunk = (
            np.empty((n_trim, nc), dtype=np.float64) if stat_ext is not None else None
        )
        for il, ig in enumerate(trim_global):
            plane = read_plane(path, flux_ext, int(ig))
            spec_chunk[il, :] = plane.reshape(-1)[i0:i1]
            if stat_ext is not None and stat_chunk is not None:
                sp = read_plane(path, stat_ext, int(ig))
                stat_chunk[il, :] = sp.reshape(-1)[i0:i1]

        def nan_frac(idxs: np.ndarray) -> np.ndarray:
            if len(idxs) == 0:
                return np.zeros(nc)
            return np.mean(np.isnan(spec_chunk[idxs, :]), axis=0)

        nf_all = nan_frac(all_local)
        nf_snr = nan_frac(snr)
        nf_no_nad = nan_frac(snr_no_nad)
        nf_no_laser = nan_frac(snr_no_laser)

        hist_all += np.histogram(nf_all, bins=hist_bins)[0]
        hist_snr += np.histogram(nf_snr, bins=hist_bins)[0]
        hist_no_nad += np.histogram(nf_no_nad, bins=hist_bins)[0]
        hist_no_laser += np.histogram(nf_no_laser, bins=hist_bins)[0]

        for t in DEFUNCT_THRESHOLDS:
            key = str(t)
            defunct_all[key] += int(np.sum(nf_all > t))
            defunct_snr[key] += int(np.sum(nf_snr > t))
            defunct_no_nad[key] += int(np.sum(nf_no_nad > t))
            defunct_no_laser[key] += int(np.sum(nf_no_laser > t))

        if len(snr) > 0 and stat_chunk is not None:
            sig = np.nanmedian(spec_chunk[snr, :], axis=0)
            noise = np.sqrt(np.nanmedian(stat_chunk[snr, :], axis=0))
            snr_val = np.nanmedian(
                spec_chunk[snr, :]
                / np.sqrt(np.maximum(stat_chunk[snr, :], 1e-30)),
                axis=0,
            )
            n_finite_snr += int(
                np.sum(np.isfinite(sig) & np.isfinite(noise) & np.isfinite(snr_val))
            )
        elif len(snr) > 0:
            sig = np.nanmedian(spec_chunk[snr, :], axis=0)
            n_finite_snr += int(np.sum(np.isfinite(sig)))

    spaxel_stats = pd.DataFrame({
        "mask_domain": [
            "trim_all",
            "snr_band",
            "snr_no_nad",
            "snr_no_laser",
        ],
        "n_channels": [n_trim, len(snr), len(snr_no_nad), len(snr_no_laser)],
        "defunct_at_0pct": [
            defunct_all["0.0"],
            defunct_snr["0.0"],
            defunct_no_nad["0.0"],
            defunct_no_laser["0.0"],
        ],
        "defunct_at_1pct": [
            defunct_all["0.01"],
            defunct_snr["0.01"],
            defunct_no_nad["0.01"],
            defunct_no_laser["0.01"],
        ],
        "defunct_at_5pct": [
            defunct_all["0.05"],
            defunct_snr["0.05"],
            defunct_no_nad["0.05"],
            defunct_no_laser["0.05"],
        ],
        "frac_defunct_at_1pct": [
            defunct_all["0.01"] / n_spax,
            defunct_snr["0.01"] / n_spax,
            defunct_no_nad["0.01"] / n_spax,
            defunct_no_laser["0.01"] / n_spax,
        ],
    })
    spaxel_stats.to_csv(out_dir / f"{stem}_spaxel_nan_stats.csv", index=False)

    pd.DataFrame({
        "bin_lo": hist_bins[:-1],
        "bin_hi": hist_bins[1:],
        "count_trim_all": hist_all,
        "count_snr_band": hist_snr,
        "count_snr_no_nad": hist_no_nad,
        "count_snr_no_laser": hist_no_laser,
    }).to_csv(out_dir / f"{stem}_nan_frac_hist.csv", index=False)

    spax_summary = {
        "n_spaxels": n_spax,
        "n_wave_trim": n_trim,
        "n_wave_snr": len(snr),
        "n_wave_nad": len(nad),
        "n_wave_laser": len(laser),
        "defunct_count_trim": defunct_all,
        "defunct_count_snr_no_nad": defunct_no_nad,
        "frac_defunct_trim": {k: v / n_spax for k, v in defunct_all.items()},
        "frac_defunct_snr_no_nad": {k: v / n_spax for k, v in defunct_no_nad.items()},
        "n_finite_snr_scalars": n_finite_snr,
        "frac_finite_snr_scalars": n_finite_snr / n_spax,
    }
    with open(out_dir / f"{stem}_spaxel_summary.json", "w") as f:
        json.dump(spax_summary, f, indent=2)

    meta = {
        "path": str(path),
        "hdus": hdu_list,
        "flux_ext": flux_ext,
        "stat_ext": stat_ext,
        "shape": [n3, ny, nx],
        "redshift": redshift,
        "lmin_tot": lmin_tot,
        "lmax_tot": lmax_tot,
        "lmin_snr": lmin_snr,
        "lmax_snr": lmax_snr,
        "n_hot_channels_flux": int((ch_df["frac_nan_flux"] >= hot_frac).sum()),
        "n_hot_channels_stat": int((ch_df["frac_nan_stat"] >= hot_frac).sum())
        if stat_ext is not None
        else 0,
        "mean_frac_nan_flux_trim": float(
            ch_df.loc[ch_df["in_trim"], "frac_nan_flux"].mean()
        )
        if ch_df["in_trim"].any()
        else np.nan,
        "mean_frac_nan_in_nad": float(
            ch_df.loc[ch_df["in_nad_gap"], "frac_nan_flux"].mean()
        )
        if ch_df["in_nad_gap"].any()
        else np.nan,
        "mean_frac_nan_in_laser": float(
            ch_df.loc[ch_df["in_laser_lgs"], "frac_nan_flux"].mean()
        )
        if ch_df["in_laser_lgs"].any()
        else np.nan,
        "frac_defunct": spax_summary["frac_defunct_trim"],
        **{k: v for k, v in spax_summary.items() if k != "frac_defunct_trim"},
    }
    with open(out_dir / f"{stem}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    nad_mean = meta.get("mean_frac_nan_in_nad") or 0.0
    d1 = defunct_all["0.01"]
    d1_nad = defunct_no_nad["0.01"]
    lines = [
        f"File: {path.name}",
        f"HDUs: flux ext {flux_ext} ({meta_flux.get('extname','')}), stat ext {stat_ext}",
        f"Shape (nwave, ny, nx): ({n3}, {ny}, {nx})",
        f"Trimmed channels: {n_trim}; SNR: {len(snr)}; NaD: {len(nad)}; laser: {len(laser)}",
        f"Defunct@1% (trim_all nan_frac): {d1} spaxels ({d1/n_spax:.2%})",
        f"Defunct@1% (snr_no_nad nan_frac): {d1_nad} spaxels ({d1_nad/n_spax:.2%})",
        f"Hot channels (>{hot_frac:.0%} NaN flux): {meta['n_hot_channels_flux']}",
        f"Mean channel NaN fraction in NaD window: {nad_mean:.4f}",
        f"Mean channel NaN fraction in laser window: {meta.get('mean_frac_nan_in_laser', 0):.4f}",
    ]
    if nad_mean > 0.3 and d1 > d1_nad * 1.5:
        lines.append(
            "VERDICT: NaD/laser gap NaNs likely inflate defunct@1% on trim_all; "
            "snr_no_nad domain reduces rejections — spectral mask for defunct/SNR recommended."
        )
    elif d1 > 0.05 * n_spax and nad_mean < 0.1:
        lines.append(
            "VERDICT: Broad NaN pattern (not NaD-localized); investigate reduction, not only laser mask."
        )
    else:
        lines.append("VERDICT: Review channel_nan.csv and hot_channels.csv for wavelength structure.")
    (out_dir / f"{stem}_diagnosis.txt").write_text("\n".join(lines) + "\n")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cube", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--redshift", type=float, default=0.0)
    ap.add_argument("--lmin-tot", type=float, default=4800.0)
    ap.add_argument("--lmax-tot", type=float, default=7000.0)
    ap.add_argument("--lmin-snr", type=float, default=4750.0)
    ap.add_argument("--lmax-snr", type=float, default=7100.0)
    ap.add_argument("--spaxel-chunk", type=int, default=SPAXEL_CHUNK)
    args = ap.parse_args()
    audit_cube(
        args.cube,
        args.out_dir,
        redshift=args.redshift,
        lmin_tot=args.lmin_tot,
        lmax_tot=args.lmax_tot,
        lmin_snr=args.lmin_snr,
        lmax_snr=args.lmax_snr,
        spaxel_chunk=args.spaxel_chunk,
    )
    print(f"Done: {args.cube.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
