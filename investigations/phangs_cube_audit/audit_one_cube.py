#!/usr/bin/env python3
"""Fitsio audit: 10 random spaxels, all wavelength channels, per MUSE cube."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fitsio_cube import (
    find_data_hdu,
    list_3d_hdus,
    list_hdus,
    read_header_wcs,
    read_spectra_for_spaxels,
    read_spectra_in_hdu,
    sample_spaxel_indices,
)
from wavelength_masks import (
    LASER_LGS_HI,
    LASER_LGS_LO,
    NAD_GAP_HI,
    NAD_GAP_LO,
    build_wave_grid,
    compile_masks,
    detect_edge_blank_channels,
    mask_indices,
    trim_indices_excluding_edge_blanks,
)

DEFUNCT_THRESHOLDS = (0.0, 0.01, 0.05)
DEFAULT_N_SAMPLE = 10


def _nan_frac_along_wave(spec_1d: np.ndarray, idxs: np.ndarray) -> float:
    if len(idxs) == 0:
        return 0.0
    return float(np.mean(np.isnan(spec_1d[idxs])))


def _defunct_counts(nf: np.ndarray, thresholds: tuple[float, ...]) -> dict[str, int]:
    return {str(t): int(np.sum(nf > t)) for t in thresholds}


def _safe_hdu_label(ext: int, name: str) -> str:
    label = (name or f"EXT{ext}").strip() or f"EXT{ext}"
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in label)


def _channel_nan_dataframe(
    spec_all: np.ndarray,
    wave_obs: np.ndarray,
    wave_rest: np.ndarray,
    trim_global: np.ndarray,
    n_samp: int,
    lmin_snr: float,
    lmax_snr: float,
    hdu_ext: int,
    hdu_name: str,
) -> pd.DataFrame:
    n3 = spec_all.shape[0]
    trim_set = set(trim_global.tolist())
    rows = []
    for ig in range(n3):
        lam_r = float(wave_rest[ig])
        in_trim = ig in trim_set
        col = spec_all[ig, :]
        n_nan = int(np.sum(~np.isfinite(col)))
        rows.append({
            "hdu_ext": hdu_ext,
            "hdu_name": hdu_name,
            "iwave": ig,
            "lambda_obs": float(wave_obs[ig]),
            "lambda_rest": lam_r,
            "frac_nan_in_sample": n_nan / n_samp,
            "in_trim": in_trim,
            "in_snr_band": (lmin_snr <= lam_r <= lmax_snr) and in_trim,
            "in_nad_gap": (NAD_GAP_LO <= lam_r <= NAD_GAP_HI) and in_trim,
            "in_laser_lgs": (LASER_LGS_LO <= lam_r <= LASER_LGS_HI) and in_trim,
        })
    return pd.DataFrame(rows)


def _audit_all_hdus(
    path: Path,
    out_dir: Path,
    stem: str,
    flat_idx: np.ndarray,
    nx: int,
    ny: int,
    n_samp: int,
    redshift: float,
    lmin_tot: float,
    lmax_tot: float,
    lmin_snr: float,
    lmax_snr: float,
    hot_frac: float,
    flux_ext: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Read 10-sample spectra from every 3D HDU; write per-HDU channel CSVs."""
    hdu_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for h in list_3d_hdus(path):
        ext = h["ext"]
        name = h["name"] or ""
        label = _safe_hdu_label(ext, name)
        row_base = {"hdu_ext": ext, "hdu_name": name, "hdu_label": label}

        try:
            meta_h = read_header_wcs(path, ext)
        except (KeyError, ValueError, TypeError) as e:
            summary_rows.append({
                **row_base,
                "status": "skip",
                "reason": f"header: {e}",
            })
            continue

        n3, hny, hnx = meta_h["naxis3"], meta_h["naxis2"], meta_h["naxis1"]
        if (hny, hnx) != (ny, nx):
            summary_rows.append({
                **row_base,
                "status": "skip",
                "reason": f"shape mismatch ({hnx},{hny}) vs flux ({nx},{ny})",
            })
            continue

        try:
            spec = read_spectra_in_hdu(path, ext, flat_idx, nx, n3)
        except Exception as e:
            summary_rows.append({
                **row_base,
                "status": "fail",
                "reason": str(e),
            })
            continue

        wave_obs = build_wave_grid(meta_h["crval3"], meta_h["cdelt3"], n3)
        wave_rest = wave_obs / (1.0 + redshift)
        trim_global = compile_masks(
            wave_rest, lmin_tot, lmax_tot, lmin_snr, lmax_snr
        )["trim"]

        ch_df = _channel_nan_dataframe(
            spec,
            wave_obs,
            wave_rest,
            trim_global,
            n_samp,
            lmin_snr,
            lmax_snr,
            ext,
            name,
        )
        ch_df.to_csv(out_dir / f"{stem}_ext{ext}_{label}_channel_nan.csv", index=False)
        hdu_frames.append(ch_df)

        in_trim = ch_df["in_trim"] == True  # noqa: E712
        summary_rows.append({
            **row_base,
            "status": "ok",
            "naxis3": n3,
            "is_primary_flux": ext == flux_ext,
            "mean_frac_nan_trim_in_sample": float(
                ch_df.loc[in_trim, "frac_nan_in_sample"].mean()
            )
            if in_trim.any()
            else np.nan,
            "mean_frac_nan_in_nad": float(
                ch_df.loc[ch_df["in_nad_gap"], "frac_nan_in_sample"].mean()
            )
            if ch_df["in_nad_gap"].any()
            else np.nan,
            "n_hot_channels_in_sample": int(
                (ch_df["frac_nan_in_sample"] >= hot_frac).sum()
            ),
        })

    all_hdus = pd.concat(hdu_frames, ignore_index=True) if hdu_frames else pd.DataFrame()
    if len(all_hdus):
        all_hdus.to_csv(out_dir / f"{stem}_channel_nan_all_hdus.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / f"{stem}_hdu_summary.csv", index=False)
    return all_hdus, summary_df, summary_rows


def audit_cube(
    path: Path,
    out_dir: Path,
    redshift: float = 0.0,
    lmin_tot: float = 4800.0,
    lmax_tot: float = 7000.0,
    lmin_snr: float = 4750.0,
    lmax_snr: float = 7100.0,
    hot_frac: float = 0.5,
    n_sample_spaxels: int = DEFAULT_N_SAMPLE,
    sample_seed: int = 42,
) -> dict:
    path = path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    hdu_list = list_hdus(path)
    flux_ext, stat_ext = find_data_hdu(path)
    meta_flux = read_header_wcs(path, flux_ext)
    n3, ny, nx = meta_flux["naxis3"], meta_flux["naxis2"], meta_flux["naxis1"]
    n_spax = ny * nx

    flat_idx = sample_spaxel_indices(n_spax, n_sample_spaxels, sample_seed)
    n_samp = len(flat_idx)

    wave_obs = build_wave_grid(meta_flux["crval3"], meta_flux["cdelt3"], n3)
    wave_rest = wave_obs / (1.0 + redshift)

    flux_all, stat_all = read_spectra_for_spaxels(
        path, flux_ext, stat_ext, flat_idx, nx, n3
    )

    _, hdu_summary_df, hdu_summary_rows = _audit_all_hdus(
        path,
        out_dir,
        stem,
        flat_idx,
        nx,
        ny,
        n_samp,
        redshift,
        lmin_tot,
        lmax_tot,
        lmin_snr,
        lmax_snr,
        hot_frac,
        flux_ext,
    )

    masks_full = compile_masks(wave_rest, lmin_tot, lmax_tot, lmin_snr, lmax_snr)
    trim_global = masks_full["trim"]
    w_trim = wave_rest[trim_global]
    n_trim = len(trim_global)

    spec_trim = flux_all[trim_global, :]
    stat_trim = stat_all[trim_global, :] if stat_all is not None else None

    snr = mask_indices(w_trim, lmin_snr, lmax_snr)
    nad = mask_indices(w_trim, NAD_GAP_LO, NAD_GAP_HI)
    laser = mask_indices(w_trim, LASER_LGS_LO, LASER_LGS_HI)
    nad_set, laser_set = set(nad.tolist()), set(laser.tolist())
    snr_no_nad = np.array([i for i in snr if i not in nad_set], dtype=np.int64)
    snr_no_laser = np.array([i for i in snr if i not in laser_set], dtype=np.int64)
    all_local = np.arange(n_trim, dtype=np.int64)

    # --- Primary flux HDU channel table (backward-compatible columns + stat) ---
    ch_df = _channel_nan_dataframe(
        flux_all,
        wave_obs,
        wave_rest,
        trim_global,
        n_samp,
        lmin_snr,
        lmax_snr,
        flux_ext,
        meta_flux["extname"],
    )
    ch_df = ch_df.rename(columns={"frac_nan_in_sample": "frac_nan_flux_in_sample"})
    if stat_all is not None:
        stat_frac = []
        for ig in range(n3):
            n_nan_s = int(np.sum(~np.isfinite(stat_all[ig, :])))
            stat_frac.append(n_nan_s / n_samp)
        ch_df["frac_nan_stat_in_sample"] = stat_frac
    else:
        ch_df["frac_nan_stat_in_sample"] = np.nan
    ch_df.to_csv(out_dir / f"{stem}_channel_nan.csv", index=False)

    hot = ch_df[
        (ch_df["frac_nan_flux_in_sample"] >= hot_frac)
        | (ch_df["frac_nan_stat_in_sample"] >= hot_frac)
    ]
    hot.to_csv(out_dir / f"{stem}_hot_channels.csv", index=False)

    edge = detect_edge_blank_channels(
        ch_df["frac_nan_flux_in_sample"].to_numpy(),
        wave_rest,
    )
    trim_no_edge_global = trim_indices_excluding_edge_blanks(
        trim_global, edge["valid_channel_mask"]
    )
    edge_in_trim = int(
        np.sum(~edge["valid_channel_mask"][trim_global])
    ) if len(trim_global) else 0
    pd.DataFrame([{
        "n_leading_blank": edge["n_leading_blank"],
        "n_trailing_blank": edge["n_trailing_blank"],
        "n_blank_in_config_trim": edge_in_trim,
        "has_edge_blank": edge["has_edge_blank"],
        "lambda_rest_first_valid": edge["lambda_rest_first_valid"],
        "lambda_rest_last_valid": edge["lambda_rest_last_valid"],
        "suggested_lmin_tot": edge["suggested_lmin_tot"],
        "suggested_lmax_tot": edge["suggested_lmax_tot"],
        "config_lmin_tot": lmin_tot,
        "config_lmax_tot": lmax_tot,
        "example_phangs_cube": (
            "NGC1087_PHANGS_DATACUBE_native.fits (not all PHANGS cubes)"
        ),
    }]).to_csv(out_dir / f"{stem}_edge_blank.csv", index=False)

    # Map global trim-no-edge indices to local trim indices
    trim_set = set(trim_global.tolist())
    trim_no_edge_local = np.array(
        [
            np.where(trim_global == g)[0][0]
            for g in trim_no_edge_global
            if g in trim_set
        ],
        dtype=np.int64,
    )

    # --- Per sampled spaxel: nan_frac by mask domain (trim grid) ---
    spaxel_rows = []
    nf_all = np.zeros(n_samp)
    nf_trim_no_edge = np.zeros(n_samp)
    nf_snr = np.zeros(n_samp)
    nf_no_nad = np.zeros(n_samp)
    nf_no_laser = np.zeros(n_samp)
    n_finite_snr = 0

    for j in range(n_samp):
        spec_j = spec_trim[:, j]
        stat_j = stat_trim[:, j] if stat_trim is not None else None
        nf_all[j] = _nan_frac_along_wave(spec_j, all_local)
        nf_trim_no_edge[j] = _nan_frac_along_wave(spec_j, trim_no_edge_local)
        nf_snr[j] = _nan_frac_along_wave(spec_j, snr)
        nf_no_nad[j] = _nan_frac_along_wave(spec_j, snr_no_nad)
        nf_no_laser[j] = _nan_frac_along_wave(spec_j, snr_no_laser)

        sig = noise = snr_val = np.nan
        if len(snr) > 0:
            sig = float(np.nanmedian(spec_j[snr]))
            if stat_j is not None:
                noise = float(np.sqrt(np.nanmedian(stat_j[snr])))
                snr_val = float(
                    np.nanmedian(
                        spec_j[snr] / np.sqrt(np.maximum(stat_j[snr], 1e-30))
                    )
                )
            if np.isfinite(sig) and np.isfinite(noise) and np.isfinite(snr_val):
                n_finite_snr += 1

        spaxel_rows.append({
            "spaxel_index": int(flat_idx[j]),
            "iy": int(flat_idx[j] // nx),
            "ix": int(flat_idx[j] % nx),
            "nan_frac_trim_all": nf_all[j],
            "nan_frac_trim_no_edge_blank": nf_trim_no_edge[j],
            "nan_frac_snr_band": nf_snr[j],
            "nan_frac_snr_no_nad": nf_no_nad[j],
            "nan_frac_snr_no_laser": nf_no_laser[j],
            "signal_snr_scalar": sig,
            "noise_snr_scalar": noise,
            "snr_scalar": snr_val,
            "defunct_at_0pct": nf_all[j] > 0.0,
            "defunct_at_1pct": nf_all[j] > 0.01,
            "defunct_at_5pct": nf_all[j] > 0.05,
        })
    pd.DataFrame(spaxel_rows).to_csv(out_dir / f"{stem}_sample_spaxels.csv", index=False)

    defunct_all = _defunct_counts(nf_all, DEFUNCT_THRESHOLDS)
    defunct_no_edge = _defunct_counts(nf_trim_no_edge, DEFUNCT_THRESHOLDS)
    defunct_snr = _defunct_counts(nf_snr, DEFUNCT_THRESHOLDS)
    defunct_no_nad = _defunct_counts(nf_no_nad, DEFUNCT_THRESHOLDS)
    defunct_no_laser = _defunct_counts(nf_no_laser, DEFUNCT_THRESHOLDS)

    spaxel_stats = pd.DataFrame({
        "mask_domain": [
            "trim_all",
            "trim_no_edge_blank",
            "snr_band",
            "snr_no_nad",
            "snr_no_laser",
        ],
        "n_channels": [
            n_trim,
            len(trim_no_edge_local),
            len(snr),
            len(snr_no_nad),
            len(snr_no_laser),
        ],
        "defunct_at_0pct_in_sample": [
            defunct_all["0.0"],
            defunct_no_edge["0.0"],
            defunct_snr["0.0"],
            defunct_no_nad["0.0"],
            defunct_no_laser["0.0"],
        ],
        "defunct_at_1pct_in_sample": [
            defunct_all["0.01"],
            defunct_no_edge["0.01"],
            defunct_snr["0.01"],
            defunct_no_nad["0.01"],
            defunct_no_laser["0.01"],
        ],
        "defunct_at_5pct_in_sample": [
            defunct_all["0.05"],
            defunct_no_edge["0.05"],
            defunct_snr["0.05"],
            defunct_no_nad["0.05"],
            defunct_no_laser["0.05"],
        ],
        "frac_defunct_at_1pct_in_sample": [
            defunct_all["0.01"] / n_samp,
            defunct_no_edge["0.01"] / n_samp,
            defunct_snr["0.01"] / n_samp,
            defunct_no_nad["0.01"] / n_samp,
            defunct_no_laser["0.01"] / n_samp,
        ],
    })
    spaxel_stats.to_csv(out_dir / f"{stem}_spaxel_nan_stats.csv", index=False)

    hist_bins = np.linspace(0, 0.05, 26)
    pd.DataFrame({
        "bin_lo": hist_bins[:-1],
        "bin_hi": hist_bins[1:],
        "count_trim_all": np.histogram(nf_all, bins=hist_bins)[0],
        "count_trim_no_edge_blank": np.histogram(nf_trim_no_edge, bins=hist_bins)[0],
        "count_snr_band": np.histogram(nf_snr, bins=hist_bins)[0],
        "count_snr_no_nad": np.histogram(nf_no_nad, bins=hist_bins)[0],
        "count_snr_no_laser": np.histogram(nf_no_laser, bins=hist_bins)[0],
    }).to_csv(out_dir / f"{stem}_nan_frac_hist.csv", index=False)

    meta = {
        "path": str(path),
        "hdus": hdu_list,
        "hdu_audit": hdu_summary_rows,
        "flux_ext": flux_ext,
        "stat_ext": stat_ext,
        "shape": [n3, ny, nx],
        "n_spaxels_field": n_spax,
        "n_sample_spaxels": n_samp,
        "sample_seed": sample_seed,
        "sample_spaxel_indices": flat_idx.tolist(),
        "sampling_note": "Metrics are for random sample only, not full field",
        "redshift": redshift,
        "lmin_tot": lmin_tot,
        "lmax_tot": lmax_tot,
        "n_wave_all": n3,
        "n_wave_trim": n_trim,
        "n_hot_channels_flux_in_sample": int(
            (ch_df["frac_nan_flux_in_sample"] >= hot_frac).sum()
        ),
        "n_hot_channels_stat_in_sample": int(
            (ch_df["frac_nan_stat_in_sample"] >= hot_frac).sum()
        )
        if stat_all is not None
        else 0,
        "mean_frac_nan_flux_trim_in_sample": float(
            ch_df.loc[ch_df["in_trim"], "frac_nan_flux_in_sample"].mean()
        )
        if ch_df["in_trim"].any()
        else np.nan,
        "mean_frac_nan_in_nad": float(
            ch_df.loc[ch_df["in_nad_gap"], "frac_nan_flux_in_sample"].mean()
        )
        if ch_df["in_nad_gap"].any()
        else np.nan,
        "mean_frac_nan_in_laser": float(
            ch_df.loc[ch_df["in_laser_lgs"], "frac_nan_flux_in_sample"].mean()
        )
        if ch_df["in_laser_lgs"].any()
        else np.nan,
        "edge_blank": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in edge.items()
            if k != "valid_channel_mask"
        },
        "frac_defunct": {k: v / n_samp for k, v in defunct_all.items()},
        "frac_defunct_trim_no_edge": {
            k: v / n_samp for k, v in defunct_no_edge.items()
        },
        "frac_defunct_snr_no_nad": {k: v / n_samp for k, v in defunct_no_nad.items()},
        "n_finite_snr_scalars_in_sample": n_finite_snr,
        "frac_finite_snr_scalars_in_sample": n_finite_snr / n_samp,
    }
    with open(out_dir / f"{stem}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    nad_mean = meta.get("mean_frac_nan_in_nad") or 0.0
    d1 = defunct_all["0.01"]
    d1_no_edge = defunct_no_edge["0.01"]
    d1_nad = defunct_no_nad["0.01"]
    lines = [
        f"File: {path.name}",
        f"HDUs: flux ext {flux_ext}, stat ext {stat_ext}",
        f"3D HDUs audited (10 spaxels each, all channels): {len(hdu_summary_df)}",
    ]
    for _, hr in hdu_summary_df.iterrows():
        st = hr["status"]
        extra = f" — {hr['reason']}" if st != "ok" and pd.notna(hr.get("reason")) else ""
        lines.append(
            f"  ext {int(hr['hdu_ext'])} {hr['hdu_name']!r}: {st}{extra}"
        )
    lines += [
        f"Shape (nwave, ny, nx): ({n3}, {ny}, {nx}); field spaxels: {n_spax}",
        f"Sample: {n_samp} random spaxels (seed={sample_seed}), same indices in every HDU",
        f"Trimmed channels: {n_trim}; SNR: {len(snr)}; NaD: {len(nad)}; laser: {len(laser)}",
        (
            f"Edge blank pads: {edge['n_leading_blank']} blue + "
            f"{edge['n_trailing_blank']} red "
            f"({edge_in_trim} inside config trim); "
            f"suggested LMIN/LMAX_TOT ≈ "
            f"{edge['suggested_lmin_tot']:.1f}–{edge['suggested_lmax_tot']:.1f} Å"
            if edge["has_edge_blank"]
            else "Edge blank pads: none detected on native axis"
        ),
        f"Defunct@1% in sample (trim_all nan_frac): {d1}/{n_samp} spaxels",
        f"Defunct@1% in sample (trim_no_edge_blank): {d1_no_edge}/{n_samp} spaxels",
        f"Defunct@1% in sample (snr_no_nad): {d1_nad}/{n_samp} spaxels",
        f"Hot channels (>{hot_frac:.0%} NaN in sample): flux={meta['n_hot_channels_flux_in_sample']}",
        f"Mean frac sample spaxels NaN per channel in NaD window: {nad_mean:.4f}",
        f"Mean frac sample spaxels NaN per channel in laser window: {meta.get('mean_frac_nan_in_laser', 0):.4f}",
    ]
    if edge["has_edge_blank"] and edge_in_trim > 0:
        msg = (
            "VERDICT: All-NaN edge pads on native axis "
            f"({edge_in_trim} channels inside config trim; "
            f"suggest LMIN/LMAX_TOT ≈ {edge['suggested_lmin_tot']:.0f}–"
            f"{edge['suggested_lmax_tot']:.0f} Å). "
            "Wavelength-range trim fixes cubes like "
            "NGC1087_PHANGS_DATACUBE_native.fits (not all PHANGS galaxies). "
            "See edge_blank.csv."
        )
        if d1 > d1_no_edge:
            msg += f" Dropping pads lowers defunct@1% sample count ({d1}→{d1_no_edge})."
        elif nad_mean > 0.3:
            msg += " NaD/laser NaNs may still dominate defunct after edge trim."
        lines.append(msg)
    elif edge["has_edge_blank"] and d1 > d1_no_edge:
        lines.append(
            "VERDICT: Leading/trailing all-NaN channels inflate defunct; "
            "tighten LMIN_TOT/LMAX_TOT (edge_blank.csv)."
        )
    elif nad_mean > 0.3 and d1 > d1_nad:
        lines.append(
            "VERDICT: NaD/laser channels often NaN in sample; snr_no_nad lowers "
            "per-spaxel nan_frac — spectral mask for defunct/SNR recommended."
        )
    elif d1 >= n_samp * 0.5 and nad_mean < 0.1:
        lines.append("VERDICT: Broad NaNs in sample (not NaD-localized).")
    else:
        lines.append(
            "VERDICT: Review channel_nan_all_hdus.csv, channel_nan.csv, sample_spaxels.csv."
        )
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
    ap.add_argument(
        "--n-sample",
        type=int,
        default=DEFAULT_N_SAMPLE,
        help="Number of random spaxels to read (all channels)",
    )
    ap.add_argument("--sample-seed", type=int, default=42)
    args = ap.parse_args()
    audit_cube(
        args.cube,
        args.out_dir,
        redshift=args.redshift,
        lmin_tot=args.lmin_tot,
        lmax_tot=args.lmax_tot,
        lmin_snr=args.lmin_snr,
        lmax_snr=args.lmax_snr,
        n_sample_spaxels=args.n_sample,
        sample_seed=args.sample_seed,
    )
    print(f"Done: {args.cube.name} ({args.n_sample} spaxels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
