#!/usr/bin/env python3
"""Aggregate per-cube audit outputs into PHANGS vs MAUVE summary + HTML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def stack_channel_profiles(audit_root: Path, index: pd.DataFrame) -> pd.DataFrame:
    """Mean frac_nan_flux vs lambda_rest per survey (trimmed channels)."""
    stacks = []
    for survey in ("PHANGS", "MAUVE"):
        sub = index[index["survey"] == survey]
        acc: dict[float, list[float]] = {}
        for _, r in sub.iterrows():
            ch_path = Path(r["out_dir"]) / (
                Path(r["path"]).stem + "_channel_nan.csv"
            )
            if not ch_path.is_file():
                continue
            ch = pd.read_csv(ch_path)
            ch = ch[ch["in_trim"] == True]  # noqa: E712
            for _, row in ch.iterrows():
                lam = round(float(row["lambda_rest"]), 2)
                acc.setdefault(lam, []).append(float(row["frac_nan_flux"]))
        for lam, vals in sorted(acc.items()):
            stacks.append({
                "survey": survey,
                "lambda_rest": lam,
                "mean_frac_nan_flux": float(np.mean(vals)),
                "n_cubes": len(vals),
            })
    return pd.DataFrame(stacks)


def write_html(
    out_path: Path,
    summary_df: pd.DataFrame,
    stack_df: pd.DataFrame,
    interpretation: str,
) -> None:
    rows = ""
    for _, r in summary_df.iterrows():
        rows += (
            f"<tr><td>{r['survey']}</td><td>{r['file']}</td>"
            f"<td>{r.get('frac_defunct_1pct', '')}</td>"
            f"<td>{r.get('mean_frac_nan_nad', '')}</td>"
            f"<td>{r.get('frac_defunct_1pct_no_nad', '')}</td>"
            f"<td>{r.get('n_hot_flux', '')}</td></tr>\n"
        )
    stack_rows = ""
    for _, r in stack_df.head(200).iterrows():
        stack_rows += (
            f"<tr><td>{r['survey']}</td><td>{r['lambda_rest']}</td>"
            f"<td>{r['mean_frac_nan_flux']:.4f}</td><td>{r['n_cubes']}</td></tr>\n"
        )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>PHANGS vs MAUVE NaN audit</title>
<style>body{{font-family:sans-serif;margin:1.5em}} table{{border-collapse:collapse}}
th,td{{border:1px solid #ccc;padding:4px 8px}} th{{background:#eee}}</style></head>
<body>
<h1>PHANGS vs MAUVE cube NaN audit</h1>
<p>{interpretation}</p>
<h2>Per-cube summary</h2>
<table><tr><th>Survey</th><th>File</th><th>Defunct@1% trim</th>
<th>Mean NaN NaD</th><th>Defunct@1% snr_no_nad</th><th>Hot ch.</th></tr>
{rows}</table>
<h2>Stacked channel profile (trim, mean frac NaN flux)</h2>
<table><tr><th>Survey</th><th>lambda_rest</th><th>mean_frac_nan</th><th>n_cubes</th></tr>
{stack_rows}</table>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="Markdown summary path")
    args = ap.parse_args()

    index_path = args.audit_root / "audit_index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    idx = pd.read_csv(index_path)
    idx_ok = idx[idx["status"] == "ok"].copy()

    rows = []
    hot_ranges = []
    for _, r in idx_ok.iterrows():
        stem = Path(r["path"]).stem
        out = Path(r["out_dir"])
        meta_path = out / f"{stem}_meta.json"
        if not meta_path.is_file():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        stats_path = out / f"{stem}_spaxel_nan_stats.csv"
        frac_no_nad = np.nan
        if stats_path.is_file():
            st = pd.read_csv(stats_path)
            row = st[st["mask_domain"] == "snr_no_nad"]
            if len(row):
                frac_no_nad = float(row.iloc[0]["frac_defunct_at_1pct"])

        rows.append({
            "survey": r["survey"],
            "galaxy": r.get("galaxy_guess", ""),
            "file": Path(r["path"]).name,
            "flux_ext": meta.get("flux_ext"),
            "stat_ext": meta.get("stat_ext"),
            "frac_defunct_0": meta.get("frac_defunct", {}).get("0.0"),
            "frac_defunct_1pct": meta.get("frac_defunct", {}).get("0.01"),
            "frac_defunct_5pct": meta.get("frac_defunct", {}).get("0.05"),
            "frac_defunct_1pct_no_nad": frac_no_nad,
            "mean_frac_nan_nad": meta.get("mean_frac_nan_in_nad"),
            "mean_frac_nan_laser": meta.get("mean_frac_nan_in_laser"),
            "n_hot_flux": meta.get("n_hot_channels_flux"),
        })

        hot_path = out / f"{stem}_hot_channels.csv"
        if hot_path.is_file():
            hot = pd.read_csv(hot_path)
            if len(hot):
                hot_ranges.append({
                    "survey": r["survey"],
                    "file": Path(r["path"]).name,
                    "lambda_min": hot["lambda_rest"].min(),
                    "lambda_max": hot["lambda_rest"].max(),
                    "n_hot": len(hot),
                })

    df = pd.DataFrame(rows)
    stack_df = stack_channel_profiles(args.audit_root, idx_ok)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    stack_df.to_csv(args.audit_root / "stacked_channel_profile.csv", index=False)
    if hot_ranges:
        pd.DataFrame(hot_ranges).to_csv(args.audit_root / "hot_channel_ranges.csv", index=False)

    lines = [
        "# PHANGS vs MAUVE cube NaN audit summary\n",
        "\n## Pipeline context\n",
        "- Defunct mask uses `nan_frac` over **full** `LMIN_TOT`–`LMAX_TOT` trim (`ngistPipeline/spatialMasking/default.py`).\n",
        "- `MUSE_WFM` does **not** exclude or infill NaD/laser gap; `MUSE_WFMAON` does (`ngistPipeline/readData/`).\n",
        "\n## Per-survey medians\n",
    ]
    interpretation = ""
    for survey in ["PHANGS", "MAUVE"]:
        sub = df[df["survey"] == survey]
        if len(sub) == 0:
            continue
        lines.append(f"\n### {survey} ({len(sub)} cubes)\n")
        lines.append(
            f"- Median defunct@1% (trim_all): **{sub['frac_defunct_1pct'].median():.4f}**\n"
        )
        lines.append(
            f"- Median defunct@1% (snr_no_nad): **{sub['frac_defunct_1pct_no_nad'].median():.4f}**\n"
        )
        lines.append(
            f"- Median mean channel NaN in NaD window: **{sub['mean_frac_nan_nad'].median():.4f}**\n"
        )
        lines.append(
            f"- Median hot channels (flux): **{sub['n_hot_flux'].median():.0f}**\n"
        )

    if len(df[df["survey"] == "PHANGS"]) and len(df[df["survey"] == "MAUVE"]):
        p = df[df["survey"] == "PHANGS"]["frac_defunct_1pct"].median()
        m = df[df["survey"] == "MAUVE"]["frac_defunct_1pct"].median()
        pn = df[df["survey"] == "PHANGS"]["frac_defunct_1pct_no_nad"].median()
        lines.append("\n## Interpretation\n")
        if p > max(m * 1.5, 0.02):
            interpretation = (
                "PHANGS cubes show higher defunct@1% on trim_all than MAUVE. "
                "If defunct@1% drops on snr_no_nad, NaD/laser spectral mask is the clean fix."
            )
            lines.append(
                f"- PHANGS median defunct@1% ({p:.3f}) > MAUVE ({m:.3f}).\n"
            )
            lines.append(
                f"- PHANGS defunct@1% on snr_no_nad median: {pn:.3f} "
                f"(delta vs trim_all suggests NaD/laser channels).\n"
            )
            lines.append(
                "\n**Recommendation:** Add config-driven wavelength exclusions for "
                "defunct `nan_frac` and/or SNR scalars in `MUSE_WFM`; optional LGS infill like WFMAON.\n"
            )
        else:
            interpretation = "No strong PHANGS vs MAUVE split; inspect per-file channel_nan.csv."
            lines.append("- No strong survey-level split in defunct@1%; inspect per-cube channel maps.\n")

    if len(hot_ranges):
        lines.append("\n## Hot channel wavelength ranges (per file)\n")
        for h in hot_ranges[:10]:
            lines.append(
                f"- {h['survey']} {h['file']}: {h['lambda_min']:.1f}–{h['lambda_max']:.1f} Å "
                f"({h['n_hot']} channels)\n"
            )

    lines.append("\n## Artifacts\n")
    lines.append("- `audit_index.csv`, `stacked_channel_profile.csv`, `report.html`\n")
    lines.append("- Per cube: `*_channel_nan.csv`, `*_spaxel_nan_stats.csv`, `*_diagnosis.txt`\n")

    md = "".join(lines)
    args.out.write_text(md, encoding="utf-8")
    write_html(args.audit_root / "report.html", df, stack_df, interpretation)
    print(f"Wrote {args.out}, {args.out.with_suffix('.csv')}, report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
