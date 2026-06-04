#!/usr/bin/env python3
"""Aggregate per-cube audit outputs into PHANGS vs MAUVE summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    index_path = args.audit_root / "audit_index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    idx = pd.read_csv(index_path)
    idx = idx[idx["status"] == "ok"]

    rows = []
    for _, r in idx.iterrows():
        meta_path = Path(r["out_dir"]) / (Path(r["path"]).stem + "_meta.json")
        if meta_path.is_file():
            with open(meta_path) as f:
                meta = json.load(f)
            rows.append({
                "survey": r["survey"],
                "galaxy": r.get("galaxy_guess", ""),
                "file": Path(r["path"]).name,
                "frac_defunct_0": meta.get("frac_defunct", {}).get("0.0"),
                "frac_defunct_1pct": meta.get("frac_defunct", {}).get("0.01"),
                "frac_defunct_5pct": meta.get("frac_defunct", {}).get("0.05"),
                "mean_frac_nan_nad": meta.get("mean_frac_nan_in_nad"),
                "mean_frac_nan_laser": meta.get("mean_frac_nan_in_laser"),
                "n_hot_flux": meta.get("n_hot_channels_flux"),
            })

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out.with_suffix(".csv"), index=False)

    lines = ["# PHANGS vs MAUVE cube NaN audit summary\n"]
    for survey in ["PHANGS", "MAUVE"]:
        sub = df[df["survey"] == survey]
        if len(sub) == 0:
            continue
        lines.append(f"## {survey} ({len(sub)} cubes)\n")
        lines.append(f"- Median frac defunct @1%: {sub['frac_defunct_1pct'].median():.4f}\n")
        lines.append(f"- Median mean NaN frac in NaD window: {sub['mean_frac_nan_nad'].median():.4f}\n")
        lines.append(f"- Median hot channels (flux): {sub['n_hot_flux'].median():.0f}\n")

    if len(df[df["survey"] == "PHANGS"]) and len(df[df["survey"] == "MAUVE"]):
        p = df[df["survey"] == "PHANGS"]["frac_defunct_1pct"].median()
        m = df[df["survey"] == "MAUVE"]["frac_defunct_1pct"].median()
        lines.append("\n## Interpretation\n")
        if p > m * 2:
            lines.append(
                "PHANGS cubes show higher defunct@1% rates; check NaD/laser channel_nan.csv "
                "spikes. MUSE_WFM does not exclude/infill laser gap (unlike WFMAON).\n"
            )
        else:
            lines.append("No strong PHANGS vs MAUVE split in defunct@1%; investigate per-file channel maps.\n")

    md = "".join(lines)
    args.out.write_text(md)
    print(f"Wrote {args.out} and {args.out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
