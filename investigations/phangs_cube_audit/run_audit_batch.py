#!/usr/bin/env python3
"""Run cube audit on catalog subset (pilot or full)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from audit_one_cube import audit_cube


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--survey", choices=["PHANGS", "MAUVE", "ALL"], default="ALL")
    ap.add_argument("--max-cubes", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--smallest-first",
        action="store_true",
        help="Pilot: smallest files first",
    )
    ap.add_argument("--redshift", type=float, default=0.0)
    ap.add_argument("--lmin-tot", type=float, default=4800.0)
    ap.add_argument("--lmax-tot", type=float, default=7000.0)
    ap.add_argument("--lmin-snr", type=float, default=4750.0)
    ap.add_argument("--lmax-snr", type=float, default=7100.0)
    args = ap.parse_args()

    df = pd.read_csv(args.catalog)
    if args.survey != "ALL":
        df = df[df["survey"] == args.survey]
    df = df.sort_values("size_gb", ascending=args.smallest_first)
    if args.max_cubes > 0:
        df = df.head(args.max_cubes)

    args.out_root.mkdir(parents=True, exist_ok=True)
    index_path = args.out_root / "audit_index.csv"
    if index_path.is_file():
        prior = pd.read_csv(index_path)
        if args.survey != "ALL":
            prior = prior[prior["survey"] != args.survey]
        index_rows = prior.to_dict("records")
    else:
        index_rows = []

    for _, row in df.iterrows():
        cube = Path(row["path"])
        sub = args.out_root / row["survey"] / cube.stem
        try:
            meta = audit_cube(
                cube,
                sub,
                redshift=args.redshift,
                lmin_tot=args.lmin_tot,
                lmax_tot=args.lmax_tot,
                lmin_snr=args.lmin_snr,
                lmax_snr=args.lmax_snr,
            )
            status = "ok"
            err = ""
        except Exception as e:
            meta = {}
            status = "fail"
            err = str(e)

        frac_no_nad = None
        stats_p = sub / f"{cube.stem}_spaxel_nan_stats.csv"
        if stats_p.is_file():
            st = pd.read_csv(stats_p)
            m = st[st["mask_domain"] == "snr_no_nad"]
            if len(m):
                frac_no_nad = float(m.iloc[0]["frac_defunct_at_1pct"])

        index_rows.append({
            "survey": row["survey"],
            "path": row["path"],
            "galaxy_guess": row.get("galaxy_guess", ""),
            "out_dir": str(sub),
            "status": status,
            "error": err,
            "frac_defunct_0.01": meta.get("frac_defunct", {}).get("0.01"),
            "frac_defunct_1pct_no_nad": frac_no_nad,
            "mean_frac_nan_in_nad": meta.get("mean_frac_nan_in_nad"),
            "mean_frac_nan_in_laser": meta.get("mean_frac_nan_in_laser"),
        })
        print(f"[{status}] {cube.name}")

    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    print(f"Wrote {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
