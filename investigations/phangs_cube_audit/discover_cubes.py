#!/usr/bin/env python3
"""Catalog PHANGS and MAUVE cube FITS files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def guess_galaxy(name: str) -> str:
    m = re.search(r"(NGC\d+|IC\d+|UGC\d+|MCG[^\._]+)", name, re.I)
    return m.group(1).upper() if m else Path(name).stem


def catalog_dir(root: Path, survey: str) -> list[dict]:
    rows = []
    for p in sorted(root.rglob("*.fits")):
        if p.is_file():
            rows.append({
                "survey": survey,
                "path": str(p.resolve()),
                "filename": p.name,
                "size_gb": p.stat().st_size / 1e9,
                "galaxy_guess": guess_galaxy(p.name),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phangs-dir", type=Path, required=True)
    ap.add_argument("--mauve-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = catalog_dir(args.phangs_dir, "PHANGS") + catalog_dir(args.mauve_dir, "MAUVE")
    df = pd.DataFrame(rows).sort_values(["survey", "size_gb"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} cubes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
