#!/usr/bin/env python3
"""Scan arc config trees for READ_DATA / SPATIAL settings (optional CANFAR helper)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_yaml_snippet(text: str) -> dict:
    out: dict[str, str | float] = {}
    for key in (
        "METHOD",
        "LMIN_TOT",
        "LMAX_TOT",
        "LMIN_SNR",
        "LMAX_SNR",
        "MIN_SNR",
        "TARGET_SNR",
    ):
        m = re.search(rf"^\s*{key}\s*:\s*(\S+)", text, re.M | re.I)
        if m:
            val = m.group(1).strip()
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val
    return out


def scan_configs(root: Path) -> list[dict]:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix in (".yaml", ".yml", "") and p.is_file():
            if p.name in ("CONFIG", "MasterConfig.yaml", "config.yaml") or "config" in p.name.lower():
                try:
                    text = p.read_text(errors="replace")
                except OSError:
                    continue
                if "READ_DATA" not in text and "LMIN" not in text:
                    continue
                parsed = parse_yaml_snippet(text)
                if parsed:
                    rows.append({"path": str(p), **parsed})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-root", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for root in args.config_root:
        if root.is_dir():
            rows.extend(scan_configs(root))
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} config rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
