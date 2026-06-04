# PHANGS vs MAUVE cube NaN audit (CANFAR)

Isolated investigation for NaN / NaD / laser-gap patterns in MUSE cubes.
Does **not** modify nGIST pipeline code on this branch.

## Sampling strategy

Each cube is audited by reading **10 random spaxels** (configurable) over **all**
wavelength channels (`NAXIS3`). Fitsio reads one spectrum per sampled spaxel
(`[:, iy, ix]`) — no full-cube load into RAM.

Channel metrics report the fraction of those 10 spaxels that are NaN at each λ.
**Every 3D HDU** in the file gets the same 10 spaxel indices (one spectrum read
per spaxel per HDU). Defunct / SNR simulations use the primary flux (+ stat) HDU
on the trimmed wavelength range.

## Hypothesis

PHANGS cubes can show two distinct NaN patterns (not every galaxy has both):

1. **Edge blank pads** — leading/trailing channels that are all-NaN on every
   spaxel (native axis wider than science range). Example:
   `.../phangs-muse/cubes/NGC1087_PHANGS_DATACUBE_native.fits`. Tightening
   `LMIN_TOT` / `LMAX_TOT` (or dropping blank ends before read) fixes defunct
   because `maskDefunctSpaxels` rejects on `np.any(np.isnan(spec))`.
2. **NaD / LGS gap** — partial NaNs in the laser window; `MUSE_WFM` does not
   exclude or infill (unlike `MUSE_WFMAON`).

The audit reports `*_edge_blank.csv` (pad counts, suggested λ range) and compares
defunct counts for `trim_all` vs `trim_no_edge_blank`.

## Requirements

- Python 3.9+
- `fitsio`, `numpy`, `pandas` (`requirements-canfar.txt`)
- RAM: O(`NAXIS3` × `n_sample`) per cube (default 10 spaxels × ~7k channels)

## CANFAR execution

```bash
cd /path/to/gist-geckos
git checkout investigate/phangs-cube-nan-audit

export PHANGS_DIR=/arc/projects/mauve/toby_sandbox/multiwavelength/phangs/phangs-muse/cubes
export MAUVE_DIR=/arc/projects/mauve/cubes/v3.0
export SCRATCH=/scratch/$USER/phangs_nan_audit
export N_SAMPLE_SPAXELS=10
export SAMPLE_SEED=42

bash investigations/phangs_cube_audit/canfar_job.sh pilot
bash investigations/phangs_cube_audit/canfar_job.sh full
```

Local smoke test:

```bash
SCRATCH=/tmp/$USER/phangs_nan_audit bash investigations/phangs_cube_audit/canfar_job.sh synthetic-test
```

## Outputs

| File | Content |
|------|---------|
| `{stem}_channel_nan.csv` | Primary flux HDU (+ stat columns if present) |
| `{stem}_channel_nan_all_hdus.csv` | All 3D HDUs stacked (`hdu_ext`, `hdu_name`, …) |
| `{stem}_ext{N}_{name}_channel_nan.csv` | Per-HDU channel NaN fractions |
| `{stem}_hdu_summary.csv` | Per-HDU read status and trim/NaD means |
| `{stem}_edge_blank.csv` | Leading/trailing all-NaN pads + suggested `LMIN/LMAX_TOT` |
| `{stem}_sample_spaxels.csv` | Per sampled spaxel: nan_frac by mask domain (flux) |
| `{stem}_spaxel_nan_stats.csv` | Defunct counts in sample (out of 10) |
| `{stem}_meta.json` | HDU info, `hdu_audit`, sample indices, seed |

## Single cube

```bash
python3 audit_one_cube.py /arc/.../cube.fits \
  --out-dir /scratch/$USER/audit_one \
  --n-sample 10 --sample-seed 42
```
