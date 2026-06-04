# PHANGS vs MAUVE cube NaN audit (CANFAR)

Isolated investigation for NaN / NaD / laser-gap patterns in MUSE cubes.
Does **not** modify nGIST pipeline code on this branch.

## Hypothesis

PHANGS cubes may have NaNs concentrated in the NaD / LGS wavelength gap.
[`MUSE_WFM`](../../ngistPipeline/readData/MUSE_WFM.py) computes defunct `nan_frac` over the
full trimmed spectrum and does **not** exclude or infill the laser gap (unlike
[`MUSE_WFMAON`](../../ngistPipeline/readData/MUSE_WFMAON.py)).

## Requirements

- Python 3.9+
- `fitsio`, `numpy`, `pandas` (`requirements-canfar.txt`)
- ~30 GB RAM: one wavelength plane at a time; spaxel chunks checked against 25 GB cap

## CANFAR execution

```bash
cd /path/to/gist-geckos
git fetch origin investigate/phangs-cube-nan-audit
git checkout investigate/phangs-cube-nan-audit

export PHANGS_DIR=/arc/projects/mauve/toby_sandbox/multiwavelength/phangs/phangs-muse/cubes
export MAUVE_DIR=/arc/projects/mauve/cubes/v3.0
export SCRATCH=/scratch/$USER/phangs_nan_audit

# Pilot: 3 smallest PHANGS + 3 smallest MAUVE cubes
bash investigations/phangs_cube_audit/canfar_job.sh pilot

# Full catalog
bash investigations/phangs_cube_audit/canfar_job.sh full
```

Local validation (no arc paths):

```bash
bash investigations/phangs_cube_audit/canfar_job.sh synthetic-test
```

## Scripts

| Script | Role |
|--------|------|
| `discover_cubes.py` | Catalog FITS under PHANGS/MAUVE roots |
| `discover_configs.py` | Optional: scan config trees for READ_DATA / LMIN_SNR |
| `audit_one_cube.py` | Per-file channel + spaxel audit (fitsio) |
| `run_audit_batch.py` | Batch driver; merges `audit_index.csv` per survey |
| `aggregate_report.py` | Summary MD, CSV, HTML, stacked λ profile |
| `make_synthetic_cubes.py` | Tiny test cubes for CI/local smoke test |
| `canfar_job.sh` | End-to-end driver |

## Outputs (`$SCRATCH/<job_id>/reports/`)

| File | Content |
|------|---------|
| `cube_catalog.csv` | All discovered cubes |
| `config_catalog.csv` | Optional nGIST config snippets |
| `audit_index.csv` | Per-cube status + key metrics |
| `PHANGS_vs_MAUVE_summary.md` | Cross-survey report + recommendation |
| `report.html` | HTML summary |
| `stacked_channel_profile.csv` | Mean frac NaN vs λ per survey |
| `hot_channel_ranges.csv` | Wavelength ranges of hot channels |
| `{survey}/{stem}_channel_nan.csv` | Per-channel NaN fractions + mask flags |
| `{survey}/{stem}_hot_channels.csv` | Channels with >50% NaN spaxels |
| `{survey}/{stem}_spaxel_nan_stats.csv` | Defunct simulation by mask domain |
| `{survey}/{stem}_meta.json` | HDU indices, shapes, metrics |
| `{survey}/{stem}_diagnosis.txt` | Short verdict |

## Wavelength masks (rest frame, Å)

| Domain | Range | Use |
|--------|-------|-----|
| `snr_default` | 4750–7100 (CLI) | Matches typical MasterConfig SNR band |
| `nad_gap` | 5860–5900 | NaD / PHANGS non-observed |
| `laser_lgs` | 5770–6050 | WFMAON LGS exclusion |
| `snr_no_nad` | SNR minus NaD | Proposed clean defunct/SNR statistics |

## Single cube

```bash
python3 audit_one_cube.py /arc/.../cube.fits --out-dir /scratch/$USER/audit_one --redshift 0.005
```
