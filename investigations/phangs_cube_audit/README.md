# PHANGS vs MAUVE cube NaN audit (CANFAR)

Isolated investigation for NaN / NaD / laser-gap patterns in MUSE cubes.
Does not modify nGIST pipeline code.

## Hypothesis

PHANGS cubes may have NaNs concentrated in the NaD / LGS wavelength gap.
[`MUSE_WFM`](../../ngistPipeline/readData/MUSE_WFM.py) computes defunct `nan_frac` over the
full trimmed spectrum and does **not** exclude or infill the laser gap (unlike
[`MUSE_WFMAON`](../../ngistPipeline/readData/MUSE_WFMAON.py)).

## Requirements

- Python 3.9+
- `fitsio`, `numpy`, `pandas` (see `requirements-canfar.txt`)
- ~30 GB RAM: scripts read one wavelength plane at a time

## CANFAR execution

```bash
cd /path/to/gist-geckos
git checkout investigate/phangs-cube-nan-audit

bash investigations/phangs_cube_audit/canfar_job.sh pilot
# then:
bash investigations/phangs_cube_audit/canfar_job.sh full
```

Override paths if needed:

```bash
export PHANGS_DIR=/arc/projects/mauve/toby_sandbox/multiwavelength/phangs/phangs-muse/cubes
export MAUVE_DIR=/arc/projects/mauve/cubes/v3.0
export SCRATCH=/scratch/$USER/phangs_nan_audit
bash investigations/phangs_cube_audit/canfar_job.sh pilot
```

## Outputs (under `$SCRATCH/<job_id>/reports/`)

| File | Content |
|------|---------|
| `cube_catalog.csv` | All FITS cubes in both trees |
| `audit_index.csv` | Per-cube status and key metrics |
| `PHANGS_vs_MAUVE_summary.md` | Cross-survey comparison |
| `<survey>/<stem>_channel_nan.csv` | Per-wavelength NaN fractions + mask flags |
| `<survey>/<stem>_hot_channels.csv` | Channels with >50% NaN spaxels |
| `<survey>/<stem>_meta.json` | HDU indices, defunct simulation counts |
| `<survey>/<stem>_diagnosis.txt` | Short verdict |

## Single cube

```bash
python3 audit_one_cube.py /arc/.../some_cube.fits --out-dir /scratch/$USER/audit_one
```
