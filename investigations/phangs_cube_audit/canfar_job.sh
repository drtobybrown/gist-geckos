#!/bin/bash
# CANFAR / arc batch driver for PHANGS vs MAUVE cube NaN audit.
# Usage: bash canfar_job.sh [pilot|full]
set -euo pipefail

MODE="${1:-pilot}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
AUDIT_DIR="${REPO_ROOT}/investigations/phangs_cube_audit"
PHANGS_DIR="${PHANGS_DIR:-/arc/projects/mauve/toby_sandbox/multiwavelength/phangs/phangs-muse/cubes}"
MAUVE_DIR="${MAUVE_DIR:-/arc/projects/mauve/cubes/v3.0}"
SCRATCH="${SCRATCH:-/scratch/${USER}/phangs_nan_audit}"
JOB_ID="$(date +%Y%m%d_%H%M%S)_${MODE}"
WORK="${SCRATCH}/${JOB_ID}"
REPORT="${WORK}/reports"

mkdir -p "${WORK}" "${REPORT}"
cd "${AUDIT_DIR}"

python3 -m venv "${WORK}/venv"
source "${WORK}/venv/bin/activate"
pip install -q -r requirements-canfar.txt

export OMP_NUM_THREADS=1

echo "==> Discover cubes"
python3 discover_cubes.py \
  --phangs-dir "${PHANGS_DIR}" \
  --mauve-dir "${MAUVE_DIR}" \
  --out "${WORK}/cube_catalog.csv"

MAX_PHANGS=3
MAX_MAUVE=3
if [[ "${MODE}" == "full" ]]; then
  MAX_PHANGS=0
  MAX_MAUVE=0
fi

echo "==> Audit PHANGS (mode=${MODE})"
python3 run_audit_batch.py \
  --catalog "${WORK}/cube_catalog.csv" \
  --out-root "${REPORT}" \
  --survey PHANGS \
  --max-cubes "${MAX_PHANGS}" \
  --smallest-first \
  --lmin-tot 4800 --lmax-tot 7000

echo "==> Audit MAUVE (mode=${MODE})"
python3 run_audit_batch.py \
  --catalog "${WORK}/cube_catalog.csv" \
  --out-root "${REPORT}" \
  --survey MAUVE \
  --max-cubes "${MAX_MAUVE}" \
  --smallest-first \
  --lmin-tot 4800 --lmax-tot 7000

echo "==> Aggregate"
python3 aggregate_report.py \
  --audit-root "${REPORT}" \
  --out "${REPORT}/PHANGS_vs_MAUVE_summary.md"

echo "Done. Reports: ${REPORT}"
echo "  audit_index.csv"
echo "  PHANGS_vs_MAUVE_summary.md"
echo "  <survey>/<cube_stem>/*_channel_nan.csv, *_meta.json, *_diagnosis.txt"
