#!/usr/bin/env bash
set -euo pipefail

INPUT_TSV="${1:-candidate_sets_v1.tsv}"
PARALLELISM="${PARALLELISM:-10}"
RUNS="${RUNS:-1}"
BUNDLE_ROOT="${BUNDLE_ROOT:-logs_v1}"
export RUNS
export BUNDLE_ROOT
RUNSTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-run_logs_v1/${RUNSTAMP}}"
export LOG_ROOT
mkdir -p "${LOG_ROOT}"

if [[ ! -f "$INPUT_TSV" ]]; then
  echo "Input TSV not found: $INPUT_TSV" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python is required but not found in PATH" >&2
  exit 1
fi

echo "Starting candidate-set runs"
echo "Input: $INPUT_TSV"
echo "Parallel workers: $PARALLELISM"
echo "Runs per testcase: $RUNS"
echo "Bundle root: $BUNDLE_ROOT"
echo "Log root: $LOG_ROOT"

python - "$INPUT_TSV" <<'PY' | xargs -0 -n 3 -P "$PARALLELISM" bash -c '
set -euo pipefail
row_id="$1"
gene="$2"
species="$3"
slug_gene="$(printf "%s" "$gene" | tr -cs "[:alnum:]" "_" | tr "[:upper:]" "[:lower:]")"
log_file="${LOG_ROOT}/${row_id}_${slug_gene}.log"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] START row=${row_id} gene=${gene}"
if python pipeline/archive/run_full_pipeline.py --species-list "$species" --run-label "$gene" --runs "$RUNS" --bundle-root "$BUNDLE_ROOT" >"$log_file" 2>&1; then
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] DONE row=${row_id} gene=${gene}"
  : > "${LOG_ROOT}/success_${row_id}_${slug_gene}"
else
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] FAIL row=${row_id} gene=${gene} (see ${log_file})" >&2
  : > "${LOG_ROOT}/fail_${row_id}_${slug_gene}"
fi
' _
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    required = {"Gene", "Species"}
    if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
        raise SystemExit(f"TSV must contain headers: {sorted(required)}")

    for idx, row in enumerate(reader, start=1):
        gene = (row.get("Gene") or "").strip()
        species = (row.get("Species") or "").strip()
        if not gene or not species:
            continue
        sys.stdout.write(str(idx))
        sys.stdout.write("\0")
        sys.stdout.write(gene)
        sys.stdout.write("\0")
        sys.stdout.write(species)
        sys.stdout.write("\0")
PY

success_count="$(find "$LOG_ROOT" -maxdepth 1 -type f -name 'success_*' | wc -l | tr -d ' ')"
fail_count="$(find "$LOG_ROOT" -maxdepth 1 -type f -name 'fail_*' | wc -l | tr -d ' ')"
echo "All jobs completed. success=${success_count} fail=${fail_count}"
if [[ "$fail_count" != "0" ]]; then
  echo "Failed jobs logs are under: $LOG_ROOT" >&2
  exit 1
fi
