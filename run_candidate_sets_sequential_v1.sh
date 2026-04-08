#!/usr/bin/env bash
set -euo pipefail

INPUT_TSV="${1:-candidate_sets_v1.tsv}"
RUNS="${RUNS:-1}"
BUNDLE_ROOT_BASE="${BUNDLE_ROOT:-logs_v1}"
RUNSTAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_ROOT="${BUNDLE_ROOT_BASE}/${RUNSTAMP}"
export RUNS
export BUNDLE_ROOT
LOG_ROOT="${LOG_ROOT:-run_logs_v1/${RUNSTAMP}}"
# Shared Chroma store at the base level so embeddings persist across re-runs.
SHARED_CHROMA="${BUNDLE_ROOT_BASE}/shared_chroma_store_ollama"
mkdir -p "${LOG_ROOT}" "${BUNDLE_ROOT}" "${SHARED_CHROMA}"

if [[ ! -f "$INPUT_TSV" ]]; then
  echo "Input TSV not found: $INPUT_TSV" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python is required but not found in PATH" >&2
  exit 1
fi

echo "Starting candidate-set runs (sequential)"
echo "Input:             $INPUT_TSV"
echo "Runs per testcase: $RUNS"
echo "Bundle root:       $BUNDLE_ROOT"
echo "Shared chroma:     $SHARED_CHROMA"
echo "Log root:          $LOG_ROOT"

success_count=0
fail_count=0

python - "$INPUT_TSV" <<'PY' | while IFS=$'\0' read -r -d $'\0' row_id && IFS=$'\0' read -r -d $'\0' gene && IFS=$'\0' read -r -d $'\0' species; do
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
  slug_gene="$(printf "%s" "$gene" | tr -cs "[:alnum:]" "_" | tr "[:upper:]" "[:lower:]")"
  log_file="${LOG_ROOT}/${row_id}_${slug_gene}.log"
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] START row=${row_id} gene=${gene}"
  if python pipeline/archive/run_full_pipeline.py \
    --species-list "$species" \
    --run-label "$gene" \
    --runs "$RUNS" \
    --bundle-root "$BUNDLE_ROOT" \
    --chroma-dir "$SHARED_CHROMA" \
    >"$log_file" 2>&1; then
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] DONE  row=${row_id} gene=${gene}"
    : > "${LOG_ROOT}/success_${row_id}_${slug_gene}"
    ((success_count++)) || true
  else
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] FAIL  row=${row_id} gene=${gene} (see ${log_file})" >&2
    : > "${LOG_ROOT}/fail_${row_id}_${slug_gene}"
    ((fail_count++)) || true
  fi
done

echo "All jobs completed. success=${success_count} fail=${fail_count}"
if [[ "$fail_count" != "0" ]]; then
  echo "Failed job logs are under: $LOG_ROOT" >&2
  exit 1
fi
