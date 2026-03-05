#!/usr/bin/env bash
set -euo pipefail

INPUT_TSV="${1:-candidate_sets_v1.tsv}"
PARALLELISM="${PARALLELISM:-10}"
RUNS="${RUNS:-10}"
BUNDLE_ROOT="${BUNDLE_ROOT:-logs_v1}"

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

python - "$INPUT_TSV" <<'PY' | xargs -0 -n 2 -P "$PARALLELISM" bash -c '
set -euo pipefail
gene="$1"
species="$2"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] START gene=${gene}"
python run_full_pipeline.py \
  --species-list "$species" \
  --run-label "$gene" \
  --runs "$RUNS" \
  --bundle-root "$BUNDLE_ROOT"
echo "[$(date +"%Y-%m-%d %H:%M:%S")] DONE gene=${gene}"
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

    for row in reader:
        gene = (row.get("Gene") or "").strip()
        species = (row.get("Species") or "").strip()
        if not gene or not species:
            continue
        sys.stdout.write(gene)
        sys.stdout.write("\0")
        sys.stdout.write(species)
        sys.stdout.write("\0")
PY

echo "All jobs completed."
