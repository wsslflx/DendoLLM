#!/usr/bin/env bash
# Run testcases 1, 3, 4, 5, 6 sequentially through the graph pipeline.
# Purpose: collect new entity-type stats, dedup stats, and chunk-cap logging
# introduced in the latest changes.
#
# Output: logs_graph/<timestamp>-testcases-logging/<testcase_N>/
# Shared Chroma and PDF cache are reused across all runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE="$REPO_ROOT/pipeline/run_graph_pipeline.py"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
BATCH_ROOT="$REPO_ROOT/logs_graph/${TIMESTAMP}-testcases-logging"
SHARED_CHROMA="$REPO_ROOT/shared_chroma"
SHARED_PDF="$REPO_ROOT/shared_pdfs"

TESTCASES=("testcase1" "testcase3" "testcase4" "testcase5" "testcase6")

mkdir -p "$BATCH_ROOT" "$SHARED_CHROMA" "$SHARED_PDF"
echo "[run_testcases_logging] Batch root: $BATCH_ROOT"
echo "[run_testcases_logging] Running ${#TESTCASES[@]} testcases: ${TESTCASES[*]}"

FAILED=()
BATCH_START=$SECONDS

for TC in "${TESTCASES[@]}"; do
    SPECIES_FILE="$REPO_ROOT/testcases/${TC}.json"
    if [ ! -f "$SPECIES_FILE" ]; then
        echo "[run_testcases_logging] WARNING: $SPECIES_FILE not found — skipping."
        continue
    fi

    LOG_FILE="$BATCH_ROOT/${TC}.log"
    echo ""
    echo "[run_testcases_logging] === Starting $TC ==="
    TC_START=$SECONDS

    if python "$PIPELINE" --species-file "$SPECIES_FILE" --run-label "$TC" --bundle-root "$BATCH_ROOT" --shared-chroma-dir "$SHARED_CHROMA" --pdf-dir "$SHARED_PDF" --map-workers 4 --index-workers 4 --mapping-model "granite4.1:8b" >"$LOG_FILE" 2>&1; then
        ELAPSED=$((SECONDS - TC_START))
        echo "[run_testcases_logging] $TC done in $((ELAPSED / 60))m $((ELAPSED % 60))s. Log: $LOG_FILE"
    else
        ELAPSED=$((SECONDS - TC_START))
        echo "[run_testcases_logging] WARNING: $TC FAILED after $((ELAPSED / 60))m $((ELAPSED % 60))s. Log: $LOG_FILE"
        FAILED+=("$TC")
    fi
done

TOTAL=$((SECONDS - BATCH_START))
echo ""
echo "[run_testcases_logging] =============================="
echo "[run_testcases_logging] All done in $((TOTAL / 60))m $((TOTAL % 60))s."
echo "[run_testcases_logging] Output: $BATCH_ROOT"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[run_testcases_logging] Failed: ${FAILED[*]}"
else
    echo "[run_testcases_logging] All testcases completed successfully."
fi
