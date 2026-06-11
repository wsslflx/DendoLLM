#!/usr/bin/env python3
"""
One-time offline script: compute global_fan_in for every OntologyTerm node in Neo4j.

global_fan_in = number of distinct OntologyTerm nodes that have an IS_A*1..
path leading to this ancestor (i.e. how many terms roll up into it).

High fan_in = generic (e.g. "abnormal anatomical structure" — thousands of terms
              in the ontology are descendants of this)
Low fan_in  = specific (e.g. "cerebellar hypoplasia" — only a handful of leaf
              terms reach it via IS_A ancestry)

Used by graph_synthesizer._query_tier1() to rank Tier 1 candidates by IC-style
specificity instead of raw species count.

Run once after the ontology has been imported into Neo4j:
    python -m kg.precompute_fan_in

Re-run whenever the uPheno OWL file is updated and re-imported.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from kg.neo4j_client import run_query, run_write_query

# Count how many distinct descendant OntologyTerm nodes reach each ancestor
# via one or more IS_A hops. This is the "fan-in" of the ancestor.
_COUNT_CYPHER = """
MATCH (descendant:OntologyTerm)-[:IS_A*1..]->(ancestor:OntologyTerm)
WITH ancestor, count(DISTINCT descendant) AS fan_in
RETURN ancestor.term_id AS term_id, fan_in
ORDER BY fan_in DESC
"""

# Write fan_in values back onto each OntologyTerm node in chunks via UNWIND
_WRITE_CYPHER = """
UNWIND $rows AS row
MATCH (t:OntologyTerm {term_id: row.term_id})
SET t.global_fan_in = row.fan_in
"""

_CHUNK_SIZE = 1000


def run() -> None:
    print("[FanIn] Counting IS_A descendants for all OntologyTerm nodes...")
    rows = run_query(_COUNT_CYPHER, {})
    if not rows:
        print("[FanIn] No OntologyTerm nodes found — is Neo4j running and the ontology imported?")
        return

    print(f"[FanIn] {len(rows)} ancestor terms to annotate.")
    if rows:
        print(f"[FanIn] Most generic term: {rows[0]['term_id']} fan_in={rows[0]['fan_in']}")
        print(f"[FanIn] Most specific term: {rows[-1]['term_id']} fan_in={rows[-1]['fan_in']}")

    written = 0
    for i in range(0, len(rows), _CHUNK_SIZE):
        chunk = rows[i : i + _CHUNK_SIZE]
        run_write_query(_WRITE_CYPHER, {"rows": chunk})
        written += len(chunk)
        print(f"[FanIn] Written {written}/{len(rows)}")

    print("[FanIn] Done. global_fan_in is now set on all OntologyTerm nodes.")
    print("[FanIn] Verify with: MATCH (t:OntologyTerm) WHERE t.global_fan_in IS NOT NULL RETURN count(t)")


if __name__ == "__main__":
    run()
