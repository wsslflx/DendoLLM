#!/usr/bin/env python3
"""
Thin wrapper around the neo4j Python driver.
Non-fatal: if Neo4j is not reachable, all methods log and return without raising.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running from project root or kg/ subdirectory
sys.path.insert(0, str(Path(__file__).parents[1]))

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

_SCHEMA_QUERIES = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Species) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:OntologyTerm) REQUIRE t.term_id IS UNIQUE",
    "CREATE INDEX IF NOT EXISTS FOR (tr:Trait) ON (tr.raw_trait)",
]


def _get_driver():
    """Return a neo4j Driver or None if the server is not reachable."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except Exception as exc:
        print(
            f"[KG] Neo4j is not reachable at {NEO4J_URI}.\n"
            f"     Is the Neo4j Docker container running?\n"
            f"     Start it with: docker start neo4j\n"
            f"     Or launch fresh — see Neo4j Setup in the README.\n"
            f"     Skipping KG Neo4j push — local JSON outputs will still be written.\n"
            f"     (Error: {exc})"
        )
        return None


def _init_schema(driver) -> None:
    with driver.session() as session:
        for q in _SCHEMA_QUERIES:
            try:
                session.run(q)
            except Exception as exc:
                print(f"[KG] Schema init warning (non-fatal): {exc}")


def push_graph(nodes: list[dict], edges: list[dict], run_id: str) -> None:
    """
    Batch-insert nodes and edges using UNWIND.
    Uses MERGE on unique IDs to avoid duplicates across runs.
    Tags all nodes/edges with run_id.
    """
    driver = _get_driver()
    if driver is None:
        return
    try:
        _init_schema(driver)
        with driver.session() as session:
            # --- nodes by label ---
            for label in ("Species", "Trait", "OntologyTerm"):
                label_nodes = [n for n in nodes if n.get("_label") == label]
                if not label_nodes:
                    continue
                # strip internal _label key before writing
                props_list = [{k: v for k, v in n.items() if k != "_label"} for n in label_nodes]
                if label == "Species":
                    session.run(
                        "UNWIND $props AS p MERGE (n:Species {name: p.name}) SET n += p",
                        props=props_list,
                    )
                elif label == "OntologyTerm":
                    session.run(
                        "UNWIND $props AS p MERGE (n:OntologyTerm {term_id: p.term_id}) SET n += p",
                        props=props_list,
                    )
                else:  # Trait — no unique constraint, use raw_trait + run_id
                    session.run(
                        "UNWIND $props AS p MERGE (n:Trait {raw_trait: p.raw_trait, run_id: p.run_id}) SET n += p",
                        props=props_list,
                    )

            # --- edges ---
            for edge in edges:
                etype = edge.get("type", "RELATED")
                props = {k: v for k, v in edge.items() if k not in ("from_id", "to_id", "from_label", "to_label", "type")}
                props["run_id"] = run_id
                from_label = edge.get("from_label", "")
                to_label   = edge.get("to_label",   "")
                from_id    = edge.get("from_id")
                to_id      = edge.get("to_id")

                if etype == "HAS_TRAIT":
                    session.run(
                        "MATCH (a:Species {name: $fid}), (b:Trait {raw_trait: $tid, run_id: $run_id}) "
                        "MERGE (a)-[r:HAS_TRAIT {run_id: $run_id}]->(b) SET r += $props",
                        fid=from_id, tid=to_id, run_id=run_id, props=props,
                    )
                elif etype == "MAPPED_TO":
                    session.run(
                        "MATCH (a:Trait {raw_trait: $fid, run_id: $run_id}), (b:OntologyTerm {term_id: $tid}) "
                        "MERGE (a)-[r:MAPPED_TO {run_id: $run_id}]->(b) SET r += $props",
                        fid=from_id, tid=to_id, run_id=run_id, props=props,
                    )
                elif etype in ("IS_A", "PART_OF"):
                    session.run(
                        f"MATCH (a:OntologyTerm {{term_id: $fid}}), (b:OntologyTerm {{term_id: $tid}}) "
                        f"MERGE (a)-[r:{etype} {{run_id: $run_id}}]->(b) SET r += $props",
                        fid=from_id, tid=to_id, run_id=run_id, props=props,
                    )
                elif etype == "SIMILAR_TO":
                    session.run(
                        "MATCH (a:Trait {raw_trait: $fid, run_id: $run_id}), "
                        "      (b:Trait {raw_trait: $tid, run_id: $run_id}) "
                        "MERGE (a)-[r:SIMILAR_TO {run_id: $run_id}]->(b) SET r += $props",
                        fid=from_id, tid=to_id, run_id=run_id, props=props,
                    )
                elif etype == "SYNONYM_OF":
                    session.run(
                        "MATCH (a:Trait {raw_trait: $fid, run_id: $run_id}), "
                        "      (b:Trait {raw_trait: $tid, run_id: $run_id}) "
                        "MERGE (a)-[r:SYNONYM_OF {run_id: $run_id}]->(b) SET r += $props",
                        fid=from_id, tid=to_id, run_id=run_id, props=props,
                    )

        print(f"[KG] Pushed {len(nodes)} nodes and {len(edges)} edges to Neo4j (run_id={run_id})")
    except Exception as exc:
        print(f"[KG] Neo4j push failed (non-fatal): {exc}")
    finally:
        driver.close()


def run_query(cypher: str, params: dict = {}) -> list[dict]:
    """Execute a read query and return results as list of dicts."""
    driver = _get_driver()
    if driver is None:
        return []
    try:
        with driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]
    except Exception as exc:
        print(f"[KG] Query failed (non-fatal): {exc}")
        return []
    finally:
        driver.close()


def clear_run(run_id: str) -> None:
    """Delete all nodes/edges tagged with this run_id."""
    driver = _get_driver()
    if driver is None:
        return
    try:
        with driver.session() as session:
            session.run(
                "MATCH (n) WHERE n.run_id = $run_id DETACH DELETE n",
                run_id=run_id,
            )
        print(f"[KG] Cleared all nodes/edges for run_id={run_id}")
    except Exception as exc:
        print(f"[KG] clear_run failed (non-fatal): {exc}")
    finally:
        driver.close()


if __name__ == "__main__":
    print("[KG] Testing Neo4j connection...")
    driver = _get_driver()
    if driver:
        print("[KG] Connection successful.")
        _init_schema(driver)
        print("[KG] Schema initialized.")
        driver.close()
    else:
        print("[KG] Connection failed — graceful failure confirmed.")
