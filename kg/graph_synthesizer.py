#!/usr/bin/env python3
"""
Graph Synthesizer — Step 4 of the GraphRAG pipeline.

Performs three-tier cross-species synthesis using the enriched document graph:
  Tier 1 — uPheno ancestor convergence (Cypher query, high precision)
  Tier 2 — DOC_SIMILAR_TO embedding similarity clusters (Cypher + Python, medium precision)
  Tier 3 — LLM semantic bridging + interpretation + hypothesis (structured input, no raw chunks)

Output: graph_synthesis.json with a communities[] list and gene_function_hypothesis.

Usage (standalone, after enrichment):
    python kg/graph_synthesizer.py \
        --species "talpa europaea,nannospalax ehrenbergi,chrysochloris asiatica" \
        --display "Talpa europaea,Nannospalax ehrenbergi,Chrysochloris asiatica"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

SYNTHESIS_PROMPT_FILE = "Prompts/prompt_graph_synthesis.txt"
MAX_TIER1_RESULTS = 30
MAX_TIER2_CLUSTERS = 30
MAX_SYNTHESIS_RETRIES = 2


# ---------------------------------------------------------------------------
# Tier 1 — uPheno ancestor convergence
# ---------------------------------------------------------------------------

def _query_tier1(species_norms: list[str], min_species: int) -> list[dict]:
    """
    Find OntologyTerm ancestors shared by entities from >= min_species species.
    Returns list of {term_id, term_name, species_list, entity_forms, term_names}.
    """
    cypher = (
        "MATCH (e:DocEntity)-[:DOC_MAPPED_TO]->(t:OntologyTerm)-[:IS_A*0..3]->(ancestor:OntologyTerm) "
        "MATCH (c:DocChunk)-[:MENTIONS]->(e) "
        "WHERE c.species_norm IN $sn "
        "WITH ancestor, "
        "     collect(DISTINCT c.species_norm) AS species_list, "
        "     collect(DISTINCT e.surface_form) AS entity_forms, "
        "     collect(DISTINCT t.term_name)    AS term_names "
        "WHERE size(species_list) >= $min_sp "
        "RETURN ancestor.term_id  AS term_id, "
        "       ancestor.term_name AS term_name, "
        "       species_list, entity_forms, term_names "
        "ORDER BY size(species_list) DESC, size(entity_forms) DESC "
        f"LIMIT {MAX_TIER1_RESULTS}"
    )
    try:
        from kg.neo4j_client import run_query
        rows = run_query(cypher, {"sn": species_norms, "min_sp": min_species})
    except Exception as exc:
        print(f"[GraphSynthesizer] Tier 1 query failed: {exc}")
        return []

    # Filter out overly generic top-level terms
    # Heuristic: if a term appears in ALL species it's likely a generic ancestor
    n_species = len(species_norms)
    filtered = []
    for r in rows:
        sp_count = len(r.get("species_list", []))
        # Skip if it covers ALL species and has a very short/generic name (likely root term)
        if sp_count == n_species and len(r.get("term_name", "")) < 10:
            continue
        filtered.append(r)

    print(f"[GraphSynthesizer] Tier 1: {len(filtered)} communities found "
          f"(after filtering {len(rows) - len(filtered)} generic terms).")
    return filtered


def _format_tier1_text(tier1: list[dict], species_display_map: dict[str, str]) -> str:
    if not tier1:
        return "(No Tier 1 communities found — uPheno mapping may be incomplete.)"
    lines = []
    for i, r in enumerate(tier1, 1):
        sp_display = [species_display_map.get(s, s) for s in r.get("species_list", [])]
        lines.append(
            f"  [{i}] {r.get('term_name', '?')} ({r.get('term_id', '?')})\n"
            f"      Species ({len(sp_display)}): {', '.join(sp_display)}\n"
            f"      Entity forms: {', '.join(r.get('entity_forms', [])[:8])}\n"
            f"      Via terms: {', '.join(r.get('term_names', [])[:5])}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tier 2 — DOC_SIMILAR_TO clustering
# ---------------------------------------------------------------------------

def _query_tier2_pairs(species_norms: list[str]) -> list[dict]:
    """
    Pull all DOC_SIMILAR_TO edges between entities from different species.
    Returns list of {sf1, sf2, score, sp1, sp2, eid1, eid2}.
    """
    cypher = (
        "MATCH (e1:DocEntity)-[r:DOC_SIMILAR_TO]->(e2:DocEntity) "
        "MATCH (c1:DocChunk)-[:MENTIONS]->(e1) "
        "MATCH (c2:DocChunk)-[:MENTIONS]->(e2) "
        "WHERE c1.species_norm IN $sn AND c2.species_norm IN $sn "
        "  AND c1.species_norm <> c2.species_norm "
        "RETURN DISTINCT "
        "  e1.entity_id AS eid1, e1.surface_form AS sf1, "
        "  e2.entity_id AS eid2, e2.surface_form AS sf2, "
        "  r.score AS score, "
        "  c1.species_norm AS sp1, c2.species_norm AS sp2 "
        "ORDER BY score DESC "
        "LIMIT 500"
    )
    try:
        from kg.neo4j_client import run_query
        return run_query(cypher, {"sn": species_norms})
    except Exception as exc:
        print(f"[GraphSynthesizer] Tier 2 query failed: {exc}")
        return []


def _cluster_tier2(pairs: list[dict], min_species: int) -> list[dict]:
    """
    Greedy clustering of similar entity pairs.
    Returns list of clusters: {members: [(sf, sp)...], species_set, pairs}.
    """
    if not pairs:
        return []

    # Group pairs by (eid1, eid2) and collect all species combinations
    pair_info: dict[tuple, dict] = {}
    for p in pairs:
        key = tuple(sorted([p["eid1"], p["eid2"]]))
        sf_key = tuple(sorted([p["sf1"], p["sf2"]]))
        if key not in pair_info:
            pair_info[key] = {
                "eid_a": p["eid1"], "sf_a": p["sf1"],
                "eid_b": p["eid2"], "sf_b": p["sf2"],
                "score": p["score"],
                "species_set": set(),
                "sf_pair": sf_key,
            }
        pair_info[key]["species_set"].add(p["sp1"])
        pair_info[key]["species_set"].add(p["sp2"])

    # Filter to pairs spanning >= min_species
    valid_pairs = [v for v in pair_info.values() if len(v["species_set"]) >= min_species]
    valid_pairs.sort(key=lambda x: (-len(x["species_set"]), -x["score"]))

    # Greedy clustering: merge pairs that share an entity
    clusters: list[dict] = []
    eid_to_cluster: dict[str, int] = {}

    for pair in valid_pairs:
        eid_a, eid_b = pair["eid_a"], pair["eid_b"]
        ci_a = eid_to_cluster.get(eid_a)
        ci_b = eid_to_cluster.get(eid_b)

        if ci_a is None and ci_b is None:
            # New cluster
            ci = len(clusters)
            clusters.append({
                "eids": {eid_a, eid_b},
                "surface_forms": {pair["sf_a"], pair["sf_b"]},
                "species_set": set(pair["species_set"]),
                "pairs": [(pair["sf_a"], pair["sf_b"], pair["score"])],
            })
            eid_to_cluster[eid_a] = ci
            eid_to_cluster[eid_b] = ci
        elif ci_a is not None and ci_b is None:
            clusters[ci_a]["eids"].add(eid_b)
            clusters[ci_a]["surface_forms"].add(pair["sf_b"])
            clusters[ci_a]["species_set"].update(pair["species_set"])
            clusters[ci_a]["pairs"].append((pair["sf_a"], pair["sf_b"], pair["score"]))
            eid_to_cluster[eid_b] = ci_a
        elif ci_b is not None and ci_a is None:
            clusters[ci_b]["eids"].add(eid_a)
            clusters[ci_b]["surface_forms"].add(pair["sf_a"])
            clusters[ci_b]["species_set"].update(pair["species_set"])
            clusters[ci_b]["pairs"].append((pair["sf_a"], pair["sf_b"], pair["score"]))
            eid_to_cluster[eid_a] = ci_b
        elif ci_a != ci_b:
            # Merge two clusters
            keep, drop = (ci_a, ci_b) if ci_a < ci_b else (ci_b, ci_a)
            clusters[keep]["eids"].update(clusters[drop]["eids"])
            clusters[keep]["surface_forms"].update(clusters[drop]["surface_forms"])
            clusters[keep]["species_set"].update(clusters[drop]["species_set"])
            clusters[keep]["pairs"].extend(clusters[drop]["pairs"])
            for eid in clusters[drop]["eids"]:
                eid_to_cluster[eid] = keep
            clusters[drop] = None  # mark as merged

    # Remove merged (None) clusters, filter by min_species, cap
    result = [
        c for c in clusters
        if c is not None and len(c["species_set"]) >= min_species
    ]
    result.sort(key=lambda x: (-len(x["species_set"]), -len(x["pairs"])))
    result = result[:MAX_TIER2_CLUSTERS]

    print(f"[GraphSynthesizer] Tier 2: {len(result)} clusters found from {len(valid_pairs)} valid pairs.")
    return result


def _format_tier2_text(clusters: list[dict], species_display_map: dict[str, str]) -> str:
    if not clusters:
        return "(No Tier 2 clusters found — DOC_SIMILAR_TO edges may be missing.)"
    lines = []
    for i, c in enumerate(clusters, 1):
        sp_display = [species_display_map.get(s, s) for s in c["species_set"]]
        top_pairs = c["pairs"][:5]
        pairs_str = "; ".join(f'"{a}" ~ "{b}" ({sc:.2f})' for a, b, sc in top_pairs)
        lines.append(
            f"  [{i}] Entities: {', '.join(sorted(c['surface_forms'])[:8])}\n"
            f"      Species ({len(sp_display)}): {', '.join(sp_display)}\n"
            f"      Top pairs: {pairs_str}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entity lists per species (for Tier 3 gap-finding)
# ---------------------------------------------------------------------------

def _query_entity_lists(species_norms: list[str]) -> dict[str, list[str]]:
    """Pull all entity surface forms per species norm."""
    cypher = (
        "MATCH (c:DocChunk)-[:MENTIONS]->(e:DocEntity) "
        "WHERE c.species_norm IN $sn "
        "RETURN c.species_norm AS sp, "
        "       collect(DISTINCT e.surface_form) AS forms "
    )
    try:
        from kg.neo4j_client import run_query
        rows = run_query(cypher, {"sn": species_norms})
        return {r["sp"]: r["forms"] for r in rows}
    except Exception as exc:
        print(f"[GraphSynthesizer] Entity list query failed: {exc}")
        return {}


def _format_entity_lists(entity_lists: dict[str, list[str]], species_display_map: dict[str, str]) -> str:
    if not entity_lists:
        return "(No entity lists available.)"
    lines = []
    for sp_norm, forms in entity_lists.items():
        display = species_display_map.get(sp_norm, sp_norm)
        lines.append(f"  {display}: {', '.join(sorted(forms)[:40])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tier 3 — LLM synthesis
# ---------------------------------------------------------------------------

def _extract_json_text(payload: object) -> str:
    text = str(payload).strip()
    if text.startswith("```"):
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()
    return text


def _validate_synthesis(data: object) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Synthesis output must be a JSON object")

    communities = data.get("communities", [])
    if not isinstance(communities, list):
        raise ValueError("communities must be a list")

    clean_communities = []
    for c in communities:
        if not isinstance(c, dict):
            continue
        label = c.get("label", "")
        if not isinstance(label, str) or not label.strip():
            continue
        tier = c.get("tier", "llm")
        sp = c.get("species_coverage", [])
        if not isinstance(sp, list):
            sp = []
        ents = c.get("supporting_entities", [])
        if not isinstance(ents, list):
            ents = []
        clean_communities.append({
            "label": label.strip(),
            "tier": tier if tier in ("upheno", "embedding", "merged", "llm") else "llm",
            "species_coverage": [str(s) for s in sp],
            "species_count": len(sp),
            "supporting_entities": [str(e) for e in ents],
            "tier1_evidence": c.get("tier1_evidence"),
            "tier2_evidence": c.get("tier2_evidence"),
            "tier3_interpretation": str(c.get("tier3_interpretation", "")),
        })

    hyp = data.get("gene_function_hypothesis", "")
    return {
        "communities": clean_communities,
        "gene_function_hypothesis": str(hyp),
    }


def _run_llm_synthesis(
    tier1: list[dict],
    tier2_clusters: list[dict],
    entity_lists: dict[str, list[str]],
    species_norms: list[str],
    species_display_names: list[str],
    species_display_map: dict[str, str],
    min_species: int,
    model: str | None,
) -> dict:
    """Call LLM with structured Tier 1 + Tier 2 evidence."""
    try:
        prompt_text = pathlib.Path(SYNTHESIS_PROMPT_FILE).read_text(encoding="utf-8")
    except Exception as exc:
        print(f"[GraphSynthesizer] Cannot load prompt {SYNTHESIS_PROMPT_FILE}: {exc}")
        return {"communities": [], "gene_function_hypothesis": ""}

    tier1_text = _format_tier1_text(tier1, species_display_map)
    tier2_text = _format_tier2_text(tier2_clusters, species_display_map)
    entity_lists_text = _format_entity_lists(entity_lists, species_display_map)

    prompt = (
        prompt_text
        .replace("{n_species}", str(len(species_display_names)))
        .replace("{species_names}", ", ".join(species_display_names))
        .replace("{tier1_text}", tier1_text)
        .replace("{tier2_text}", tier2_text)
        .replace("{entity_lists_text}", entity_lists_text)
        .replace("{min_species}", str(min_species))
    )

    from core.llm_backend import make_chat_llm
    from langchain_core.messages import HumanMessage
    llm = make_chat_llm(model=model, temperature=0.0)

    for attempt in range(1, MAX_SYNTHESIS_RETRIES + 2):
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            raw = resp.content if hasattr(resp, "content") else str(resp)
            parsed = json.loads(_extract_json_text(raw))
            result = _validate_synthesis(parsed)
            print(f"[GraphSynthesizer] LLM synthesis succeeded on attempt {attempt}.")
            return result
        except Exception as exc:
            print(f"[GraphSynthesizer] LLM attempt {attempt} failed: {exc}")
            if attempt == MAX_SYNTHESIS_RETRIES + 1:
                print("[GraphSynthesizer] All attempts failed — returning empty synthesis.")
                return {"communities": [], "gene_function_hypothesis": ""}

    return {"communities": [], "gene_function_hypothesis": ""}


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def run_synthesis(
    species_norms: list[str],
    species_display_names: list[str],
    min_species: int = 2,
    model: str | None = None,
    log_dir: pathlib.Path | None = None,
) -> dict:
    """
    Three-tier cross-species synthesis.

    species_norms: lowercase species names matching DocChunk.species_norm
    species_display_names: display names (same order as species_norms)
    min_species: minimum species a community must span to be included

    Returns communities dict.
    """
    print(f"\n[GraphSynthesizer] ================================================")
    print(f"[GraphSynthesizer] Starting synthesis for {len(species_norms)} species.")
    print(f"[GraphSynthesizer] Min species threshold: {min_species}")
    print(f"[GraphSynthesizer] ================================================")

    species_display_map = dict(zip(species_norms, species_display_names))

    # Tier 1
    print("[GraphSynthesizer] Running Tier 1 (uPheno convergence)...")
    tier1 = _query_tier1(species_norms, min_species)

    # Tier 2
    print("[GraphSynthesizer] Running Tier 2 (embedding similarity clusters)...")
    pairs = _query_tier2_pairs(species_norms)
    tier2_clusters = _cluster_tier2(pairs, min_species)

    # Entity lists for Tier 3
    print("[GraphSynthesizer] Fetching entity lists per species...")
    entity_lists = _query_entity_lists(species_norms)

    if not tier1 and not tier2_clusters:
        print("[GraphSynthesizer] No Tier 1 or Tier 2 evidence — enrichment may not have run.")

    # Tier 3: LLM
    print("[GraphSynthesizer] Running Tier 3 (LLM synthesis)...")
    synthesis = _run_llm_synthesis(
        tier1=tier1,
        tier2_clusters=tier2_clusters,
        entity_lists=entity_lists,
        species_norms=species_norms,
        species_display_names=species_display_names,
        species_display_map=species_display_map,
        min_species=min_species,
        model=model,
    )

    # Attach metadata
    synthesis["species_norms"] = species_norms
    synthesis["species_display_names"] = species_display_names
    synthesis["min_species_threshold"] = min_species
    synthesis["tier1_raw"] = tier1
    synthesis["tier2_raw"] = [
        {
            "surface_forms": sorted(c["surface_forms"]),
            "species_count": len(c["species_set"]),
            "species_set": sorted(c["species_set"]),
            "top_pairs": c["pairs"][:5],
        }
        for c in tier2_clusters
    ]

    n_comm = len(synthesis.get("communities", []))
    print(f"\n[GraphSynthesizer] ================================================")
    print(f"[GraphSynthesizer] Synthesis complete.")
    print(f"[GraphSynthesizer]   Communities found:  {n_comm}")
    print(f"[GraphSynthesizer]   Tier 1 candidates:  {len(tier1)}")
    print(f"[GraphSynthesizer]   Tier 2 clusters:    {len(tier2_clusters)}")
    if synthesis.get("gene_function_hypothesis"):
        print(f"[GraphSynthesizer]   Hypothesis: {synthesis['gene_function_hypothesis'][:120]}...")
    print(f"[GraphSynthesizer] ================================================\n")

    if log_dir:
        prompt_log = {
            "tier1": tier1,
            "tier2_clusters": synthesis.get("tier2_raw", []),
            "entity_lists": {k: v[:20] for k, v in entity_lists.items()},
        }
        try:
            (log_dir / "graph_synthesis_evidence.json").write_text(
                json.dumps(prompt_log, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            print(f"[GraphSynthesizer] Could not write evidence log: {exc}")

    return synthesis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run three-tier cross-species synthesis.")
    parser.add_argument("--species", required=True,
                        help="Comma-separated species norms.")
    parser.add_argument("--display", required=True,
                        help="Comma-separated display names (same order as --species).")
    parser.add_argument("--min-species", type=int, default=2)
    parser.add_argument("--model", default=None)
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    norms = [s.strip().lower() for s in args.species.split(",") if s.strip()]
    display = [s.strip() for s in args.display.split(",") if s.strip()]
    log_dir = pathlib.Path(args.log_dir) if args.log_dir else None

    result = run_synthesis(
        species_norms=norms,
        species_display_names=display,
        min_species=args.min_species,
        model=args.model,
        log_dir=log_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
