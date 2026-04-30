#!/usr/bin/env python3
"""
Cypher queries against the KG + LLM hypothesis generation.

Query 1 — shared ontology terms across >= min_species species
Query 2 — inferred ENVO environmental stressors via ancestor traversal
Query 3 — full evidence paths for top 5 shared traits
Query 4 — LLM gene-function hypothesis generation
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1]))


# ---------------------------------------------------------------------------
# Cypher queries
# ---------------------------------------------------------------------------

def query_shared_terms(run_id: str, min_species: int = 2) -> list[dict]:
    """Query 1: uPheno terms shared across >= min_species species."""
    from kg import neo4j_client
    cypher = """
    MATCH (s:Species)-[:HAS_TRAIT]->(t:Trait)-[:MAPPED_TO]->(ot:OntologyTerm)
    WHERE s.run_id = $run_id
    WITH ot, collect(DISTINCT s.name) AS species_list
    WHERE size(species_list) >= $min_species
    RETURN ot.term_id AS term_id, ot.term_name AS term_name,
           ot.ontology_source AS ontology_source,
           species_list, size(species_list) AS species_count
    ORDER BY species_count DESC
    """
    return neo4j_client.run_query(cypher, {"run_id": run_id, "min_species": min_species})


def query_synonym_clusters(run_id: str, min_species: int = 2) -> list[dict]:
    """
    Query 1c: LLM-derived synonym groups (SYNONYM_OF edges) spanning >= min_species species.
    Returns one entry per synonym pair with the species covered.
    """
    from kg import neo4j_client
    cypher = """
    MATCH (s1:Species)-[:HAS_TRAIT]->(t1:Trait)-[:SYNONYM_OF]->(t2:Trait)<-[:HAS_TRAIT]-(s2:Species)
    WHERE s1.run_id = $run_id AND s2.run_id = $run_id AND s1.name <> s2.name
      AND t1.run_id = $run_id AND t2.run_id = $run_id
      AND id(t1) < id(t2)
    WITH t1.raw_trait AS trait_a, t2.raw_trait AS trait_b,
         collect(DISTINCT s1.name) + collect(DISTINCT s2.name) AS all_species
    WITH trait_a, trait_b,
         [x IN all_species WHERE x IS NOT NULL] AS all_species
    WHERE size(all_species) >= $min_species
    RETURN trait_a, trait_b, all_species, size(all_species) AS species_count
    ORDER BY species_count DESC
    """
    return neo4j_client.run_query(cypher, {"run_id": run_id, "min_species": min_species})


def query_similar_trait_clusters(run_id: str, min_species: int = 2) -> list[dict]:
    """
    Query 1b: clusters of semantically similar traits (via SIMILAR_TO edges)
    that span >= min_species different species.
    Returns one entry per similar pair group with the species covered.
    """
    from kg import neo4j_client
    cypher = """
    MATCH (s1:Species)-[:HAS_TRAIT]->(t1:Trait)-[r:SIMILAR_TO]->(t2:Trait)<-[:HAS_TRAIT]-(s2:Species)
    WHERE s1.run_id = $run_id AND s2.run_id = $run_id AND s1.name <> s2.name
      AND t1.run_id = $run_id AND t2.run_id = $run_id
      AND id(t1) < id(t2)
    WITH t1.raw_trait AS trait_a, t1.normalized_trait AS norm_a,
         t2.raw_trait AS trait_b, t2.normalized_trait AS norm_b,
         r.similarity_score AS score,
         collect(DISTINCT s1.name) + collect(DISTINCT s2.name) AS all_species
    WITH trait_a, norm_a, trait_b, norm_b, score,
         [x IN all_species WHERE x IS NOT NULL] AS all_species
    WHERE size(all_species) >= $min_species
    RETURN trait_a, norm_a, trait_b, norm_b, score,
           all_species, size(all_species) AS species_count
    ORDER BY score DESC
    """
    return neo4j_client.run_query(cypher, {"run_id": run_id, "min_species": min_species})


def query_inferred_stressors(run_id: str, shared_term_ids: list[str]) -> list[dict]:
    """Query 2: ENVO ancestor nodes reachable from shared terms."""
    from kg import neo4j_client
    if not shared_term_ids:
        return []
    cypher = """
    MATCH (ot:OntologyTerm)-[:IS_A*1..3]->(ancestor:OntologyTerm)
    WHERE ot.term_id IN $shared_term_ids
      AND ancestor.term_id STARTS WITH 'ENVO:'
    RETURN DISTINCT ancestor.term_id AS term_id, ancestor.term_name AS term_name,
           collect(DISTINCT ot.term_id) AS via_terms
    """
    return neo4j_client.run_query(cypher, {"shared_term_ids": shared_term_ids})


def query_shared_ancestor_terms(run_id: str, min_species: int = 2) -> list[dict]:
    """
    Query 2b: Find ontology ancestor terms reachable (via IS_A/PART_OF) from
    traits of >= min_species DIFFERENT species, even via different leaf terms.
    Captures convergent adaptations (e.g. diving + altitude both reaching
    'response to hypoxia phenotype').
    Filters to UPHENO and ENVO ancestors only — HP/MP/GO are too specific or too broad.
    """
    from kg import neo4j_client
    cypher = """
    MATCH (s:Species)-[:HAS_TRAIT]->(t:Trait)-[:MAPPED_TO]->(ot:OntologyTerm)
    WHERE s.run_id = $run_id
    MATCH (ot)-[:IS_A*1..3]->(ancestor:OntologyTerm)
    WHERE (ancestor.term_id STARTS WITH 'UPHENO:' OR ancestor.term_id STARTS WITH 'ENVO:')
    WITH ancestor,
         collect(DISTINCT s.name) AS species_list,
         collect(DISTINCT ot.term_id) AS leaf_term_ids,
         collect(DISTINCT ot.term_name) AS leaf_term_names
    WHERE size(species_list) >= $min_species
    RETURN ancestor.term_id AS ancestor_id,
           ancestor.term_name AS ancestor_name,
           ancestor.ontology_source AS ontology_source,
           species_list,
           leaf_term_ids,
           leaf_term_names,
           size(species_list) AS species_count
    ORDER BY species_count DESC, ancestor_id
    """
    return neo4j_client.run_query(cypher, {"run_id": run_id, "min_species": min_species})


def query_evidence_paths(run_id: str, top_n: int = 5) -> list[dict]:
    """
    Query 3: Evidence paths for the top N most-shared traits.
    Returns simplified path summaries (not full Neo4j path objects).
    """
    from kg import neo4j_client
    # First find the top shared traits
    top_traits_cypher = """
    MATCH (s:Species)-[:HAS_TRAIT]->(t:Trait)-[:MAPPED_TO]->(ot:OntologyTerm)
    WHERE s.run_id = $run_id
    WITH t.raw_trait AS trait, collect(DISTINCT s.name) AS species_list,
         collect(DISTINCT ot.term_id) AS term_ids
    WHERE size(species_list) >= 2
    RETURN trait, species_list, term_ids, size(species_list) AS support
    ORDER BY support DESC
    LIMIT $top_n
    """
    top_traits = neo4j_client.run_query(top_traits_cypher, {"run_id": run_id, "top_n": top_n})

    paths = []
    for row in top_traits:
        trait = row.get("trait", "")
        # Simplified path: species → trait → term → ancestors
        path_cypher = """
        MATCH (s:Species)-[:HAS_TRAIT]->(t:Trait {raw_trait: $trait})-[:MAPPED_TO]->(ot:OntologyTerm)
        WHERE s.run_id = $run_id
        OPTIONAL MATCH (ot)-[:IS_A|PART_OF*1..3]->(anc:OntologyTerm)
        RETURN s.name AS species, t.raw_trait AS raw_trait,
               ot.term_id AS term_id, ot.term_name AS term_name,
               collect(DISTINCT anc.term_id) AS ancestor_ids
        """
        path_rows = neo4j_client.run_query(path_cypher, {"run_id": run_id, "trait": trait})
        paths.append({
            "trait": trait,
            "species_list": row.get("species_list", []),
            "support": row.get("support", 0),
            "paths": path_rows,
        })
    return paths


# ---------------------------------------------------------------------------
# Query 4 — LLM hypothesis generation
# ---------------------------------------------------------------------------

_HYPOTHESIS_SYSTEM = (
    "You are a comparative biologist specializing in gene function inference. "
    "Generate concise scientific hypotheses from knowledge graph evidence."
)

_HYPOTHESIS_TEMPLATE = """\
Species set: {species_names}
Shared biological traits (ontology-mapped): {term_names_and_ids}
Synonym trait pairs across species (LLM-verified same biological phenomenon): {synonym_summary}
Semantically similar traits across species (embedding-based, unverified): {similar_summary}
Inferred environmental stressors: {envo_terms}
Evidence traversal summary: {path_summary}

Generate a 2-3 sentence hypothesis about what biological function the shared gene \
of interest likely performs. Name the pathway or mechanism where possible.

Reply as JSON only:
{{
  "hypothesis": "...",
  "confidence": "high|medium|low",
  "key_evidence": ["...", "..."],
  "suggested_pathways": ["..."]
}}
"""


def generate_hypotheses(
    run_id: str,
    species_names: list[str],
    shared_terms: list[dict],
    stressors: list[dict],
    evidence_paths: list[dict],
    similar_clusters: list[dict] | None = None,
    synonym_clusters: list[dict] | None = None,
    model=None,
) -> list[dict]:
    """Query 4: LLM generates gene-function hypotheses from assembled KG context."""
    from core.llm_backend import make_chat_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    if not shared_terms and not similar_clusters and not synonym_clusters:
        print("[KG] No shared terms, synonym clusters, or similar clusters found — skipping hypothesis generation.")
        return []

    term_names_and_ids = "; ".join(
        f"{t.get('term_name', '')} ({t.get('term_id', '')})"
        for t in shared_terms[:10]
    ) or "none"
    envo_str = "; ".join(
        f"{s.get('term_name', '')} ({s.get('term_id', '')})"
        for s in stressors[:5]
    ) or "none identified"

    # Summarise synonym clusters (LLM-verified)
    synonym_lines = []
    for sc in (synonym_clusters or [])[:8]:
        sp = ", ".join(sc.get("all_species", []))
        synonym_lines.append(
            f"'{sc.get('trait_a')}' ≡ '{sc.get('trait_b')}' (species: {sp})"
        )
    synonym_summary = "; ".join(synonym_lines) or "none"

    # Summarise top similar clusters (embedding-based)
    similar_lines = []
    for sc in (similar_clusters or [])[:8]:
        sp = ", ".join(sc.get("all_species", []))
        similar_lines.append(
            f"'{sc.get('norm_a')}' ~ '{sc.get('norm_b')}' (score={sc.get('score', 0):.2f}, species: {sp})"
        )
    similar_summary = "; ".join(similar_lines) or "none"

    # Flatten top 3 evidence paths for the summary
    path_lines = []
    for ep in evidence_paths[:3]:
        trait = ep.get("trait", "")
        sp_list = ", ".join(ep.get("species_list", []))
        path_lines.append(f"'{trait}' shared by [{sp_list}]")
    path_summary = "; ".join(path_lines) or "no paths"

    prompt = _HYPOTHESIS_TEMPLATE.format(
        species_names=", ".join(species_names),
        term_names_and_ids=term_names_and_ids,
        synonym_summary=synonym_summary,
        similar_summary=similar_summary,
        envo_terms=envo_str,
        path_summary=path_summary,
    )

    llm = make_chat_llm(model=model, temperature=0.3, timeout=120)
    hypotheses = []
    for attempt in range(2):
        try:
            resp = llm.invoke([SystemMessage(content=_HYPOTHESIS_SYSTEM), HumanMessage(content=prompt)])
            raw = resp.content.strip()
            # Strip markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            hyp = json.loads(raw)
            hyp["run_id"] = run_id
            hypotheses.append(hyp)
            print(f"[KG] Hypothesis generated (confidence={hyp.get('confidence', '?')})")
            break
        except Exception as exc:
            if attempt == 0:
                print(f"[KG] Hypothesis generation failed (attempt 1), retrying: {exc}")
                time.sleep(2)
            else:
                print(f"[KG] Hypothesis generation failed: {exc}")

    return hypotheses


# ---------------------------------------------------------------------------
# Run all queries
# ---------------------------------------------------------------------------

def run_all_queries(
    run_id: str,
    species_names: list[str],
    min_species: int = 2,
    model=None,
    generate_hypotheses_flag: bool = False,
) -> dict:
    """
    Run all four queries and return structured results dict.
    All queries are non-fatal — failures produce empty lists.
    """
    print(f"[KG] Running queries for run_id={run_id}...")

    print("[KG] Query 1: shared ontology terms...")
    try:
        shared_terms = query_shared_terms(run_id, min_species=min_species)
        print(f"[KG]   Found {len(shared_terms)} shared terms.")
    except Exception as exc:
        print(f"[KG] Query 1 failed (non-fatal): {exc}")
        shared_terms = []

    print("[KG] Query 1b: LLM synonym clusters...")
    try:
        synonym_clusters = query_synonym_clusters(run_id, min_species=min_species)
        print(f"[KG]   Found {len(synonym_clusters)} cross-species synonym pairs.")
        for sc in synonym_clusters[:5]:
            print(f"[KG]     '{sc.get('trait_a')}' ≡ '{sc.get('trait_b')}' "
                  f"(species={sc.get('all_species')})")
    except Exception as exc:
        print(f"[KG] Query 1b failed (non-fatal): {exc}")
        synonym_clusters = []

    print("[KG] Query 1c: semantically similar trait clusters...")
    try:
        similar_clusters = query_similar_trait_clusters(run_id, min_species=min_species)
        print(f"[KG]   Found {len(similar_clusters)} cross-species similar trait pairs.")
        for sc in similar_clusters[:5]:
            print(f"[KG]     '{sc.get('norm_a')}' ~ '{sc.get('norm_b')}' "
                  f"(score={sc.get('score', 0):.2f}, species={sc.get('all_species')})")
    except Exception as exc:
        print(f"[KG] Query 1c failed (non-fatal): {exc}")
        similar_clusters = []

    shared_term_ids = [t.get("term_id") for t in shared_terms if t.get("term_id")]

    print("[KG] Query 2: inferred stressors...")
    try:
        stressors = query_inferred_stressors(run_id, shared_term_ids)
        print(f"[KG]   Found {len(stressors)} ENVO stressor terms.")
    except Exception as exc:
        print(f"[KG] Query 2 failed (non-fatal): {exc}")
        stressors = []

    print("[KG] Query 2b: shared ancestor term convergence...")
    try:
        ancestor_convergence = query_shared_ancestor_terms(run_id, min_species=min_species)
        print(f"[KG]   Found {len(ancestor_convergence)} shared ancestor terms.")
        for ac in ancestor_convergence[:5]:
            sp = ", ".join(ac.get("species_list", []))
            print(f"[KG]     {ac.get('ancestor_name')} ({ac.get('ancestor_id')}) "
                  f"— {ac.get('species_count')} species: {sp}")
    except Exception as exc:
        print(f"[KG] Query 2b failed (non-fatal): {exc}")
        ancestor_convergence = []

    print("[KG] Query 3: evidence paths...")
    try:
        evidence_paths = query_evidence_paths(run_id, top_n=5)
        print(f"[KG]   Found {len(evidence_paths)} evidence paths.")
    except Exception as exc:
        print(f"[KG] Query 3 failed (non-fatal): {exc}")
        evidence_paths = []

    if generate_hypotheses_flag:
        print("[KG] Query 4: LLM hypothesis generation...")
        try:
            hypotheses = generate_hypotheses(
                run_id=run_id,
                species_names=species_names,
                shared_terms=shared_terms,
                stressors=stressors,
                evidence_paths=evidence_paths,
                similar_clusters=similar_clusters,
                synonym_clusters=synonym_clusters,
                model=model,
            )
        except Exception as exc:
            print(f"[KG] Query 4 failed (non-fatal): {exc}")
            hypotheses = []
    else:
        print("[KG] Query 4: LLM hypothesis generation skipped (use --kg-hypotheses to enable).")
        hypotheses = []

    return {
        "shared_terms": shared_terms,
        "synonym_clusters": synonym_clusters,
        "similar_clusters": similar_clusters,
        "stressors": stressors,
        "ancestor_convergence": ancestor_convergence,
        "evidence_paths": evidence_paths,
        "hypotheses": hypotheses,
    }
