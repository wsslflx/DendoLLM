#!/usr/bin/env python3
"""
KG pipeline step — orchestration entry point.

Called by run_full_pipeline_v4.py immediately after step 4 (hybrid normalization).
Reads hybrid_profile.json files, maps traits to uPheno, builds and pushes the graph,
runs queries, and writes all output files to the bundle directory.

Usage (standalone):
    python kg/kg_pipeline_step.py --bundle-dir logs_v4/<run>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from core.llm_backend import resolve_chat_model, make_chat_llm


# ---------------------------------------------------------------------------
# Inline LLM synonym grouping
# ---------------------------------------------------------------------------

_SYNONYM_SYSTEM = (
    "You are grouping biological trait phrases that mean the same underlying condition "
    "across different species."
)

_SYNONYM_PROMPT = """\
Below are biological traits extracted per species. Group traits from DIFFERENT species \
that describe the same biological phenomenon (true synonyms or rephrasings).

Rules:
- Only group when meaning is essentially identical (true synonyms / rephrasings).
- If unsure, do NOT group — keep the trait in its own singleton group.
- Do NOT merge different abstraction levels (behavior vs morphology vs physiology).
- Every input trait must appear in exactly one group's members list.
- Output JSON only (no markdown fences, no commentary).

Output format:
{{
  "groups": [
    {{"canonical": "<representative phrase>", "members": ["<trait 1>", "<trait 2>", ...]}}
  ]
}}

TRAITS BY SPECIES:
{trait_block}
"""


def _group_synonyms_inline(
    species_trait_map: dict[str, list[dict]],
    model: str,
) -> list[list[str]]:
    """
    Ask the LLM to group synonym traits across species.
    Returns list of groups, each group is a list of raw_trait strings.
    Singletons are excluded (no cross-species synonym found).
    """
    import re

    # Build trait block: "Species: trait1, trait2, ..."
    lines = []
    all_traits = []
    for species_name, trait_results in species_trait_map.items():
        traits = [tr.get("raw_trait", "") for tr in trait_results if tr.get("raw_trait")]
        if traits:
            lines.append(f"{species_name}: {'; '.join(traits)}")
            all_traits.extend(traits)

    if len(all_traits) < 2:
        return []

    trait_block = "\n".join(lines)
    prompt = _SYNONYM_PROMPT.format(trait_block=trait_block)

    print(f"[KG] Running inline synonym grouping for {len(all_traits)} traits across "
          f"{len(species_trait_map)} species...")

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        llm = make_chat_llm(model=model, temperature=0.0)
        resp = llm.invoke([SystemMessage(content=_SYNONYM_SYSTEM), HumanMessage(content=prompt)])
        raw = resp.content.strip()
        # Strip markdown fences if present
        fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if fence:
            raw = fence.group(1).strip()
        parsed = json.loads(raw)
        groups = parsed.get("groups", [])
        # Return only multi-member groups (singletons have no synonym to link)
        result = [g["members"] for g in groups if len(g.get("members", [])) > 1]
        print(f"[KG] Synonym grouping complete: {len(result)} synonym groups found.")
        for g in result:
            print(f"[KG]   Group: {g}")
        return result
    except Exception as exc:
        print(f"[KG] Inline synonym grouping failed (non-fatal): {exc}")
        return []


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def discover_hybrid_profiles(bundle_dir: Path) -> dict[str, list[dict]]:
    """
    Scan bundle_dir for hybrid_profile.json files.
    Returns { species_name: hybrid_profile_dict }.

    Directory structure:
        bundle_dir/runs/run_NN/<species_slug>/hybrid_profile.json
    """
    profiles: dict[str, dict] = {}
    runs_dir = bundle_dir / "runs"
    if not runs_dir.exists():
        print(f"[KG] No runs/ directory found in {bundle_dir}")
        return {}

    for profile_path in sorted(runs_dir.glob("*/*/hybrid_profile.json")):
        species_slug = profile_path.parent.name
        try:
            data = json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[KG] Failed to read {profile_path}: {exc}")
            continue
        # Merge if same species appears in multiple runs — take last (or all, deduplicated)
        if species_slug not in profiles:
            profiles[species_slug] = data
        # else: skip duplicates from multiple runs (same species)

    print(f"[KG] Discovered {len(profiles)} species profiles.")
    return profiles


def _extract_traits_from_profile(profile: dict) -> list[str]:
    """
    Extract the raw trait strings from a hybrid_profile dict.
    Prefers normalized_tags > open_traits.
    Real profile shape: { open_traits: [...], normalized_tags: [...], latent_factors: [...] }
    """
    traits: list[str] = []

    # Prefer normalized tags (deduplicated, canonical)
    normalized_tags = profile.get("normalized_tags", [])
    if normalized_tags:
        for tag in normalized_tags:
            if isinstance(tag, dict):
                tag_name = tag.get("tag", "")
                if tag_name:
                    traits.append(tag_name)

    # Fall back to open_traits if no normalized tags
    if not traits:
        for ot in profile.get("open_traits", []):
            if isinstance(ot, dict):
                t = ot.get("trait", "")
                if t:
                    traits.append(t)

    return list(dict.fromkeys(traits))  # deduplicate preserving order


def _deslugify(slug: str) -> str:
    """Convert species_slug back to 'Species name' (best effort)."""
    return slug.replace("_", " ").strip().title()


# ---------------------------------------------------------------------------
# Output file writing (all atomic)
# ---------------------------------------------------------------------------

def _write_json(path: Path, data) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _write_summary(path: Path, data: dict) -> None:
    lines = [
        "KG Pipeline Step Summary",
        "=" * 40,
        f"Run ID:               {data.get('run_id', '?')}",
        f"Species:              {len(data.get('species_names', []))}",
        f"Traits mapped:        {data.get('traits_mapped', 0)} / {data.get('traits_total', 0)}",
        f"Shared terms:         {len(data.get('shared_terms', []))}",
        f"Similar trait pairs:  {len(data.get('similar_clusters', []))}",
        f"ENVO stressors:       {len(data.get('stressors', []))}",
        f"Hypotheses:           {len(data.get('hypotheses', []))}",
        "",
        "Shared ontology terms:",
    ]
    for t in data.get("shared_terms", [])[:10]:
        lines.append(f"  {t.get('term_id','?')} — {t.get('term_name','?')} "
                     f"({t.get('species_count',0)} species)")
    lines += ["", "Semantically similar trait pairs:"]
    for sc in data.get("similar_clusters", [])[:10]:
        sp = ", ".join(sc.get("all_species", []))
        lines.append(f"  '{sc.get('norm_a')}' ~ '{sc.get('norm_b')}' "
                     f"(score={sc.get('score', 0):.2f}, species: {sp})")
    lines += [
        "",
        "Inferred stressors:",
    ]
    for s in data.get("stressors", [])[:5]:
        lines.append(f"  {s.get('term_id','?')} — {s.get('term_name','?')}")
    lines += ["", "Hypotheses:"]
    for h in data.get("hypotheses", []):
        lines.append(f"  [{h.get('confidence','?')}] {h.get('hypothesis','')}")
    tmp = Path(str(path) + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_kg_step(bundle_dir_str: str, model: str | None = None, generate_hypotheses: bool = False) -> None:
    """
    Full KG pipeline step. Non-fatal at the top level — any unhandled exception
    is caught by the caller in run_full_pipeline_v4.py.
    """
    bundle_dir = Path(bundle_dir_str)
    run_id = bundle_dir.name
    model = resolve_chat_model(model)

    print(f"\n[KG] ============================================================")
    print(f"[KG] Starting KG pipeline step for bundle: {bundle_dir.name}")
    print(f"[KG] Model: {model}")
    print(f"[KG] ============================================================")

    # Step 1: discover hybrid profiles
    profiles = discover_hybrid_profiles(bundle_dir)
    if not profiles:
        print("[KG] No hybrid profiles found — skipping KG step.")
        return

    species_names = [_deslugify(slug) for slug in profiles.keys()]
    print(f"[KG] Species: {species_names}")

    # Step 2: check / build ontology index
    from kg.ontology_index import build_index
    print("[KG] Checking ontology index cache...")
    build_index(force=False)  # no-op if cache is fresh

    # Step 3: extract and map traits per species
    from kg.trait_mapper import map_traits_batch
    no_match_path = bundle_dir / "kg_no_match.jsonl"

    species_trait_map: dict[str, list[dict]] = {}
    all_mapped: list[dict] = []

    for slug, profile in profiles.items():
        species_display = _deslugify(slug)
        raw_traits = _extract_traits_from_profile(profile)
        print(f"[KG] Mapping {len(raw_traits)} traits for {species_display}...")

        # Add cluster_id context from normalized_tags
        tag_to_cluster: dict[str, str] = {}
        for i, tag in enumerate(profile.get("normalized_tags", [])):
            if isinstance(tag, dict):
                tag_name = tag.get("tag", "")
                if tag_name:
                    tag_to_cluster[tag_name] = f"cluster_{i}"

        mapped = map_traits_batch(raw_traits, no_match_out_path=None, model=model)

        # Annotate with species and cluster_id
        for m in mapped:
            m["species"] = species_display
            m["cluster_id"] = tag_to_cluster.get(m.get("raw_trait", ""), "")
        all_mapped.extend(mapped)
        species_trait_map[species_display] = mapped

    # Write no-match file (aggregate across all species)
    no_matches = [m for m in all_mapped if not m.get("mapped")]
    if no_matches:
        _write_jsonl(no_match_path, no_matches)
        print(f"[KG] {len(no_matches)} no-match traits → {no_match_path}")

    # Write mapped traits
    _write_json(bundle_dir / "kg_mapped_traits.json", all_mapped)
    print(f"[KG] Mapped traits written: {bundle_dir / 'kg_mapped_traits.json'}")

    # Step 3b: inline LLM synonym grouping across species
    synonym_groups = _group_synonyms_inline(species_trait_map, model=model)

    # Step 4: build graph
    from kg.kg_builder import build_graph, push_and_serialize
    print("[KG] Building graph...")
    nodes, edges = build_graph(
        species_trait_map=species_trait_map,
        run_id=run_id,
        input_file=str(bundle_dir / "species_input.json"),
        synonym_groups=synonym_groups,
    )

    # Step 5: push to Neo4j + write GraphML
    graphml_path = bundle_dir / "kg_graph.graphml"
    push_and_serialize(nodes, edges, run_id, graphml_path)

    # Step 6: run queries
    from kg.kg_query import run_all_queries
    query_results = run_all_queries(
        run_id=run_id,
        species_names=species_names,
        min_species=max(2, len(species_names) // 2),
        model=model,
        generate_hypotheses_flag=generate_hypotheses,
    )

    # Step 7: write output files
    _write_json(bundle_dir / "kg_shared_terms.json", query_results["shared_terms"])
    _write_json(bundle_dir / "kg_synonym_clusters.json", query_results.get("synonym_clusters", []))
    _write_json(bundle_dir / "kg_similar_clusters.json", query_results.get("similar_clusters", []))
    _write_json(bundle_dir / "kg_inferred_stressors.json", query_results["stressors"])
    _write_json(bundle_dir / "kg_hypotheses.json", query_results["hypotheses"])
    _write_json(bundle_dir / "kg_evidence_paths.json", query_results["evidence_paths"])

    # Summary
    traits_total = len(all_mapped)
    traits_mapped = sum(1 for m in all_mapped if m.get("mapped"))
    summary_data = {
        "run_id": run_id,
        "species_names": species_names,
        "traits_total": traits_total,
        "traits_mapped": traits_mapped,
        "shared_terms": query_results["shared_terms"],
        "similar_clusters": query_results.get("similar_clusters", []),
        "stressors": query_results["stressors"],
        "hypotheses": query_results["hypotheses"],
    }
    _write_summary(bundle_dir / "kg_summary.txt", summary_data)

    synonym_clusters = query_results.get("synonym_clusters", [])
    similar_clusters = query_results.get("similar_clusters", [])
    print(f"\n[KG] ============================================================")
    print(f"[KG] KG step complete.")
    print(f"[KG]   Traits mapped:        {traits_mapped}/{traits_total}")
    print(f"[KG]   Shared terms:         {len(query_results['shared_terms'])}")
    print(f"[KG]   Synonym pairs:        {len(synonym_clusters)}")
    print(f"[KG]   Similar trait pairs:  {len(similar_clusters)}")
    print(f"[KG]   ENVO stressors:       {len(query_results['stressors'])}")
    print(f"[KG]   Hypotheses:           {len(query_results['hypotheses'])}")
    print(f"[KG]   Output dir:           {bundle_dir}")
    print(f"[KG] ============================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG pipeline step (standalone).")
    parser.add_argument("--bundle-dir", required=True, help="Path to pipeline bundle directory.")
    parser.add_argument("--model", default=None, help="Override LLM model name.")
    parser.add_argument(
        "--kg-hypotheses",
        action="store_true",
        default=False,
        help="Enable LLM hypothesis generation (Query 4). Off by default.",
    )
    args = parser.parse_args()
    run_kg_step(args.bundle_dir, model=args.model, generate_hypotheses=args.kg_hypotheses)
