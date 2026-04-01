#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any

from build_testcase_json import build_entries, parse_species_arg
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

from llm_backend import DEFAULT_CHAT_MODEL, make_chat_llm, make_embeddings

load_dotenv()

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

SYNTHESIS_PROMPT_FILE = "Prompts/prompt_synthesis_v3.txt"
INFERENCE_PROMPT_FILE = "Prompts/prompt_inference_v3.txt"
VERIFIER_PROMPT_FILE = "Prompts/prompt_verifier_v3.txt"


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip().casefold()


def extract_json_from_text(text: str) -> str:
    text = text.strip()
    match = FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text


def parse_json_text(text: str) -> Any:
    return json.loads(extract_json_from_text(text))


def read_json_from_text_file(path: pathlib.Path) -> Any:
    return parse_json_text(path.read_text(encoding="utf-8"))


def load_prompt(path: str) -> str:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8")


def resolve_species_file(
    species_file_arg: str | None,
    species_list_arg: str | None,
    generated_species_file_arg: str | None,
    default_generated_out: pathlib.Path,
) -> tuple[pathlib.Path, str]:
    if species_file_arg:
        species_file = pathlib.Path(species_file_arg)
        if not species_file.exists():
            raise SystemExit(f"Species file not found: {species_file}")
        return species_file, "file"

    species_list = parse_species_arg(species_list_arg or "")
    if not species_list:
        raise SystemExit("No species provided in --species-list.")

    out_path = pathlib.Path(generated_species_file_arg) if generated_species_file_arg else default_generated_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    entries = build_entries(species_list)
    out_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated species file: {out_path}")
    return out_path, "list"


def load_species_entries(path: pathlib.Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Species file must contain a list: {path}")
    return [d for d in data if isinstance(d, dict)]


def build_species_maps(entries: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    slug_to_norm: dict[str, str] = {}
    slug_to_label: dict[str, str] = {}
    for entry in entries:
        canonical = entry.get("canonical")
        if not isinstance(canonical, str) or not canonical.strip():
            continue
        label = canonical.strip()
        slug = slugify(label)
        slug_to_norm[slug] = label.lower().strip()
        slug_to_label[slug] = label
    return slug_to_norm, slug_to_label


def write_run_list(run_dirs: list[pathlib.Path], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(p) for p in run_dirs) + "\n", encoding="utf-8")


def list_species_dirs(run_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted([p for p in run_dir.iterdir() if p.is_dir()])


def run_inventory_stage(
    species_file: pathlib.Path,
    bundle_dir: pathlib.Path,
    runs: int,
    reuse_traits: bool,
    skip_ingest_after_first: bool,
    skip_ingest_all: bool,
) -> list[pathlib.Path]:
    run_dirs: list[pathlib.Path] = []
    for i in range(runs):
        run_dir = bundle_dir / f"run_{i + 1:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "inventory_single_3.py",
            "--species-file",
            str(species_file),
            "--log-run",
            "--log-dir",
            str(run_dir),
        ]
        if reuse_traits:
            cmd.append("--reuse-traits")
        if skip_ingest_all or (skip_ingest_after_first and i > 0):
            cmd.append("--skip-ingest")
        print(f"Run {i + 1}/{runs}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        run_dirs.append(run_dir)
    return run_dirs


def load_inventory_traits(answer_path: pathlib.Path) -> list[dict[str, Any]]:
    if not answer_path.exists():
        return []
    try:
        data = read_json_from_text_file(answer_path)
    except Exception:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def build_inventories_for_run(run_dir: pathlib.Path, slug_to_label: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    inventories: dict[str, list[dict[str, Any]]] = {}
    for species_dir in list_species_dirs(run_dir):
        canonical = slug_to_label.get(species_dir.name, species_dir.name.replace("_", " "))
        inventories[canonical] = load_inventory_traits(species_dir / "answer.txt")
    return inventories


def sanitize_trait_items(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        trait = item.get("trait")
        if not isinstance(trait, str) or not trait.strip():
            continue
        sources = item.get("sources")
        source_list: list[str] = []
        if isinstance(sources, list):
            for src in sources:
                if isinstance(src, str) and src.strip():
                    source_list.append(src.strip())
        confidence = item.get("confidence", 0.5)
        try:
            conf = float(confidence)
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        out.append(
            {
                "trait": " ".join(trait.split()).strip(),
                "sources": sorted(set(source_list)),
                "confidence": conf,
            }
        )
    return out


def sanitize_synthesis_output(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(raw, dict):
        return {
            "synthesis_common_traits": [],
            "synthesis_subgroup_traits": [],
            "synthesis_mechanism_traits": [],
        }
    return {
        "synthesis_common_traits": sanitize_trait_items(raw.get("synthesis_common_traits")),
        "synthesis_subgroup_traits": sanitize_trait_items(raw.get("synthesis_subgroup_traits")),
        "synthesis_mechanism_traits": sanitize_trait_items(raw.get("synthesis_mechanism_traits")),
    }


def sanitize_inference_output(raw: Any, min_candidates: int, max_candidates: int) -> dict[str, Any]:
    fallback = {
        "mechanism_candidates": [
            {
                "mechanism": "shared biological constraint adaptation",
                "confidence_prior": 0.1,
            }
        ]
    }
    if not isinstance(raw, dict):
        return fallback
    items = raw.get("mechanism_candidates")
    if not isinstance(items, list):
        return fallback

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        mechanism = item.get("mechanism")
        if not isinstance(mechanism, str) or not mechanism.strip():
            continue
        key = normalize_text(mechanism)
        if key in seen:
            continue
        seen.add(key)
        confidence = item.get("confidence_prior", 0.5)
        try:
            conf = float(confidence)
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        out.append(
            {
                "mechanism": " ".join(mechanism.split()).strip(),
                "confidence_prior": conf,
            }
        )

    out.sort(key=lambda x: x["confidence_prior"], reverse=True)
    out = out[:max(1, max_candidates)]
    if not out:
        return fallback
    if len(out) < min_candidates:
        return {"mechanism_candidates": out}
    return {"mechanism_candidates": out}


def build_trait_records(
    inventories: dict[str, list[dict[str, Any]]],
    max_traits_per_species: int = 80,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    trait_id = 1
    for species, traits in inventories.items():
        species_norm = species.lower().strip()
        for item in traits[:max_traits_per_species]:
            trait = item.get("trait")
            if not isinstance(trait, str) or not trait.strip():
                continue
            sources = item.get("sources")
            source_list: list[str] = []
            if isinstance(sources, list):
                for src in sources:
                    if isinstance(src, str) and src.strip():
                        source_list.append(src.strip())
            confidence = item.get("confidence", 0.5)
            try:
                conf = float(confidence)
            except Exception:
                conf = 0.5
            out.append(
                {
                    "trait_id": f"T{trait_id:04d}",
                    "species": species,
                    "species_norm": species_norm,
                    "trait": " ".join(trait.split()).strip(),
                    "sources": sorted(set(source_list)),
                    "confidence": max(0.0, min(1.0, conf)),
                }
            )
            trait_id += 1
    return out


def llm_invoke_json(llm: Any, template_text: str, variables: dict[str, Any]) -> Any:
    prompt = PromptTemplate(input_variables=list(variables.keys()), template=template_text)
    rendered = prompt.format(**variables)
    raw = llm.invoke(rendered)
    content = raw.content if hasattr(raw, "content") else raw
    return parse_json_text(str(content))


def is_wiki_source(value: str | None) -> bool:
    if not value:
        return False
    return value.startswith("wiki:") or value.startswith("wikipedia:")


def aggregate_source_stats(run_dirs: list[pathlib.Path]) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    total_papers_fetched: set[str] = set()
    total_papers_used: set[str] = set()
    total_wiki_fetched: set[str] = set()
    total_wiki_used: set[str] = set()

    for run_dir in run_dirs:
        papers_fetched: set[str] = set()
        papers_used: set[str] = set()
        wiki_fetched: set[str] = set()
        wiki_used: set[str] = set()

        for species_dir in list_species_dirs(run_dir):
            ingested_path = species_dir / "ingested_docs.json"
            if ingested_path.exists():
                try:
                    ingested_docs = json.loads(ingested_path.read_text(encoding="utf-8"))
                except Exception:
                    ingested_docs = []
                if isinstance(ingested_docs, list):
                    for entry in ingested_docs:
                        if not isinstance(entry, str):
                            continue
                        if is_wiki_source(entry):
                            wiki_fetched.add(entry)
                        else:
                            papers_fetched.add(entry)

            used_path = species_dir / "used_chunks.json"
            if used_path.exists():
                try:
                    used_chunks = json.loads(used_path.read_text(encoding="utf-8"))
                except Exception:
                    used_chunks = []
                if isinstance(used_chunks, list):
                    for chunk in used_chunks:
                        if not isinstance(chunk, dict):
                            continue
                        source_path = chunk.get("source_path")
                        doc_id = chunk.get("doc_id")
                        source_key = (
                            str(doc_id)
                            if isinstance(doc_id, str) and doc_id
                            else (str(source_path) if isinstance(source_path, str) else "")
                        )
                        if not source_key:
                            continue
                        if is_wiki_source(source_key) or is_wiki_source(str(source_path) if isinstance(source_path, str) else ""):
                            wiki_used.add(source_key)
                        else:
                            papers_used.add(source_key)

        total_papers_fetched.update(papers_fetched)
        total_papers_used.update(papers_used)
        total_wiki_fetched.update(wiki_fetched)
        total_wiki_used.update(wiki_used)
        per_run.append(
            {
                "run_dir": str(run_dir),
                "papers_fetched": len(papers_fetched),
                "papers_used": len(papers_used),
                "wiki_fetched": len(wiki_fetched),
                "wiki_used": len(wiki_used),
            }
        )

    return {
        "overall": {
            "papers_fetched": len(total_papers_fetched),
            "papers_used": len(total_papers_used),
            "wiki_fetched": len(total_wiki_fetched),
            "wiki_used": len(total_wiki_used),
        },
        "per_run": per_run,
    }


def normalize_ref(value: str) -> set[str]:
    base = value.strip()
    if not base:
        return set()
    out = {base, base.casefold()}
    try:
        out.add(pathlib.Path(base).name)
        out.add(pathlib.Path(base).name.casefold())
    except Exception:
        pass
    return out


def metadata_ref_keys(meta: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in ("doc_id", "source_path", "openalex_id"):
        value = meta.get(field)
        if isinstance(value, str) and value.strip():
            keys.update(normalize_ref(value))
    return keys


def citation_from_meta(meta: dict[str, Any]) -> str:
    base = meta.get("doc_id") or meta.get("source_path") or meta.get("openalex_id") or "unknown"
    idx = meta.get("chunk_index", "na")
    return f"{base}|chunk:{idx}"


def build_run_doc_allowlists(run_dir: pathlib.Path, slug_to_norm: dict[str, str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for species_dir in list_species_dirs(run_dir):
        refs: set[str] = set()
        ingested_path = species_dir / "ingested_docs.json"
        if ingested_path.exists():
            try:
                data = json.loads(ingested_path.read_text(encoding="utf-8"))
            except Exception:
                data = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str) and item.strip():
                        refs.add(item.strip())

        used_path = species_dir / "used_chunks.json"
        if used_path.exists():
            try:
                data = json.loads(used_path.read_text(encoding="utf-8"))
            except Exception:
                data = []
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    for key in ("doc_id", "source_path"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            refs.add(value.strip())

        species_norm = slug_to_norm.get(species_dir.name, species_dir.name.replace("_", " ").strip().lower())
        out[species_norm] = refs
    return out


def get_all_chunks_for_species(vectorstore: Chroma, species_norm: str, allow_refs: set[str]) -> list[dict[str, Any]]:
    try:
        raw = vectorstore._collection.get(where={"specie": species_norm}, include=["documents", "metadatas"])  # type: ignore[attr-defined]
    except Exception:
        try:
            raw = vectorstore.get(where={"specie": species_norm}, include=["documents", "metadatas"])
        except Exception:
            raw = {"documents": [], "metadatas": []}

    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    n = min(len(docs), len(metas))
    allow_norm: set[str] = set()
    for ref in allow_refs:
        allow_norm.update(normalize_ref(ref))

    rows: list[dict[str, Any]] = []
    for i in range(n):
        text = docs[i]
        meta = metas[i] if isinstance(metas[i], dict) else {}
        if not isinstance(text, str) or not text.strip():
            continue
        if allow_norm:
            keys = metadata_ref_keys(meta)
            if not (keys & allow_norm):
                continue
        rows.append(
            {
                "species": species_norm,
                "citation": citation_from_meta(meta),
                "text": text.strip(),
            }
        )
    return rows


def build_query_variants(mechanism: str) -> list[str]:
    queries = [
        mechanism,
        f"{mechanism} adaptation physiology ecology",
        f"evidence for {mechanism}",
    ]
    out: list[str] = []
    for q in queries:
        q = " ".join(q.split()).strip()
        if q and q not in out:
            out.append(q)
    return out


def retrieve_chunks_for_mechanism(
    vectorstore: Chroma,
    species_norms: list[str],
    allowlists: dict[str, set[str]],
    mechanism: str,
    k_per_query: int,
    max_chunks_per_species: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    queries = build_query_variants(mechanism)
    combined: list[dict[str, Any]] = []
    per_species_counts: dict[str, int] = {}

    for species_norm in species_norms:
        allow_refs = allowlists.get(species_norm, set())
        all_chunks = get_all_chunks_for_species(vectorstore, species_norm, allow_refs)
        best: dict[str, dict[str, Any]] = {}
        for query in queries:
            try:
                results = vectorstore.similarity_search_with_score(
                    query,
                    k=k_per_query,
                    filter={"specie": species_norm},
                )
            except Exception:
                results = []
            for doc, _score in results:
                meta = doc.metadata or {}
                if allow_refs:
                    keys = metadata_ref_keys(meta)
                    allow_norm = set().union(*(normalize_ref(r) for r in allow_refs))
                    if not (keys & allow_norm):
                        continue
                citation = citation_from_meta(meta)
                best[citation] = {
                    "species": species_norm,
                    "citation": citation,
                    "text": (doc.page_content or "").strip(),
                }
        picked = list(best.values())[:max_chunks_per_species]
        if not picked and all_chunks:
            picked = all_chunks[: min(2, len(all_chunks))]
        per_species_counts[species_norm] = len(picked)
        combined.extend(picked)
    return combined, per_species_counts


def sanitize_verifier_output(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "supporting_trait_ids": [],
            "contradicting_species": [],
            "contradicting_citations": [],
            "verdict_note": "invalid verifier payload",
        }

    def clean_str_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return sorted(set(out))

    return {
        "supporting_trait_ids": clean_str_list(raw.get("supporting_trait_ids")),
        "contradicting_species": clean_str_list(raw.get("contradicting_species")),
        "contradicting_citations": clean_str_list(raw.get("contradicting_citations")),
        "verdict_note": str(raw.get("verdict_note", "")).strip(),
    }


def classify_verifier_status(species_support: int, citation_support: int, contradiction_species: int) -> str:
    if contradiction_species >= 2:
        return "rejected"
    if species_support >= 2 and citation_support >= 2:
        return "supported"
    if species_support > 0 and citation_support > 0:
        return "weak"
    return "rejected"


def aggregate_trait_outputs(
    per_run_sections: list[dict[str, Any]],
    section_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for run_payload in per_run_sections:
        run_id = run_payload.get("run_id")
        items = run_payload.get(section_name, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            trait = item.get("trait")
            if not isinstance(trait, str) or not trait.strip():
                continue
            key = normalize_text(trait)
            state = grouped.setdefault(
                key,
                {
                    "trait": trait.strip(),
                    "sources": set(),
                    "runs": set(),
                    "confidence": 0.0,
                },
            )
            if len(trait.strip()) < len(state["trait"]):
                state["trait"] = trait.strip()
            for src in item.get("sources", []):
                if isinstance(src, str) and src:
                    state["sources"].add(src)
            if isinstance(run_id, str) and run_id:
                state["runs"].add(run_id)
            try:
                state["confidence"] = max(float(item.get("confidence", 0.0)), state["confidence"])
            except Exception:
                pass

    out = []
    for state in grouped.values():
        out.append(
            {
                "trait": state["trait"],
                "sources": sorted(state["sources"]),
                "confidence": round(float(state["confidence"]), 3),
                "run_support_count": len(state["runs"]),
                "runs": sorted(state["runs"]),
            }
        )
    out.sort(key=lambda x: (x["run_support_count"], len(x["sources"])), reverse=True)
    return out


def aggregate_inference_outputs(
    per_run_sections: list[dict[str, Any]],
    section_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for run_payload in per_run_sections:
        run_id = run_payload.get("run_id")
        items = run_payload.get(section_name, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            mechanism = item.get("mechanism")
            if not isinstance(mechanism, str) or not mechanism.strip():
                continue
            key = normalize_text(mechanism)
            state = grouped.setdefault(
                key,
                {
                    "mechanism": mechanism.strip(),
                    "supporting_species": set(),
                    "trait_level_citations": set(),
                    "runs": set(),
                    "supporting_traits": [],
                    "contradicting_species": set(),
                    "contradicting_citations": set(),
                    "status": item.get("status", section_name.replace("inference_", "").replace("_mechanisms", "")),
                    "reasons": set(),
                },
            )
            if len(mechanism.strip()) < len(state["mechanism"]):
                state["mechanism"] = mechanism.strip()
            for s in item.get("supporting_species", []):
                if isinstance(s, str) and s:
                    state["supporting_species"].add(s)
            for c in item.get("trait_level_citations", []):
                if isinstance(c, str) and c:
                    state["trait_level_citations"].add(c)
            for s in item.get("contradicting_species", []):
                if isinstance(s, str) and s:
                    state["contradicting_species"].add(s)
            for c in item.get("contradicting_citations", []):
                if isinstance(c, str) and c:
                    state["contradicting_citations"].add(c)
            if isinstance(run_id, str) and run_id:
                state["runs"].add(run_id)
            reason = item.get("reason")
            if isinstance(reason, str) and reason.strip():
                state["reasons"].add(reason.strip())
            supporting_traits = item.get("supporting_traits", [])
            if isinstance(supporting_traits, list):
                state["supporting_traits"].extend([t for t in supporting_traits if isinstance(t, dict)])

    out = []
    for state in grouped.values():
        out.append(
            {
                "mechanism": state["mechanism"],
                "supporting_species": sorted(state["supporting_species"]),
                "trait_level_citations": sorted(state["trait_level_citations"]),
                "supporting_traits": state["supporting_traits"],
                "contradicting_species": sorted(state["contradicting_species"]),
                "contradicting_citations": sorted(state["contradicting_citations"]),
                "status": state["status"],
                "reason": "; ".join(sorted(state["reasons"])) if state["reasons"] else None,
                "run_support_count": len(state["runs"]),
                "runs": sorted(state["runs"]),
            }
        )
    out.sort(key=lambda x: (x["run_support_count"], len(x["trait_level_citations"])), reverse=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v3 pipeline: inventory -> conservative synthesis + species-name-informed inference -> verifier."
    )
    species_group = parser.add_mutually_exclusive_group(required=True)
    species_group.add_argument("--species-file", help="Path to species JSON file.")
    species_group.add_argument("--species-list", help="Comma-separated scientific names.")
    parser.add_argument("--generated-species-file", help="Output path for generated species JSON when using --species-list.")
    parser.add_argument("--runs", type=int, default=1, help="Number of inventory runs.")
    parser.add_argument("--reuse-traits", action="store_true", help="Reuse existing traits/<species>.json if present.")
    parser.add_argument(
        "--skip-ingest-after-first",
        action="store_true",
        default=True,
        help="Only ingest/download on the first run; subsequent runs skip ingestion (default: on).",
    )
    parser.add_argument("--skip-ingest-all", action="store_true", help="Skip ingestion/download for all runs.")
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL, help="Chat model name for v3 layers.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    parser.add_argument(
        "--inference-temperature",
        type=float,
        default=0.4,
        help="LLM temperature for zero-shot inference proposal step (default: 0.4).",
    )
    parser.add_argument("--min-inference-candidates", type=int, default=1, help="Minimum number of inference candidates.")
    parser.add_argument("--max-inference-candidates", type=int, default=5, help="Maximum number of inference candidates.")
    parser.add_argument("--verifier-k-per-query", type=int, default=60, help="Retrieved chunk candidates per query.")
    parser.add_argument("--verifier-max-chunks-per-species", type=int, default=6, help="Max verifier chunks per species.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    label = slugify(pathlib.Path(args.species_file).stem) if args.species_file else "generated_species"
    bundle_dir = pathlib.Path("logs_v3") / f"{timestamp}-{label}"
    summary_dir = bundle_dir / "summary"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    species_file, species_input_mode = resolve_species_file(
        args.species_file,
        args.species_list,
        args.generated_species_file,
        bundle_dir / "species_input.json",
    )
    species_entries = load_species_entries(species_file)
    slug_to_norm, slug_to_label = build_species_maps(species_entries)

    run_dirs = run_inventory_stage(
        species_file=species_file,
        bundle_dir=bundle_dir,
        runs=args.runs,
        reuse_traits=args.reuse_traits,
        skip_ingest_after_first=args.skip_ingest_after_first,
        skip_ingest_all=args.skip_ingest_all,
    )
    run_list_path = bundle_dir / "run_list.txt"
    write_run_list(run_dirs, run_list_path)
    print(f"Run list written to: {run_list_path}")

    inventories_per_run: list[dict[str, Any]] = []
    synthesis_per_run: list[dict[str, Any]] = []
    inference_raw_per_run: list[dict[str, Any]] = []
    inference_verified_per_run: list[dict[str, Any]] = []

    llm = make_chat_llm(model=args.model, temperature=args.temperature)
    llm_inference = make_chat_llm(model=args.model, temperature=args.inference_temperature)
    synthesis_prompt = load_prompt(SYNTHESIS_PROMPT_FILE)
    inference_prompt = load_prompt(INFERENCE_PROMPT_FILE)
    verifier_prompt = load_prompt(VERIFIER_PROMPT_FILE)
    vectorstore = Chroma(
        collection_name="bunch_of_docs",
        embedding_function=make_embeddings(),
        persist_directory="./chroma_store_ollama",
    )

    for run_dir in run_dirs:
        run_id = run_dir.name
        inventories = build_inventories_for_run(run_dir, slug_to_label)
        inventories_per_run.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "inventories": inventories,
            }
        )

        inventories_json = json.dumps(inventories, ensure_ascii=False, indent=2)
        species_names = [
            str(name).strip()
            for name in inventories.keys()
            if isinstance(name, str) and str(name).strip()
        ]
        species_list_json = json.dumps(species_names, ensure_ascii=False, indent=2)

        try:
            synthesis_raw = llm_invoke_json(
                llm,
                synthesis_prompt,
                {"extracted_species_lists": inventories_json},
            )
        except Exception:
            synthesis_raw = {}
        synthesis_output = sanitize_synthesis_output(synthesis_raw)
        synthesis_per_run.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                **synthesis_output,
            }
        )

        try:
            inference_raw = llm_invoke_json(
                llm_inference,
                inference_prompt,
                {
                    "min_candidates": args.min_inference_candidates,
                    "max_candidates": args.max_inference_candidates,
                    "species_list": species_list_json,
                },
            )
        except Exception:
            inference_raw = {}
        inference_output = sanitize_inference_output(
            inference_raw,
            min_candidates=args.min_inference_candidates,
            max_candidates=args.max_inference_candidates,
        )
        inference_raw_per_run.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                **inference_output,
            }
        )

        trait_records = build_trait_records(inventories)
        trait_by_id = {
            str(record["trait_id"]): record
            for record in trait_records
            if isinstance(record.get("trait_id"), str)
        }
        allowlists = build_run_doc_allowlists(run_dir, slug_to_norm)
        species_norms = sorted(allowlists.keys())

        supported: list[dict[str, Any]] = []
        weak: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for candidate in inference_output.get("mechanism_candidates", []):
            if not isinstance(candidate, dict):
                continue
            mechanism = candidate.get("mechanism")
            if not isinstance(mechanism, str) or not mechanism.strip():
                continue
            mechanism = mechanism.strip()

            retrieved_chunks, per_species_counts = retrieve_chunks_for_mechanism(
                vectorstore=vectorstore,
                species_norms=species_norms,
                allowlists=allowlists,
                mechanism=mechanism,
                k_per_query=args.verifier_k_per_query,
                max_chunks_per_species=args.verifier_max_chunks_per_species,
            )

            try:
                verifier_raw = llm_invoke_json(
                    llm,
                    verifier_prompt,
                    {
                        "mechanism": mechanism,
                        "trait_records": json.dumps(trait_records, ensure_ascii=False, indent=2),
                        "retrieved_chunks": json.dumps(retrieved_chunks, ensure_ascii=False, indent=2),
                    },
                )
            except Exception:
                verifier_raw = {}
            verifier = sanitize_verifier_output(verifier_raw)

            supporting_trait_ids = [
                tid for tid in verifier["supporting_trait_ids"] if tid in trait_by_id
            ]
            supporting_traits = [trait_by_id[tid] for tid in supporting_trait_ids]
            supporting_species = sorted(
                {
                    str(item.get("species"))
                    for item in supporting_traits
                    if isinstance(item.get("species"), str)
                }
            )
            trait_level_citations = sorted(
                {
                    src
                    for item in supporting_traits
                    for src in (item.get("sources") if isinstance(item.get("sources"), list) else [])
                    if isinstance(src, str) and src
                }
            )
            contradicting_species = sorted(
                {
                    s for s in verifier["contradicting_species"] if isinstance(s, str) and s.strip()
                }
            )
            contradicting_citations = sorted(
                {
                    c for c in verifier["contradicting_citations"] if isinstance(c, str) and c.strip()
                }
            )

            status = classify_verifier_status(
                species_support=len(supporting_species),
                citation_support=len(trait_level_citations),
                contradiction_species=len(contradicting_species),
            )
            item = {
                "mechanism": mechanism,
                "supporting_species": supporting_species,
                "supporting_traits": [
                    {
                        "species": t.get("species"),
                        "trait": t.get("trait"),
                        "sources": t.get("sources", []),
                    }
                    for t in supporting_traits
                ],
                "trait_level_citations": trait_level_citations,
                "contradicting_species": contradicting_species,
                "contradicting_citations": contradicting_citations,
                "status": status,
                "confidence_prior": candidate.get("confidence_prior", 0.5),
                "verdict_note": verifier.get("verdict_note", ""),
                "verification_context": {
                    "track_a_trait_record_count": len(trait_records),
                    "retrieved_chunk_count": len(retrieved_chunks),
                    "per_species_chunk_counts": per_species_counts,
                },
            }
            if status == "supported":
                supported.append(item)
            elif status == "weak":
                weak.append(item)
            else:
                item["reason"] = "explicit contradiction in >=2 species" if len(contradicting_species) >= 2 else "insufficient support"
                rejected.append(item)

        inference_verified_per_run.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "inference_supported_mechanisms": supported,
                "inference_weak_mechanisms": weak,
                "inference_rejected_mechanisms": rejected,
            }
        )

    inventory_traits_payload = {"per_run": inventories_per_run}
    inventory_traits_path = summary_dir / "inventory_traits.json"
    inventory_traits_path.write_text(json.dumps(inventory_traits_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {inventory_traits_path}")

    synthesis_aggregated = {
        "synthesis_common_traits": aggregate_trait_outputs(synthesis_per_run, "synthesis_common_traits"),
        "synthesis_subgroup_traits": aggregate_trait_outputs(synthesis_per_run, "synthesis_subgroup_traits"),
        "synthesis_mechanism_traits": aggregate_trait_outputs(synthesis_per_run, "synthesis_mechanism_traits"),
    }
    synthesis_output_payload = {
        "per_run": synthesis_per_run,
        "aggregated": synthesis_aggregated,
    }
    synthesis_output_path = summary_dir / "synthesis_output.json"
    synthesis_output_path.write_text(json.dumps(synthesis_output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {synthesis_output_path}")

    inference_raw_payload = {"per_run": inference_raw_per_run}
    inference_raw_path = summary_dir / "inference_raw.json"
    inference_raw_path.write_text(json.dumps(inference_raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {inference_raw_path}")

    inference_verified_aggregated = {
        "inference_supported_mechanisms": aggregate_inference_outputs(
            inference_verified_per_run, "inference_supported_mechanisms"
        ),
        "inference_weak_mechanisms": aggregate_inference_outputs(
            inference_verified_per_run, "inference_weak_mechanisms"
        ),
        "inference_rejected_mechanisms": aggregate_inference_outputs(
            inference_verified_per_run, "inference_rejected_mechanisms"
        ),
    }
    inference_verified_payload = {
        "per_run": inference_verified_per_run,
        "aggregated": inference_verified_aggregated,
    }
    inference_verified_path = summary_dir / "inference_verified.json"
    inference_verified_path.write_text(
        json.dumps(inference_verified_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {inference_verified_path}")

    final_report = {
        "synthesis_common_traits": synthesis_aggregated["synthesis_common_traits"],
        "synthesis_subgroup_traits": synthesis_aggregated["synthesis_subgroup_traits"],
        "synthesis_mechanism_traits": synthesis_aggregated["synthesis_mechanism_traits"],
        "inference_supported_mechanisms": inference_verified_aggregated["inference_supported_mechanisms"],
        "inference_weak_mechanisms": inference_verified_aggregated["inference_weak_mechanisms"],
        "inference_rejected_mechanisms": inference_verified_aggregated["inference_rejected_mechanisms"],
    }
    final_report_path = summary_dir / "final_report.json"
    final_report_path.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {final_report_path}")

    meta = {
        "timestamp": timestamp,
        "species_input_mode": species_input_mode,
        "species_file": str(species_file),
        "species_list": args.species_list if args.species_list else None,
        "runs_requested": args.runs,
        "runs_completed": len(run_dirs),
        "reuse_traits": args.reuse_traits,
        "skip_ingest_after_first": args.skip_ingest_after_first,
        "skip_ingest_all": args.skip_ingest_all,
        "inventory_script": "inventory_single_3.py",
        "model": args.model,
        "temperature": args.temperature,
        "inference_temperature": args.inference_temperature,
        "thresholds": {
            "min_inference_candidates": args.min_inference_candidates,
            "max_inference_candidates": args.max_inference_candidates,
            "min_support_species": 2,
            "min_support_citations": 2,
            "reject_if_contradiction_species_ge": 2,
        },
        "run_dirs_v3": [str(p) for p in run_dirs],
        "artifacts": {
            "run_list": str(run_list_path),
            "inventory_traits": str(inventory_traits_path),
            "synthesis_output": str(synthesis_output_path),
            "inference_raw": str(inference_raw_path),
            "inference_verified": str(inference_verified_path),
            "final_report": str(final_report_path),
        },
        "source_stats": aggregate_source_stats(run_dirs),
    }
    meta_path = summary_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
