#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from analyze_runs import analyze_runs as analyze_run_files
from build_testcase_json import build_entries, parse_species_arg
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

load_dotenv()

INFERENCE_PROMPT_FILE = "Prompts/prompt_inference.txt"


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def list_log_dirs(base: pathlib.Path) -> set[pathlib.Path]:
    if not base.exists():
        return set()
    return {p for p in base.iterdir() if p.is_dir()}


def extract_json_from_text(text: str) -> str:
    text = text.strip()
    m = FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text


def read_json_from_text_file(path: pathlib.Path) -> Any:
    raw = path.read_text(encoding="utf-8").strip()
    payload = extract_json_from_text(raw)
    return json.loads(payload)


def normalize_trait(text: str) -> str:
    return " ".join(text.split()).casefold()


def load_inference_prompt() -> str:
    path = pathlib.Path(INFERENCE_PROMPT_FILE)
    if not path.exists():
        raise FileNotFoundError(f"Inference prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


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


def detect_created_log_dir(
    before: set[pathlib.Path], after: set[pathlib.Path], run_label: str
) -> pathlib.Path:
    new_dirs = sorted(after - before)
    if len(new_dirs) == 1:
        return new_dirs[0]
    label_dirs = [p for p in after if p.name.endswith(f"-{run_label}")]
    if label_dirs:
        return max(label_dirs, key=lambda p: p.stat().st_mtime)
    if after:
        return max(after, key=lambda p: p.stat().st_mtime)
    raise RuntimeError("No log directories found after inventory run.")


def run_inventory_stage(
    species_file: pathlib.Path,
    runs: int,
    reuse_traits: bool,
    skip_ingest_after_first: bool,
    skip_ingest_all: bool,
) -> list[pathlib.Path]:
    created: list[pathlib.Path] = []
    run_label = slugify(species_file.stem)
    logs_root = pathlib.Path("logs")
    for i in range(runs):
        before = list_log_dirs(logs_root)
        cmd = [sys.executable, "inventory_single_2.py", "--species-file", str(species_file), "--log-run"]
        if reuse_traits:
            cmd.append("--reuse-traits")
        if skip_ingest_all or (skip_ingest_after_first and i > 0):
            cmd.append("--skip-ingest")
        print(f"Run {i + 1}/{runs}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        after = list_log_dirs(logs_root)
        created.append(detect_created_log_dir(before, after, run_label))
    return created


def copy_run_logs_to_bundle(src_runs: list[pathlib.Path], bundle_dir: pathlib.Path) -> list[pathlib.Path]:
    copied: list[pathlib.Path] = []
    for i, src in enumerate(src_runs, start=1):
        dst = bundle_dir / f"run_{i:02d}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        copied.append(dst)
    return copied


def write_run_list(run_dirs: list[pathlib.Path], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(p) for p in run_dirs) + "\n", encoding="utf-8")


def is_wiki_source(value: str | None) -> bool:
    if not value:
        return False
    return value.startswith("wiki:") or value.startswith("wikipedia:")


def aggregate_source_stats(run_dirs: list[pathlib.Path]) -> dict:
    per_run: list[dict] = []
    total_papers_fetched: set[str] = set()
    total_papers_used: set[str] = set()
    total_wiki_fetched: set[str] = set()
    total_wiki_used: set[str] = set()

    for run_dir in run_dirs:
        papers_fetched: set[str] = set()
        papers_used: set[str] = set()
        wiki_fetched: set[str] = set()
        wiki_used: set[str] = set()

        for species_dir in run_dir.iterdir():
            if not species_dir.is_dir() or species_dir.name == "synthesis":
                continue

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
                        if is_wiki_source(source_key) or is_wiki_source(
                            str(source_path) if isinstance(source_path, str) else ""
                        ):
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


def iter_synthesis_items(data: Any) -> list[tuple[str, dict]]:
    if isinstance(data, list):
        out: list[tuple[str, dict]] = []
        for item in data:
            if isinstance(item, dict):
                out.append(("synthesis_list", item))
        return out

    if isinstance(data, dict):
        out = []
        for section in ("strict_common_traits", "subgroup_common_traits", "mechanism_hypotheses"):
            items = data.get(section, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    out.append((section, item))
        return out

    return []


def build_evidence_graph(run_dirs: list[pathlib.Path]) -> list[dict]:
    edges: list[dict] = []
    for run_dir in run_dirs:
        run_id = run_dir.name
        for species_dir in run_dir.iterdir():
            if not species_dir.is_dir():
                continue

            if species_dir.name == "synthesis":
                synthesis_file = species_dir / "synthesis_answer.txt"
                if not synthesis_file.exists():
                    continue
                try:
                    syn_data = read_json_from_text_file(synthesis_file)
                except Exception:
                    continue
                for section, item in iter_synthesis_items(syn_data):
                    trait = item.get("trait")
                    if not isinstance(trait, str) or not trait.strip():
                        continue
                    confidence = item.get("confidence")
                    sources = item.get("sources", [])
                    if not isinstance(sources, list):
                        sources = []
                    if not sources:
                        edges.append(
                            {
                                "run_id": run_id,
                                "species": None,
                                "section": section,
                                "trait_text": trait,
                                "normalized_trait": normalize_trait(trait),
                                "source_tag": None,
                                "confidence": confidence,
                                "evidence_type": "explicit",
                                "derived_from_trait": None,
                                "inference_rule_id": None,
                                "contradiction_flags": [],
                            }
                        )
                    for src in sources:
                        if not isinstance(src, str):
                            continue
                        edges.append(
                            {
                                "run_id": run_id,
                                "species": None,
                                "section": section,
                                "trait_text": trait,
                                "normalized_trait": normalize_trait(trait),
                                "source_tag": src,
                                "confidence": confidence,
                                "evidence_type": "explicit",
                                "derived_from_trait": None,
                                "inference_rule_id": None,
                                "contradiction_flags": [],
                            }
                        )
                continue

            answer_path = species_dir / "answer.txt"
            if not answer_path.exists():
                continue
            try:
                inv_data = read_json_from_text_file(answer_path)
            except Exception:
                continue
            if not isinstance(inv_data, list):
                continue
            for item in inv_data:
                if not isinstance(item, dict):
                    continue
                trait = item.get("trait")
                if not isinstance(trait, str) or not trait.strip():
                    continue
                confidence = item.get("confidence")
                sources = item.get("sources", [])
                if not isinstance(sources, list):
                    sources = []
                if not sources:
                    edges.append(
                        {
                            "run_id": run_id,
                            "species": species_dir.name,
                            "section": "inventory",
                            "trait_text": trait,
                            "normalized_trait": normalize_trait(trait),
                            "source_tag": None,
                            "confidence": confidence,
                            "evidence_type": "explicit",
                            "derived_from_trait": None,
                            "inference_rule_id": None,
                            "contradiction_flags": [],
                        }
                    )
                for src in sources:
                    if not isinstance(src, str):
                        continue
                    edges.append(
                        {
                            "run_id": run_id,
                            "species": species_dir.name,
                            "section": "inventory",
                            "trait_text": trait,
                            "normalized_trait": normalize_trait(trait),
                            "source_tag": src,
                            "confidence": confidence,
                            "evidence_type": "explicit",
                            "derived_from_trait": None,
                            "inference_rule_id": None,
                            "contradiction_flags": [],
                        }
                    )
    return edges


def build_source_to_species(edges: list[dict]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge.get("section") != "inventory":
            continue
        source = edge.get("source_tag")
        species = edge.get("species")
        if isinstance(source, str) and source and isinstance(species, str) and species:
            out[source].add(species)
    return out


def chunked(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def infer_trait_semantics_with_llm(
    traits: list[str],
    model: str,
    temperature: float,
    batch_size: int = 120,
    min_confidence: float = 0.6,
    max_inferences_per_trait: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    if not traits:
        return {}

    llm = ChatOpenAI(model_name=model, temperature=temperature)
    prompt = PromptTemplate(
        input_variables=["trait_list"],
        template=load_inference_prompt(),
    )
    out: dict[str, list[dict[str, Any]]] = {}

    for batch in chunked(traits, batch_size):
        try:
            rendered = prompt.format(trait_list=json.dumps(batch, ensure_ascii=False, indent=2))
            raw = llm.invoke(rendered)
            payload_text = raw.content if hasattr(raw, "content") else raw
            parsed = json.loads(extract_json_from_text(str(payload_text)))
        except Exception:
            continue

        items = parsed.get("traits") if isinstance(parsed, dict) else None
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            trait = item.get("trait")
            if not isinstance(trait, str) or not trait.strip():
                continue
            inferences = item.get("inferences", [])
            if not isinstance(inferences, list):
                continue
            clean: list[dict[str, Any]] = []
            for inf in inferences:
                if not isinstance(inf, dict):
                    continue
                context = inf.get("context")
                challenge = inf.get("challenge")
                mechanism = inf.get("mechanism")
                confidence = inf.get("confidence", 0.5)
                if not isinstance(context, str) or not context.strip():
                    continue
                if not isinstance(challenge, str) or not challenge.strip():
                    continue
                if not isinstance(mechanism, str) or not mechanism.strip():
                    mechanism = f"adaptation related to {challenge}"
                try:
                    conf = float(confidence)
                except Exception:
                    conf = 0.5
                conf = max(0.0, min(1.0, conf))
                if conf < min_confidence:
                    continue
                clean.append(
                    {
                        "context": context.strip(),
                        "challenge": challenge.strip(),
                        "mechanism": mechanism.strip(),
                        "confidence": conf,
                    }
                )
            clean.sort(key=lambda x: x["confidence"], reverse=True)
            out[normalize_trait(trait)] = clean[:max_inferences_per_trait]
    return out


def build_inference_layer(
    explicit_edges: list[dict],
    model: str,
    temperature: float,
    min_inferred_traits: int = 2,
    max_inferred_traits: int = 7,
) -> dict[str, Any]:
    unique_inventory_traits = sorted(
        {
            str(edge.get("trait_text"))
            for edge in explicit_edges
            if edge.get("section") == "inventory" and isinstance(edge.get("trait_text"), str)
        }
    )
    trait_inference_map = infer_trait_semantics_with_llm(
        unique_inventory_traits,
        model=model,
        temperature=temperature,
    )

    context_edges: list[dict] = []
    seen_context: set[tuple] = set()

    for edge in explicit_edges:
        if edge.get("section") != "inventory":
            continue
        trait_text = edge.get("trait_text")
        normalized_trait = edge.get("normalized_trait")
        if not isinstance(trait_text, str) or not isinstance(normalized_trait, str):
            continue
        inferences = trait_inference_map.get(normalized_trait, [])
        for inf in inferences:
            context = inf.get("context")
            challenge = inf.get("challenge")
            mechanism = inf.get("mechanism")
            if not isinstance(context, str) or not context:
                continue
            if not isinstance(challenge, str) or not challenge:
                continue
            if not isinstance(mechanism, str) or not mechanism:
                mechanism = f"adaptation related to {challenge}"
            key = (
                edge.get("run_id"),
                edge.get("species"),
                normalize_trait(challenge),
                normalize_trait(context),
                edge.get("source_tag"),
            )
            if key in seen_context:
                continue
            seen_context.add(key)
            context_edges.append(
                {
                    "run_id": edge.get("run_id"),
                    "species": edge.get("species"),
                    "section": "inferred_contexts",
                    "trait_text": context,
                    "normalized_trait": normalize_trait(context),
                    "source_tag": edge.get("source_tag"),
                    "confidence": inf.get("confidence", 0.5),
                    "evidence_type": "inferred",
                    "derived_from_trait": trait_text,
                    "inference_rule_id": "llm_semantic_inference",
                    "challenge": normalize_trait(challenge),
                    "challenge_label": challenge,
                    "context_type": normalize_trait(context),
                    "context_label": context,
                    "mechanism_label": mechanism,
                    "contradiction_flags": [],
                }
            )

    by_run_challenge: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for edge in context_edges:
        run_id = edge.get("run_id")
        challenge = edge.get("challenge")
        if isinstance(run_id, str) and isinstance(challenge, str):
            by_run_challenge[(run_id, challenge)].append(edge)

    mechanism_edges: list[dict] = []
    summary: list[dict] = []
    seen_mechanism: set[tuple] = set()
    for (run_id, challenge), edges in by_run_challenge.items():
        species = sorted({str(e.get("species")) for e in edges if isinstance(e.get("species"), str)})
        context_types = sorted({str(e.get("context_label")) for e in edges if isinstance(e.get("context_label"), str)})
        sources = sorted({str(e.get("source_tag")) for e in edges if isinstance(e.get("source_tag"), str)})
        mechanism_labels = [
            str(e.get("mechanism_label"))
            for e in edges
            if isinstance(e.get("mechanism_label"), str) and e.get("mechanism_label")
        ]
        mechanism_label = (
            Counter(mechanism_labels).most_common(1)[0][0]
            if mechanism_labels
            else f"adaptation related to {challenge.replace('_', ' ')}"
        )

        supports = len(species) >= 2 and len(context_types) >= 2
        summary.append(
            {
                "run_id": run_id,
                "challenge": challenge,
                "mechanism_label": mechanism_label,
                "species_support": len(species),
                "context_type_support": len(context_types),
                "source_support": len(sources),
                "species": species,
                "context_types": context_types,
                "supports_inferred_mechanism": supports,
            }
        )
        if not supports:
            continue

        label = mechanism_label
        source_list = sources or [None]
        for source in source_list:
            key = (run_id, challenge, source)
            if key in seen_mechanism:
                continue
            seen_mechanism.add(key)
            mechanism_edges.append(
                {
                    "run_id": run_id,
                    "species": None,
                    "section": "inferred_mechanism_hypotheses",
                    "trait_text": label,
                    "normalized_trait": normalize_trait(label),
                    "source_tag": source,
                    "confidence": 0.5,
                    "evidence_type": "inferred",
                    "derived_from_trait": ", ".join(context_types),
                    "inference_rule_id": "llm_cross_context_convergence",
                    "challenge": challenge,
                    "context_types": context_types,
                    "contradiction_flags": [],
                }
            )

    # Rank inferred mechanisms and keep only the strongest set (default max 7).
    mechanism_rank: dict[str, dict[str, Any]] = {}
    for row in summary:
        label = row.get("mechanism_label")
        if not isinstance(label, str) or not label.strip():
            continue
        key = normalize_trait(label)
        score = (
            (2.0 * float(row.get("species_support", 0)))
            + float(row.get("context_type_support", 0))
            + (0.5 * float(row.get("source_support", 0)))
        )
        state = mechanism_rank.setdefault(
            key,
            {
                "trait": label,
                "normalized_trait": key,
                "score": 0.0,
                "runs": set(),
                "supports_count": 0,
            },
        )
        state["score"] += score
        state["runs"].add(row.get("run_id"))
        if row.get("supports_inferred_mechanism"):
            state["supports_count"] += 1
        if len(label) < len(state["trait"]):
            state["trait"] = label

    ranked = sorted(
        (
            {
                "trait": v["trait"],
                "normalized_trait": v["normalized_trait"],
                "score": round(float(v["score"]), 3),
                "run_support": len(v["runs"]),
                "supports_count": int(v["supports_count"]),
            }
            for v in mechanism_rank.values()
        ),
        key=lambda x: (x["supports_count"], x["score"], x["run_support"]),
        reverse=True,
    )
    ranked_supported = [r for r in ranked if r["supports_count"] > 0]

    if max_inferred_traits < 1:
        max_inferred_traits = 1
    if min_inferred_traits < 0:
        min_inferred_traits = 0
    if min_inferred_traits > max_inferred_traits:
        min_inferred_traits = max_inferred_traits

    selected_ranked = ranked_supported[:max_inferred_traits]
    if len(selected_ranked) < min_inferred_traits and ranked_supported:
        selected_ranked = ranked_supported[: min(min_inferred_traits, len(ranked_supported))]
    selected_mechanisms = {r["normalized_trait"] for r in selected_ranked}
    if selected_mechanisms:
        mechanism_edges = [e for e in mechanism_edges if e.get("normalized_trait") in selected_mechanisms]
        context_edges = [
            e for e in context_edges
            if normalize_trait(str(e.get("mechanism_label", ""))) in selected_mechanisms
        ]
        filtered_trait_map: dict[str, list[dict[str, Any]]] = {}
        for trait_key, inferences in trait_inference_map.items():
            kept = [
                inf for inf in inferences
                if normalize_trait(str(inf.get("mechanism", ""))) in selected_mechanisms
            ]
            if kept:
                filtered_trait_map[trait_key] = kept
        trait_inference_map = filtered_trait_map
        summary = [
            row for row in summary
            if normalize_trait(str(row.get("mechanism_label", ""))) in selected_mechanisms
        ]
    else:
        mechanism_edges = []
        context_edges = []
        trait_inference_map = {}
        summary = []

    return {
        "trait_inference_map": trait_inference_map,
        "context_edges": context_edges,
        "mechanism_edges": mechanism_edges,
        "selected_inferred_traits": selected_ranked,
        "summary": sorted(summary, key=lambda x: (x["run_id"], x["challenge"])),
    }


def contradiction_hits_for_trait(candidate: str, all_traits: set[str]) -> list[str]:
    rules = [
        ("nocturnal", ["diurnal", "crepuscular"]),
        ("diurnal", ["nocturnal"]),
        ("aquatic", ["terrestrial"]),
        ("terrestrial", ["aquatic", "marine"]),
        ("high altitude", ["lowland", "sea level"]),
        ("low oxygen", ["normoxia", "high oxygen"]),
    ]
    hits: list[str] = []
    for token, opposites in rules:
        if token in candidate:
            for term in opposites:
                if any(term in t for t in all_traits):
                    hits.append(f"{token} vs {term}")
    return sorted(set(hits))


def mechanism_specificity_metrics(candidate: str) -> dict[str, Any]:
    generic_tokens = [
        "survival",
        "environment",
        "environments",
        "competitive",
        "adaptation",
        "adaptive",
        "resilience",
        "ecological",
        "conditions",
    ]
    function_markers = [
        "oxygen",
        "hypoxi",
        "respirat",
        "metabol",
        "thermal",
        "thermoreg",
        "osmotic",
        "osmoreg",
        "detox",
        "toxin",
        "immune",
        "energy",
        "cardio",
        "circulat",
        "hemoglobin",
        "myoglobin",
        "pressure",
        "buoyanc",
        "pigment",
        "vision",
        "auditory",
    ]
    generic_hits = sorted({g for g in generic_tokens if g in candidate})
    function_hits = sorted({f for f in function_markers if f in candidate})

    # Generic wording is acceptable, but should be penalized when no concrete functional signal exists.
    if function_hits:
        specificity_penalty = round(0.2 * len(generic_hits), 3)
    else:
        specificity_penalty = round(0.9 * len(generic_hits), 3)

    informative_patterns = [
        r"\b(regulation|tolerance|capacity|management|homeostasis)\b",
        r"\b(low[- ]oxygen|hypoxi)\b",
    ]
    pattern_hit = any(re.search(p, candidate) for p in informative_patterns)
    informative = bool(function_hits or pattern_hit)
    return {
        "generic_hits": generic_hits,
        "function_hits": function_hits,
        "specificity_penalty": specificity_penalty,
        "informative": informative,
    }


def score_mechanism_hypotheses(
    edges: list[dict],
    min_species_support: int = 2,
    min_source_support: int = 2,
) -> dict:
    mechanism_sections = {"mechanism_hypotheses", "inferred_mechanism_hypotheses"}
    mechanism_edges = [e for e in edges if e.get("section") in mechanism_sections]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for edge in mechanism_edges:
        key = edge.get("normalized_trait")
        if isinstance(key, str) and key:
            grouped[key].append(edge)

    source_to_species = build_source_to_species(edges)
    all_norm_traits = {str(e.get("normalized_trait")) for e in edges if isinstance(e.get("normalized_trait"), str)}

    candidates: list[dict] = []
    for normalized, group in grouped.items():
        labels = Counter(str(e.get("trait_text", normalized)) for e in group)
        trait_label = labels.most_common(1)[0][0]
        runs = sorted({str(e.get("run_id")) for e in group if isinstance(e.get("run_id"), str)})
        sources = sorted({str(e.get("source_tag")) for e in group if isinstance(e.get("source_tag"), str)})
        sections = sorted({str(e.get("section")) for e in group if isinstance(e.get("section"), str)})
        evidence_types = sorted({str(e.get("evidence_type")) for e in group if isinstance(e.get("evidence_type"), str)})
        species_support: set[str] = set()
        for src in sources:
            species_support.update(source_to_species.get(src, set()))

        contradictions = contradiction_hits_for_trait(normalized, all_norm_traits - {normalized})
        penalty = float(len(contradictions))
        specificity = mechanism_specificity_metrics(normalized)
        score_raw = (2.0 * len(species_support)) + (0.5 * len(sources)) + (1.0 * len(runs))
        score_final = round(score_raw - penalty - float(specificity["specificity_penalty"]), 3)
        meets_support = len(species_support) >= min_species_support and len(sources) >= min_source_support
        meets = meets_support and bool(specificity["informative"])

        candidates.append(
            {
                "trait": trait_label,
                "normalized_trait": normalized,
                "species_support": len(species_support),
                "species": sorted(species_support),
                "source_support": len(sources),
                "sources": sources,
                "run_support": len(runs),
                "runs": runs,
                "sections": sections,
                "evidence_types": evidence_types,
                "contradiction_hits": contradictions,
                "generic_hits": specificity["generic_hits"],
                "function_hits": specificity["function_hits"],
                "specificity_penalty": specificity["specificity_penalty"],
                "informative": specificity["informative"],
                "contradiction_penalty": penalty,
                "score_raw": round(score_raw, 3),
                "score_final": score_final,
                "meets_thresholds": meets,
                "meets_support_thresholds": meets_support,
            }
        )

    candidates.sort(
        key=lambda c: (
            c["score_final"],
            c["species_support"],
            c["source_support"],
            c["run_support"],
        ),
        reverse=True,
    )
    return {
        "thresholds": {
            "min_species_support": min_species_support,
            "min_source_support": min_source_support,
            "contradiction_penalty_per_hit": 1.0,
            "informative_mechanism_required": True,
        },
        "candidates": candidates,
        "ranked_passing": [c for c in candidates if c["meets_thresholds"]],
    }


def write_hypothesis_report(
    out_path: pathlib.Path,
    mechanism_scores: dict,
    analyze_report: dict,
    source_stats: dict,
) -> None:
    lines: list[str] = []
    lines.append("# Hypothesis Report v2")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Runs analyzed: {analyze_report.get('n_runs', 0)}")
    lines.append(f"- Traits in variance report: {len(analyze_report.get('trait_frequency', []))}")
    overall = source_stats.get("overall", {})
    lines.append(f"- Papers fetched: {overall.get('papers_fetched', 0)}")
    lines.append(f"- Papers used: {overall.get('papers_used', 0)}")
    lines.append(f"- Wiki fetched: {overall.get('wiki_fetched', 0)}")
    lines.append(f"- Wiki used: {overall.get('wiki_used', 0)}")
    lines.append("")
    lines.append("## Ranked Mechanism Hypotheses")
    passing = mechanism_scores.get("ranked_passing", [])
    if not passing:
        lines.append("- No mechanism hypotheses passed thresholds.")
    else:
        for i, item in enumerate(passing, start=1):
            lines.append(f"{i}. **{item['trait']}**")
            total_penalty = float(item.get("contradiction_penalty", 0.0)) + float(item.get("specificity_penalty", 0.0))
            lines.append(
                f"   - score: {item['score_final']} "
                f"(raw={item['score_raw']}, penalty={round(total_penalty, 3)})"
            )
            lines.append(
                f"   - support: species={item['species_support']}, "
                f"sources={item['source_support']}, runs={item['run_support']}"
            )
            if item.get("evidence_types"):
                lines.append(f"   - evidence types: {', '.join(item['evidence_types'])}")
            if item.get("sections"):
                lines.append(f"   - sections: {', '.join(item['sections'])}")
            if item.get("generic_hits"):
                lines.append(f"   - generic wording hits: {', '.join(item['generic_hits'])}")
            if item.get("function_hits"):
                lines.append(f"   - functional markers: {', '.join(item['function_hits'])}")
            if item["contradiction_hits"]:
                lines.append(f"   - contradictions: {', '.join(item['contradiction_hits'])}")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v2 pipeline: inventory runs + deterministic evidence/scoring under logs_v2."
    )
    species_group = parser.add_mutually_exclusive_group(required=True)
    species_group.add_argument("--species-file", help="Path to species JSON file.")
    species_group.add_argument("--species-list", help="Comma-separated scientific names.")
    parser.add_argument(
        "--generated-species-file",
        help="Output path for generated species JSON when using --species-list.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of inventory runs.")
    parser.add_argument("--reuse-traits", action="store_true", help="Reuse existing traits/<species>.json if present.")
    parser.add_argument(
        "--skip-ingest-after-first",
        action="store_true",
        default=True,
        help="Only ingest/download on the first run; subsequent runs skip ingestion (default: on).",
    )
    parser.add_argument(
        "--skip-ingest-all",
        action="store_true",
        help="Skip ingestion/download for all runs, including the first run.",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name for grouping.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature for grouping.")
    parser.add_argument(
        "--group-traits",
        action="store_true",
        help="Run trait grouping LLM stage and write trait_groups.json (default: off).",
    )
    parser.add_argument("--min-species-support", type=int, default=2, help="Minimum species support for mechanisms.")
    parser.add_argument("--min-source-support", type=int, default=2, help="Minimum unique sources for mechanisms.")
    parser.add_argument(
        "--min-inferred-traits",
        type=int,
        default=2,
        help="Soft minimum number of inferred mechanism traits to keep.",
    )
    parser.add_argument(
        "--max-inferred-traits",
        type=int,
        default=7,
        help="Maximum number of inferred mechanism traits to keep.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    label = slugify(pathlib.Path(args.species_file).stem) if args.species_file else "generated_species"
    bundle_dir = pathlib.Path("logs_v2") / f"{timestamp}-{label}"
    summary_dir = bundle_dir / "summary"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    species_file, species_input_mode = resolve_species_file(
        args.species_file,
        args.species_list,
        args.generated_species_file,
        bundle_dir / "species_input.json",
    )

    run_dirs_original = run_inventory_stage(
        species_file=species_file,
        runs=args.runs,
        reuse_traits=args.reuse_traits,
        skip_ingest_after_first=args.skip_ingest_after_first,
        skip_ingest_all=args.skip_ingest_all,
    )
    run_dirs_v2 = copy_run_logs_to_bundle(run_dirs_original, bundle_dir)

    run_list_path = bundle_dir / "run_list.txt"
    write_run_list(run_dirs_v2, run_list_path)
    print(f"Run list written to: {run_list_path}")

    run_files = [run_dir / "synthesis" / "synthesis_answer.txt" for run_dir in run_dirs_v2]
    analyze_report = analyze_run_files(run_files)
    analyze_report_path = summary_dir / "analyze_report.json"
    analyze_report_path.write_text(json.dumps(analyze_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {analyze_report_path}")

    trait_groups_path = summary_dir / "trait_groups.json"
    if args.group_traits:
        subprocess.run(
            [
                sys.executable,
                "group_traits_llm.py",
                str(analyze_report_path),
                "--model",
                args.model,
                "--temperature",
                str(args.temperature),
                "--out",
                str(trait_groups_path),
            ],
            check=True,
        )
    else:
        trait_groups_path.write_text(json.dumps({"groups": []}, ensure_ascii=False, indent=2), encoding="utf-8")

    explicit_edges = build_evidence_graph(run_dirs_v2)
    inference_layer = build_inference_layer(
        explicit_edges,
        model=args.model,
        temperature=args.temperature,
        min_inferred_traits=args.min_inferred_traits,
        max_inferred_traits=args.max_inferred_traits,
    )
    evidence_graph = explicit_edges + inference_layer["context_edges"] + inference_layer["mechanism_edges"]
    evidence_graph_path = summary_dir / "evidence_graph.json"
    evidence_graph_path.write_text(json.dumps(evidence_graph, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {evidence_graph_path}")

    inference_layer_path = summary_dir / "inference_layer.json"
    inference_layer_payload = {
        "trait_inference_map": inference_layer["trait_inference_map"],
        "context_edges_count": len(inference_layer["context_edges"]),
        "mechanism_edges_count": len(inference_layer["mechanism_edges"]),
        "selected_inferred_traits": inference_layer["selected_inferred_traits"],
        "summary": inference_layer["summary"],
    }
    inference_layer_path.write_text(
        json.dumps(inference_layer_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {inference_layer_path}")

    mechanism_scores = score_mechanism_hypotheses(
        edges=evidence_graph,
        min_species_support=args.min_species_support,
        min_source_support=args.min_source_support,
    )
    mechanism_scores_path = summary_dir / "mechanism_scores.json"
    mechanism_scores_path.write_text(json.dumps(mechanism_scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {mechanism_scores_path}")

    source_stats = aggregate_source_stats(run_dirs_v2)
    report_path = summary_dir / "hypothesis_report.md"
    write_hypothesis_report(report_path, mechanism_scores, analyze_report, source_stats)
    print(f"Wrote {report_path}")

    meta = {
        "timestamp": timestamp,
        "species_input_mode": species_input_mode,
        "species_file": str(species_file),
        "species_list": args.species_list if args.species_list else None,
        "runs_requested": args.runs,
        "runs_completed": len(run_dirs_v2),
        "reuse_traits": args.reuse_traits,
        "skip_ingest_after_first": args.skip_ingest_after_first,
        "skip_ingest_all": args.skip_ingest_all,
        "inventory_script": "inventory_single_2.py",
        "model": args.model,
        "temperature": args.temperature,
        "group_traits_enabled": args.group_traits,
        "thresholds": {
            "min_species_support": args.min_species_support,
            "min_source_support": args.min_source_support,
            "min_inferred_traits": args.min_inferred_traits,
            "max_inferred_traits": args.max_inferred_traits,
        },
        "run_dirs_original": [str(p) for p in run_dirs_original],
        "run_dirs_v2": [str(p) for p in run_dirs_v2],
        "artifacts": {
            "run_list": str(run_list_path),
            "analyze_report": str(analyze_report_path),
            "trait_groups": str(trait_groups_path),
            "evidence_graph": str(evidence_graph_path),
            "inference_layer": str(inference_layer_path),
            "mechanism_scores": str(mechanism_scores_path),
            "hypothesis_report": str(report_path),
        },
        "inference_layer_counts": {
            "explicit_edges": len(explicit_edges),
            "inferred_context_edges": len(inference_layer["context_edges"]),
            "inferred_mechanism_edges": len(inference_layer["mechanism_edges"]),
            "total_edges": len(evidence_graph),
        },
        "source_stats": source_stats,
    }
    meta_path = summary_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
