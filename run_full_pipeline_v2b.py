#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

from llm_backend import DEFAULT_CHAT_MODEL, make_chat_llm, make_embeddings

load_dotenv()

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

DEFAULT_PROPOSER_PROMPT = "Prompts/prompt_track_b_proposer_v2b.txt"
DEFAULT_VERIFIER_PROMPT = "Prompts/prompt_track_b_verifier_v2b.txt"

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "these",
    "those",
    "into",
    "their",
    "there",
    "where",
    "which",
    "while",
    "about",
    "among",
    "across",
    "species",
    "trait",
    "traits",
    "mechanism",
    "mechanisms",
    "adaptation",
    "adaptations",
    "animal",
    "animals",
    "biological",
    "biology",
    "evidence",
    "based",
    "common",
    "shared",
    "environment",
    "environments",
}


# -------------------------
# Generic helpers
# -------------------------


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
    payload = extract_json_from_text(text)
    return json.loads(payload)


def read_json_path(path: pathlib.Path) -> Any:
    return parse_json_text(path.read_text(encoding="utf-8"))


def load_prompt(path: str) -> str:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8")


def list_dirs(base: pathlib.Path) -> set[pathlib.Path]:
    if not base.exists():
        return set()
    return {p for p in base.iterdir() if p.is_dir()}


def detect_created_bundle(before: set[pathlib.Path], after: set[pathlib.Path], run_label: str) -> pathlib.Path:
    new_dirs = sorted(after - before)
    if len(new_dirs) == 1:
        return new_dirs[0]

    label_dirs = [p for p in after if p.name.endswith(f"-{run_label}")]
    if label_dirs:
        return max(label_dirs, key=lambda p: p.stat().st_mtime)

    if new_dirs:
        return max(new_dirs, key=lambda p: p.stat().st_mtime)

    if after:
        return max(after, key=lambda p: p.stat().st_mtime)

    raise RuntimeError("No logs_v2 bundle found after Track A run.")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def tokenize_for_similarity(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


def token_jaccard(a: str, b: str) -> float:
    ta = tokenize_for_similarity(a)
    tb = tokenize_for_similarity(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def llm_invoke_json(
    llm: Any,
    template_text: str,
    variables: dict[str, Any],
) -> Any:
    prompt = PromptTemplate(input_variables=list(variables.keys()), template=template_text)
    rendered = prompt.format(**variables)
    raw = llm.invoke(rendered)
    content = raw.content if hasattr(raw, "content") else raw
    return parse_json_text(str(content))


# -------------------------
# Track A orchestration
# -------------------------


def build_track_a_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "run_full_pipeline_v2.py"]
    if args.species_file:
        cmd += ["--species-file", args.species_file]
    else:
        cmd += ["--species-list", args.species_list]

    if args.generated_species_file:
        cmd += ["--generated-species-file", args.generated_species_file]

    cmd += [
        "--runs",
        str(args.runs),
        "--model",
        args.model,
        "--temperature",
        str(args.temperature),
        "--min-species-support",
        str(args.min_species_support),
        "--min-source-support",
        str(args.min_source_support),
        "--min-inferred-traits",
        str(args.min_inferred_traits),
        "--max-inferred-traits",
        str(args.max_inferred_traits),
    ]

    if args.reuse_traits:
        cmd.append("--reuse-traits")
    if args.skip_ingest_after_first:
        cmd.append("--skip-ingest-after-first")
    if args.skip_ingest_all:
        cmd.append("--skip-ingest-all")
    if args.group_traits:
        cmd.append("--group-traits")

    return cmd


def run_track_a_and_get_bundle(args: argparse.Namespace) -> pathlib.Path:
    logs_v2 = pathlib.Path("logs_v2")
    before = list_dirs(logs_v2)

    cmd = build_track_a_cmd(args)
    print(f"[v2b] Running Track A: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    after = list_dirs(logs_v2)
    run_label = slugify(pathlib.Path(args.species_file).stem) if args.species_file else "generated_species"
    bundle = detect_created_bundle(before, after, run_label)
    print(f"[v2b] Track A bundle: {bundle}")
    return bundle


# -------------------------
# Bundle parsing
# -------------------------


def load_meta(bundle_dir: pathlib.Path) -> dict[str, Any]:
    meta_path = bundle_dir / "summary" / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing Track A meta: {meta_path}")
    data = read_json_path(meta_path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid meta JSON: {meta_path}")
    return data


def load_species_entries(meta: dict[str, Any], bundle_dir: pathlib.Path) -> list[dict[str, Any]]:
    species_file_value = meta.get("species_file")
    if isinstance(species_file_value, str) and species_file_value:
        species_file = pathlib.Path(species_file_value)
        if not species_file.is_absolute():
            species_file = pathlib.Path.cwd() / species_file
        if species_file.exists():
            data = read_json_path(species_file)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]

    generated_file = bundle_dir / "species_input.json"
    if generated_file.exists():
        data = read_json_path(generated_file)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]

    return []


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


def read_run_dirs(bundle_dir: pathlib.Path) -> list[pathlib.Path]:
    run_list = bundle_dir / "run_list.txt"
    if run_list.exists():
        out: list[pathlib.Path] = []
        for line in run_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = pathlib.Path(line)
            if p.exists() and p.is_dir():
                out.append(p)
        if out:
            return out

    return sorted([p for p in bundle_dir.glob("run_*") if p.is_dir()])


def list_species_dirs(run_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name != "synthesis"])


def iter_synthesis_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        out: list[dict[str, Any]] = []
        for section in ("strict_common_traits", "subgroup_common_traits", "mechanism_hypotheses"):
            items = data.get(section)
            if isinstance(items, list):
                out.extend([x for x in items if isinstance(x, dict)])
        return out

    return []


def load_inventory_traits(answer_path: pathlib.Path) -> list[dict[str, Any]]:
    if not answer_path.exists():
        return []
    try:
        data = read_json_path(answer_path)
    except Exception:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def build_trait_digest_for_run(run_dir: pathlib.Path, max_items: int = 30) -> list[dict[str, Any]]:
    mentions = Counter()
    species_support: dict[str, set[str]] = defaultdict(set)
    label_map: dict[str, str] = {}

    for species_dir in list_species_dirs(run_dir):
        species_slug = species_dir.name
        for item in load_inventory_traits(species_dir / "answer.txt"):
            trait = item.get("trait")
            if not isinstance(trait, str) or not trait.strip():
                continue
            key = normalize_text(trait)
            mentions[key] += 1
            species_support[key].add(species_slug)
            best = label_map.get(key)
            if not best or len(trait.strip()) < len(best):
                label_map[key] = trait.strip()

    syn_path = run_dir / "synthesis" / "synthesis_answer.txt"
    if syn_path.exists():
        try:
            syn_data = read_json_path(syn_path)
            for item in iter_synthesis_items(syn_data):
                trait = item.get("trait")
                if not isinstance(trait, str) or not trait.strip():
                    continue
                key = normalize_text(trait)
                mentions[key] += 1
                best = label_map.get(key)
                if not best or len(trait.strip()) < len(best):
                    label_map[key] = trait.strip()
        except Exception:
            pass

    rows: list[dict[str, Any]] = []
    for key, count in mentions.items():
        rows.append(
            {
                "trait": label_map.get(key, key),
                "species_support": len(species_support.get(key, set())),
                "mentions": int(count),
            }
        )

    rows.sort(key=lambda x: (x["species_support"], x["mentions"], x["trait"]), reverse=True)
    return rows[:max_items]


def build_track_a_trait_records(
    run_dir: pathlib.Path,
    slug_to_norm: dict[str, str],
    slug_to_label: dict[str, str],
    max_traits_per_species: int = 80,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    record_id = 1

    for species_dir in list_species_dirs(run_dir):
        species_slug = species_dir.name
        species_norm = slug_to_norm.get(species_slug, species_slug.replace("_", " ").strip().lower())
        species_label = slug_to_label.get(species_slug, species_slug.replace("_", " ").strip())
        items = load_inventory_traits(species_dir / "answer.txt")[:max_traits_per_species]

        for item in items:
            trait = item.get("trait")
            if not isinstance(trait, str) or not trait.strip():
                continue
            sources = item.get("sources")
            source_list: list[str] = []
            if isinstance(sources, list):
                for src in sources:
                    if isinstance(src, str) and src.strip():
                        source_list.append(src.strip())
            source_list = sorted(set(source_list))
            confidence = item.get("confidence")
            try:
                conf = float(confidence)
            except Exception:
                conf = 0.5
            conf = max(0.0, min(1.0, conf))

            records.append(
                {
                    "trait_id": f"T{record_id:04d}",
                    "species_norm": species_norm,
                    "species": species_label,
                    "trait": " ".join(trait.split()).strip(),
                    "citations": source_list,
                    "confidence": conf,
                }
            )
            record_id += 1

    return records


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


# -------------------------
# Track B retrieval + verify
# -------------------------


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


def get_all_chunks_for_species(
    vectorstore: Chroma,
    species_norm: str,
    allow_refs: set[str],
) -> list[dict[str, Any]]:
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
    rows_unfiltered: list[dict[str, Any]] = []

    for i in range(n):
        text = docs[i]
        meta = metas[i] if isinstance(metas[i], dict) else {}
        if not isinstance(text, str) or not text.strip():
            continue

        citation = citation_from_meta(meta)
        row = {
            "species": species_norm,
            "citation": citation,
            "text": text.strip(),
            "distance": None,
        }
        rows_unfiltered.append(row)

        if allow_norm:
            keys = metadata_ref_keys(meta)
            if not (keys & allow_norm):
                continue
        rows.append(row)

    if rows:
        return rows
    if allow_norm:
        return rows_unfiltered
    return rows


def build_query_variants(mechanism: str, rationale: str) -> list[str]:
    queries: list[str] = []
    for q in (
        mechanism,
        rationale,
        f"{mechanism} physiology ecology",
        f"evidence for {mechanism}",
    ):
        q = " ".join(q.split()).strip()
        if q and q not in queries:
            queries.append(q)
    return queries


def semantic_hits_for_species(
    vectorstore: Chroma,
    species_norm: str,
    queries: list[str],
    allow_refs: set[str],
    k_per_query: int,
    max_keep: int,
) -> list[dict[str, Any]]:
    allow_norm: set[str] = set()
    for ref in allow_refs:
        allow_norm.update(normalize_ref(ref))

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

        for doc, score in results:
            meta = doc.metadata or {}
            if allow_norm:
                keys = metadata_ref_keys(meta)
                if not (keys & allow_norm):
                    continue

            citation = citation_from_meta(meta)
            row = {
                "species": species_norm,
                "citation": citation,
                "text": (doc.page_content or "").strip(),
                "distance": float(score),
            }
            prev = best.get(citation)
            if prev is None or float(score) < float(prev.get("distance", 1e9)):
                best[citation] = row

    ranked = sorted(best.values(), key=lambda x: float(x.get("distance", 1e9)))
    return ranked[:max_keep]


def lexical_terms(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def lexical_hits_from_all_chunks(
    all_chunks: list[dict[str, Any]],
    mechanism: str,
    rationale: str,
    max_keep: int,
) -> list[dict[str, Any]]:
    terms = set(lexical_terms(mechanism + " " + rationale))
    if not terms:
        return []

    scored: list[tuple[int, dict[str, Any]]] = []
    for row in all_chunks:
        text = str(row.get("text", "")).lower()
        if not text:
            continue
        overlap = sum(1 for t in terms if t in text)
        if overlap <= 0:
            continue
        scored.append((overlap, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:max_keep]]


def build_evidence_chunks_for_proposal(
    vectorstore: Chroma,
    species_norms: list[str],
    allowlists: dict[str, set[str]],
    mechanism: str,
    rationale: str,
    k_per_query: int,
    max_chunks_per_species: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    queries = build_query_variants(mechanism, rationale)
    combined: list[dict[str, Any]] = []
    per_species_counts: dict[str, int] = {}

    for species_norm in species_norms:
        allow_refs = allowlists.get(species_norm, set())
        all_chunks = get_all_chunks_for_species(vectorstore, species_norm, allow_refs)

        semantic = semantic_hits_for_species(
            vectorstore,
            species_norm,
            queries,
            allow_refs,
            k_per_query=k_per_query,
            max_keep=max_chunks_per_species,
        )
        lexical = lexical_hits_from_all_chunks(
            all_chunks,
            mechanism,
            rationale,
            max_keep=max_chunks_per_species,
        )

        dedup: dict[str, dict[str, Any]] = {}
        for row in semantic + lexical:
            citation = str(row.get("citation"))
            prev = dedup.get(citation)
            if prev is None:
                dedup[citation] = row
            else:
                prev_dist = prev.get("distance")
                this_dist = row.get("distance")
                if isinstance(this_dist, (int, float)) and (
                    prev_dist is None or (isinstance(prev_dist, (int, float)) and this_dist < prev_dist)
                ):
                    dedup[citation] = row

        picked = list(dedup.values())
        picked.sort(key=lambda x: (x.get("distance") is None, x.get("distance", 1e9)))
        if not picked and all_chunks:
            # fallback: provide minimal context for this species
            picked = all_chunks[: min(2, len(all_chunks))]

        picked = picked[:max_chunks_per_species]
        per_species_counts[species_norm] = len(picked)
        combined.extend(picked)

    return combined, per_species_counts


def sanitize_proposals(raw: Any, min_n: int, max_n: int) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    proposals = raw.get("proposals")
    if not isinstance(proposals, list):
        return []

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in proposals:
        if not isinstance(item, dict):
            continue
        mechanism = item.get("mechanism")
        rationale = item.get("rationale", "")
        confidence = item.get("confidence_prior", 0.5)

        if not isinstance(mechanism, str) or not mechanism.strip():
            continue
        if not isinstance(rationale, str):
            rationale = ""

        key = normalize_text(mechanism)
        if key in seen:
            continue
        seen.add(key)

        try:
            conf = float(confidence)
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        out.append(
            {
                "mechanism": " ".join(mechanism.split()).strip(),
                "rationale": " ".join(rationale.split()).strip(),
                "confidence_prior": conf,
            }
        )

    out.sort(key=lambda x: x["confidence_prior"], reverse=True)
    out = out[:max(1, max_n)]
    if out:
        if len(out) < min_n:
            return out
        return out

    # Fallback to keep Track B operable even when proposer JSON is malformed.
    return [
        {
            "mechanism": "shared adaptation to recurring ecological constraints",
            "rationale": "fallback proposal due to invalid proposer output",
            "confidence_prior": 0.1,
        }
    ][: max(1, min(max_n, min_n))]


def sanitize_verifier_result(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "supporting_trait_ids": [],
            "contradicting_trait_ids": [],
            "premise_groups": [],
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

    premise_groups_in = raw.get("premise_groups")
    premise_groups_out: list[dict[str, Any]] = []
    if isinstance(premise_groups_in, list):
        for group in premise_groups_in:
            if not isinstance(group, dict):
                continue
            premise = group.get("premise")
            if not isinstance(premise, str) or not premise.strip():
                continue
            trait_ids = clean_str_list(group.get("trait_ids"))
            premise_groups_out.append(
                {
                    "premise": " ".join(premise.split()).strip(),
                    "trait_ids": trait_ids,
                }
            )

    return {
        "supporting_trait_ids": clean_str_list(raw.get("supporting_trait_ids")),
        "contradicting_trait_ids": clean_str_list(raw.get("contradicting_trait_ids")),
        "premise_groups": premise_groups_out,
        "verdict_note": str(raw.get("verdict_note", "")).strip(),
    }


def mechanism_specificity_penalty(mechanism: str) -> float:
    candidate = mechanism.lower()
    generic_tokens = {
        "survival",
        "environment",
        "environments",
        "competitive",
        "adaptation",
        "adaptive",
        "resilience",
        "ecological",
        "conditions",
        "general",
        "versatility",
    }
    function_markers = {
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
        "vision",
        "auditory",
        "pigment",
    }

    generic_hits = sum(1 for token in generic_tokens if token in candidate)
    function_hits = sum(1 for token in function_markers if token in candidate)

    if function_hits > 0:
        penalty = 0.2 * generic_hits
    else:
        penalty = 0.9 * generic_hits

    return round(float(penalty), 3)


def classify_proposal_status(
    species_support: int,
    source_support: int,
    premise_type_support: int,
    contradiction_species: int,
    min_species_support: int,
    min_source_support: int,
    min_premise_types: int,
) -> str:
    # Contradiction dominance: reject when opposition is as large as support.
    if contradiction_species > 0 and contradiction_species >= max(1, species_support):
        return "rejected"

    if (
        species_support >= min_species_support
        and source_support >= min_source_support
        and premise_type_support >= min_premise_types
    ):
        return "supported"

    if species_support > 0 and source_support > 0:
        return "weak"

    return "rejected"


def score_track_b_candidate(
    species_support: int,
    run_support: int,
    source_support: int,
    contradiction: int,
    specificity_penalty: float,
) -> float:
    score = (
        (2.0 * species_support)
        + (1.0 * run_support)
        + (0.5 * source_support)
        - (1.0 * contradiction)
        - (0.5 * float(specificity_penalty))
    )
    return round(float(score), 3)


# -------------------------
# Track B aggregation + merge
# -------------------------


def cluster_mechanisms(
    labels: list[str],
    embed_threshold: float,
    token_threshold: float,
) -> dict[str, str]:
    if not labels:
        return {}

    unique = list(dict.fromkeys(labels))
    mapping: dict[str, str] = {}

    try:
        embedder = make_embeddings()
        vectors = embedder.embed_documents(unique)
        arr = {label: np.array(vec, dtype=float) for label, vec in zip(unique, vectors)}
    except Exception:
        arr = {}

    clusters: list[dict[str, Any]] = []
    for label in unique:
        assigned = False
        for cluster in clusters:
            rep = cluster["rep"]
            sim = 0.0
            if label in arr and rep in arr:
                sim = cosine_similarity(arr[label], arr[rep])
            tok = token_jaccard(label, rep)
            if sim >= embed_threshold or tok >= token_threshold:
                cluster["members"].append(label)
                assigned = True
                break
        if not assigned:
            clusters.append({"rep": label, "members": [label]})

    for cluster in clusters:
        members = cluster["members"]
        canonical = min(members, key=lambda x: (len(x), x))
        for m in members:
            mapping[m] = canonical

    return mapping


def aggregate_verified_across_runs(
    per_run_verified: list[dict[str, Any]],
    min_species_support: int,
    min_source_support: int,
    min_premise_types: int,
    embed_threshold: float,
    token_threshold: float,
) -> dict[str, Any]:
    labels = [str(item["mechanism"]) for item in per_run_verified if isinstance(item.get("mechanism"), str)]
    canonical_map = cluster_mechanisms(labels, embed_threshold=embed_threshold, token_threshold=token_threshold)

    grouped: dict[str, dict[str, Any]] = {}
    for item in per_run_verified:
        mechanism = item.get("mechanism")
        if not isinstance(mechanism, str) or not mechanism.strip():
            continue
        canonical = canonical_map.get(mechanism, mechanism)

        state = grouped.setdefault(
            canonical,
            {
                "mechanism": canonical,
                "aliases": set(),
                "supporting_citations": set(),
                "contradicting_citations": set(),
                "supporting_species": set(),
                "contradicting_species": set(),
                "run_support": set(),
                "statuses": [],
                "evidence_chain": [],
                "premise_types": set(),
            },
        )

        state["aliases"].add(mechanism)
        state["statuses"].append(str(item.get("status", "rejected")))

        for c in item.get("supporting_citations", []):
            if isinstance(c, str) and c:
                state["supporting_citations"].add(c)
        for c in item.get("contradicting_citations", []):
            if isinstance(c, str) and c:
                state["contradicting_citations"].add(c)
        for s in item.get("supporting_species", []):
            if isinstance(s, str) and s:
                state["supporting_species"].add(s)
        for s in item.get("contradicting_species", []):
            if isinstance(s, str) and s:
                state["contradicting_species"].add(s)

        if item.get("status") in {"supported", "weak"}:
            run_id = item.get("run_id")
            if isinstance(run_id, str) and run_id:
                state["run_support"].add(run_id)

        for e in item.get("evidence_chain", []):
            if isinstance(e, dict):
                state["evidence_chain"].append(e)
                premise = e.get("premise_type")
                if isinstance(premise, str) and premise.strip():
                    state["premise_types"].add(premise.strip())

    verified: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for mechanism, state in grouped.items():
        species_support = len(state["supporting_species"])
        source_support = len(state["supporting_citations"])
        run_support = len(state["run_support"])
        contradiction = len(state["contradicting_species"])
        premise_type_support = len(state["premise_types"])
        specificity = mechanism_specificity_penalty(mechanism)

        status = classify_proposal_status(
            species_support=species_support,
            source_support=source_support,
            premise_type_support=premise_type_support,
            contradiction_species=contradiction,
            min_species_support=min_species_support,
            min_source_support=min_source_support,
            min_premise_types=min_premise_types,
        )

        score = score_track_b_candidate(
            species_support=species_support,
            run_support=run_support,
            source_support=source_support,
            contradiction=contradiction,
            specificity_penalty=specificity,
        )

        payload = {
            "mechanism": mechanism,
            "aliases": sorted(state["aliases"]),
            "status": status,
            "species_support_count": species_support,
            "source_support_count": source_support,
            "run_support_count": run_support,
            "premise_type_support_count": premise_type_support,
            "premise_types": sorted(state["premise_types"]),
            "supporting_species": sorted(state["supporting_species"]),
            "contradicting_species": sorted(state["contradicting_species"]),
            "supporting_citations": sorted(state["supporting_citations"]),
            "contradicting_citations": sorted(state["contradicting_citations"]),
            "contradiction_penalty": contradiction,
            "specificity_penalty": specificity,
            "score_final": score,
            "evidence_chain": state["evidence_chain"],
        }

        if status == "rejected":
            rejected.append(
                {
                    "mechanism": mechanism,
                    "reason": "insufficient support|contradicted",
                    "details": payload,
                }
            )
        else:
            verified.append(payload)

    verified.sort(key=lambda x: (x["score_final"], x["species_support_count"], x["source_support_count"]), reverse=True)
    rejected.sort(key=lambda x: x["mechanism"])
    return {
        "verified": verified,
        "rejected": rejected,
    }


def load_track_a_mechanisms(summary_dir: pathlib.Path) -> list[dict[str, Any]]:
    path = summary_dir / "mechanism_scores.json"
    if not path.exists():
        return []
    try:
        data = read_json_path(path)
    except Exception:
        return []

    if not isinstance(data, dict):
        return []

    ranked = data.get("ranked_passing")
    if isinstance(ranked, list) and ranked:
        return [r for r in ranked if isinstance(r, dict)]

    candidates = data.get("candidates")
    if isinstance(candidates, list):
        return [r for r in candidates if isinstance(r, dict)][:10]

    return []


def match_track_b_to_track_a(
    track_a: list[dict[str, Any]],
    track_b: list[dict[str, Any]],
    embed_threshold: float,
    token_threshold: float,
) -> dict[int, int]:
    if not track_a or not track_b:
        return {}

    a_labels = [str(a.get("trait", "")).strip() for a in track_a]
    b_labels = [str(b.get("mechanism", "")).strip() for b in track_b]

    vectors: dict[str, np.ndarray] = {}
    all_labels = [x for x in a_labels + b_labels if x]
    try:
        embedder = make_embeddings()
        embs = embedder.embed_documents(all_labels)
        vectors = {label: np.array(vec, dtype=float) for label, vec in zip(all_labels, embs)}
    except Exception:
        vectors = {}

    out: dict[int, int] = {}
    used_a: set[int] = set()

    for b_idx, b_label in enumerate(b_labels):
        if not b_label:
            continue
        best = (-1, 0.0, 0.0, 0.0)
        for a_idx, a_label in enumerate(a_labels):
            if a_idx in used_a or not a_label:
                continue

            sim = 0.0
            if b_label in vectors and a_label in vectors:
                sim = cosine_similarity(vectors[b_label], vectors[a_label])
            tok = token_jaccard(b_label, a_label)
            score = max(sim, tok)
            if score > best[1]:
                best = (a_idx, score, sim, tok)

        if best[0] >= 0 and (best[2] >= embed_threshold or best[3] >= token_threshold):
            out[b_idx] = best[0]
            used_a.add(best[0])

    return out


def merge_track_a_and_track_b(
    track_a: list[dict[str, Any]],
    track_b_verified: list[dict[str, Any]],
    embed_threshold: float,
    token_threshold: float,
) -> dict[str, Any]:
    merged: list[dict[str, Any]] = []

    match_map = match_track_b_to_track_a(
        track_a,
        track_b_verified,
        embed_threshold=embed_threshold,
        token_threshold=token_threshold,
    )

    used_a: set[int] = set()
    for b_idx, b in enumerate(track_b_verified):
        if b_idx in match_map:
            a_idx = match_map[b_idx]
            used_a.add(a_idx)
            a = track_a[a_idx]

            citations = sorted(
                set(a.get("sources", []) if isinstance(a.get("sources"), list) else [])
                | set(b.get("supporting_citations", []) if isinstance(b.get("supporting_citations"), list) else [])
            )
            a_score = float(a.get("score_final", 0.0))
            b_score = float(b.get("score_final", 0.0))

            merged.append(
                {
                    "mechanism": b.get("mechanism") or a.get("trait"),
                    "source_track": "A+B",
                    "score": round(a_score + b_score + 1.0, 3),
                    "supporting_citations": citations,
                    "notes": (
                        f"Track A score={round(a_score, 3)}; "
                        f"Track B status={b.get('status')} score={round(b_score, 3)}"
                    ),
                    "track_a_trait": a.get("trait"),
                    "track_b_mechanism": b.get("mechanism"),
                    "track_b_status": b.get("status"),
                }
            )
        else:
            citations = b.get("supporting_citations", []) if isinstance(b.get("supporting_citations"), list) else []
            merged.append(
                {
                    "mechanism": b.get("mechanism"),
                    "source_track": "B",
                    "score": float(b.get("score_final", 0.0)),
                    "supporting_citations": citations,
                    "notes": f"Track B status={b.get('status')}",
                    "track_b_status": b.get("status"),
                }
            )

    for a_idx, a in enumerate(track_a):
        if a_idx in used_a:
            continue
        citations = a.get("sources", []) if isinstance(a.get("sources"), list) else []
        merged.append(
            {
                "mechanism": a.get("trait"),
                "source_track": "A",
                "score": float(a.get("score_final", 0.0)),
                "supporting_citations": citations,
                "notes": "Track A only",
            }
        )

    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return {"merged_ranked": merged}


# -------------------------
# Track B execution
# -------------------------


def run_track_b(bundle_dir: pathlib.Path, args: argparse.Namespace) -> dict[str, Any]:
    summary_dir = bundle_dir / "summary"
    track_b_dir = summary_dir / "track_b"
    track_b_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(bundle_dir)
    species_entries = load_species_entries(meta, bundle_dir)
    slug_to_norm, slug_to_label = build_species_maps(species_entries)

    proposer_prompt = load_prompt(args.track_b_proposer_prompt)
    verifier_prompt = load_prompt(args.track_b_verifier_prompt)

    llm = make_chat_llm(model=args.model, temperature=args.temperature)

    run_dirs = read_run_dirs(bundle_dir)
    if not run_dirs:
        raise RuntimeError(f"No run directories found in bundle: {bundle_dir}")

    proposals_per_run: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        species_dirs = list_species_dirs(run_dir)
        species_labels = [slug_to_label.get(s.name, s.name.replace("_", " ")) for s in species_dirs]
        track_a_trait_records = build_track_a_trait_records(
            run_dir=run_dir,
            slug_to_norm=slug_to_norm,
            slug_to_label=slug_to_label,
        )

        try:
            raw = llm_invoke_json(
                llm=llm,
                template_text=proposer_prompt,
                variables={
                    "min_proposals": args.track_b_min_proposals,
                    "max_proposals": args.track_b_max_proposals,
                    "species_list": json.dumps(species_labels, ensure_ascii=False, indent=2),
                },
            )
        except Exception:
            raw = {}
        proposals = sanitize_proposals(raw, args.track_b_min_proposals, args.track_b_max_proposals)

        proposals_per_run.append(
            {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "species": species_labels,
                "track_a_trait_count": len(track_a_trait_records),
                "proposals": proposals,
            }
        )

    proposal_artifact = {
        "model": args.model,
        "temperature": args.temperature,
        "proposal_input_mode": "species_only",
        "proposal_count_range": [args.track_b_min_proposals, args.track_b_max_proposals],
        "per_run": proposals_per_run,
        "all_proposals": [
            {
                "run_id": row["run_id"],
                **proposal,
            }
            for row in proposals_per_run
            for proposal in row.get("proposals", [])
        ],
    }
    (track_b_dir / "zero_shot_proposals.json").write_text(
        json.dumps(proposal_artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    verified_per_run: list[dict[str, Any]] = []

    for row in proposals_per_run:
        run_dir = pathlib.Path(row["run_dir"])
        run_id = str(row["run_id"])
        species_norms = sorted(
            {
                slug_to_norm.get(species_dir.name, species_dir.name.replace("_", " ").strip().lower())
                for species_dir in list_species_dirs(run_dir)
            }
        )
        track_a_trait_records = build_track_a_trait_records(
            run_dir=run_dir,
            slug_to_norm=slug_to_norm,
            slug_to_label=slug_to_label,
        )
        trait_by_id = {
            str(record.get("trait_id")): record
            for record in track_a_trait_records
            if isinstance(record.get("trait_id"), str)
        }

        for proposal in row.get("proposals", []):
            mechanism = str(proposal.get("mechanism", "")).strip()
            rationale = str(proposal.get("rationale", "")).strip()
            if not mechanism:
                continue

            try:
                verifier_raw = llm_invoke_json(
                    llm=llm,
                    template_text=verifier_prompt,
                    variables={
                        "mechanism": mechanism,
                        "rationale": rationale,
                        "species_list": json.dumps(species_norms, ensure_ascii=False, indent=2),
                        "trait_records": json.dumps(track_a_trait_records, ensure_ascii=False, indent=2),
                    },
                )
            except Exception:
                verifier_raw = {}
            verifier = sanitize_verifier_result(verifier_raw)

            supporting_trait_ids = [
                trait_id
                for trait_id in verifier["supporting_trait_ids"]
                if trait_id in trait_by_id
            ]
            contradicting_trait_ids = [
                trait_id
                for trait_id in verifier["contradicting_trait_ids"]
                if trait_id in trait_by_id
            ]

            supporting_traits = [trait_by_id[tid] for tid in supporting_trait_ids]
            contradicting_traits = [trait_by_id[tid] for tid in contradicting_trait_ids]

            supporting_species = sorted(
                {
                    str(item.get("species_norm"))
                    for item in supporting_traits
                    if isinstance(item.get("species_norm"), str)
                }
            )
            contradicting_species = sorted(
                {
                    str(item.get("species_norm"))
                    for item in contradicting_traits
                    if isinstance(item.get("species_norm"), str)
                }
            )

            supporting_citations = sorted(
                {
                    src
                    for item in supporting_traits
                    for src in (item.get("citations") if isinstance(item.get("citations"), list) else [])
                    if isinstance(src, str) and src
                }
            )
            contradicting_citations = sorted(
                {
                    src
                    for item in contradicting_traits
                    for src in (item.get("citations") if isinstance(item.get("citations"), list) else [])
                    if isinstance(src, str) and src
                }
            )

            premise_types: set[str] = set()
            premise_type_by_trait_id: dict[str, str] = {}
            for group in verifier.get("premise_groups", []):
                if not isinstance(group, dict):
                    continue
                premise = group.get("premise")
                if not isinstance(premise, str) or not premise.strip():
                    continue
                normalized_premise = " ".join(premise.split()).strip()
                trait_ids = group.get("trait_ids")
                if not isinstance(trait_ids, list):
                    continue
                valid_trait_ids = [tid for tid in trait_ids if isinstance(tid, str) and tid in supporting_trait_ids]
                if not valid_trait_ids:
                    continue
                premise_types.add(normalized_premise)
                for tid in valid_trait_ids:
                    premise_type_by_trait_id[tid] = normalized_premise

            status = classify_proposal_status(
                species_support=len(supporting_species),
                source_support=len(set(supporting_citations)),
                premise_type_support=len(premise_types),
                contradiction_species=len(contradicting_species),
                min_species_support=args.track_b_min_species_support,
                min_source_support=args.track_b_min_source_support,
                min_premise_types=args.track_b_min_premise_types,
            )

            run_support = 1 if status in {"supported", "weak"} else 0
            specificity_penalty = mechanism_specificity_penalty(mechanism)
            score_final = score_track_b_candidate(
                species_support=len(supporting_species),
                run_support=run_support,
                source_support=len(set(supporting_citations)),
                contradiction=len(contradicting_species),
                specificity_penalty=specificity_penalty,
            )

            evidence_chain = []
            for trait in supporting_traits:
                citations = trait.get("citations") if isinstance(trait.get("citations"), list) else []
                premise_label = premise_type_by_trait_id.get(str(trait.get("trait_id")))
                for citation in citations:
                    if not isinstance(citation, str) or not citation.strip():
                        continue
                    evidence_chain.append(
                        {
                            "species": trait.get("species"),
                            "premise_trait": trait.get("trait"),
                            "premise_type": premise_label,
                            "citation": citation.strip(),
                        }
                    )

            verified_per_run.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "mechanism": mechanism,
                    "rationale": rationale,
                    "confidence_prior": proposal.get("confidence_prior", 0.5),
                    "status": status,
                    "species_support_count": len(supporting_species),
                    "source_support_count": len(set(supporting_citations)),
                    "premise_type_support_count": len(premise_types),
                    "premise_types": sorted(premise_types),
                    "run_support_count": run_support,
                    "supporting_species": supporting_species,
                    "contradicting_species": contradicting_species,
                    "supporting_trait_ids": supporting_trait_ids,
                    "contradicting_trait_ids": contradicting_trait_ids,
                    "supporting_citations": sorted(set(supporting_citations)),
                    "contradicting_citations": sorted(set(contradicting_citations)),
                    "contradiction_penalty": len(contradicting_species),
                    "specificity_penalty": specificity_penalty,
                    "score_final": score_final,
                    "verdict_note": verifier.get("verdict_note", ""),
                    "evidence_chain": evidence_chain,
                    "verification_context": {
                        "species_norms": species_norms,
                        "track_a_trait_record_count": len(track_a_trait_records),
                    },
                }
            )

    aggregated = aggregate_verified_across_runs(
        per_run_verified=verified_per_run,
        min_species_support=args.track_b_min_species_support,
        min_source_support=args.track_b_min_source_support,
        min_premise_types=args.track_b_min_premise_types,
        embed_threshold=args.track_b_embed_merge_threshold,
        token_threshold=args.track_b_token_merge_threshold,
    )

    verified_artifact = {
        "model": args.model,
        "temperature": args.temperature,
        "thresholds": {
            "min_species_support": args.track_b_min_species_support,
            "min_source_support": args.track_b_min_source_support,
            "min_premise_types": args.track_b_min_premise_types,
            "score_formula": "2*species_support + 1*run_support + 0.5*source_support - 1*contradiction - 0.5*specificity_penalty",
        },
        "per_run": verified_per_run,
        "verified": aggregated["verified"],
        "rejected": aggregated["rejected"],
    }
    (track_b_dir / "zero_shot_verified.json").write_text(
        json.dumps(verified_artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    track_a = load_track_a_mechanisms(summary_dir)
    merged = merge_track_a_and_track_b(
        track_a=track_a,
        track_b_verified=aggregated["verified"],
        embed_threshold=args.track_b_embed_merge_threshold,
        token_threshold=args.track_b_token_merge_threshold,
    )
    (track_b_dir / "hypothesis_merged.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    track_b_meta = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "bundle_dir": str(bundle_dir),
        "run_count": len(run_dirs),
        "proposals_total": sum(len(x.get("proposals", [])) for x in proposals_per_run),
        "verified_total": len(aggregated["verified"]),
        "rejected_total": len(aggregated["rejected"]),
        "artifacts": {
            "zero_shot_proposals": str(track_b_dir / "zero_shot_proposals.json"),
            "zero_shot_verified": str(track_b_dir / "zero_shot_verified.json"),
            "hypothesis_merged": str(track_b_dir / "hypothesis_merged.json"),
        },
    }
    (track_b_dir / "meta.json").write_text(json.dumps(track_b_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return track_b_meta


# -------------------------
# CLI
# -------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "v2b pipeline: run Track A via run_full_pipeline_v2.py, then Track B "
            "(zero-shot proposals + citation-grounded verification + merge)."
        )
    )

    species_group = parser.add_mutually_exclusive_group(required=True)
    species_group.add_argument("--species-file", help="Path to species testcase JSON.")
    species_group.add_argument("--species-list", help="Comma-separated scientific names.")

    parser.add_argument("--generated-species-file", help="Output path for generated species file (when using --species-list).")

    # Track A passthrough options
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--reuse-traits", action="store_true")
    parser.add_argument(
        "--skip-ingest-after-first",
        action="store_true",
        default=True,
        help="Pass through to Track A (default on).",
    )
    parser.add_argument("--skip-ingest-all", action="store_true")
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--group-traits", action="store_true")
    parser.add_argument("--min-species-support", type=int, default=2)
    parser.add_argument("--min-source-support", type=int, default=2)
    parser.add_argument("--min-inferred-traits", type=int, default=2)
    parser.add_argument("--max-inferred-traits", type=int, default=7)

    # Track B options
    parser.add_argument("--skip-track-b", action="store_true", help="Run Track A only.")
    parser.add_argument("--track-b-min-proposals", type=int, default=1)
    parser.add_argument("--track-b-max-proposals", type=int, default=5)
    parser.add_argument("--track-b-k-per-query", type=int, default=80)
    parser.add_argument("--track-b-max-chunks-per-species", type=int, default=8)
    parser.add_argument("--track-b-min-species-support", type=int, default=2)
    parser.add_argument("--track-b-min-source-support", type=int, default=2)
    parser.add_argument(
        "--track-b-min-premise-types",
        type=int,
        default=2,
        help="Minimum distinct biological premise types required for a supported proposal.",
    )
    parser.add_argument("--track-b-embed-merge-threshold", type=float, default=0.86)
    parser.add_argument("--track-b-token-merge-threshold", type=float, default=0.60)
    parser.add_argument("--track-b-proposer-prompt", default=DEFAULT_PROPOSER_PROMPT)
    parser.add_argument("--track-b-verifier-prompt", default=DEFAULT_VERIFIER_PROMPT)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bundle_dir = run_track_a_and_get_bundle(args)
    track_b_dir = bundle_dir / "summary" / "track_b"
    track_b_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_track_b:
        status = {
            "status": "skipped",
            "reason": "--skip-track-b set",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        (track_b_dir / "meta.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[v2b] Track B skipped. Status written: {track_b_dir / 'meta.json'}")
        return

    try:
        meta = run_track_b(bundle_dir=bundle_dir, args=args)
        print(f"[v2b] Track B finished. Verified={meta.get('verified_total', 0)}")
        print(f"[v2b] Merged output: {track_b_dir / 'hypothesis_merged.json'}")
    except Exception as exc:
        status = {
            "status": "skipped",
            "reason": "track_b_error",
            "error": str(exc),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        (track_b_dir / "meta.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[v2b] Track B failed but Track A output remains available: {exc}")


if __name__ == "__main__":
    main()
