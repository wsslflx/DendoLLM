#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import re
from collections import defaultdict
from typing import Any

import numpy as np

from core.llm_backend import make_chat_llm, make_embeddings

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "without",
    "ability",
    "adapted",
    "adaptation",
    "associated",
    "feature",
    "features",
    "shows",
    "showing",
    "trait",
    "traits",
}

LATENT_FACTORS_PROMPT_FILE = "Prompts/prompt_latent_factors_v4.txt"
MAX_LATENT_FACTOR_RETRIES = 3


def _clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def _normalize_key(text: str) -> str:
    return _clean_text(text).casefold()


def _simple_stem(token: str) -> str:
    tok = token.casefold()
    for suffix in ("ingly", "edly", "tion", "ness", "ment", "able", "ible", "ing", "edly", "ed", "ly", "es", "s"):
        if tok.endswith(suffix) and len(tok) > len(suffix) + 2:
            return tok[: -len(suffix)]
    return tok


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    out: list[str] = []
    for tok in tokens:
        if tok in _STOPWORDS:
            continue
        stem = _simple_stem(tok)
        if stem and stem not in _STOPWORDS:
            out.append(stem)
    return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _sanitize_open_traits(open_traits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in open_traits:
        if not isinstance(item, dict):
            continue
        trait = item.get("trait")
        if not isinstance(trait, str) or not trait.strip():
            continue
        sources_raw = item.get("sources")
        sources: list[str] = []
        if isinstance(sources_raw, list):
            for src in sources_raw:
                if isinstance(src, str) and src.strip():
                    sources.append(src.strip())
        confidence_raw = item.get("confidence", 0.5)
        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 0.5
        out.append(
            {
                "trait": _clean_text(trait),
                "sources": sorted(set(sources)),
                "confidence": max(0.0, min(1.0, confidence)),
            }
        )
    return out


def _cluster_by_semantics(
    traits: list[dict[str, Any]],
    similarity_threshold: float = 0.82,
) -> list[list[int]]:
    if not traits:
        return []

    texts = [t["trait"] for t in traits]
    try:
        embeddings = make_embeddings()
        vectors = np.array(embeddings.embed_documents(texts), dtype=float)
    except Exception:
        # Fallback: exact normalized key grouping when embeddings are unavailable.
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, text in enumerate(texts):
            groups[_normalize_key(text)].append(idx)
        return list(groups.values())

    # Normalize vectors once for stable cosine comparisons.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    vectors = vectors / norms

    clusters: list[list[int]] = []
    centroids: list[np.ndarray] = []

    for idx, vec in enumerate(vectors):
        best_cluster = -1
        best_score = -1.0
        for cluster_idx, centroid in enumerate(centroids):
            score = _cosine(vec, centroid)
            if score > best_score:
                best_score = score
                best_cluster = cluster_idx

        if best_cluster >= 0 and best_score >= similarity_threshold:
            clusters[best_cluster].append(idx)
            members = vectors[clusters[best_cluster]]
            centroid = members.mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids[best_cluster] = centroid / norm if norm != 0.0 else centroid
        else:
            clusters.append([idx])
            centroids.append(vec.copy())

    return clusters


def _canonical_from_cluster(traits: list[dict[str, Any]], cluster: list[int]) -> str:
    if len(cluster) == 1:
        return traits[cluster[0]]["trait"]

    texts = [traits[i]["trait"] for i in cluster]
    token_sets = [set(_tokenize(text)) for text in texts]

    # Pick the member with highest average lexical overlap as a robust canonical label.
    best_idx = 0
    best_score = -1.0
    for i, tokens_i in enumerate(token_sets):
        scores: list[float] = []
        for j, tokens_j in enumerate(token_sets):
            if i == j:
                continue
            union = tokens_i | tokens_j
            if not union:
                scores.append(0.0)
            else:
                scores.append(len(tokens_i & tokens_j) / len(union))
        avg_score = sum(scores) / len(scores) if scores else 0.0
        if avg_score > best_score:
            best_score = avg_score
            best_idx = i
    return texts[best_idx]


def _build_normalized_tags(
    traits: list[dict[str, Any]],
    clusters: list[list[int]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cluster in clusters:
        members = [traits[i] for i in cluster]
        canonical = _canonical_from_cluster(traits, cluster)
        all_sources: set[str] = set()
        confidences: list[float] = []
        member_texts: list[str] = []
        for member in members:
            member_texts.append(member["trait"])
            all_sources.update(member.get("sources", []))
            confidences.append(float(member.get("confidence", 0.5)))
        out.append(
            {
                "tag": canonical,
                "members": sorted(set(member_texts)),
                "sources": sorted(all_sources),
                "support_count": len(members),
                "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.5,
            }
        )

    out.sort(key=lambda x: (-int(x["support_count"]), x["tag"].casefold()))
    return out


def _extract_json_text(payload: object) -> str:
    text_payload = str(payload).strip()
    if text_payload.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", text_payload, re.DOTALL | re.IGNORECASE)
        if match:
            text_payload = match.group(1).strip()
    return text_payload


def _load_latent_factors_prompt() -> str:
    path = pathlib.Path(LATENT_FACTORS_PROMPT_FILE)
    if not path.exists():
        raise FileNotFoundError(f"Latent factors prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def _validate_latent_factors(
    data: object,
    open_traits: list[dict[str, Any]],
    min_factors: int = 2,
    max_factors: int = 7,
) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    items = data.get("latent_factors")
    if not isinstance(items, list):
        return []

    trait_to_sources: dict[str, list[str]] = {}
    for item in open_traits:
        trait = item.get("trait")
        if isinstance(trait, str):
            trait_to_sources[trait] = [s for s in item.get("sources", []) if isinstance(s, str)]

    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        factor = item.get("factor")
        if not isinstance(factor, str) or not factor.strip():
            continue
        factor_clean = _clean_text(factor)
        factor_key = factor_clean.casefold()
        if factor_key in seen:
            continue
        seen.add(factor_key)

        supports_raw = item.get("supporting_traits")
        supporting_traits: list[str] = []
        if isinstance(supports_raw, list):
            for s in supports_raw:
                if isinstance(s, str) and s in trait_to_sources:
                    supporting_traits.append(s)
        supporting_traits = sorted(set(supporting_traits))

        confidence_raw = item.get("confidence", 0.5)
        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        srcs: set[str] = set()
        for trait in supporting_traits:
            srcs.update(trait_to_sources.get(trait, []))

        cleaned.append(
            {
                "factor": factor_clean,
                "supporting_traits": supporting_traits,
                "sources": sorted(srcs),
                "support_count": len(supporting_traits),
                "confidence": confidence,
            }
        )

    cleaned.sort(key=lambda x: (x["support_count"], x["confidence"]), reverse=True)
    cleaned = cleaned[:max_factors]
    if cleaned and len(cleaned) < min_factors:
        return cleaned
    return cleaned


def _infer_latent_factors(
    open_traits: list[dict[str, Any]],
    min_factors: int = 2,
    max_factors: int = 7,
) -> list[dict[str, Any]]:
    if not open_traits:
        return []
    prompt = _load_latent_factors_prompt()
    llm = make_chat_llm(model=None, temperature=0.3, format="json")
    traits_json = json.dumps(open_traits, ensure_ascii=False, indent=2)

    for _ in range(MAX_LATENT_FACTOR_RETRIES):
        rendered = prompt.format(open_traits_json=traits_json)
        raw = llm.invoke(rendered)
        payload = raw.content if hasattr(raw, "content") else raw
        try:
            parsed = json.loads(_extract_json_text(payload))
        except Exception:
            continue
        validated = _validate_latent_factors(
            parsed,
            open_traits=open_traits,
            min_factors=min_factors,
            max_factors=max_factors,
        )
        if validated:
            return validated
    return []


def build_hybrid_species_profile(
    open_traits_raw: list[dict[str, Any]],
    similarity_threshold: float = 0.82,
) -> dict[str, Any]:
    open_traits = _sanitize_open_traits(open_traits_raw)
    clusters = _cluster_by_semantics(open_traits, similarity_threshold=similarity_threshold)
    normalized_tags = _build_normalized_tags(open_traits, clusters)
    latent_factors = _infer_latent_factors(open_traits)

    return {
        "open_traits": open_traits,
        "normalized_tags": normalized_tags,
        "latent_factors": latent_factors,
    }
