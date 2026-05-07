#!/usr/bin/env python3
"""
Three-stage trait → uPheno ontology term mapping.

Stage 1: LLM normalizes raw trait into ontology-style phrase.
Stage 2: Cosine similarity against term_embeddings.npy, return top 10.
Stage 3: LLM picks the best candidate or returns no_match.

Confidence policy:
  high / medium → accept
  low           → try parent broadening via OAK; accept if high/medium; else no_match
  no_match      → no_match
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1]))

from core.llm_backend import make_chat_llm, make_embeddings, resolve_embed_model
from kg.ontology_index import CACHE_DIR, load_index  # CACHE_DIR kept for backward compat

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _invoke_llm(prompt: str, system: str, model=None, retries: int = 1) -> str:
    """Call the LLM with a 120s timeout, retry once on failure."""
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = make_chat_llm(model=model, temperature=0.0, timeout=120)
    for attempt in range(retries + 1):
        try:
            resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
            return resp.content.strip()
        except Exception as exc:
            if attempt < retries:
                print(f"[KG] LLM call failed (attempt {attempt+1}), retrying: {exc}")
                time.sleep(2)
            else:
                print(f"[KG] LLM call failed after {retries+1} attempts: {exc}")
                return ""
    return ""


# ---------------------------------------------------------------------------
# Stage 1 — LLM normalization
# ---------------------------------------------------------------------------

_NORM_SYSTEM = (
    "You are a biological ontology expert. Rewrite the given trait description "
    "in the formal style used by phenotype ontology term names (e.g. uPheno, HPO, MP). "
    "Use neutral, precise biological language. Output only the rewritten phrase — "
    "no explanation, no punctuation at the end, no articles."
)


def normalize_trait(raw_trait: str, model=None) -> str:
    result = _invoke_llm(f'Trait: "{raw_trait}"', _NORM_SYSTEM, model=model)
    return result if result else raw_trait


# ---------------------------------------------------------------------------
# Stage 2 — cosine similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(a, b_matrix) -> "np.ndarray":
    import numpy as np
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norms = b_matrix / (np.linalg.norm(b_matrix, axis=1, keepdims=True) + 1e-10)
    return b_norms @ a_norm


def find_top_candidates(normalized_trait: str, embeddings, terms: list[dict], top_k: int = 10, embed_backend: str | None = None) -> list[dict]:
    import numpy as np
    embedder = make_embeddings(embed_backend=embed_backend)
    vec = np.array(embedder.embed_query(normalized_trait), dtype="float32")
    sims = _cosine_similarity(vec, embeddings)
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [
        {**terms[i], "cosine_score": float(sims[i])}
        for i in top_idx
    ]


# ---------------------------------------------------------------------------
# Stage 3 — LLM verification
# ---------------------------------------------------------------------------

_VERIFY_SYSTEM = (
    "You are a biological ontology expert. Select the best matching ontology term "
    "for the given trait. Reply in JSON only — no other text."
)

_VERIFY_TEMPLATE = """\
Given the biological trait: "{raw_trait}"
(normalized as: "{normalized_trait}")

Select the single best matching ontology term from the candidates below.
Reply in this exact JSON format — no other text:
{{
  "term_id": "UPHENO:XXXXXXX",
  "term_name": "...",
  "confidence": "high|medium|low",
  "reasoning": "one sentence max"
}}
If none are a reasonable match, reply exactly:
{{"term_id": null, "confidence": "no_match", "reasoning": "..."}}

Candidates:
{candidates_text}
"""


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        defn = c.get("definition", "")[:120]
        lines.append(f"{i}. {c['id']} — {c.get('name', '')} — \"{defn}\"")
    return "\n".join(lines)


def verify_candidate(raw_trait: str, normalized_trait: str, candidates: list[dict], model=None) -> dict:
    candidates_text = _format_candidates(candidates)
    prompt = _VERIFY_TEMPLATE.format(
        raw_trait=raw_trait,
        normalized_trait=normalized_trait,
        candidates_text=candidates_text,
    )
    raw = _invoke_llm(prompt, _VERIFY_SYSTEM, model=model)
    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"term_id": None, "confidence": "no_match", "reasoning": "LLM output parse error"}


# ---------------------------------------------------------------------------
# Parent broadening
# ---------------------------------------------------------------------------

def _get_parents(term_id: str) -> list[str]:
    """Fetch direct is_a parents via owlready2."""
    try:
        import owlready2
        quadstore = CACHE_DIR / "owlready2_quadstore.db"
        if not quadstore.exists():
            return []
        world = owlready2.World()
        world.set_backend(filename=str(quadstore), exclusive=False)
        frag = term_id.replace(":", "_")
        for cls in world.classes():
            if cls.iri and frag in cls.iri:
                parents = []
                for parent in cls.is_a:
                    piri = getattr(parent, "iri", None)
                    if not piri:
                        continue
                    for sep in ("_", "/"):
                        if sep in piri:
                            last = piri.rsplit(sep, 1)[-1]
                            prefix = piri.rsplit(sep, 1)[0].rsplit("/", 1)[-1]
                            pid = f"{prefix}:{last}"
                            from kg.ontology_index import _is_allowed
                            if _is_allowed(pid):
                                parents.append(pid)
                            break
                return parents
        return []
    except Exception as exc:
        print(f"[KG] Parent lookup failed for {term_id}: {exc}")
        return []


def broaden_to_parents(term_id: str, embeddings, terms: list[dict]) -> list[dict]:
    """Re-rank using parent terms as new candidates."""
    parents = _get_parents(term_id)
    if not parents:
        return []
    term_by_id = {t["id"]: t for t in terms}
    return [term_by_id[p] for p in parents if p in term_by_id]


# ---------------------------------------------------------------------------
# Main mapping function
# ---------------------------------------------------------------------------

def map_trait(
    raw_trait: str,
    embeddings,
    terms: list[dict],
    model=None,
    embed_backend: str | None = None,
) -> dict:
    """
    Full three-stage mapping for a single raw trait string.
    Returns the standardized result dict.
    """
    # Stage 1
    normalized = normalize_trait(raw_trait, model=model)
    print(f"[KG]   Normalized: '{raw_trait}' → '{normalized}'")

    # Stage 2
    candidates = find_top_candidates(normalized, embeddings, terms, top_k=10, embed_backend=embed_backend)
    if not candidates:
        return _no_match_result(raw_trait, normalized)

    # Stage 3
    result = verify_candidate(raw_trait, normalized, candidates, model=model)
    confidence = result.get("confidence", "no_match")

    if confidence in ("high", "medium"):
        matched_term = _find_term(result.get("term_id"), terms)
        cosine = _find_cosine(result.get("term_id"), candidates)
        return {
            "raw_trait": raw_trait,
            "normalized_trait": normalized,
            "term_id": result.get("term_id"),
            "term_name": result.get("term_name") or (matched_term.get("name") if matched_term else None),
            "confidence": confidence,
            "cosine_score": cosine,
            "reasoning": result.get("reasoning", ""),
            "mapped": True,
            "broadened": False,
        }

    if confidence == "low" and result.get("term_id"):
        # Parent broadening
        print(f"[KG]   Low confidence for {result.get('term_id')} — trying parent broadening...")
        parent_candidates = broaden_to_parents(result["term_id"], embeddings, terms)
        if parent_candidates:
            result2 = verify_candidate(raw_trait, normalized, parent_candidates, model=model)
            confidence2 = result2.get("confidence", "no_match")
            if confidence2 in ("high", "medium"):
                matched_term = _find_term(result2.get("term_id"), terms)
                cosine = _find_cosine(result2.get("term_id"), parent_candidates)
                return {
                    "raw_trait": raw_trait,
                    "normalized_trait": normalized,
                    "term_id": result2.get("term_id"),
                    "term_name": result2.get("term_name") or (matched_term.get("name") if matched_term else None),
                    "confidence": confidence2,
                    "cosine_score": cosine,
                    "reasoning": result2.get("reasoning", ""),
                    "mapped": True,
                    "broadened": True,
                }

    return _no_match_result(raw_trait, normalized, reasoning=result.get("reasoning", ""))


def _no_match_result(raw_trait: str, normalized: str, reasoning: str = "") -> dict:
    return {
        "raw_trait": raw_trait,
        "normalized_trait": normalized,
        "term_id": None,
        "term_name": None,
        "confidence": "no_match",
        "cosine_score": None,
        "reasoning": reasoning,
        "mapped": False,
        "broadened": False,
    }


def _find_term(term_id: str | None, terms: list[dict]) -> dict | None:
    if not term_id:
        return None
    for t in terms:
        if t["id"] == term_id:
            return t
    return None


def _find_cosine(term_id: str | None, candidates: list[dict]) -> float | None:
    if not term_id:
        return None
    for c in candidates:
        if c.get("id") == term_id:
            return c.get("cosine_score")
    return None


def map_traits_batch(
    raw_traits: list[str],
    no_match_out_path: Path | None = None,
    model=None,
    embed_backend: str | None = None,
) -> list[dict]:
    """Map a list of traits, writing no-matches to a JSONL file."""
    embeddings, terms = load_index(embed_backend=embed_backend)
    results = []
    no_matches = []

    for i, trait in enumerate(raw_traits):
        print(f"[KG] Mapping trait {i+1}/{len(raw_traits)}: '{trait}'")
        result = map_trait(trait, embeddings, terms, model=model, embed_backend=embed_backend)
        results.append(result)
        if not result["mapped"]:
            no_matches.append(result)

    if no_match_out_path and no_matches:
        tmp = Path(str(no_match_out_path) + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for nm in no_matches:
                f.write(json.dumps(nm, ensure_ascii=False) + "\n")
        import os
        os.replace(tmp, no_match_out_path)
        print(f"[KG] {len(no_matches)} unmatched traits written to {no_match_out_path}")

    mapped = sum(1 for r in results if r["mapped"])
    print(f"[KG] Mapped {mapped}/{len(results)} traits successfully.")
    return results
