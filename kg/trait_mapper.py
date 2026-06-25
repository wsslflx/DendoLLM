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

Performance optimizations:
  - Persistent SQLite cache: seen surface forms skip all LLM calls
  - Cosine early exit: skip LLM verify for obvious matches (>= 0.95) or non-matches (< 0.50)
  - Parallel execution: ThreadPoolExecutor in map_traits_batch()
  - Batch Stage 1 normalization: send N traits in one LLM call (norm_batch_size > 1)
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1]))

from core.llm_backend import make_chat_llm, make_embeddings, resolve_embed_model
from kg.ontology_index import CACHE_DIR, load_index  # CACHE_DIR kept for backward compat

# ---------------------------------------------------------------------------
# Cosine early-exit thresholds
# ---------------------------------------------------------------------------

# Top-1 cosine >= this → auto-accept as "high" confidence, skip LLM verify
COSINE_AUTO_ACCEPT = 0.95
# Top-1 cosine < this → auto no_match, skip LLM verify
COSINE_AUTO_REJECT = 0.50

# ---------------------------------------------------------------------------
# Thread-safe per-batch stats (reset at start of each map_traits_batch call)
# ---------------------------------------------------------------------------

_bstats_lock = Lock()
_bstats: dict = {
    "cache_hits": 0,
    "stage1_calls": 0,
    "stage1_time_s": 0.0,
    "stage2_calls": 0,
    "stage2_time_s": 0.0,
    "stage3_calls": 0,
    "stage3_time_s": 0.0,
    "auto_accept": 0,
    "auto_reject": 0,
}


def _reset_bstats() -> None:
    global _bstats
    with _bstats_lock:
        _bstats = {k: type(v)() for k, v in _bstats.items()}


def _bstats_add(**kwargs) -> None:
    with _bstats_lock:
        for k, v in kwargs.items():
            _bstats[k] = _bstats.get(k, type(v)()) + v


# ---------------------------------------------------------------------------
# Persistent SQLite mapping cache
# ---------------------------------------------------------------------------

_CACHE_DB = CACHE_DIR / "trait_map_cache.db"
_cache_lock = Lock()


def _init_cache() -> None:
    """Create the cache table if it doesn't exist. Called once at module level."""
    _CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(_CACHE_DB)) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute(
            "CREATE TABLE IF NOT EXISTS trait_map ("
            "  surface_form TEXT PRIMARY KEY,"
            "  result_json  TEXT NOT NULL"
            ")"
        )
        con.commit()


def _cache_get(raw_trait: str) -> dict | None:
    """Return cached mapping result for raw_trait, or None if not cached."""
    try:
        with sqlite3.connect(str(_CACHE_DB)) as con:
            con.execute("PRAGMA journal_mode=WAL")
            row = con.execute(
                "SELECT result_json FROM trait_map WHERE surface_form = ?",
                (raw_trait,),
            ).fetchone()
        if row:
            return json.loads(row[0])
    except Exception as exc:
        print(f"[KG] Cache read error (non-fatal): {exc}")
    return None


def _cache_put(raw_trait: str, result: dict) -> None:
    """Store a mapping result. Thread-safe via WAL + INSERT OR REPLACE."""
    try:
        with _cache_lock:
            with sqlite3.connect(str(_CACHE_DB)) as con:
                con.execute("PRAGMA journal_mode=WAL")
                con.execute(
                    "INSERT OR REPLACE INTO trait_map (surface_form, result_json) VALUES (?, ?)",
                    (raw_trait, json.dumps(result, ensure_ascii=False)),
                )
                con.commit()
    except Exception as exc:
        print(f"[KG] Cache write error (non-fatal): {exc}")


# Initialise cache DB on import
try:
    _init_cache()
except Exception as _exc:
    print(f"[KG] Could not initialise trait map cache (non-fatal): {_exc}")


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


_NORM_BATCH_SYSTEM = (
    "You are a biological ontology expert. Rewrite each numbered trait description "
    "in the formal style used by phenotype ontology term names (e.g. uPheno, HPO, MP). "
    "Use neutral, precise biological language. No articles, no trailing punctuation. "
    'Reply with ONLY a JSON object like {"1": "...", "2": "...", ...} — no other text.'
)

_NORM_BATCH_TEMPLATE = """\
Normalize these {n} biological trait descriptions into ontology-style phrases.
Reply with ONLY a JSON object mapping each number (as string) to the rewritten phrase.

{numbered_traits}
"""


def normalize_traits_batch(raw_traits: list[str], model=None, batch_size: int = 10) -> list[str]:
    """
    Normalize a list of traits using batched LLM calls.
    batch_size=1 falls back to one-by-one (same as normalize_trait()).
    Returns a list of normalized strings in the same order as raw_traits.
    """
    if batch_size <= 1:
        return [normalize_trait(t, model=model) for t in raw_traits]

    results: list[str] = []
    for start in range(0, len(raw_traits), batch_size):
        batch = raw_traits[start:start + batch_size]
        normalized = _normalize_batch(batch, model=model)
        results.extend(normalized)
        print(f"[KG] Batch-normalized traits {start + 1}–{start + len(batch)}/{len(raw_traits)}")
    return results


def _normalize_batch(traits: list[str], model=None) -> list[str]:
    """
    Send a batch of traits to the LLM in a single call, parse the JSON response.
    Falls back to individual normalize_trait() calls if the response is unparseable.
    """
    numbered = "\n".join(f'{i + 1}. "{t}"' for i, t in enumerate(traits))
    prompt = _NORM_BATCH_TEMPLATE.format(n=len(traits), numbered_traits=numbered)
    raw = _invoke_llm(prompt, _NORM_BATCH_SYSTEM, model=model)

    # Strip markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
        normalized = []
        for i, original in enumerate(traits):
            val = parsed.get(str(i + 1), "")
            normalized.append(val.strip() if val.strip() else original)
        return normalized
    except Exception as exc:
        print(f"[KG] Batch normalization parse failed ({exc}) — falling back to individual calls.")
        return [normalize_trait(t, model=model) for t in traits]


# ---------------------------------------------------------------------------
# Stage 2 — cosine similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(a, b_matrix) -> "np.ndarray":
    import numpy as np
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norms = b_matrix / (np.linalg.norm(b_matrix, axis=1, keepdims=True) + 1e-10)
    return b_norms @ a_norm


def find_top_candidates(
    normalized_trait: str,
    embeddings,
    terms: list[dict],
    top_k: int = 10,
    embed_backend: str | None = None,
    embedder=None,
) -> list[dict]:
    """
    Return top_k ontology terms by cosine similarity.
    Pass a pre-created `embedder` to avoid re-instantiating per call in parallel mode.
    """
    import numpy as np
    if embedder is None:
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
    embedder=None,
    normalized_trait: str | None = None,
) -> dict:
    """
    Full three-stage mapping for a single raw trait string.
    Returns the standardized result dict.

    Performance shortcuts (in order):
      1. SQLite cache hit → return immediately, zero LLM calls
      2. normalized_trait provided → skip Stage 1 LLM call
      3. Cosine auto-accept (>= COSINE_AUTO_ACCEPT) → skip Stage 3 LLM call
      4. Cosine auto-reject (< COSINE_AUTO_REJECT) → skip Stage 3 LLM call
    """
    # --- Cache check (skip all stages if seen before) ---
    cached = _cache_get(raw_trait)
    if cached is not None:
        _bstats_add(cache_hits=1)
        print(f"[KG]   Cache hit: '{raw_trait}'")
        return cached

    # Stage 1 — normalize (skip if pre-normalized by batch call)
    if normalized_trait is not None:
        normalized = normalized_trait
    else:
        _t1 = time.monotonic()
        normalized = normalize_trait(raw_trait, model=model)
        _bstats_add(stage1_calls=1, stage1_time_s=time.monotonic() - _t1)
    print(f"[KG]   Normalized: '{raw_trait}' → '{normalized}'")

    # Stage 2 — cosine similarity
    _t2 = time.monotonic()
    candidates = find_top_candidates(
        normalized, embeddings, terms, top_k=10,
        embed_backend=embed_backend, embedder=embedder,
    )
    _bstats_add(stage2_calls=1, stage2_time_s=time.monotonic() - _t2)
    if not candidates:
        result = _no_match_result(raw_trait, normalized)
        _cache_put(raw_trait, result)
        return result

    top_score = candidates[0]["cosine_score"]

    # Cosine early exit — auto-accept
    if top_score >= COSINE_AUTO_ACCEPT:
        best = candidates[0]
        _bstats_add(auto_accept=1)
        print(f"[KG]   Auto-accept (cosine={top_score:.3f}): '{raw_trait}' → {best['id']}")
        result = {
            "raw_trait": raw_trait,
            "normalized_trait": normalized,
            "term_id": best["id"],
            "term_name": best.get("name", ""),
            "confidence": "high",
            "cosine_score": top_score,
            "reasoning": f"auto-accepted: cosine score {top_score:.3f} >= {COSINE_AUTO_ACCEPT}",
            "mapped": True,
            "broadened": False,
        }
        _cache_put(raw_trait, result)
        return result

    # Cosine early exit — auto-reject
    if top_score < COSINE_AUTO_REJECT:
        _bstats_add(auto_reject=1)
        print(f"[KG]   Auto-reject (cosine={top_score:.3f}): '{raw_trait}'")
        result = _no_match_result(
            raw_trait, normalized,
            reasoning=f"auto-rejected: top cosine score {top_score:.3f} < {COSINE_AUTO_REJECT}",
        )
        _cache_put(raw_trait, result)
        return result

    # Stage 3 — LLM verification
    _t3 = time.monotonic()
    result = verify_candidate(raw_trait, normalized, candidates, model=model)
    _bstats_add(stage3_calls=1, stage3_time_s=time.monotonic() - _t3)
    confidence = result.get("confidence", "no_match")

    if confidence in ("high", "medium"):
        matched_term = _find_term(result.get("term_id"), terms)
        cosine = _find_cosine(result.get("term_id"), candidates)
        out = {
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
        _cache_put(raw_trait, out)
        return out

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
                out = {
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
                _cache_put(raw_trait, out)
                return out

    out = _no_match_result(raw_trait, normalized, reasoning=result.get("reasoning", ""))
    _cache_put(raw_trait, out)
    return out


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
    max_workers: int = 1,
    norm_batch_size: int = 1,
) -> list[dict]:
    """
    Map a list of traits in parallel, writing no-matches to a JSONL file.

    norm_batch_size > 1: pre-normalize all non-cached traits in batches via a
    single LLM call per batch before Stage 2/3, saving ~(batch_size-1)/batch_size
    of Stage 1 LLM calls. norm_batch_size=1 behaves exactly as before.

    The embedder and ontology index are loaded once and shared across threads
    (both are read-only after initialisation).
    """
    _reset_bstats()
    _batch_wall_start = time.monotonic()

    embeddings, terms = load_index(embed_backend=embed_backend)
    # Pre-create embedder once — avoids re-instantiating per thread call
    embedder = make_embeddings(embed_backend=embed_backend)

    n = len(raw_traits)

    # --- Batch Stage 1: pre-normalize non-cached traits ---
    # Check cache first so we don't waste LLM calls on already-known traits.
    pre_normalized: dict[str, str] = {}  # raw_trait → normalized
    if norm_batch_size > 1:
        uncached = [t for t in raw_traits if _cache_get(t) is None]
        if uncached:
            print(f"[KG] Batch-normalizing {len(uncached)} uncached traits "
                  f"(batch_size={norm_batch_size})...")
            _t1_batch = time.monotonic()
            norm_results = normalize_traits_batch(uncached, model=model, batch_size=norm_batch_size)
            _bstats_add(stage1_calls=len(uncached), stage1_time_s=time.monotonic() - _t1_batch)
            pre_normalized = dict(zip(uncached, norm_results))
        cached_count = n - len(uncached)
        if cached_count:
            print(f"[KG] Skipping normalization for {cached_count} cached traits.")

    results: list[dict | None] = [None] * n
    completed = 0

    def _map_one(idx_trait):
        idx, trait = idx_trait
        norm = pre_normalized.get(trait)  # None if batch_size=1 or cache hit
        return idx, map_trait(
            trait, embeddings, terms,
            model=model, embed_backend=embed_backend, embedder=embedder,
            normalized_trait=norm,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_map_one, (i, t)): i for i, t in enumerate(raw_traits)}
        for future in as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
            except Exception as exc:
                idx = futures[future]
                print(f"[KG] map_trait failed for trait index {idx}: {exc}")
                results[idx] = _no_match_result(raw_traits[idx], raw_traits[idx], reasoning=str(exc))
            completed += 1
            if completed % 50 == 0 or completed == n:
                mapped_so_far = sum(1 for r in results if r is not None and r.get("mapped"))
                print(f"[KG] Progress: {completed}/{n} done, {mapped_so_far} mapped so far.")

    # Fill any None slots that somehow slipped through
    for i, r in enumerate(results):
        if r is None:
            results[i] = _no_match_result(raw_traits[i], raw_traits[i], reasoning="unknown error")

    no_matches = [r for r in results if not r["mapped"]]

    if no_match_out_path and no_matches:
        tmp = Path(str(no_match_out_path) + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for nm in no_matches:
                f.write(json.dumps(nm, ensure_ascii=False) + "\n")
        os.replace(tmp, no_match_out_path)
        print(f"[KG] {len(no_matches)} unmatched traits written to {no_match_out_path}")

    mapped = sum(1 for r in results if r["mapped"])
    print(f"[KG] Mapped {mapped}/{n} traits successfully.")

    # Per-stage timing and outcome summary
    with _bstats_lock:
        st = dict(_bstats)
    batch_wall_s = time.monotonic() - _batch_wall_start
    non_cached = n - st["cache_hits"]
    stage3_trigger_rate = st["stage3_calls"] / non_cached if non_cached > 0 else 0.0
    s1s, s2s, s3s = st["stage1_time_s"], st["stage2_time_s"], st["stage3_time_s"]
    stage_total = s1s + s2s + s3s
    print(f"[KG] ── Mapping stage stats (thread-summed time) ──────────────────")
    print(f"[KG]   Total traits:      {n}  |  cache hits: {st['cache_hits']} ({st['cache_hits']/n:.0%})  |  wall time: {batch_wall_s:.1f}s")
    print(f"[KG]   Stage 1 (LLM norm):   {st['stage1_calls']:4d} calls  {s1s:7.1f}s  ({s1s/stage_total:.0%} of stage time)" if stage_total else f"[KG]   Stage 1 (LLM norm):   {st['stage1_calls']:4d} calls")
    print(f"[KG]   Stage 2 (cosine):     {st['stage2_calls']:4d} calls  {s2s:7.1f}s  ({s2s/stage_total:.0%} of stage time)" if stage_total else f"[KG]   Stage 2 (cosine):     {st['stage2_calls']:4d} calls")
    print(f"[KG]   Stage 3 (LLM verify): {st['stage3_calls']:4d} calls  {s3s:7.1f}s  ({s3s/stage_total:.0%} of stage time)  trigger rate: {stage3_trigger_rate:.0%}" if stage_total else f"[KG]   Stage 3 (LLM verify): {st['stage3_calls']:4d} calls  trigger rate: {stage3_trigger_rate:.0%}")
    print(f"[KG]   Auto-accept: {st['auto_accept']}  |  auto-reject: {st['auto_reject']}  |  stage time total: {stage_total:.1f}s")
    print(f"[KG] ────────────────────────────────────────────────────────────────")

    return results
