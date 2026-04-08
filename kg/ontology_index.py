#!/usr/bin/env python3
"""
Build and cache the uPheno ontology index.

Usage:
    python -m kg.ontology_index --build [--force-rebuild]

On first run: downloads upheno.owl, parses with owlready2, embeds all terms.
Takes 15-45 minutes. Subsequent runs load from cache instantly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow running from project root or kg/ subdirectory
sys.path.insert(0, str(Path(__file__).parents[1]))

CACHE_DIR    = Path(__file__).parent / "cache"
OWL_PATH     = CACHE_DIR / "upheno.owl"
EMBED_NPY    = CACHE_DIR / "term_embeddings.npy"
METADATA_JSONL = CACHE_DIR / "term_metadata.jsonl"
MANIFEST_JSON  = CACHE_DIR / "manifest.json"

# Only keep terms from these namespaces
ALLOWED_PREFIXES = ("UPHENO:", "HP:", "MP:", "GO:", "ENVO:", "UBERON:", "ZP:", "XPO:")

UPHENO_OWL_URL = (
    "https://github.com/obophenotype/upheno/releases/latest/download/upheno.owl"
)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_owl(force: bool = False) -> None:
    if OWL_PATH.exists() and not force:
        print(f"[KG] upheno.owl already present at {OWL_PATH} — skipping download.")
        return
    print(f"[KG] Downloading upheno.owl from GitHub releases...")
    print(f"     URL: {UPHENO_OWL_URL}")
    import urllib.request
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OWL_PATH.with_suffix(".owl.tmp")
    try:
        urllib.request.urlretrieve(UPHENO_OWL_URL, tmp)
        os.replace(tmp, OWL_PATH)
        print(f"[KG] Downloaded upheno.owl ({OWL_PATH.stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Failed to download upheno.owl: {exc}") from exc


# ---------------------------------------------------------------------------
# Parse terms with owlready2 (no CLI tools needed)
# ---------------------------------------------------------------------------

def _is_allowed(term_id: str) -> bool:
    return any(term_id.startswith(p) for p in ALLOWED_PREFIXES)


def _iri_to_curie(iri: str) -> str | None:
    """Convert an OBO-style IRI to a CURIE, e.g. .../UPHENO_0001234 → UPHENO:0001234"""
    # OBO IRIs end with Namespace_LocalID or Namespace/LocalID
    for sep in ("_", "/"):
        if sep in iri:
            last = iri.rsplit(sep, 1)[-1]
            prefix = iri.rsplit(sep, 1)[0].rsplit("/", 1)[-1]
            curie = f"{prefix}:{last}"
            if _is_allowed(curie):
                return curie
    # Fallback: try splitting on last /
    part = iri.rsplit("/", 1)[-1].replace("_", ":")
    if _is_allowed(part):
        return part
    return None


def load_terms() -> list[dict]:
    """Parse all allowed terms from upheno.owl using owlready2."""
    import owlready2
    print(f"[KG] Loading upheno.owl with owlready2 (this may take a few minutes)...")

    # owlready2 uses a SQLite world internally — point it at cache dir to persist
    world = owlready2.World()
    world.set_backend(filename=str(CACHE_DIR / "owlready2_quadstore.db"), exclusive=False)

    owl_iri = OWL_PATH.absolute().as_uri()
    onto = world.get_ontology(owl_iri).load()
    print(f"[KG] Ontology loaded. Extracting terms...")

    # Collect OBO synonym annotation properties via IRI lookup
    HAS_EXACT_SYN_IRI   = "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"
    HAS_RELATED_SYN_IRI = "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym"
    has_exact_syn   = world[HAS_EXACT_SYN_IRI]
    has_related_syn = world[HAS_RELATED_SYN_IRI]

    terms = []
    scanned = 0

    for cls in onto.classes():
        scanned += 1
        iri = cls.iri or ""
        curie = _iri_to_curie(iri)
        if not curie:
            continue

        try:
            # Label
            label = ""
            if cls.label:
                label = str(cls.label[0]) if cls.label else ""

            # Definition (IAO_0000115 or comment)
            defn = ""
            try:
                defn_prop = world["http://purl.obolibrary.org/obo/IAO_0000115"]
                if defn_prop:
                    vals = defn_prop[cls]
                    defn = str(vals[0]) if vals else ""
            except Exception:
                pass
            if not defn and cls.comment:
                defn = str(cls.comment[0])

            # Synonyms
            syns: list[str] = []
            for prop in (has_exact_syn, has_related_syn):
                if prop is None:
                    continue
                try:
                    for s in prop[cls]:
                        if s:
                            syns.append(str(s))
                except Exception:
                    pass

            # Parents (is_a only, allowed namespaces)
            parents: list[str] = []
            try:
                for parent in cls.is_a:
                    parent_iri = getattr(parent, "iri", None)
                    if parent_iri:
                        pc = _iri_to_curie(parent_iri)
                        if pc:
                            parents.append(pc)
            except Exception:
                pass

            rich_text = f"{label}. {' | '.join(syns)}. {defn}".strip(". ")
            if not rich_text.strip():
                rich_text = curie

            terms.append({
                "id": curie,
                "name": label,
                "ontology": curie.split(":")[0],
                "synonyms": syns,
                "definition": defn,
                "parents": parents,
                "rich_text": rich_text,
            })
        except Exception:
            continue

        if scanned % 10000 == 0:
            print(f"[KG]   Scanned {scanned} classes, collected {len(terms)} allowed terms...")

    print(f"[KG] Parsed {len(terms)} allowed terms from {scanned} total classes.")
    return terms


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_terms(terms: list[dict]) -> "np.ndarray":
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from core.llm_backend import make_embeddings

    embedder = make_embeddings()
    batch_size = 256
    all_vecs = []

    try:
        from tqdm import tqdm
        iter_batches = tqdm(range(0, len(terms), batch_size), desc="[KG] Embedding terms")
    except ImportError:
        iter_batches = range(0, len(terms), batch_size)

    for start in iter_batches:
        batch = terms[start:start + batch_size]
        texts = [t["rich_text"] for t in batch]
        try:
            vecs = embedder.embed_documents(texts)
            all_vecs.extend(vecs)
        except Exception as exc:
            print(f"[KG] Embedding batch {start}-{start+len(batch)} failed: {exc} — using zeros")
            dim = len(all_vecs[0]) if all_vecs else 1024
            all_vecs.extend([[0.0] * dim] * len(batch))

        if not hasattr(iter_batches, "__iter__") or isinstance(iter_batches, range):
            if (start // batch_size + 1) % 10 == 0:
                print(f"[KG]   Embedded {len(all_vecs)}/{len(terms)} terms...")

    return np.array(all_vecs, dtype="float32")


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _read_manifest() -> dict:
    if MANIFEST_JSON.exists():
        try:
            return json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _is_cache_fresh() -> bool:
    if not (EMBED_NPY.exists() and METADATA_JSONL.exists() and MANIFEST_JSON.exists()):
        return False
    manifest = _read_manifest()
    if not manifest:
        return False
    # Rebuild if OWL file is newer than the last build
    if OWL_PATH.exists():
        owl_mtime = OWL_PATH.stat().st_mtime
        manifest_mtime = manifest.get("owl_mtime", 0)
        if owl_mtime > manifest_mtime:
            print("[KG] upheno.owl has been updated — rebuilding index.")
            return False
    return True


def _write_manifest(terms: list[dict], embed_model: str) -> None:
    manifest = {
        "term_count": len(terms),
        "embedding_model": embed_model,
        "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "owl_mtime": OWL_PATH.stat().st_mtime if OWL_PATH.exists() else 0,
    }
    tmp = MANIFEST_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, MANIFEST_JSON)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(force: bool = False) -> None:
    """Full build: download → convert → parse → embed → persist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force and _is_cache_fresh():
        manifest = _read_manifest()
        print(f"[KG] Ontology index is fresh ({manifest.get('term_count', '?')} terms). Loading from cache.")
        return

    print(
        "[KG] Building ontology index for the first time. This will take 15–45 minutes.\n"
        "     Subsequent runs will load from cache instantly."
    )

    download_owl(force=force)
    terms = load_terms()

    if not terms:
        raise RuntimeError("[KG] No terms extracted from uPheno — aborting index build.")

    print(f"[KG] Embedding {len(terms)} terms...")
    import numpy as np
    vecs = embed_terms(terms)

    # Persist embeddings
    tmp_npy = EMBED_NPY.with_suffix(".npy.tmp")
    np.save(tmp_npy, vecs)
    os.replace(tmp_npy, EMBED_NPY)
    print(f"[KG] Saved embeddings: {EMBED_NPY}")

    # Persist metadata
    tmp_jsonl = METADATA_JSONL.with_suffix(".jsonl.tmp")
    with open(tmp_jsonl, "w", encoding="utf-8") as f:
        for t in terms:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    os.replace(tmp_jsonl, METADATA_JSONL)
    print(f"[KG] Saved metadata: {METADATA_JSONL}")

    from core.llm_backend import resolve_embed_model
    embed_model = resolve_embed_model()
    _write_manifest(terms, embed_model)
    print(f"[KG] Index build complete. {len(terms)} terms indexed.")


def load_index() -> tuple["np.ndarray", list[dict]]:
    """Load cached embeddings and metadata. Raises if cache missing."""
    import numpy as np
    if not (EMBED_NPY.exists() and METADATA_JSONL.exists()):
        raise RuntimeError(
            "[KG] Ontology index cache not found. Run: python -m kg.ontology_index --build"
        )
    vecs = np.load(EMBED_NPY)
    terms = []
    with open(METADATA_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                terms.append(json.loads(line))
    print(f"[KG] Loaded ontology index: {len(terms)} terms, embedding dim={vecs.shape[1]}")
    return vecs, terms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build uPheno ontology index.")
    parser.add_argument("--build", action="store_true", help="Build (or check) the index.")
    parser.add_argument("--force-rebuild", action="store_true", help="Force full rebuild even if cache is fresh.")
    args = parser.parse_args()
    if args.build or args.force_rebuild:
        build_index(force=args.force_rebuild)
    else:
        parser.print_help()
