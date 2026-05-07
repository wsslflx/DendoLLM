#!/usr/bin/env python3
"""
Build and cache the ontology index for uPheno, GO, and ENVO.

Usage:
    python -m kg.ontology_index --build [--force-rebuild]

On first run: downloads upheno.owl, go.owl, and envo.owl, parses with owlready2,
embeds all terms. Takes 15–60 minutes. Subsequent runs load from cache instantly.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Allow running from project root or kg/ subdirectory
sys.path.insert(0, str(Path(__file__).parents[1]))

# CACHE_DIR is the shared base for OWL files (model-independent).
# Embedding index files live in per-model subdirectories — see _get_embed_cache_dir().
CACHE_DIR     = Path(__file__).parent / "cache"
OWL_PATH      = CACHE_DIR / "upheno.owl"
GO_OWL_PATH   = CACHE_DIR / "go.owl"
ENVO_OWL_PATH = CACHE_DIR / "envo.owl"


def _get_embed_cache_dir(embed_backend: str | None = None) -> Path:
    """
    Return the per-model embedding cache directory.
    e.g. kg/cache/snowflake-arctic-embed2_latest/ or kg/cache/text-embedding-ada-002/
    """
    from core.llm_backend import resolve_embed_model_name
    model_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", resolve_embed_model_name(embed_backend))
    d = CACHE_DIR / model_slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_paths(embed_backend: str | None = None) -> tuple[Path, Path, Path]:
    """Return (EMBED_NPY, METADATA_JSONL, MANIFEST_JSON) for the given backend."""
    d = _get_embed_cache_dir(embed_backend)
    return d / "term_embeddings.npy", d / "term_metadata.jsonl", d / "manifest.json"

# Only keep terms from these namespaces
ALLOWED_PREFIXES = ("UPHENO:", "HP:", "MP:", "GO:", "ENVO:", "UBERON:", "ZP:", "XPO:")

UPHENO_OWL_URL = (
    "https://github.com/obophenotype/upheno/releases/latest/download/upheno.owl"
)
GO_OWL_URL   = "http://purl.obolibrary.org/obo/go.owl"
ENVO_OWL_URL = "http://purl.obolibrary.org/obo/envo.owl"

# All OWL sources: (local_path, download_url, label)
_OWL_SOURCES = [
    (OWL_PATH,      UPHENO_OWL_URL, "upheno.owl"),
    (GO_OWL_PATH,   GO_OWL_URL,     "go.owl"),
    (ENVO_OWL_PATH, ENVO_OWL_URL,   "envo.owl"),
]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_all_owls(force: bool = False) -> None:
    """Download upheno.owl, go.owl, and envo.owl if not already present."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import urllib.request
    for owl_path, owl_url, label in _OWL_SOURCES:
        if owl_path.exists() and not force:
            print(f"[KG] {label} already present at {owl_path} — skipping download.")
            continue
        print(f"[KG] Downloading {label} from {owl_url} ...")
        tmp = owl_path.with_suffix(".owl.tmp")
        try:
            urllib.request.urlretrieve(owl_url, tmp)
            os.replace(tmp, owl_path)
            print(f"[KG] Downloaded {label} ({owl_path.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            raise RuntimeError(f"Failed to download {label}: {exc}") from exc


def download_owl(force: bool = False) -> None:
    """Backwards-compatible wrapper — downloads all OWL files."""
    download_all_owls(force=force)


# ---------------------------------------------------------------------------
# Parse terms with owlready2 (no CLI tools needed)
# ---------------------------------------------------------------------------

def _is_allowed(term_id: str) -> bool:
    return any(term_id.startswith(p) for p in ALLOWED_PREFIXES)


def _iri_to_curie(iri: str) -> str | None:
    """Convert an OBO-style IRI to a CURIE, e.g. .../UPHENO_0001234 → UPHENO:0001234"""
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


def _load_terms_from_owl(owl_path: Path) -> list[dict]:
    """
    Parse all allowed terms from a single OWL file using a fresh owlready2 world.
    Uses no quadstore backend — in-memory only, no side effects on owlready2_quadstore.db.
    """
    import owlready2
    label = owl_path.name
    print(f"[KG] Loading {label} with owlready2...")

    # Fresh in-memory world per OWL file — no shared state, no SQLite writes
    world = owlready2.World()
    owl_iri = owl_path.absolute().as_uri()
    onto = world.get_ontology(owl_iri).load()
    print(f"[KG] {label} loaded. Extracting terms...")

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
            label_val = ""
            if cls.label:
                label_val = str(cls.label[0]) if cls.label else ""

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

            rich_text = f"{label_val}. {' | '.join(syns)}. {defn}".strip(". ")
            if not rich_text.strip():
                rich_text = curie

            terms.append({
                "id": curie,
                "name": label_val,
                "ontology": curie.split(":")[0],
                "synonyms": syns,
                "definition": defn,
                "parents": parents,
                "rich_text": rich_text,
            })
        except Exception:
            continue

        if scanned % 10000 == 0:
            print(f"[KG]   {owl_path.name}: scanned {scanned}, collected {len(terms)} terms...")

    print(f"[KG] {owl_path.name}: parsed {len(terms)} allowed terms from {scanned} total classes.")
    return terms


def load_terms() -> list[dict]:
    """
    Parse all allowed terms from upheno.owl, go.owl, and envo.owl.
    Deduplicates by term ID — first occurrence wins.
    Skips OWL files that are not yet downloaded.
    """
    seen: dict[str, dict] = {}
    for owl_path, _, label in _OWL_SOURCES:
        if not owl_path.exists():
            print(f"[KG] {label} not found in cache — skipping.")
            continue
        for term in _load_terms_from_owl(owl_path):
            if term["id"] not in seen:
                seen[term["id"]] = term
    terms = list(seen.values())
    print(f"[KG] Total unique terms after merge: {len(terms)}")
    return terms


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_terms(terms: list[dict], embed_backend: str | None = None) -> "np.ndarray":
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from core.llm_backend import make_embeddings

    embedder = make_embeddings(embed_backend=embed_backend)
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

def _read_manifest(embed_backend: str | None = None) -> dict:
    _, _, manifest_json = _index_paths(embed_backend)
    if manifest_json.exists():
        try:
            return json.loads(manifest_json.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _is_cache_fresh(embed_backend: str | None = None) -> bool:
    embed_npy, metadata_jsonl, manifest_json = _index_paths(embed_backend)
    if not (embed_npy.exists() and metadata_jsonl.exists() and manifest_json.exists()):
        return False
    manifest = _read_manifest(embed_backend)
    if not manifest:
        return False
    owl_mtimes = manifest.get("owl_mtimes", {})
    for owl_path, _, label in _OWL_SOURCES:
        if owl_path.exists():
            recorded = owl_mtimes.get(label, 0)
            if owl_path.stat().st_mtime > recorded:
                print(f"[KG] {label} has been updated — rebuilding index.")
                return False
    return True


def _write_manifest(terms: list[dict], embed_model: str, embed_backend: str | None = None) -> None:
    _, _, manifest_json = _index_paths(embed_backend)
    owl_mtimes = {}
    for owl_path, _, label in _OWL_SOURCES:
        if owl_path.exists():
            owl_mtimes[label] = owl_path.stat().st_mtime
    manifest = {
        "term_count": len(terms),
        "embedding_model": embed_model,
        "embed_backend": embed_backend or "ollama",
        "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "owl_mtimes": owl_mtimes,
    }
    tmp = manifest_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, manifest_json)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(force: bool = False, embed_backend: str | None = None) -> None:
    """Full build: download → parse → embed → persist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    embed_npy, metadata_jsonl, _ = _index_paths(embed_backend)

    if not force and _is_cache_fresh(embed_backend):
        manifest = _read_manifest(embed_backend)
        print(f"[KG] Ontology index is fresh ({manifest.get('term_count', '?')} terms). Loading from cache.")
        return

    backend_label = (embed_backend or "ollama").upper()
    print(
        f"[KG] Building ontology index (uPheno + GO + ENVO) with {backend_label} embeddings.\n"
        "     This will take 15–60 minutes. Subsequent runs load from cache instantly."
    )

    download_all_owls(force=force)
    terms = load_terms()

    if not terms:
        raise RuntimeError("[KG] No terms extracted from ontologies — aborting index build.")

    print(f"[KG] Embedding {len(terms)} terms with {backend_label}...")
    import numpy as np
    vecs = embed_terms(terms, embed_backend=embed_backend)

    # Persist embeddings
    tmp_npy = embed_npy.with_suffix(".npy.tmp")
    with open(tmp_npy, "wb") as _f:
        np.save(_f, vecs)
    os.replace(tmp_npy, embed_npy)
    print(f"[KG] Saved embeddings: {embed_npy}")

    # Persist metadata
    tmp_jsonl = metadata_jsonl.with_suffix(".jsonl.tmp")
    with open(tmp_jsonl, "w", encoding="utf-8") as f:
        for t in terms:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    os.replace(tmp_jsonl, metadata_jsonl)
    print(f"[KG] Saved metadata: {metadata_jsonl}")

    from core.llm_backend import resolve_embed_model_name
    embed_model = resolve_embed_model_name(embed_backend)
    _write_manifest(terms, embed_model, embed_backend)
    print(f"[KG] Index build complete. {len(terms)} terms indexed.")


def load_index(embed_backend: str | None = None) -> tuple["np.ndarray", list[dict]]:
    """Load cached embeddings and metadata. Raises if cache missing."""
    import numpy as np
    embed_npy, metadata_jsonl, _ = _index_paths(embed_backend)
    if not (embed_npy.exists() and metadata_jsonl.exists()):
        raise RuntimeError(
            "[KG] Ontology index cache not found. "
            f"Run: python -m kg.ontology_index --build --embed-backend {embed_backend or 'ollama'}"
        )
    vecs = np.load(embed_npy)
    terms = []
    with open(metadata_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                terms.append(json.loads(line))
    print(f"[KG] Loaded ontology index: {len(terms)} terms, embedding dim={vecs.shape[1]}")
    return vecs, terms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build uPheno + GO + ENVO ontology index.")
    parser.add_argument("--build", action="store_true", help="Build (or check) the index.")
    parser.add_argument("--force-rebuild", action="store_true", help="Force full rebuild even if cache is fresh.")
    parser.add_argument(
        "--embed-backend",
        default=os.getenv("EMBED_BACKEND", "ollama"),
        choices=["ollama", "openai"],
        help="Embedding backend to use (default: ollama).",
    )
    args = parser.parse_args()
    if args.build or args.force_rebuild:
        build_index(force=args.force_rebuild, embed_backend=args.embed_backend)
    else:
        parser.print_help()
