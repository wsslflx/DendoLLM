"""Shared small helpers used across pipeline/, kg/, and core/.

Consolidated from duplicated copies in inventory_single_2.py, run_graph_pipeline.py,
graph_inventory_single.py, rag_cli.py, graph_synthesizer.py, graph_indexer.py,
graph_enricher.py, and kg_builder.py.
"""

from __future__ import annotations

import json
import pathlib
import re
from datetime import datetime


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def extract_json_text(payload: object) -> str:
    """Strip a ```json ... ``` markdown fence from an LLM response, if present."""
    text = str(payload).strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
    return text


def load_species_file(path: str) -> list[dict]:
    """Load species/alias mapping from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Species file must contain a list of mappings")
    return data


def ontology_source_from_term_id(term_id: str) -> str:
    """Derive the ontology prefix (e.g. 'HP', 'UBERON') from a term id like 'HP:0001250'."""
    return term_id.split(":")[0] if ":" in term_id else "UNKNOWN"


def init_log_dir(
    log_runs: bool,
    log_root: pathlib.Path | None = None,
    subdir: str | None = None,
    default_base: str = "logs",
) -> pathlib.Path | None:
    if not log_runs:
        return None
    base = pathlib.Path(log_root) if log_root else pathlib.Path(default_base)
    log_dir = base / subdir if subdir else base
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def init_run_log_dir(
    log_runs: bool,
    run_label: str,
    explicit_log_dir: pathlib.Path | None = None,
    default_base: str = "logs",
) -> pathlib.Path | None:
    if not log_runs:
        return None
    if explicit_log_dir is not None:
        explicit_log_dir.mkdir(parents=True, exist_ok=True)
        return explicit_log_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pathlib.Path(default_base) / f"{timestamp}-{slugify(run_label)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
