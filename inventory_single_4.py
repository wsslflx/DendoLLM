#!/usr/bin/env python3
"""
V4 inventory runner: v1 extraction + hybrid normalization before synthesis.
Keeps v1 scripts unchanged.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from datetime import datetime
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

import inventory_single_2 as v1
from hybrid_normalization import build_hybrid_species_profile
from llm_backend import make_chat_llm

SYNTHESIS_PROMPT_FILE = "Prompts/prompt_synthesis_v4.txt"
MAX_SYNTHESIS_RETRIES = 3


def init_log_dir(
    log_runs: bool, log_root: pathlib.Path | None = None, subdir: str | None = None
) -> pathlib.Path | None:
    if not log_runs:
        return None
    base = pathlib.Path(log_root) if log_root else pathlib.Path("logs")
    log_dir = base / subdir if subdir else base
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def init_run_log_dir(
    log_runs: bool,
    run_label: str,
    explicit_log_dir: pathlib.Path | None = None,
) -> pathlib.Path | None:
    if not log_runs:
        return None
    if explicit_log_dir is not None:
        explicit_log_dir.mkdir(parents=True, exist_ok=True)
        return explicit_log_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pathlib.Path("logs") / f"{timestamp}-{v1.slugify(run_label)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _extract_json_text(payload: object) -> str:
    text_payload = str(payload).strip()
    if text_payload.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", text_payload, re.DOTALL | re.IGNORECASE)
        if match:
            text_payload = match.group(1).strip()
    return text_payload


def _validate_synthesis_json(data: object) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(data, dict):
        raise ValueError("Synthesis output must be a JSON object")
    required = ("strict_common_traits", "subgroup_common_traits", "mechanism_hypotheses")
    out: dict[str, list[dict[str, Any]]] = {}
    for key in required:
        value = data.get(key)
        if not isinstance(value, list):
            raise ValueError(f"Synthesis output field '{key}' must be a list")
        cleaned: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            trait = item.get("trait")
            sources = item.get("sources")
            confidence = item.get("confidence")
            if not isinstance(trait, str) or not trait.strip():
                continue
            if not isinstance(sources, list):
                sources = []
            clean_sources = [str(s) for s in sources if isinstance(s, str) and s.strip()]
            try:
                conf = float(confidence)
            except Exception:
                conf = 0.0
            cleaned.append(
                {
                    "trait": " ".join(trait.split()).strip(),
                    "sources": sorted(set(clean_sources)),
                    "confidence": max(0.0, min(1.0, conf)),
                }
            )
        out[key] = cleaned
    return out


def run_synthesis_hybrid(
    species_profiles: dict[str, dict[str, Any]],
    log_runs: bool = False,
    log_root: pathlib.Path | None = None,
) -> None:
    prompt = PromptTemplate(
        input_variables=["extracted_species_profiles"],
        template=pathlib.Path(SYNTHESIS_PROMPT_FILE).read_text(encoding="utf-8"),
    )
    llm = make_chat_llm(model=None, temperature=0.0, format="json")
    chain = {"extracted_species_profiles": RunnablePassthrough()} | prompt | llm

    profiles_json = json.dumps(species_profiles, ensure_ascii=False, indent=2)
    parsed_output: dict[str, list[dict[str, Any]]] | None = None
    raw_attempts: list[str] = []

    for attempt in range(1, MAX_SYNTHESIS_RETRIES + 1):
        payload = chain.invoke(profiles_json)
        content = payload.content if hasattr(payload, "content") else payload
        raw_text = str(content)
        raw_attempts.append(raw_text)
        try:
            parsed = json.loads(_extract_json_text(content))
            parsed_output = _validate_synthesis_json(parsed)
            break
        except Exception as exc:
            print(f"Debug: invalid synthesis JSON on attempt {attempt}/{MAX_SYNTHESIS_RETRIES}: {exc}")
            if attempt == MAX_SYNTHESIS_RETRIES:
                raise RuntimeError("Synthesis failed: model did not return valid JSON after retries") from exc

    assert parsed_output is not None
    content = json.dumps(parsed_output, ensure_ascii=False, indent=2)

    log_dir = init_log_dir(log_runs, log_root=log_root, subdir="synthesis")
    if log_dir:
        with open(log_dir / "synthesis_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt.format(extracted_species_profiles=profiles_json))
        with open(log_dir / "synthesis_input_profiles.json", "w", encoding="utf-8") as f:
            f.write(profiles_json)
        with open(log_dir / "synthesis_answer.txt", "w", encoding="utf-8") as f:
            f.write(content)
        with open(log_dir / "synthesis_raw_attempts.txt", "w", encoding="utf-8") as f:
            f.write("\n\n--- attempt ---\n\n".join(raw_attempts))

    print(content)


def load_species_file(path: str) -> list[dict]:
    return v1.load_species_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run v4 inventory (v1 extraction + hybrid normalization) and synthesis."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--species", help="Canonical species name to process.")
    group.add_argument("--species-file", help="Path to JSON file with canonical/aliases mappings.")
    parser.add_argument("--aliases", help="Comma-separated aliases to use for ingestion/search.")
    parser.add_argument("--log-run", action="store_true", help="Log prompt and answer to logs/<timestamp>/.")
    parser.add_argument("--log-dir", help="Explicit directory for run logs.")
    parser.add_argument("--traits-dir", help="Directory to write per-species traits JSON files.")
    parser.add_argument("--pdf-dir", help="Directory to store downloaded PDFs.")
    parser.add_argument("--chroma-dir", help="Directory for persisted Chroma vectorstore.")
    parser.add_argument(
        "--ingest-lock-file",
        default=".ingest.lock",
        help="Global lock file path to serialize ingestion across parallel processes.",
    )
    parser.add_argument(
        "--reuse-traits",
        action="store_true",
        help="Reuse existing traits/<species>.json if present (skip ingestion and prompting).",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion/download; use existing vectorstore content only.",
    )
    parser.add_argument(
        "--hybrid-sim-threshold",
        type=float,
        default=0.82,
        help="Cosine similarity threshold for semantic trait clustering (default: 0.82).",
    )
    args = parser.parse_args()

    explicit_log_dir = pathlib.Path(args.log_dir) if args.log_dir else None
    traits_dir = pathlib.Path(args.traits_dir) if args.traits_dir else None
    pdf_dir = pathlib.Path(args.pdf_dir) if args.pdf_dir else None
    chroma_dir = pathlib.Path(args.chroma_dir) if args.chroma_dir else None
    ingest_lock_file = pathlib.Path(args.ingest_lock_file) if args.ingest_lock_file else None

    run_log_dir = None
    if args.log_run:
        run_label = pathlib.Path(args.species_file).stem if args.species_file else args.species.strip()
        run_log_dir = init_run_log_dir(True, run_label, explicit_log_dir=explicit_log_dir)

    if args.species_file:
        species_groups = load_species_file(args.species_file)
        species_profiles: dict[str, dict[str, Any]] = {}
        for entry in species_groups:
            canonical = (entry.get("canonical") or "").strip()
            aliases = [a.strip() for a in entry.get("aliases", []) if a.strip()]
            if not canonical:
                continue
            open_traits = v1.run_inventory(
                canonical,
                aliases,
                log_runs=args.log_run,
                reuse_traits=args.reuse_traits,
                log_root=run_log_dir,
                skip_ingest=args.skip_ingest,
                traits_dir=traits_dir,
                pdf_dir=pdf_dir,
                chroma_dir=chroma_dir,
                ingest_lock_file=ingest_lock_file,
            )
            profile = build_hybrid_species_profile(open_traits, similarity_threshold=args.hybrid_sim_threshold)
            species_profiles[canonical] = profile

            if run_log_dir is not None:
                species_log_dir = run_log_dir / v1.slugify(canonical)
                species_log_dir.mkdir(parents=True, exist_ok=True)
                with open(species_log_dir / "hybrid_profile.json", "w", encoding="utf-8") as f:
                    json.dump(profile, f, ensure_ascii=False, indent=2)

        if species_profiles:
            run_synthesis_hybrid(species_profiles, log_runs=args.log_run, log_root=run_log_dir)
        return

    aliases = [a.strip() for a in (args.aliases or "").split(",") if a.strip()]
    open_traits = v1.run_inventory(
        args.species.strip(),
        aliases,
        log_runs=args.log_run,
        reuse_traits=args.reuse_traits,
        log_root=run_log_dir,
        skip_ingest=args.skip_ingest,
        traits_dir=traits_dir,
        pdf_dir=pdf_dir,
        chroma_dir=chroma_dir,
        ingest_lock_file=ingest_lock_file,
    )
    profile = build_hybrid_species_profile(open_traits, similarity_threshold=args.hybrid_sim_threshold)
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
