#!/usr/bin/env python3
"""
Zero-shot common-traits baseline (no retrieval, no citations).
Takes a list of species and asks the model for shared observable traits.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parents[1]))

import argparse
import json
import pathlib
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

from core.llm_backend import make_chat_llm

load_dotenv()

PROMPT_FILE = "Prompts/prompt_zero_shot_common.txt"


def load_species_file(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Species file must contain a list of mappings")
    return data


def format_species_list(species_groups: list[dict]) -> str:
    lines = []
    for entry in species_groups:
        canonical = (entry.get("canonical") or "").strip()
        aliases = [a.strip() for a in entry.get("aliases", []) if a.strip()]
        if not canonical:
            continue
        if aliases:
            lines.append(f"- {canonical} (aliases: {', '.join(aliases)})")
        else:
            lines.append(f"- {canonical}")
    return "\n".join(lines) if lines else "- (no species provided)"


def run_zero_shot(species_groups: list[dict]) -> tuple[str, str]:
    prompt = PromptTemplate(
        input_variables=["species_list"],
        template=pathlib.Path(PROMPT_FILE).read_text(encoding="utf-8"),
    )
    llm = make_chat_llm(model=None, temperature=0.0)
    formatted_prompt = prompt.format(species_list=format_species_list(species_groups))
    chain = (
        {
            "species_list": RunnableLambda(lambda _: format_species_list(species_groups)),
        }
        | prompt
        | llm
    )
    raw = chain.invoke({})
    payload = raw.content if hasattr(raw, "content") else raw
    print(payload)
    return formatted_prompt, str(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot common traits baseline (no RAG).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--species-file", help="Path to JSON file with canonical/aliases mappings.")
    group.add_argument("--species", help="Comma-separated list of species names.")
    parser.add_argument("--log-run", action="store_true", help="Log prompt and answer to logs/<timestamp>-<label>/.")
    args = parser.parse_args()

    if args.species_file:
        species_groups = load_species_file(args.species_file)
    else:
        names = [s.strip() for s in args.species.split(",") if s.strip()]
        species_groups = [{"canonical": n, "aliases": []} for n in names]

    formatted_prompt, payload = run_zero_shot(species_groups)
    if args.log_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = pathlib.Path(args.species_file).stem if args.species_file else "zero_shot"
        log_dir = pathlib.Path("logs") / f"{timestamp}-{label}"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "prompt.txt", "w", encoding="utf-8") as f:
            f.write(formatted_prompt)
        with open(log_dir / "answer.txt", "w", encoding="utf-8") as f:
            f.write(payload)


if __name__ == "__main__":
    main()
