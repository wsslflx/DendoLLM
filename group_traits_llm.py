#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import re
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

PROMPT_FILE = "Prompts/prompt_trait_grouping.txt"
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> str:
    text = text.strip()
    m = FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text


def load_trait_frequency(path: pathlib.Path) -> list[tuple[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    freq = data.get("trait_frequency")
    if not isinstance(freq, list):
        raise ValueError("Input JSON missing trait_frequency list")
    parsed: list[tuple[str, int]] = []
    for item in freq:
        if (
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], int)
        ):
            parsed.append((item[0], item[1]))
    if not parsed:
        raise ValueError("No valid (trait, count) pairs found in trait_frequency")
    return parsed


def format_trait_list(freq: list[tuple[str, int]]) -> str:
    lines = [f"- {trait} :: {count}" for trait, count in freq]
    return "\n".join(lines)


def run_grouping(
    freq: list[tuple[str, int]],
    model: str,
    temperature: float,
) -> str:
    prompt = PromptTemplate(
        input_variables=["trait_list"],
        template=pathlib.Path(PROMPT_FILE).read_text(encoding="utf-8"),
    )
    llm = ChatOpenAI(model_name=model, temperature=temperature)
    chain = (
        {"trait_list": RunnableLambda(lambda _: format_trait_list(freq))}
        | prompt
        | llm
    )
    raw = chain.invoke({})
    payload = raw.content if hasattr(raw, "content") else raw
    return str(payload)

def add_group_counts(
    groups: dict,
    freq: list[tuple[str, int]],
) -> dict:
    count_map = {trait: count for trait, count in freq}
    if not isinstance(groups, dict):
        raise ValueError("LLM output must be a JSON object with a 'groups' list")
    group_list = groups.get("groups")
    if not isinstance(group_list, list):
        raise ValueError("LLM output missing 'groups' list")

    for group in group_list:
        members = group.get("members", [])
        if not isinstance(members, list):
            continue
        group["count"] = sum(count_map.get(m, 0) for m in members)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group synonymous traits via LLM using trait_frequency from a report JSON."
    )
    parser.add_argument("report_json", help="Path to analyze_report/trait_variance_report_*.json")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0)",
    )
    parser.add_argument(
        "--out",
        help="Output JSON path (default: analyze_report/trait_groups_<timestamp>.json)",
    )
    args = parser.parse_args()

    report_path = pathlib.Path(args.report_json)
    freq = load_trait_frequency(report_path)

    raw = run_grouping(freq, model=args.model, temperature=args.temperature)
    payload = extract_json(raw)
    groups = json.loads(payload)
    groups = add_group_counts(groups, freq)

    out_path: pathlib.Path
    if args.out:
        out_path = pathlib.Path(args.out)
    else:
        out_dir = pathlib.Path("group_reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = out_dir / f"trait_groups_{timestamp}.json"

    out_path.write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
