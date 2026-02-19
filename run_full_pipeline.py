#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from datetime import datetime

from build_testcase_json import build_entries, parse_species_arg


def list_log_dirs() -> set[pathlib.Path]:
    base = pathlib.Path("logs")
    if not base.exists():
        return set()
    return {p for p in base.iterdir() if p.is_dir()}


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def newest_report(report_dir: pathlib.Path, prefix: str) -> pathlib.Path | None:
    if not report_dir.exists():
        return None
    candidates = [p for p in report_dir.iterdir() if p.is_file() and p.name.startswith(prefix)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_inventory(
    species_file: pathlib.Path,
    runs: int,
    reuse_traits: bool,
    skip_ingest_after_first: bool,
) -> list[pathlib.Path]:
    created: list[pathlib.Path] = []
    run_label = slugify(species_file.stem)
    for i in range(runs):
        before = list_log_dirs()
        cmd = [sys.executable, "inventory_single_2.py", "--species-file", str(species_file), "--log-run"]
        if reuse_traits:
            cmd.append("--reuse-traits")
        if skip_ingest_after_first and i > 0:
            cmd.append("--skip-ingest")
        print(f"Run {i + 1}/{runs}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        after = list_log_dirs()
        new_dirs = sorted(after - before)
        if len(new_dirs) == 1:
            created.append(new_dirs[0])
            continue
        # fallback: pick newest dir matching label
        label_dirs = [p for p in after if p.name.endswith(f"-{run_label}")]
        if label_dirs:
            newest = max(label_dirs, key=lambda p: p.stat().st_mtime)
            created.append(newest)
        else:
            newest = max(after, key=lambda p: p.stat().st_mtime)
            created.append(newest)
    return created


def write_run_list(run_dirs: list[pathlib.Path], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(p) for p in run_dirs]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_wiki_source(value: str | None) -> bool:
    if not value:
        return False
    return value.startswith("wiki:") or value.startswith("wikipedia:")


def aggregate_source_stats(run_dirs: list[pathlib.Path]) -> dict:
    per_run: list[dict] = []
    total_papers_fetched: set[str] = set()
    total_papers_used: set[str] = set()
    total_wiki_fetched: set[str] = set()
    total_wiki_used: set[str] = set()

    for run_dir in run_dirs:
        papers_fetched: set[str] = set()
        papers_used: set[str] = set()
        wiki_fetched: set[str] = set()
        wiki_used: set[str] = set()

        for species_dir in run_dir.iterdir():
            if not species_dir.is_dir() or species_dir.name == "synthesis":
                continue

            ingested_path = species_dir / "ingested_docs.json"
            if ingested_path.exists():
                try:
                    ingested_docs = json.loads(ingested_path.read_text(encoding="utf-8"))
                except Exception:
                    ingested_docs = []
                if isinstance(ingested_docs, list):
                    for entry in ingested_docs:
                        if not isinstance(entry, str):
                            continue
                        if _is_wiki_source(entry):
                            wiki_fetched.add(entry)
                        else:
                            papers_fetched.add(entry)

            used_path = species_dir / "used_chunks.json"
            if used_path.exists():
                try:
                    used_chunks = json.loads(used_path.read_text(encoding="utf-8"))
                except Exception:
                    used_chunks = []
                if isinstance(used_chunks, list):
                    for chunk in used_chunks:
                        if not isinstance(chunk, dict):
                            continue
                        source_path = chunk.get("source_path")
                        doc_id = chunk.get("doc_id")
                        source_key = (
                            str(doc_id)
                            if isinstance(doc_id, str) and doc_id
                            else (str(source_path) if isinstance(source_path, str) else "")
                        )
                        if not source_key:
                            continue
                        if _is_wiki_source(source_key) or _is_wiki_source(
                            str(source_path) if isinstance(source_path, str) else ""
                        ):
                            wiki_used.add(source_key)
                        else:
                            papers_used.add(source_key)

        total_papers_fetched.update(papers_fetched)
        total_papers_used.update(papers_used)
        total_wiki_fetched.update(wiki_fetched)
        total_wiki_used.update(wiki_used)

        per_run.append(
            {
                "run_dir": str(run_dir),
                "papers_fetched": len(papers_fetched),
                "papers_used": len(papers_used),
                "wiki_fetched": len(wiki_fetched),
                "wiki_used": len(wiki_used),
            }
        )

    return {
        "overall": {
            "papers_fetched": len(total_papers_fetched),
            "papers_used": len(total_papers_used),
            "wiki_fetched": len(total_wiki_fetched),
            "wiki_used": len(total_wiki_used),
        },
        "per_run": per_run,
    }


def resolve_species_file(
    species_file_arg: str | None,
    species_list_arg: str | None,
    generated_species_file_arg: str | None,
    timestamp: str,
) -> tuple[pathlib.Path, str]:
    if species_file_arg:
        species_file = pathlib.Path(species_file_arg)
        if not species_file.exists():
            raise SystemExit(f"Species file not found: {species_file}")
        return species_file, "file"

    species_list = parse_species_arg(species_list_arg or "")
    if not species_list:
        raise SystemExit("No species provided in --species-list.")

    if generated_species_file_arg:
        out_path = pathlib.Path(generated_species_file_arg)
    else:
        out_path = pathlib.Path("group_reports") / f"generated_species_{timestamp}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    entries = build_entries(species_list)
    out_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated species file: {out_path}")
    return out_path, "list"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inventory, analyze runs, then group traits via LLM."
    )
    species_group = parser.add_mutually_exclusive_group(required=True)
    species_group.add_argument("--species-file", help="Path to species JSON file.")
    species_group.add_argument(
        "--species-list",
        help="Comma-separated scientific names (space or underscore style).",
    )
    parser.add_argument(
        "--generated-species-file",
        help="Output path for generated species JSON when using --species-list.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of inventory runs.")
    parser.add_argument(
        "--reuse-traits",
        action="store_true",
        help="Reuse existing traits/<species>.json if present.",
    )
    parser.add_argument(
        "--skip-ingest-after-first",
        action="store_true",
        default=True,
        help="Only ingest/download on the first run; subsequent runs skip ingestion (default: on).",
    )
    parser.add_argument(
        "--run-list-out",
        help="Where to write run list (default: group_reports/run_list_<timestamp>.txt)",
    )
    parser.add_argument(
        "--group-out",
        help="Where to write grouped traits JSON (passed to group_traits_llm.py --out).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name for grouping (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature for grouping (default: 0.0).",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    species_file, species_input_mode = resolve_species_file(
        args.species_file,
        args.species_list,
        args.generated_species_file,
        timestamp,
    )

    run_dirs = run_inventory(
        species_file,
        args.runs,
        args.reuse_traits,
        args.skip_ingest_after_first,
    )
    source_stats = aggregate_source_stats(run_dirs)

    bundle_dir = pathlib.Path("group_reports") / f"{timestamp}-{slugify(species_file.stem)}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if args.run_list_out:
        run_list_path = pathlib.Path(args.run_list_out)
    else:
        run_list_path = bundle_dir / "run_list.txt"
    write_run_list(run_dirs, run_list_path)
    print(f"Run list written to: {run_list_path}")

    before_report = newest_report(pathlib.Path("analyze_report"), "trait_variance_report_")
    subprocess.run(
        [sys.executable, "analyze_runs.py", "--run-list", str(run_list_path)],
        check=True,
    )
    after_report = newest_report(pathlib.Path("analyze_report"), "trait_variance_report_")
    report_path = after_report if after_report and after_report != before_report else after_report
    if not report_path:
        raise SystemExit("No analyze_report/trait_variance_report_*.json found after analysis.")

    # copy/rename report into bundle for traceability
    bundled_report = bundle_dir / "analyze_report.json"
    bundled_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Using report: {report_path}")
    print(f"Bundled report: {bundled_report}")

    group_cmd = [
        sys.executable,
        "group_traits_llm.py",
        str(bundled_report),
        "--model",
        args.model,
        "--temperature",
        str(args.temperature),
    ]
    if args.group_out:
        group_cmd += ["--out", args.group_out]
    else:
        group_cmd += ["--out", str(bundle_dir / "trait_groups.json")]
    subprocess.run(group_cmd, check=True)

    meta = {
        "species_file": str(species_file),
        "species_input_mode": species_input_mode,
        "species_list": args.species_list if args.species_list else None,
        "runs": args.runs,
        "reuse_traits": args.reuse_traits,
        "skip_ingest_after_first": args.skip_ingest_after_first,
        "run_list": str(run_list_path),
        "analyze_report": str(bundled_report),
        "group_report": str(bundle_dir / "trait_groups.json")
        if not args.group_out
        else str(args.group_out),
        "model": args.model,
        "temperature": args.temperature,
        "timestamp": timestamp,
        "source_stats": source_stats,
    }
    (bundle_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
