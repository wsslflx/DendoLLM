#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from datetime import datetime


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inventory, analyze runs, then group traits via LLM."
    )
    parser.add_argument("--species-file", required=True, help="Path to species JSON file.")
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

    species_file = pathlib.Path(args.species_file)
    if not species_file.exists():
        raise SystemExit(f"Species file not found: {species_file}")

    run_dirs = run_inventory(
        species_file,
        args.runs,
        args.reuse_traits,
        args.skip_ingest_after_first,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
    }
    (bundle_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
