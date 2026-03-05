from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

def normalize_trait(trait: str) -> str:
    # Normalize whitespace and case so "Colonial living" == "colonial living".
    return " ".join(trait.split()).casefold()

def extract_json_from_txt(text: str) -> str:
    """
    Returns the JSON payload as a string.
    Supports either:
      - fenced ```json ... ```
      - raw JSON (no fences)
    """
    text = text.strip()
    m = FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text  # assume raw JSON

def load_run_file(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").strip()
    payload = extract_json_from_txt(raw)
    data = json.loads(payload)
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # Support synthesis object output by combining all configured sections.
        combined: list[dict] = []
        for section in ("strict_common_traits", "subgroup_common_traits", "mechanism_hypotheses"):
            items = data.get(section, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    combined.append(item)
        return combined

    raise ValueError(f"{path}: JSON top-level must be a list or synthesis object")

def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def analyze_runs(run_files: list[Path]) -> dict:
    runs: list[list[dict]] = [load_run_file(p) for p in run_files]
    n = len(runs)

    # Presence: trait -> set(run_idx)
    present: dict[str, set[int]] = defaultdict(set)

    # Citations: trait -> run_idx -> set(citations)
    cites: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))

    for i, run in enumerate(runs):
        for item in run:
            trait = item.get("trait")
            if not isinstance(trait, str) or not trait.strip():
                continue
            trait = normalize_trait(trait)
            sources = item.get("sources", [])
            if not isinstance(sources, list):
                sources = []

            present[trait].add(i)
            for s in sources:
                if isinstance(s, str):
                    cites[trait][i].add(s)

    # Trait frequency table
    freq = sorted(((t, len(idxs)) for t, idxs in present.items()),
                  key=lambda x: (-x[1], x[0].lower()))

    # Citation drift per trait
    drift = {}
    for trait, run_map in cites.items():
        appearing = sorted(present.get(trait, set()))
        per_run_sets = [run_map.get(i, set()) for i in appearing]
        union = set().union(*per_run_sets) if per_run_sets else set()

        # mean pairwise Jaccard across runs where trait appears
        if len(per_run_sets) >= 2:
            sims = []
            for a in range(len(per_run_sets)):
                for b in range(a + 1, len(per_run_sets)):
                    sims.append(jaccard(per_run_sets[a], per_run_sets[b]))
            mean_pairwise = sum(sims) / len(sims)
        else:
            mean_pairwise = 1.0 if per_run_sets else 0.0

        # mean jaccard vs union (how complete each run’s citations are relative to all seen)
        if union and per_run_sets:
            vs_union = [jaccard(s, union) for s in per_run_sets]
            mean_vs_union = sum(vs_union) / len(vs_union)
        else:
            mean_vs_union = 0.0

        # Count how many runs cite each source for this trait
        source_counts: dict[str, int] = {}
        for src in union:
            source_counts[src] = sum(1 for s in per_run_sets if src in s)

        drift[trait] = {
            "runs_present": len(appearing),
            "citation_union_size": len(union),
            "mean_pairwise_jaccard": round(mean_pairwise, 3),
            "mean_vs_union_jaccard": round(mean_vs_union, 3),
            "citation_union": sorted(union),
            "citation_counts": {
                src: source_counts[src] for src in sorted(source_counts)
            },
        }

    return {
        "n_runs": n,
        "trait_frequency": freq,   # list of (trait, count)
        "citation_drift": drift,   # dict keyed by trait
    }

def resolve_runs_from_list(list_path: Path) -> list[Path]:
    if not list_path.exists():
        raise FileNotFoundError(f"Run list not found: {list_path}")
    run_files: list[Path] = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        run_dir = Path(line)
        run_file = run_dir / "synthesis" / "synthesis_answer.txt"
        if not run_file.exists():
            raise FileNotFoundError(f"Missing synthesis_answer.txt under {run_dir}")
        run_files.append(run_file)
    return run_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze trait/citation variance across runs.")
    parser.add_argument(
        "--run-list",
        type=Path,
        help="Path to a list of run directories; each must contain synthesis/synthesis_answer.txt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output JSON path (default: analyze_report/trait_variance_report_<timestamp>.json)",
    )
    args = parser.parse_args()

    if args.run_list:
        files = resolve_runs_from_list(args.run_list)
    else:
        # Adjust pattern/path to your folder
        run_dir = Path("runs")
        files = sorted(run_dir.glob("*.txt"))
        if not files:
            raise SystemExit(f"No .txt files found in {run_dir.resolve()}")

    report = analyze_runs(files)

    # Print top traits
    print(f"Runs: {report['n_runs']}")
    print("\nTop traits (trait -> #runs):")
    for trait, count in report["trait_frequency"][:30]:
        print(f"- {trait}: {count}")

    # Save full report
    from datetime import datetime

    if args.out:
        out_path = args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path("analyze_report")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = out_dir / f"trait_variance_report_{timestamp}.json"

    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\nWrote {out_path}")
