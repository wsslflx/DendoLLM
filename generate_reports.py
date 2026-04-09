#!/usr/bin/env python3
"""
Generate HTML reports for trait extraction results.

Usage:
    python generate_reports.py <runstamp_dir>
    python generate_reports.py logs_v1/20260408_120000/

Generates:
    <runstamp_dir>/index.html            — overview of all testcases, click to navigate
    <bundle_dir>/summary/report.html     — per-testcase: species images + common traits
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Wikipedia thumbnail fetching
# ---------------------------------------------------------------------------

def fetch_thumbnail(species_name: str) -> str | None:
    """Return a Wikipedia thumbnail URL for the species, or None if unavailable."""
    title = species_name.strip().replace(" ", "_")
    url = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + urllib.parse.quote(title, safe="")
    )
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SMTB2025-ReportGenerator/1.0 (research tool)"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
            src = data.get("thumbnail", {}).get("source")
            return src
    except Exception:
        return None


def fetch_thumbnails(species_list: list[dict]) -> dict[str, str | None]:
    """Fetch thumbnails for all species, with a small delay to be polite."""
    result: dict[str, str | None] = {}
    for i, sp in enumerate(species_list):
        canonical = sp.get("canonical", "")
        if not canonical:
            continue
        print(f"  [image] {canonical} ...", end=" ", flush=True)
        # Try canonical name first, then first alias
        url = fetch_thumbnail(canonical)
        if not url:
            for alias in sp.get("aliases", []):
                url = fetch_thumbnail(alias)
                if url:
                    break
        result[canonical] = url
        print("ok" if url else "no image")
        if i < len(species_list) - 1:
            time.sleep(0.3)
    return result


# ---------------------------------------------------------------------------
# Bundle discovery
# ---------------------------------------------------------------------------

def find_bundles(root: Path) -> list[Path]:
    """
    Return all bundle directories under root.
    A bundle dir has a summary/meta.json inside it.
    """
    bundles = []
    for candidate in sorted(root.iterdir()):
        if candidate.is_dir() and (candidate / "summary" / "meta.json").exists():
            bundles.append(candidate)
    return bundles


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [warn] Could not read {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f8f9fa;
    color: #212529;
    padding: 24px;
}
a { color: #0066cc; text-decoration: none; }
a:hover { text-decoration: underline; }
h1 { font-size: 1.6rem; margin-bottom: 6px; }
h2 { font-size: 1.15rem; margin: 28px 0 12px; color: #333; border-bottom: 2px solid #dee2e6; padding-bottom: 4px; }
.meta { color: #6c757d; font-size: 0.9rem; margin-bottom: 20px; }
.back { display: inline-block; margin-bottom: 18px; font-size: 0.9rem; }

/* Species grid */
.species-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 16px;
    margin-bottom: 8px;
}
.species-card {
    background: #fff;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 14px 10px 10px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.species-card img {
    width: 130px;
    height: 130px;
    object-fit: cover;
    border-radius: 6px;
    background: #e9ecef;
}
.species-card .no-img {
    width: 130px;
    height: 130px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #e9ecef;
    border-radius: 6px;
    margin: 0 auto;
    font-size: 2rem;
    color: #adb5bd;
}
.species-card .sp-name {
    font-weight: 600;
    font-size: 0.82rem;
    margin-top: 10px;
    font-style: italic;
}
.species-card .sp-aliases {
    font-size: 0.75rem;
    color: #6c757d;
    margin-top: 4px;
    line-height: 1.4;
}

/* Trait table */
.trait-table {
    width: 100%;
    border-collapse: collapse;
    background: #fff;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    font-size: 0.88rem;
}
.trait-table th {
    background: #343a40;
    color: #fff;
    padding: 10px 14px;
    text-align: left;
    font-weight: 500;
}
.trait-table td {
    padding: 9px 14px;
    border-bottom: 1px solid #dee2e6;
    vertical-align: top;
}
.trait-table tr:last-child td { border-bottom: none; }
.trait-table tr:nth-child(even) td { background: #f8f9fa; }
.freq-badge {
    display: inline-block;
    background: #0d6efd;
    color: #fff;
    border-radius: 12px;
    padding: 2px 9px;
    font-size: 0.78rem;
    font-weight: 600;
}
.member-pill {
    display: inline-block;
    background: #e9ecef;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.78rem;
    margin: 2px 2px 2px 0;
    color: #495057;
}
.citation-count { color: #6c757d; font-size: 0.82rem; }

/* Index table */
.index-table {
    width: 100%;
    border-collapse: collapse;
    background: #fff;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    font-size: 0.88rem;
}
.index-table th {
    background: #343a40;
    color: #fff;
    padding: 10px 14px;
    text-align: left;
    font-weight: 500;
}
.index-table td {
    padding: 9px 14px;
    border-bottom: 1px solid #dee2e6;
    vertical-align: top;
}
.index-table tr:last-child td { border-bottom: none; }
.index-table tr:nth-child(even) td { background: #f8f9fa; }
.index-table tr:hover td { background: #e8f4f8; }
.top-traits { color: #495057; font-size: 0.82rem; }
.btn {
    display: inline-block;
    background: #0d6efd;
    color: #fff;
    border-radius: 5px;
    padding: 4px 12px;
    font-size: 0.82rem;
}
.btn:hover { background: #0b5ed7; text-decoration: none; }
"""


def html_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{CSS}
</style>
</head>
<body>
{body}
</body>
</html>"""


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Per-testcase report
# ---------------------------------------------------------------------------

def build_report(bundle_dir: Path) -> dict | None:
    """
    Build summary/report.html for one bundle.
    Returns a summary dict for the index page, or None on failure.
    """
    summary_dir = bundle_dir / "summary"
    meta = load_json(summary_dir / "meta.json")
    if meta is None:
        return None

    analyze = load_json(summary_dir / "analyze_report.json") or {}
    groups_data = load_json(summary_dir / "trait_groups.json") or {}
    groups = groups_data.get("groups", [])

    # Load species list — always look in bundle_dir/species_input.json first
    species_list: list[dict] = []
    species_path = bundle_dir / "species_input.json"
    if species_path.exists():
        species_list = load_json(species_path) or []

    label = meta.get("run_label") or bundle_dir.name
    timestamp = meta.get("timestamp", "")
    n_runs = meta.get("runs", 1)
    n_species = len(species_list)

    print(f"\n[Report] {label} ({n_species} species)")

    # Fetch Wikipedia thumbnails
    thumbnails = fetch_thumbnails(species_list)

    # --- Species cards ---
    cards_html = ""
    for sp in species_list:
        canonical = sp.get("canonical", "")
        aliases = [a for a in sp.get("aliases", []) if a.lower() != canonical.lower()]
        img_url = thumbnails.get(canonical)
        if img_url:
            img_tag = f'<img src="{_esc(img_url)}" alt="{_esc(canonical)}" loading="lazy">'
        else:
            img_tag = '<div class="no-img">🐾</div>'
        alias_html = (
            f'<div class="sp-aliases">{_esc(", ".join(aliases[:2]))}</div>'
            if aliases else ""
        )
        cards_html += f"""
        <div class="species-card">
            {img_tag}
            <div class="sp-name">{_esc(canonical)}</div>
            {alias_html}
        </div>"""

    # --- Trait table ---
    trait_freq: dict[str, int] = {t: f for t, f in analyze.get("trait_frequency", [])}
    citation_drift: dict = analyze.get("citation_drift", {})

    # Build lookup: member trait → group canonical
    member_to_group: dict[str, dict] = {}
    for g in groups:
        for m in g.get("members", []):
            member_to_group[m] = g

    rows_html = ""
    # Use groups as the primary display (deduplicated, grouped)
    # Fall back to raw trait_frequency if no groups
    displayed: list[dict] = []
    if groups:
        displayed = groups
    else:
        displayed = [{"canonical": t, "members": [t], "count": f}
                     for t, f in analyze.get("trait_frequency", [])]

    for g in displayed:
        canonical_trait = g.get("canonical", "")
        members = g.get("members", [])
        count = g.get("count", trait_freq.get(canonical_trait, 0))
        drift = citation_drift.get(canonical_trait, {})
        n_citations = drift.get("citation_union_size", 0)
        n_runs_present = drift.get("runs_present", 0)

        members_html = "".join(
            f'<span class="member-pill">{_esc(m)}</span>'
            for m in members if m != canonical_trait
        )
        citation_html = (
            f'<span class="citation-count">{n_citations} source(s) across {n_runs_present} run(s)</span>'
            if n_citations else ""
        )
        rows_html += f"""
        <tr>
            <td><strong>{_esc(canonical_trait)}</strong></td>
            <td><span class="freq-badge">{count}</span></td>
            <td>{members_html}</td>
            <td>{citation_html}</td>
        </tr>"""

    if not rows_html:
        rows_html = '<tr><td colspan="4" style="color:#6c757d">No traits found.</td></tr>'

    # Relative path back to index
    depth = len(bundle_dir.relative_to(bundle_dir.parent).parts)
    back_path = "../" * (depth) + "index.html"

    body = f"""
<a class="back" href="{back_path}">← Back to overview</a>
<h1>{_esc(label)}</h1>
<div class="meta">{n_runs} run(s) &nbsp;·&nbsp; {n_species} species &nbsp;·&nbsp; {_esc(timestamp)}</div>

<h2>Species</h2>
<div class="species-grid">{cards_html}
</div>

<h2>Common Traits</h2>
<table class="trait-table">
  <thead>
    <tr>
      <th>Trait</th>
      <th>Frequency</th>
      <th>Related traits (same group)</th>
      <th>Evidence</th>
    </tr>
  </thead>
  <tbody>
{rows_html}
  </tbody>
</table>
"""

    out_path = summary_dir / "report.html"
    out_path.write_text(html_page(f"{label} — Trait Report", body), encoding="utf-8")
    print(f"  → {out_path}")

    top_traits = [g.get("canonical", "") for g in displayed[:3]]
    return {
        "label": label,
        "timestamp": timestamp,
        "n_runs": n_runs,
        "n_species": n_species,
        "species_names": [sp.get("canonical", "") for sp in species_list],
        "top_traits": top_traits,
        "report_path": out_path,
    }


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

def build_index(root: Path, summaries: list[dict]) -> None:
    rows_html = ""
    for s in summaries:
        label = _esc(s["label"])
        species_str = _esc(", ".join(s["species_names"]))
        top = _esc(" · ".join(s["top_traits"])) if s["top_traits"] else "<em>none</em>"
        # relative path from root/index.html to bundle/summary/report.html
        try:
            rel = s["report_path"].relative_to(root)
        except ValueError:
            rel = s["report_path"]
        rows_html += f"""
        <tr>
            <td><strong>{label}</strong><br><small style="color:#6c757d">{_esc(s['timestamp'])}</small></td>
            <td class="top-traits">{species_str}</td>
            <td style="text-align:center">{s['n_runs']}</td>
            <td class="top-traits">{top}</td>
            <td><a class="btn" href="{rel}">View →</a></td>
        </tr>"""

    if not rows_html:
        rows_html = '<tr><td colspan="5" style="color:#6c757d;padding:20px">No testcases found.</td></tr>'

    body = f"""
<h1>Trait Extraction — Run Overview</h1>
<div class="meta">{len(summaries)} testcase(s) &nbsp;·&nbsp; {root.name}</div>

<h2>Testcases</h2>
<table class="index-table">
  <thead>
    <tr>
      <th>Gene / Label</th>
      <th>Species</th>
      <th>Runs</th>
      <th>Top Traits</th>
      <th>Report</th>
    </tr>
  </thead>
  <tbody>
{rows_html}
  </tbody>
</table>
"""

    out_path = root / "index.html"
    out_path.write_text(html_page(f"Run Overview — {root.name}", body), encoding="utf-8")
    print(f"\n[Index] → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML reports for trait extraction results.")
    parser.add_argument("runstamp_dir", help="Path to the runstamp directory (e.g. logs_v1/20260408_120000/)")
    args = parser.parse_args()

    root = Path(args.runstamp_dir).resolve()
    if not root.exists():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    bundles = find_bundles(root)
    if not bundles:
        print(f"No bundle directories found under {root}", file=sys.stderr)
        print("(Each bundle must contain summary/meta.json)")
        sys.exit(1)

    print(f"Found {len(bundles)} bundle(s) under {root}")

    summaries = []
    for bundle in bundles:
        s = build_report(bundle)
        if s:
            summaries.append(s)

    build_index(root, summaries)
    print(f"\nDone. Open in browser:\n  {root / 'index.html'}")


if __name__ == "__main__":
    main()
