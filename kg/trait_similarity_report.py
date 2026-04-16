#!/usr/bin/env python3
"""
Trait similarity report — standalone diagnostic tool.

Reads hybrid_profile.json files from a pipeline bundle, embeds all normalized
trait strings, computes the full pairwise cosine similarity matrix, and writes:
  - trait_similarity.json  — raw scores for all pairs
  - trait_similarity.html  — interactive heatmap (open in browser)

Usage:
    python kg/trait_similarity_report.py --bundle-dir logs_v4/<run>
    python kg/trait_similarity_report.py --bundle-dir logs_v4/<run> --threshold 0.78
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


def discover_traits(bundle_dir: Path) -> list[dict]:
    """
    Scan bundle for hybrid_profile.json files.
    Returns list of {species, raw_trait, normalized_trait} dicts.
    """
    entries = []
    runs_dir = bundle_dir / "runs"
    if not runs_dir.exists():
        print(f"[SIM] No runs/ directory in {bundle_dir}")
        return entries

    for profile_path in sorted(runs_dir.glob("*/*/hybrid_profile.json")):
        species_slug = profile_path.parent.name
        species_display = species_slug.replace("_", " ").title()
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[SIM] Could not read {profile_path}: {exc}")
            continue

        # Prefer normalized_tags, fall back to open_traits
        tags = profile.get("normalized_tags", [])
        if tags:
            for tag in tags:
                if isinstance(tag, dict) and tag.get("tag"):
                    entries.append({
                        "species": species_display,
                        "raw_trait": tag.get("tag", ""),
                        "normalized_trait": tag.get("tag", ""),
                    })
        else:
            for ot in profile.get("open_traits", []):
                if isinstance(ot, dict) and ot.get("trait"):
                    entries.append({
                        "species": species_display,
                        "raw_trait": ot.get("trait", ""),
                        "normalized_trait": ot.get("trait", ""),
                    })

    # Deduplicate by (species, normalized_trait)
    seen = set()
    deduped = []
    for e in entries:
        key = (e["species"], e["normalized_trait"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    print(f"[SIM] Found {len(deduped)} traits across "
          f"{len(set(e['species'] for e in deduped))} species.")
    return deduped


def compute_similarity_matrix(entries: list[dict]) -> "np.ndarray":
    import numpy as np
    from core.llm_backend import make_embeddings

    texts = [e["normalized_trait"] for e in entries]
    print(f"[SIM] Embedding {len(texts)} traits...")
    embedder = make_embeddings()
    vecs = np.array(embedder.embed_documents(texts), dtype="float32")

    # Normalise
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    sim = vecs @ vecs.T
    print(f"[SIM] Similarity matrix computed: {sim.shape}")
    return sim


def write_json(entries: list[dict], sim, out_path: Path, threshold: float) -> None:
    import numpy as np
    pairs = []
    n = len(entries)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim[i, j])
            pairs.append({
                "trait_a": entries[i]["normalized_trait"],
                "species_a": entries[i]["species"],
                "trait_b": entries[j]["normalized_trait"],
                "species_b": entries[j]["species"],
                "score": round(score, 4),
                "cross_species": entries[i]["species"] != entries[j]["species"],
                "above_threshold": score >= threshold,
            })
    pairs.sort(key=lambda x: -x["score"])

    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out_path)
    print(f"[SIM] JSON written: {out_path} ({len(pairs)} pairs)")

    # Print top cross-species pairs to console
    cross = [p for p in pairs if p["cross_species"]]
    print(f"\n[SIM] Top 20 cross-species similarity scores:")
    print(f"  {'Score':>6}  {'Threshold':>9}  Species A → Trait A  |  Species B → Trait B")
    print(f"  {'-'*6}  {'-'*9}  {'-'*60}")
    for p in cross[:20]:
        flag = "✓" if p["above_threshold"] else "✗"
        print(f"  {p['score']:>6.3f}  {flag:>9}  "
              f"{p['species_a']}: '{p['trait_a']}'  |  "
              f"{p['species_b']}: '{p['trait_b']}'")


def write_html(entries: list[dict], sim, out_path: Path, threshold: float,
               bundle_name: str) -> None:
    n = len(entries)
    labels = [f"{e['species']}<br>{e['normalized_trait']}" for e in entries]
    labels_json = json.dumps(labels)

    # Build flat z matrix row by row
    rows = []
    for i in range(n):
        row = [round(float(sim[i, j]), 4) for j in range(n)]
        rows.append(row)
    z_json = json.dumps(rows)

    # Color annotations: red border for cross-species pairs above threshold
    species_list = json.dumps([e["species"] for e in entries])

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Trait Similarity — {bundle_name}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ font-family: sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
  h1 {{ font-size: 1.3em; margin-bottom: 4px; }}
  p  {{ font-size: 0.85em; color: #aaa; margin-top: 0; }}
  #controls {{ margin: 12px 0; }}
  label {{ margin-right: 16px; }}
  input[type=range] {{ vertical-align: middle; width: 200px; }}
  #threshold-val {{ font-weight: bold; color: #f0a500; }}
  #plot {{ width: 100%; height: 800px; }}
  #pairs-table {{ margin-top: 20px; width: 100%; border-collapse: collapse; font-size: 0.82em; }}
  #pairs-table th {{ background: #333; padding: 6px 10px; text-align: left; }}
  #pairs-table td {{ padding: 5px 10px; border-bottom: 1px solid #333; }}
  .above {{ color: #4caf50; font-weight: bold; }}
  .below {{ color: #888; }}
</style>
</head>
<body>
<h1>Trait Similarity Report — {bundle_name}</h1>
<p>Cosine similarity between all normalized trait embeddings.
   Diagonal = 1.0 (self-similarity). Cross-species pairs only shown in table.</p>

<div id="controls">
  <label>Threshold: <input type="range" id="thresh" min="0.5" max="1.0" step="0.01"
    value="{threshold}" oninput="updateThreshold(this.value)">
  <span id="threshold-val">{threshold}</span></label>
  <label><input type="checkbox" id="cross-only" checked onchange="renderTable()">
    Cross-species only</label>
</div>

<div id="plot"></div>
<h2 style="margin-top:30px">Pair scores</h2>
<table id="pairs-table">
  <thead><tr><th>Score</th><th>Above threshold</th><th>Species A</th><th>Trait A</th>
  <th>Species B</th><th>Trait B</th></tr></thead>
  <tbody id="pairs-body"></tbody>
</table>

<script>
const Z = {z_json};
const labels = {labels_json};
const species = {species_list};
let threshold = {threshold};

const data = [{{
  type: 'heatmap',
  z: Z,
  x: labels,
  y: labels,
  colorscale: 'RdBu',
  zmin: 0,
  zmax: 1,
  hoverongaps: false,
  hovertemplate: 'Score: %{{z:.3f}}<extra></extra>',
}}];

const layout = {{
  paper_bgcolor: '#1a1a2e',
  plot_bgcolor: '#1a1a2e',
  font: {{ color: '#eee', size: 10 }},
  margin: {{ l: 220, r: 20, t: 20, b: 220 }},
  xaxis: {{ tickangle: -45 }},
  yaxis: {{ autorange: 'reversed' }},
}};

Plotly.newPlot('plot', data, layout, {{responsive: true}});

// Add threshold line shape on click
document.getElementById('plot').on('plotly_click', function(d) {{
  const score = d.points[0].z.toFixed(3);
  const xi = d.points[0].pointIndex[1];
  const yi = d.points[0].pointIndex[0];
  console.log(`Clicked: ${{labels[yi]}} vs ${{labels[xi]}} = ${{score}}`);
}});

function updateThreshold(val) {{
  threshold = parseFloat(val);
  document.getElementById('threshold-val').textContent = val;
  renderTable();
}}

function renderTable() {{
  const crossOnly = document.getElementById('cross-only').checked;
  const n = Z.length;
  let pairs = [];
  for (let i = 0; i < n; i++) {{
    for (let j = i+1; j < n; j++) {{
      const cross = species[i] !== species[j];
      if (crossOnly && !cross) continue;
      pairs.push({{
        score: Z[i][j],
        above: Z[i][j] >= threshold,
        sa: species[i], ta: labels[i].replace('<br>', ': '),
        sb: species[j], tb: labels[j].replace('<br>', ': '),
      }});
    }}
  }}
  pairs.sort((a,b) => b.score - a.score);
  const tbody = document.getElementById('pairs-body');
  tbody.innerHTML = pairs.map(p => `
    <tr class="${{p.above ? 'above' : 'below'}}">
      <td>${{p.score.toFixed(4)}}</td>
      <td>${{p.above ? '✓' : '✗'}}</td>
      <td>${{p.sa}}</td><td>${{p.ta.split(': ').slice(1).join(': ')}}</td>
      <td>${{p.sb}}</td><td>${{p.tb.split(': ').slice(1).join(': ')}}</td>
    </tr>`).join('');
}}

renderTable();
</script>
</body>
</html>
"""
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, out_path)
    print(f"[SIM] HTML report written: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Trait similarity diagnostic report.")
    parser.add_argument("--bundle-dir", required=True, help="Path to pipeline bundle directory.")
    parser.add_argument("--threshold", type=float, default=0.78,
                        help="Similarity threshold to highlight (default: 0.78).")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    if not bundle_dir.exists():
        print(f"[SIM] Bundle dir not found: {bundle_dir}")
        sys.exit(1)

    entries = discover_traits(bundle_dir)
    if len(entries) < 2:
        print("[SIM] Need at least 2 traits to compute similarity.")
        sys.exit(1)

    sim = compute_similarity_matrix(entries)

    out_json = bundle_dir / "trait_similarity.json"
    out_html = bundle_dir / "trait_similarity.html"

    write_json(entries, sim, out_json, threshold=args.threshold)
    write_html(entries, sim, out_html, threshold=args.threshold,
               bundle_name=bundle_dir.name)

    print(f"\n[SIM] Open the report in your browser:")
    print(f"      {out_html.resolve()}")


if __name__ == "__main__":
    main()
