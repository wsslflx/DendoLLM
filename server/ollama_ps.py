#!/usr/bin/env python3
"""Show models loaded in VRAM (/api/ps) or all available models (/api/tags) as fallback."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from core.llm_backend import ollama_base_url, ollama_headers

import httpx

base    = ollama_base_url()
headers = ollama_headers(require_api_key=False)

# Try /api/ps (active models in VRAM) — may be admin-only on some Open WebUI instances
r_ps = httpx.get(f"{base}/api/ps", headers=headers, timeout=10)
if r_ps.status_code == 200:
    models = r_ps.json().get("models", [])
    if not models:
        print("No models currently loaded in VRAM.")
    else:
        print(f"{'Model':<40} {'Size':>10}  {'VRAM':>10}  Expires")
        print("-" * 75)
        for m in models:
            name    = m.get("name", "?")
            size    = m.get("size", 0) / 1e9
            vram    = m.get("size_vram", 0) / 1e9
            expires = m.get("expires_at", "?")[:19].replace("T", " ")
            print(f"{name:<40} {size:>9.1f}G  {vram:>9.1f}G  {expires}")
else:
    print(f"/api/ps not accessible ({r_ps.status_code}) — showing all available models instead:\n")
    r_tags = httpx.get(f"{base}/api/tags", headers=headers, timeout=10)
    r_tags.raise_for_status()
    models = r_tags.json().get("models", [])
    print(f"{'Model':<40} {'Size':>10}  {'Family':<15}  {'Params'}")
    print("-" * 80)
    for m in models:
        name   = m.get("name", "?")
        size   = m.get("size", 0) / 1e9
        det    = m.get("details", {})
        family = det.get("family", "?")
        params = det.get("parameter_size", "?")
        print(f"{name:<40} {size:>9.1f}G  {family:<15}  {params}")
