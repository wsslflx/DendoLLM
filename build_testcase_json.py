#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

GBIF_BASE = "https://api.gbif.org/v1/species"


def normalize_scientific_name(name: str) -> str:
    return " ".join(name.replace("_", " ").strip().split()).lower()


def gbif_match(name: str) -> dict | None:
    resp = requests.get(
        f"{GBIF_BASE}/match",
        params={"name": name},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    usage_key = data.get("usageKey")
    if not usage_key:
        return None
    return data


def gbif_english_vernaculars(usage_key: int) -> list[str]:
    resp = requests.get(
        f"{GBIF_BASE}/{usage_key}/vernacularNames",
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    names: list[str] = []
    seen: set[str] = set()
    for item in results:
        lang = (item.get("language") or "").lower()
        vernacular = (item.get("vernacularName") or "").strip()
        if not vernacular:
            continue
        if lang not in {"eng", "en", "english"}:
            continue
        key = vernacular.casefold()
        if key in seen:
            continue
        seen.add(key)
        names.append(vernacular)
    return names


def parse_species_arg(species_raw: str) -> list[str]:
    parts = [p.strip() for p in species_raw.split(",")]
    return [p for p in parts if p]


def build_entries(species: list[str]) -> list[dict]:
    entries: list[dict] = []
    for raw_name in species:
        normalized = normalize_scientific_name(raw_name)
        match = gbif_match(normalized)
        canonical = normalized
        aliases: list[str] = []

        if match:
            canonical_name = (match.get("canonicalName") or "").strip()
            sci = canonical_name or (match.get("scientificName") or "").strip()
            if sci:
                canonical = normalize_scientific_name(sci)
            usage_key = match.get("usageKey")
            if usage_key:
                try:
                    aliases.extend(gbif_english_vernaculars(int(usage_key)))
                except Exception:
                    pass

        if canonical not in aliases:
            aliases.append(canonical)

        entries.append({"canonical": canonical, "aliases": aliases})
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build testcase JSON from a comma-separated list of scientific names using GBIF."
    )
    parser.add_argument(
        "--species",
        required=True,
        help="Comma-separated scientific names (spaces or underscores supported).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON path, e.g. testcase5.json",
    )
    args = parser.parse_args()

    species = parse_species_arg(args.species)
    if not species:
        raise SystemExit("No species provided.")

    entries = build_entries(species)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(entries)} species)")


if __name__ == "__main__":
    main()
