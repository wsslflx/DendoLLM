#!/usr/bin/env python3
"""
API connectivity test script.
Tests LangChain ChatOllama against the configured backend.
Reads credentials from .env in the project root.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from core.llm_backend import ollama_base_url, ollama_headers, resolve_chat_model

import os

BASE_URL = ollama_base_url()
API_KEY  = os.environ.get("OLLAMA_API_KEY", "")
MODEL    = resolve_chat_model(None)

print()
print("=== Config ===")
print(f"  OLLAMA_BASE_URL  : {BASE_URL}")
print(f"  OLLAMA_CHAT_MODEL: {MODEL}")
print(f"  OLLAMA_API_KEY   : {API_KEY[:8]}{'*' * (len(API_KEY) - 8) if len(API_KEY) > 8 else '(not set)'}")
print()

MESSAGES = [
    ("system",    "You are a helpful trip advisor."),
    ("human",     "What can we visit in Hamburg?"),
    ("assistant", "Hamburg has many great spots including the Speicherstadt, Reeperbahn, and Miniatur Wunderland."),
    ("human",     "How about dark tourism?"),
]

errors = []

# ── Test: LangChain ChatOllama ────────────────────────────────────────────────
print("=== Test: LangChain ChatOllama ===")
try:
    from langchain_ollama.chat_models import ChatOllama

    llm = ChatOllama(
        base_url=BASE_URL,
        model=MODEL,
        temperature=0.0,
        seed=28,
        num_ctx=4096,
        num_predict=256,
        top_k=100,
        top_p=0.95,
        client_kwargs={"headers": ollama_headers(), "timeout": 300},
    )

    print(f"  Invoking model '{MODEL}' via LangChain...")
    t0 = time.time()
    response = llm.invoke(MESSAGES)
    elapsed = time.time() - t0

    content = response.content if hasattr(response, "content") else str(response)
    print(f"  Elapsed : {elapsed:.1f}s")
    print(f"  Response: {content[:300].strip()}")
    print("  Status  : OK")
except Exception as e:
    print(f"  FAILED: {e}")
    errors.append(str(e))

print()

if errors:
    print("=== SUMMARY: FAILED ===")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("=== SUMMARY: OK ===")
