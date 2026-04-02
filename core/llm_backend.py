#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Load .env from project root so subprocesses that import this module
# pick up OLLAMA_* settings without needing manual shell exports.
_env_path = Path(__file__).parents[1] / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

from langchain_ollama import ChatOllama, OllamaEmbeddings

DEFAULT_OLLAMA_BASE_URL = "https://dev.chat.cosy.bio/ollama"
DEFAULT_CHAT_MODEL = "qwen2.5:latest"


def ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")


def ollama_headers(require_api_key: bool = True) -> dict[str, str]:
    key = os.getenv("OLLAMA_API_KEY", "").strip()
    if key:
        return {"Authorization": f"Bearer {key}"}
    if require_api_key:
        raise RuntimeError(
            "Missing OLLAMA_API_KEY. Set OLLAMA_API_KEY for authenticated access "
            "to dev.chat.cosy.bio."
        )
    return {}


def resolve_chat_model(model: str | None = None) -> str:
    if isinstance(model, str) and model.strip():
        return model.strip()
    env_model = os.getenv("OLLAMA_CHAT_MODEL", "").strip()
    if env_model:
        return env_model
    return DEFAULT_CHAT_MODEL


def resolve_embed_model(model: str | None = None) -> str:
    if isinstance(model, str) and model.strip():
        return model.strip()
    env_model = os.getenv("OLLAMA_EMBED_MODEL", "").strip()
    if env_model:
        return env_model
    raise RuntimeError(
        "Missing OLLAMA_EMBED_MODEL. Set OLLAMA_EMBED_MODEL to an embedding model "
        "served by your Ollama backend."
    )


def make_chat_llm(model: str | None, temperature: float, **kwargs: Any) -> ChatOllama:
    kwargs.setdefault("num_ctx", 16000)    # cap context to reduce VRAM pressure and proxy timeout risk
    kwargs.setdefault("num_predict", 4096) # enough for thinking tokens + JSON output
    resolved = resolve_chat_model(model)
    # Disable thinking mode for qwen3.x — without this, the model burns the entire
    # num_predict budget on thinking tokens before generating any JSON output.
    # langchain_ollama maps `reasoning` → Ollama API `think` field (chat_models.py:770).
    # Setting system="/no_think" does NOT disable thinking; reasoning=False does.
    if "qwen3" in resolved.lower():
        kwargs.setdefault("reasoning", False)
    return ChatOllama(
        base_url=ollama_base_url(),
        model=resolved,
        temperature=temperature,
        client_kwargs={"headers": ollama_headers(), "timeout": 300, **kwargs.pop("client_kwargs", {})},
        **kwargs,
    )


def make_embeddings(model: str | None = None, **kwargs: Any) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=ollama_base_url(),
        model=resolve_embed_model(model),
        client_kwargs={"headers": ollama_headers(), **kwargs.pop("client_kwargs", {})},
        **kwargs,
    )
