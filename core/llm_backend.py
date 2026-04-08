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

from ollama import Client as OllamaClient
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

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


def _prompt_value_to_messages(input_val) -> list[dict]:
    """Convert a LangChain PromptValue, message list, or plain string to ollama messages."""
    if isinstance(input_val, str):
        return [{"role": "user", "content": input_val}]
    if hasattr(input_val, "to_messages"):
        # ChatPromptValue / StringPromptValue
        lc_messages = input_val.to_messages()
    elif isinstance(input_val, list):
        lc_messages = input_val
    else:
        return [{"role": "user", "content": str(input_val)}]

    messages = []
    for m in lc_messages:
        # Support (role, content) tuples as well as LangChain message objects
        if isinstance(m, tuple) and len(m) == 2:
            messages.append({"role": m[0], "content": m[1]})
            continue
        msg_type = getattr(m, "type", None)
        if msg_type == "system" or isinstance(m, SystemMessage):
            role = "system"
        elif msg_type == "ai" or isinstance(m, AIMessage):
            role = "assistant"
        else:
            role = "user"
        messages.append({"role": role, "content": m.content})
    return messages


def make_chat_llm(model: str | None, temperature: float, **kwargs: Any) -> RunnableLambda:
    """Return a LangChain-compatible Runnable backed by the native ollama.Client.

    Uses the native ollama Python library instead of langchain_ollama.ChatOllama
    to avoid timeout/streaming issues with newer Ollama server versions.
    Supports the same | prompt | llm chain syntax and .invoke() calls.
    """
    resolved = resolve_chat_model(model)
    fmt = kwargs.pop("format", None)
    num_ctx = kwargs.pop("num_ctx", 16000)
    num_predict = kwargs.pop("num_predict", 4096)

    options = {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict}

    # Disable thinking mode for qwen3.x models
    disable_think = "qwen3" in resolved.lower()

    client = OllamaClient(host=ollama_base_url(), headers=ollama_headers())

    def _invoke(input_val):
        messages = _prompt_value_to_messages(input_val)
        chat_kwargs: dict[str, Any] = {
            "model": resolved,
            "messages": messages,
            "options": options,
        }
        if fmt:
            chat_kwargs["format"] = fmt
        if disable_think:
            chat_kwargs["think"] = False
        response = client.chat(**chat_kwargs)
        return AIMessage(content=response.message.content)

    return RunnableLambda(_invoke)


def make_embeddings(model: str | None = None, **kwargs: Any) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=ollama_base_url(),
        model=resolve_embed_model(model),
        client_kwargs={"headers": ollama_headers(), **kwargs.pop("client_kwargs", {})},
        **kwargs,
    )
