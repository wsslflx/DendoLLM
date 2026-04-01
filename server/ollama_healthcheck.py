#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parents[1]))

from core.llm_backend import make_chat_llm, make_embeddings, resolve_chat_model, resolve_embed_model


def main() -> None:
    chat_model = resolve_chat_model(None)
    embed_model = resolve_embed_model(None)

    llm = make_chat_llm(model=chat_model, temperature=0.0)
    emb = make_embeddings(embed_model)

    chat_resp = llm.invoke([("user", "Reply with exactly: ok")])
    chat_text = chat_resp.content if hasattr(chat_resp, "content") else str(chat_resp)

    vec = emb.embed_query("healthcheck")
    print("chat_model:", chat_model)
    print("embed_model:", embed_model)
    print("chat_response:", str(chat_text).strip())
    print("embedding_dim:", len(vec))
    print("status: ok")


if __name__ == "__main__":
    main()
