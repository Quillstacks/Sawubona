"""Thin wrapper over the Ollama Python client for constrained, repeatable sampling.

Imported lazily so the rest of the package imports on machines without Ollama. The
production run targets a local Ollama daemon (``OLLAMA_HOST`` honoured by the library).
"""
from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, host: str | None = None, max_retries: int = 3,
                 retry_wait: float = 2.0):
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "The 'ollama' package is required for elicitation "
                "(`pip install ollama`) plus a running Ollama daemon."
            ) from exc
        self._ollama = ollama
        self._client = ollama.Client(host=host) if host else ollama.Client()
        self.max_retries = max_retries
        self.retry_wait = retry_wait

    # ----------------------------------------------------------------- availability
    def available_models(self) -> set[str]:
        try:
            resp = self._client.list()
        except Exception as exc:  # pragma: no cover
            log.warning("ollama list failed: %s", exc)
            return set()
        names = set()
        for m in resp.get("models", []):
            name = m.get("model") or m.get("name")
            if name:
                names.add(name)
                names.add(name.split(":")[0])
        return names

    def ensure(self, tag: str, pull: bool = False) -> bool:
        avail = self.available_models()
        if tag in avail or tag.split(":")[0] in avail:
            return True
        if pull:
            log.info("Pulling %s ...", tag)
            self._client.pull(tag)
            return True
        return False

    # ----------------------------------------------------------------------- chat
    def chat(self, model: str, messages: list[dict], *, schema: dict | None = None,
             temperature: float = 0.7, seed: int | None = None,
             num_predict: int = 32) -> str:
        """Single completion; returns raw assistant text (JSON string if ``schema`` set)."""
        options: dict[str, Any] = {"temperature": temperature, "num_predict": num_predict}
        if seed is not None:
            options["seed"] = seed
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "options": options}
        if schema is not None:
            kwargs["format"] = schema   # Ollama: pass a JSON schema dict to constrain output

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.chat(**kwargs)
                return (resp.get("message") or {}).get("content", "") or ""
            except Exception as exc:  # pragma: no cover - network/daemon dependent
                last_exc = exc
                log.warning("chat attempt %d/%d for %s failed: %s",
                            attempt, self.max_retries, model, exc)
                time.sleep(self.retry_wait * attempt)
        raise RuntimeError(f"ollama chat failed for {model}: {last_exc}")
