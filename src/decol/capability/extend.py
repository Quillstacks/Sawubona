"""Capability extension for the under-served languages AfroBench does not cover well —
notably **isiNdebele (nbl)** and **Tshivenda (ven)** — using MzansiText-derived resources
(refs [4, 5]).

Each language has a small multiple-choice eval set at
``data/processed/extension/<lang>.jsonl`` (one JSON object per line):

    {"prompt": "...", "options": ["A", "B", "C", "D"], "answer": "B"}

We administer it through the same Ollama path (structured-output enum over the options),
score accuracy, chance-correct to [0, 1], and return a comparable capability number.
Populate the eval sets from MzansiText / the sentiment corpus (ref [5]) — see
data/processed/extension/README.md. Missing set -> no score (language is then ungated and
reported as a documented gap, like its human baseline).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .. import config
from ..elicit.ollama_client import OllamaClient
from ..elicit.parse import _normalize

log = logging.getLogger(__name__)

DEFAULT_EXTENSION_LANGS = ["ven", "nbl"]


def eval_set_path(lang_code: str) -> Path:
    return config.paths().data_processed / "extension" / f"{lang_code}.jsonl"


def load_eval_set(lang_code: str) -> list[dict]:
    path = eval_set_path(lang_code)
    if not path.is_file():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _schema(options: list[str]) -> dict:
    return {"type": "object",
            "properties": {"answer": {"type": "string", "enum": list(options)}},
            "required": ["answer"]}


def score_language(model_tag: str, lang_code: str, client: OllamaClient,
                   limit: int | None = None) -> float | None:
    rows = load_eval_set(lang_code)
    if not rows:
        log.info("No extension eval set for %s (%s); capability left unknown.",
                 lang_code, eval_set_path(lang_code))
        return None
    if limit:
        rows = rows[:limit]

    correct = 0
    n_classes = []
    for ex in rows:
        options = list(ex["options"])
        n_classes.append(len(options))
        messages = [
            {"role": "system", "content": "Answer the multiple-choice question. "
                                          "Reply with exactly one of the options."},
            {"role": "user", "content": ex["prompt"] + "\nOptions: " + " / ".join(options)},
        ]
        raw = client.chat(model_tag, messages, schema=_schema(options),
                          temperature=0.0, seed=0, num_predict=24)
        pred = _extract(raw, options)
        if pred is not None and _normalize(pred) == _normalize(str(ex["answer"])):
            correct += 1

    acc = correct / len(rows)
    baseline = sum(1.0 / c for c in n_classes) / len(n_classes) if n_classes else 0.0
    score = (acc - baseline) / (1.0 - baseline) if 0 < baseline < 1 else acc
    return round(min(1.0, max(0.0, score)), 3)


def _extract(raw: str, options: list[str]) -> str | None:
    s = raw.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "answer" in obj:
                return str(obj["answer"])
        except ValueError:
            pass
    low = _normalize(s)
    for opt in options:
        if _normalize(opt) in low:
            return opt
    return None


def run_model(model_tag: str, model_name: str, langs: list[str] | None = None,
              client: OllamaClient | None = None, limit: int | None = None
              ) -> dict[str, float]:
    client = client or OllamaClient()
    langs = langs or DEFAULT_EXTENSION_LANGS
    out: dict[str, float] = {}
    for lang in langs:
        score = score_language(model_tag, lang, client, limit=limit)
        if score is not None:
            out[lang] = score
    return out
