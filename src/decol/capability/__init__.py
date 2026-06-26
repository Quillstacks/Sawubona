"""Per-language language-capability scoring (Step 2).

AfroBench (subsuming IrokoBench) scores the languages it covers; an extension scores the
under-served ones (isiNdebele, Tshivenda). Scores are normalised to [0, 1] and stored as
``results/capability/scores.json`` = ``{model_name: {lang_code: score}}``.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from .. import config

log = logging.getLogger(__name__)

Scores = dict[str, dict[str, float]]


def default_path() -> Path:
    return config.paths().results / "capability" / "scores.json"


def save_scores(scores: Scores, path: Path | None = None) -> Path:
    path = path or default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(scores, fh, indent=2, ensure_ascii=False)
    log.info("Wrote capability scores -> %s", path)
    return path


def load_scores(path: Path | None = None) -> Scores:
    path = path or default_path()
    if not path.is_file():
        log.warning("No capability scores at %s; divergence will be ungated.", path)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def merge_scores(*sources: Scores) -> Scores:
    out: Scores = {}
    for src in sources:
        for model, per_lang in src.items():
            out.setdefault(model, {}).update(per_lang)
    return out


# --------------------------------------------------------------- synthetic (smoke/tests)
def synthetic(models=None, seed: int = 0) -> Scores:
    """Plausible capability scores for smoke runs (lower for under-served languages)."""
    rng = random.Random(seed)
    models = list(models) if models is not None else config.roster()
    out: Scores = {}
    for m in models:
        for lang in config.load_languages():
            base = 0.75 if lang.afrobench else 0.35
            base += 0.05 * (m.params_b ** 0.5)            # mild size effect
            base += 0.10 if (m.specialized and not lang.afrobench) else 0.0
            out.setdefault(m.name, {})[lang.code] = round(
                min(0.98, max(0.05, base + rng.uniform(-0.08, 0.08))), 3)
    return out
