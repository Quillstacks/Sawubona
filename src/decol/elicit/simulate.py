"""Offline response simulator — run the whole pipeline without Ollama or real microdata.

Writes the exact same ``results/raw/<model>/<lang>.jsonl`` records the real runner does, so
Steps 3-4 produce populated result files for development and for wiring up downstream
analysis before models/data are ready. The simulated model distribution is a mixture of
the language's own (synthetic) baseline and the English-anchored baseline, weighted by
(1 - capability): low-capability languages collapse toward English defaults — the
proposal's hypothesis, made visible so you can confirm the analysis recovers it.

NOTE: clearly a SIMULATION. The manifest records ``simulated: true`` for that survey run.
"""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

from .. import capability as capability_pkg
from .. import config
from ..config import Model
from ..dist import Distribution
from .runner import response_path

log = logging.getLogger(__name__)


def _mix(own: Distribution, anchor: Distribution, alpha: float,
         categories) -> Distribution:
    o = own.reindex(categories).probs
    a = anchor.reindex(categories).probs
    mixed = {c: (1 - alpha) * o[i] + alpha * a[i] for i, c in enumerate(categories)}
    return Distribution.from_counts(categories, mixed)


def simulate(models=None, languages=None, baselines=None, capability_scores=None,
             k: int = 8, drift: float = 1.0, seed: int = 0,
             out_dir: Path | None = None) -> Path:
    from ..data import baselines as baselines_mod

    models = list(models) if models is not None else config.roster()
    languages = languages or [l.code for l in config.load_languages()]
    baselines = baselines if baselines is not None else baselines_mod.synthetic(seed)
    capability_scores = capability_scores or capability_pkg.synthetic(models, seed)
    out_dir = out_dir or (config.paths().results / "raw")
    _, items = config.load_items()
    anchor = baselines.get("eng", {})

    rng = random.Random(seed)
    for model in models:
        caps = capability_scores.get(model.name, {})
        for lang in languages:
            own = baselines.get(lang, {})
            if not own:
                continue   # no baseline (e.g. isiNdebele) -> nothing to simulate against
            cap = caps.get(lang, 0.5)
            alpha = max(0.0, min(1.0, (1.0 - cap) * drift))
            path = response_path(out_dir, model, lang)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:   # overwrite (fresh simulation)
                for item in items:
                    if item.id not in own:
                        continue
                    cats = item.scale.ordered
                    a = anchor.get(item.id, own[item.id])
                    dist = _mix(own[item.id], a, alpha, cats)
                    weights = list(dist.reindex(cats).probs)
                    for s in range(k):
                        cat = rng.choices(cats, weights=weights, k=1)[0]
                        fh.write(json.dumps({
                            "model_tag": model.tag, "model": model.name,
                            "params_b": model.params_b, "specialized": model.specialized,
                            "lang": lang, "item_id": item.id, "scale": item.scale.name,
                            "mode": "forced", "variant_idx": 0, "sample_idx": s,
                            "seed": seed, "raw": cat, "category": cat, "status": "ok",
                            "ts": time.time(), "simulated": True,
                        }) + "\n")
    log.info("Simulated responses for %d models × %d languages -> %s",
             len(models), len(languages), out_dir)
    return out_dir
