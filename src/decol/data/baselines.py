"""Build, persist, and load the unified human-baseline table.

A baseline is, per (language, item), the speakers' marginal answer distribution on that
item's native scale. We merge the Afrobarometer (9 languages) and SASAS (Tshivenda)
sources into ``data/processed/baselines.json``. isiNdebele has no baseline by design.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from .. import config
from ..dist import Distribution
from . import afrobarometer, sasas

log = logging.getLogger(__name__)

Baselines = dict[str, dict[str, Distribution]]   # {lang: {item_id: Distribution}}


def default_path() -> Path:
    return config.paths().data_processed / "baselines.json"


def build(save: bool = True) -> Baselines:
    """Merge all human-baseline sources into one table; optionally persist to JSON."""
    merged: Baselines = {}
    try:
        ab = afrobarometer.build_baselines()
    except FileNotFoundError as exc:
        log.warning("Afrobarometer baselines unavailable: %s", exc)
        ab = {}
    for lang, items in ab.items():
        merged.setdefault(lang, {}).update(items)

    sa = sasas.build_baselines()
    for lang, items in sa.items():
        merged.setdefault(lang, {}).update(items)

    if save:
        save_baselines(merged)
    _report_coverage(merged)
    return merged


def save_baselines(baselines: Baselines, path: Path | None = None) -> Path:
    path = path or default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {lang: {iid: d.to_dict() for iid, d in items.items()}
               for lang, items in baselines.items()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    log.info("Wrote baselines -> %s", path)
    return path


def load(path: Path | None = None) -> Baselines:
    path = path or default_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Build it first: python scripts/02_run_survey.py "
            "(it builds baselines), or call decol.data.baselines.build()."
        )
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return {lang: {iid: Distribution.from_dict(d) for iid, d in items.items()}
            for lang, items in payload.items()}


def _report_coverage(baselines: Baselines) -> None:
    _, items = config.load_items()
    for lang in config.load_languages():
        if lang.baseline == "none":
            continue
        have = baselines.get(lang.code, {})
        log.info("baseline %s (%s): %d/%d items", lang.code, lang.name,
                 len(have), len(items))


# --------------------------------------------------------------- synthetic (smoke/tests)
def synthetic(seed: int = 0) -> Baselines:
    """Plausible random baselines for smoke runs / tests BEFORE real microdata is present.

    NOT for analysis — clearly labelled so results built on it can be flagged. Generates a
    distribution for every (language-with-baseline, item).
    """
    rng = random.Random(seed)
    _, items = config.load_items()
    out: Baselines = {}
    for lang in config.load_languages():
        if lang.baseline == "none":
            continue
        for item in items:
            cats = item.scale.ordered
            weights = [rng.random() + 0.1 for _ in cats]
            out.setdefault(lang.code, {})[item.id] = Distribution.from_counts(
                cats, dict(zip(cats, weights)))
    return out
