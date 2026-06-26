"""Run manifest: a single machine-readable index of every result artefact and its
provenance, written to ``results/run_manifest.json``.

Each pipeline step merges its own section (survey / capability / divergence / analysis) so
that after a full run the manifest records what was produced, when, from which models and
languages, and crucially whether any baseline or capability number was *synthetic* (smoke)
versus real microdata — so downstream work is never misled about what it is reading.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

from . import __version__, config


def path() -> Path:
    return config.paths().results / "run_manifest.json"


def _load() -> dict:
    p = path()
    if p.is_file():
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def update(section: str, payload: dict) -> Path:
    """Merge ``payload`` into ``manifest[section]`` and rewrite the file."""
    man = _load()
    man.setdefault("decol_version", __version__)
    man.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S"))
    man["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    man.setdefault("host", platform.node())
    man[section] = {**(man.get(section) or {}), **payload,
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, ensure_ascii=False)
    return p


def config_snapshot() -> dict:
    """The roster / languages / items in effect, for reproducibility."""
    defaults, _ = config.load_models()
    _, items = config.load_items()
    return {
        "languages": [l.code for l in config.load_languages()],
        "models": [{"tag": m.tag, "params_b": m.params_b,
                    "specialized": m.specialized} for m in config.roster()],
        "items": [{"id": it.id, "qcode": it.qcode, "verified": it.verified}
                  for it in items],
        "samples_per_item": defaults.samples_per_item,
        "temperature": defaults.temperature,
        "gpu_mb_budget": defaults.gpu_mb_budget,
    }
