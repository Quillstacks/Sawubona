"""Administer the survey to the models (Step 1), with checkpoint/resume.

For every (model × language × item × wording-variant × sample) we ask Ollama once and
append one JSON line to ``results/raw/<model>/<lang>.jsonl``. Re-running skips work already
on disk, so a crash or a closed SSH session resumes instead of restarting. Repeated
sampling (``k_forced``) yields the model's *answer distribution* per item — the quantity
compared against the human marginal in Step 3.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from .. import config
from ..config import Model
from . import prompts
from .ollama_client import OllamaClient
from .parse import parse_answer

log = logging.getLogger(__name__)


@dataclass
class RunConfig:
    languages: list[str] = field(default_factory=lambda: [l.code for l in config.load_languages()])
    k_forced: int = 0                 # 0 -> use model defaults.samples_per_item
    k_open: int = 2                   # open-ended cross-check samples (variant 0 only)
    temperature: float = 0.0          # 0 -> use model defaults.temperature
    include_paraphrases: bool = True
    pull_missing: bool = False
    host: str | None = None
    out_dir: Path | None = None

    def resolved(self) -> "RunConfig":
        defaults, _ = config.load_models()
        if self.k_forced <= 0:
            self.k_forced = defaults.samples_per_item
        if self.temperature <= 0:
            self.temperature = defaults.temperature
        if self.out_dir is None:
            self.out_dir = config.paths().results / "raw"
        return self


def response_path(out_dir: Path, model: Model, lang_code: str) -> Path:
    return out_dir / model.name / f"{lang_code}.jsonl"


def _seed(*parts: object) -> int:
    h = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=6)
    return int.from_bytes(h.digest(), "big") % (2 ** 31)


def _done_keys(path: Path) -> set[tuple]:
    done: set[tuple] = set()
    if not path.is_file():
        return done
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["item_id"], r["mode"], r["variant_idx"], r["sample_idx"]))
            except (ValueError, KeyError):
                continue
    return done


def _units_for_language(lang_code: str, include_paraphrases: bool, k_open: int
                        ) -> Iterator[prompts.PromptUnit]:
    _, items = config.load_items()
    for item in items:
        variants = prompts.localized_variants(item, lang_code)
        if not include_paraphrases:
            variants = variants[:1]
        for vi, text in enumerate(variants):
            yield prompts.PromptUnit(item.id, lang_code, vi, text, item.scale, "forced")
        if k_open > 0:
            yield prompts.PromptUnit(item.id, lang_code, 0, variants[0], item.scale, "open")


def run(models: Iterable[Model] | None = None, cfg: RunConfig | None = None,
        client: OllamaClient | None = None) -> None:
    cfg = (cfg or RunConfig()).resolved()
    models = list(models) if models is not None else config.roster()
    client = client or OllamaClient(host=cfg.host)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        if not client.ensure(model.tag, pull=cfg.pull_missing):
            log.warning("Model %s not available (pull_missing=%s); skipping. "
                        "Try scripts/01_pull_models.sh.", model.tag, cfg.pull_missing)
            continue
        for lang_code in cfg.languages:
            _run_one(model, lang_code, cfg, client)


def _run_one(model: Model, lang_code: str, cfg: RunConfig, client: OllamaClient) -> None:
    path = response_path(cfg.out_dir, model, lang_code)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(path)

    units = list(_units_for_language(lang_code, cfg.include_paraphrases, cfg.k_open))
    n_written = 0
    with open(path, "a", encoding="utf-8") as fh:
        for unit in units:
            k = cfg.k_open if unit.mode == "open" else cfg.k_forced
            schema = None if unit.mode == "open" else prompts.enum_schema(unit.scale)
            num_predict = 160 if unit.mode == "open" else 24
            messages = prompts.build_messages(unit)
            for s in range(k):
                key = (unit.item_id, unit.mode, unit.variant_idx, s)
                if key in done:
                    continue
                seed = _seed(model.tag, lang_code, unit.item_id, unit.mode,
                             unit.variant_idx, s)
                raw = client.chat(model.tag, messages, schema=schema,
                                  temperature=cfg.temperature, seed=seed,
                                  num_predict=num_predict)
                pr = parse_answer(raw, unit.scale, lang_code)
                rec = {
                    "model_tag": model.tag, "model": model.name,
                    "params_b": model.params_b, "specialized": model.specialized,
                    "lang": lang_code, "item_id": unit.item_id,
                    "scale": unit.scale.name, "mode": unit.mode,
                    "variant_idx": unit.variant_idx, "sample_idx": s, "seed": seed,
                    "raw": raw, "category": pr.category, "status": pr.status.value,
                    "ts": time.time(),
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                n_written += 1
    log.info("%s / %s: %d new responses (%d already on disk)",
             model.name, lang_code, n_written, len(done))


def iter_records(path: Path) -> Iterator[dict]:
    if not path.is_file():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
