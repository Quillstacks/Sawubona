"""Configuration loading: repo paths, languages, model roster, and political items.

Everything the pipeline does is driven by the YAML files under ``config/``; no roster,
language list, or item text is hardcoded in the Python. Load once and pass the resulting
dataclasses around.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- paths
def repo_root() -> Path:
    """Locate the repository root (the dir containing ``config/`` and ``pyproject.toml``).

    Overridable with the ``DECOL_ROOT`` environment variable for tests / odd layouts.
    """
    env = os.environ.get("DECOL_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "config").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    # Fallback: two levels up from src/decol/config.py
    return here.parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def translations(self) -> Path:
        return self.config / "translations"

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def afrobarometer(self) -> Path:
        return self.data_raw / "afrobarometer"

    @property
    def sasas(self) -> Path:
        return self.data_raw / "sasas"

    def ensure(self) -> "Paths":
        for p in (self.data_raw, self.data_processed, self.results,
                  self.afrobarometer, self.sasas):
            p.mkdir(parents=True, exist_ok=True)
        return self


def paths() -> Paths:
    return Paths(repo_root())


def _read_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ----------------------------------------------------------------------- languages
@dataclass(frozen=True)
class Language:
    code: str
    name: str
    native: str
    home_share: float
    baseline: str          # afrobarometer | sasas | none
    afrobench: bool
    ab_lang_code: Any = None
    anchor: bool = False

    @property
    def has_human_baseline(self) -> bool:
        return self.baseline in ("afrobarometer", "sasas")


@lru_cache(maxsize=1)
def load_languages() -> list[Language]:
    raw = _read_yaml(paths().config / "languages.yaml")["languages"]
    return [Language(
        code=e["code"], name=e["name"], native=e["native"],
        home_share=float(e["home_share"]), baseline=e["baseline"],
        afrobench=bool(e["afrobench"]), ab_lang_code=e.get("ab_lang_code"),
        anchor=bool(e.get("anchor", False)),
    ) for e in raw]


def language(code: str) -> Language:
    for lang in load_languages():
        if lang.code == code:
            return lang
    raise KeyError(f"unknown language code: {code!r}")


def anchor_language() -> Language:
    for lang in load_languages():
        if lang.anchor:
            return lang
    return language("eng")


# -------------------------------------------------------------------------- models
@dataclass(frozen=True)
class Model:
    tag: str
    family: str
    params_b: float
    approx_gpu_mb: int
    quant: str
    specialized: bool = False
    requires_conversion: bool = False
    hf_repo: Any = None

    @property
    def name(self) -> str:
        """Filesystem-safe identifier for results dirs."""
        return self.tag.replace(":", "_").replace("/", "_")


@dataclass(frozen=True)
class ModelDefaults:
    quant: str = "Q4_K_M"
    gpu_mb_budget: int = 6144
    samples_per_item: int = 8
    temperature: float = 0.7


@lru_cache(maxsize=1)
def load_models() -> tuple[ModelDefaults, list[Model]]:
    raw = _read_yaml(paths().config / "models.yaml")
    defaults = ModelDefaults(**(raw.get("defaults") or {}))
    fields = ("tag", "family", "params_b", "approx_gpu_mb", "quant",
              "specialized", "requires_conversion", "hf_repo")
    models = [Model(**{k: m.get(k) for k in fields if k in m}) for m in raw["models"]]
    return defaults, models


def roster(include_oversized: bool = False) -> list[Model]:
    """Models within the GPU-memory budget (the ≤ ~6 GB edge constraint)."""
    defaults, models = load_models()
    if include_oversized:
        return models
    return [m for m in models if m.approx_gpu_mb <= defaults.gpu_mb_budget]


# --------------------------------------------------------------------------- items
@dataclass(frozen=True)
class Scale:
    name: str
    type: str               # categorical | ordinal | binary
    ordered: list[str]

    @property
    def is_ordinal(self) -> bool:
        return self.type in ("ordinal", "binary")

    def index(self, category: str) -> int:
        return self.ordered.index(category)


@dataclass(frozen=True)
class Item:
    id: str
    qcode: str
    theme: str
    scale: Scale
    english: str
    paraphrases: list[str] = field(default_factory=list)
    verified: bool = False
    higher_means: str = ""

    def prompts(self, include_paraphrases: bool = True) -> list[str]:
        """Canonical English wording first, then paraphrases (for invariance checks)."""
        out = [self.english.strip()]
        if include_paraphrases:
            out += [p.strip() for p in self.paraphrases]
        return out


@lru_cache(maxsize=1)
def load_items() -> tuple[dict[str, Scale], list[Item]]:
    raw = _read_yaml(paths().config / "items_political.yaml")
    scales = {name: Scale(name=name, type=spec["type"], ordered=list(spec["ordered"]))
              for name, spec in raw["scales"].items()}
    items = []
    for it in raw["items"]:
        items.append(Item(
            id=it["id"], qcode=str(it.get("qcode", "")), theme=it.get("theme", ""),
            scale=scales[it["scale"]], english=it["english"],
            paraphrases=list(it.get("paraphrases", [])),
            verified=bool(it.get("verified", False)),
            higher_means=it.get("higher_means", ""),
        ))
    return scales, items


# -------------------------------------------------------------- translations / lexicon
@lru_cache(maxsize=None)
def load_translations(lang_code: str) -> dict[str, dict]:
    """Return {item_id: {'text': str, 'paraphrases': [str]}} for a language, or {}."""
    path = paths().translations / f"{lang_code}.yaml"
    if not path.is_file():
        return {}
    data = _read_yaml(path) or {}
    return data.get("items") or {}


@lru_cache(maxsize=1)
def load_answer_lexicon() -> dict[str, Any]:
    path = paths().config / "answer_lexicon.yaml"
    return _read_yaml(path) if path.is_file() else {"canonical": {}, "by_language": {}}
