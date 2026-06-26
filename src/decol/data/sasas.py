"""Load SASAS microdata for the Tshivenda human-baseline cross-check.

SASAS (HSRC) is gated, so the file is placed manually in ``data/raw/sasas/`` (see
data/README.md). The recode lives in ``config/sasas_recode.yaml`` and mirrors the
Afrobarometer schema. If either the data or the recode is missing, this returns ``{}`` and
the pipeline simply omits the Tshivenda baseline (documented gap), exactly like isiNdebele.
"""
from __future__ import annotations

import glob
import logging
from pathlib import Path

from .. import config
from ..dist import Distribution

log = logging.getLogger(__name__)


def _read_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_recode() -> dict:
    path = config.paths().config / "sasas_recode.yaml"
    if not path.is_file():
        return {}
    spec = _read_yaml(path)
    spec["global"] = spec.get("global") or {}
    spec["items"] = spec.get("items") or {}
    return spec


def find_sav() -> Path | None:
    hits = sorted(glob.glob(str(config.paths().sasas / "*.sav")))
    return Path(hits[0]) if hits else None


def build_baselines() -> dict[str, dict[str, Distribution]]:
    """Return ``{'ven': {item_id: Distribution}}`` or ``{}`` if data/recode absent."""
    recode = load_recode()
    if not recode.get("items"):
        log.info("No config/sasas_recode.yaml items — Tshivenda baseline omitted.")
        return {}
    sav = find_sav()
    if sav is None:
        log.info("No SASAS .sav in %s — Tshivenda baseline omitted.", config.paths().sasas)
        return {}

    try:
        import pyreadstat
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyreadstat is required to read SASAS microdata.") from exc

    df, _meta = pyreadstat.read_sav(str(sav), apply_value_formats=True)
    g = recode["global"]
    lang_code = g.get("language_code", "ven")
    weight_var = g.get("weight_var")
    weights = df[weight_var] if weight_var and weight_var in df.columns else 1.0

    # Optional: restrict to Tshivenda home-language speakers if a language var is given.
    lang_var, lang_value = g.get("language_var"), g.get("language_value")
    if lang_var and lang_var in df.columns and lang_value is not None:
        df = df[df[lang_var].astype(str).str.strip().str.lower()
                == str(lang_value).strip().lower()]

    df = df.assign(_w=weights)
    _, items_cfg = config.load_items()
    items_by_id = {it.id: it for it in items_cfg}

    out: dict[str, Distribution] = {}
    for item_id, spec in recode["items"].items():
        item = items_by_id.get(item_id)
        if item is None or spec["var"] not in df.columns:
            continue
        code_map = {str(k): v for k, v in spec["codes"].items()}
        missing = {str(m) for m in spec.get("missing", [])}
        col = df[spec["var"]].astype(str).str.strip()
        sub = df[~col.isin(missing)].assign(_cat=col.map(code_map)).dropna(subset=["_cat"])
        counts = sub.groupby("_cat")["_w"].sum().to_dict()
        if sum(counts.values()) > 0:
            out[item_id] = Distribution.from_counts(item.scale.ordered, counts)

    return {lang_code: out} if out else {}
