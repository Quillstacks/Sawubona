"""Load Afrobarometer Round 9 (South Africa) microdata into per-language human baselines.

The mapping from raw ``.sav`` columns/codes to our canonical scale categories lives in
``config/afrobarometer_recode.yaml`` (filled from the codebook). This module reads the
``.sav`` with ``pyreadstat`` (imported lazily so the package imports without it), filters
to South Africa, groups respondents by language of interview, recodes each configured
item, and returns weighted marginal :class:`~decol.dist.Distribution` objects.
"""
from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Any

from .. import config
from ..dist import Distribution

log = logging.getLogger(__name__)


def _read_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_recode() -> dict[str, Any]:
    spec = _read_yaml(config.paths().config / "afrobarometer_recode.yaml")
    spec["global"] = spec.get("global") or {}
    spec["items"] = spec.get("items") or {}
    return spec


def find_sav(directory: Path | None = None) -> Path:
    directory = directory or config.paths().afrobarometer
    hits = sorted(glob.glob(str(directory / "*.sav")))
    if not hits:
        raise FileNotFoundError(
            f"No Afrobarometer .sav found in {directory}. "
            "Run scripts/00_download_data.sh and see data/README.md."
        )
    if len(hits) > 1:
        log.warning("Multiple .sav files in %s; using %s", directory, hits[0])
    return Path(hits[0])


def _read_sav(path: Path):
    try:
        import pyreadstat  # lazy: only needed on the run machine
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "pyreadstat is required to read Afrobarometer microdata "
            "(`pip install pyreadstat`)."
        ) from exc
    # Read NUMERIC codes (not value-label strings) so recoding is unambiguous.
    df, meta = pyreadstat.read_sav(str(path), apply_value_formats=False)
    return df, meta


def build_baselines(sav_path: Path | None = None) -> dict[str, dict[str, Distribution]]:
    """Return ``{lang_code: {item_id: Distribution}}`` from the Afrobarometer microdata.

    Only items present in ``afrobarometer_recode.yaml`` (with a code map) are produced;
    everything else is intentionally absent (reported downstream as a gap, not guessed).
    """
    recode = load_recode()
    g = recode["global"]
    item_specs = recode["items"] or {}
    if not item_specs:
        log.warning("afrobarometer_recode.yaml has no items yet — no baselines built. "
                    "Fill it from the R9 codebook (data/README.md).")
        return {}

    import pandas as pd

    df, _meta = _read_sav(sav_path or find_sav())

    def _num(series):
        return pd.to_numeric(series, errors="coerce")

    def _as_int_key(x):
        """Map a numeric code to an int dict key, NaN -> None."""
        try:
            return int(round(float(x)))
        except (TypeError, ValueError):
            return None

    # 1. filter to South Africa (merged datasets cover many countries)
    cvar, cval = g.get("country_var"), g.get("country_value")
    if cvar and cvar in df.columns and cval is not None:
        df = df[_num(df[cvar]) == float(cval)]
        log.info("Filtered to country %s: %d rows", cval, len(df))

    # 2. language of interview (Q102) -> ISO code, via numeric map
    lang_var = g.get("language_var")
    if lang_var not in df.columns:
        raise KeyError(f"language_var {lang_var!r} not in microdata columns")
    lang_map = {int(k): v for k, v in (g.get("language_map") or {}).items()}

    # 3. survey weight, with fallbacks; unweighted if none present
    weight_var = g.get("weight_var")
    for cand in [weight_var, *(g.get("weight_var_fallbacks") or [])]:
        if cand and cand in df.columns:
            weight_var = cand
            break
    else:
        weight_var = None
    weights = _num(df[weight_var]) if weight_var else 1.0
    if not weight_var:
        log.warning("no weight variable present; using unweighted marginals")
    else:
        log.info("using weight variable %r", weight_var)

    global_missing = {int(m) for m in (recode.get("missing_codes") or [])
                      if _as_int_key(m) is not None}

    _, items_cfg = config.load_items()
    items_by_id = {it.id: it for it in items_cfg}

    out: dict[str, dict[str, Distribution]] = {}
    df = df.assign(_w=weights)
    df["_lang"] = _num(df[lang_var]).map(lambda x: lang_map.get(_as_int_key(x)))

    for item_id, spec in item_specs.items():
        item = items_by_id.get(item_id)
        if item is None:
            log.warning("recode item %r not in items_political.yaml; skipping", item_id)
            continue
        var = spec["var"]
        if var not in df.columns:
            log.warning("item %s: variable %r not in microdata; skipping", item_id, var)
            continue
        code_map = {int(k): v for k, v in spec["codes"].items()}
        missing = global_missing | {int(m) for m in spec.get("missing", [])}
        categories = item.scale.ordered

        sub = df[["_lang", "_w", var]].dropna(subset=["_lang"])
        codes = _num(sub[var]).map(_as_int_key)
        cat = codes.map(lambda c: None if c in missing else code_map.get(c))
        sub = sub.assign(_cat=cat).dropna(subset=["_cat"])

        for lang_code, grp in sub.groupby("_lang"):
            counts = grp.groupby("_cat")["_w"].sum().to_dict()
            if sum(counts.values()) <= 0:
                continue
            out.setdefault(lang_code, {})[item_id] = Distribution.from_counts(
                categories, counts)

    return out
