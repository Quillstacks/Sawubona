"""Turn raw survey responses into model answer distributions and the divergence table.

Per (model, language, item) we pool the forced-choice samples (across wording variants)
into the model's answer distribution, compare it to the speakers' human marginal with the
scale-native metrics, and aggregate to a per-(model, language) divergence — the headline
quantity. Refusal/garble rates and a capability gate are carried alongside so that a large
gap in a low-capability cell is reported as a finding, not a forced score.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .. import config
from ..config import Model
from ..dist import Distribution
from ..elicit.runner import iter_records, response_path
from . import metrics
from .. import capability as capability_pkg

log = logging.getLogger(__name__)

DEFAULT_TAU = 0.40   # AfroBench capability threshold for the low-capability regime


def model_distributions(model: Model, lang_code: str, out_dir: Path | None = None
                        ) -> dict[str, dict]:
    """{item_id: {dist, n, n_ok, refusal_rate, garble_rate}} from forced responses."""
    out_dir = out_dir or (config.paths().results / "raw")
    path = response_path(out_dir, model, lang_code)
    scales, items = config.load_items()
    item_scale = {it.id: it.scale for it in items}

    counts: dict[str, dict[str, float]] = {}
    totals: dict[str, dict[str, int]] = {}
    for rec in iter_records(path):
        if rec.get("mode") != "forced":
            continue
        iid = rec["item_id"]
        t = totals.setdefault(iid, {"n": 0, "ok": 0, "refusal": 0, "garble": 0})
        t["n"] += 1
        status = rec.get("status")
        if status == "ok" and rec.get("category"):
            t["ok"] += 1
            counts.setdefault(iid, {})
            counts[iid][rec["category"]] = counts[iid].get(rec["category"], 0.0) + 1.0
        elif status in ("refusal", "garble"):
            t[status] += 1

    result: dict[str, dict] = {}
    for iid, t in totals.items():
        scale = item_scale.get(iid)
        dist = None
        if iid in counts and sum(counts[iid].values()) > 0:
            dist = Distribution.from_counts(scale.ordered, counts[iid])
        n = max(t["n"], 1)
        result[iid] = {
            "dist": dist, "n": t["n"], "n_ok": t["ok"],
            "refusal_rate": t["refusal"] / n, "garble_rate": t["garble"] / n,
        }
    return result


def item_rows(model: Model, lang_code: str, baselines, capability: float | None,
              out_dir: Path | None = None) -> list[dict]:
    scales, items = config.load_items()
    item_by_id = {it.id: it for it in items}
    mdists = model_distributions(model, lang_code, out_dir)
    human = baselines.get(lang_code, {})

    rows: list[dict] = []
    for iid, info in mdists.items():
        item = item_by_id[iid]
        hdist = human.get(iid)
        row = {
            "model": model.name, "model_tag": model.tag, "params_b": model.params_b,
            "specialized": model.specialized, "lang": lang_code, "item_id": iid,
            "theme": item.theme, "scale": item.scale.name,
            "n": info["n"], "n_ok": info["n_ok"],
            "refusal_rate": info["refusal_rate"], "garble_rate": info["garble_rate"],
            "has_model_dist": info["dist"] is not None,
            "has_human_baseline": hdist is not None,
            "capability": capability,
            "low_capability": metrics.capability_gate(capability, DEFAULT_TAU),
            "jsd": None, "tv": None, "wasserstein": None, "mean_shift": None,
        }
        if info["dist"] is not None and hdist is not None:
            row.update(metrics.item_divergence(info["dist"], hdist, item.scale))
        rows.append(row)
    return rows


def build_table(models=None, languages=None, baselines=None, capability_scores=None,
                out_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (item-level df, language-level df) of divergences for all model×language."""
    from ..data import baselines as baselines_mod

    models = list(models) if models is not None else config.roster()
    languages = languages or [l.code for l in config.load_languages()]
    if baselines is None:
        baselines = baselines_mod.load()
    if capability_scores is None:
        capability_scores = capability_pkg.load_scores()

    all_rows: list[dict] = []
    for model in models:
        for lang in languages:
            cap = (capability_scores.get(model.name, {}) or {}).get(lang)
            all_rows.extend(item_rows(model, lang, baselines, cap, out_dir))

    item_df = pd.DataFrame(all_rows)
    lang_df = _aggregate(item_df)
    return item_df, lang_df


def _aggregate(item_df: pd.DataFrame) -> pd.DataFrame:
    if item_df.empty:
        return item_df
    scored = item_df[item_df["jsd"].notna()]
    grp = scored.groupby(["model", "model_tag", "params_b", "specialized", "lang"])
    agg = grp.agg(
        divergence_jsd=("jsd", "mean"),
        divergence_tv=("tv", "mean"),
        mean_abs_shift=("mean_shift", lambda s: s.abs().mean()),
        mean_signed_shift=("mean_shift", "mean"),
        n_items_scored=("jsd", "size"),
    ).reset_index()
    # carry capability + gating + refusal/garble (means over all attempted items)
    meta = item_df.groupby(["model", "lang"]).agg(
        capability=("capability", "first"),
        low_capability=("low_capability", "first"),
        refusal_rate=("refusal_rate", "mean"),
        garble_rate=("garble_rate", "mean"),
    ).reset_index()
    return agg.merge(meta, on=["model", "lang"], how="left")


# ----------------------------------------------------------------------- persistence
def save_tables(item_df: pd.DataFrame, lang_df: pd.DataFrame,
                out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (config.paths().results / "divergence")
    out_dir.mkdir(parents=True, exist_ok=True)
    item_df.to_csv(out_dir / "divergence_items.csv", index=False)
    lang_df.to_csv(out_dir / "divergence_by_language.csv", index=False)
    log.info("Wrote divergence tables -> %s", out_dir)
    return out_dir
