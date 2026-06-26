import json

import pandas as pd

from decol import config
from decol.dist import Distribution
from decol.divergence import compare
from decol.elicit.runner import response_path


def _write_responses(out_dir, model, lang, item_id, scale_name, categories):
    """Write forced-mode responses cycling through given categories."""
    path = response_path(out_dir, model, lang)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for i, cat in enumerate(categories):
            fh.write(json.dumps({
                "model": model.name, "model_tag": model.tag, "lang": lang,
                "item_id": item_id, "scale": scale_name, "mode": "forced",
                "variant_idx": 0, "sample_idx": i, "status": "ok", "category": cat,
            }) + "\n")


def test_model_distributions_and_divergence(tmp_path):
    out_dir = tmp_path / "raw"
    model = config.roster()[0]
    _, items = config.load_items()
    item = next(it for it in items if it.id == "media_freedom")  # two_statement, ordinal
    ordered = item.scale.ordered
    hi, lo = ordered[-1], ordered[0]   # top vs bottom of the ordinal scale

    # model answers: mostly the top category
    cats = [hi, hi, hi, ordered[2], lo]
    _write_responses(out_dir, model, "eng", item.id, item.scale.name, cats)

    dists = compare.model_distributions(model, "eng", out_dir=out_dir)
    assert item.id in dists
    md = dists[item.id]["dist"]
    assert md.as_map[hi] > md.as_map[lo]
    assert dists[item.id]["n_ok"] == 5

    # human baseline: mostly the bottom category -> clear divergence, model sits higher
    human = {"eng": {item.id: Distribution.from_counts(ordered, {lo: 8, ordered[1]: 2})}}
    rows = compare.item_rows(model, "eng", human, capability=0.7, out_dir=out_dir)
    row = next(r for r in rows if r["item_id"] == item.id)
    assert row["has_model_dist"] and row["has_human_baseline"]
    assert row["jsd"] is not None and row["jsd"] > 0
    assert row["mean_shift"] is not None and row["mean_shift"] > 0   # model higher than people
    assert row["low_capability"] is False


def test_aggregate_table_shape(tmp_path):
    out_dir = tmp_path / "raw"
    model = config.roster()[0]
    _, items = config.load_items()
    human = {"eng": {}}
    for it in items[:3]:
        _write_responses(out_dir, model, "eng", it.id, it.scale.name,
                         [it.scale.ordered[-1]] * 4)
        human["eng"][it.id] = Distribution.from_counts(
            it.scale.ordered, {it.scale.ordered[0]: 4})

    item_df, lang_df = compare.build_table(
        models=[model], languages=["eng"], baselines=human, capability_scores={},
        out_dir=out_dir)
    assert isinstance(item_df, pd.DataFrame) and not item_df.empty
    assert not lang_df.empty
    assert "divergence_jsd" in lang_df.columns
    assert lang_df.iloc[0]["n_items_scored"] == 3
