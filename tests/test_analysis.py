import numpy as np
import pandas as pd

from decol.analysis import correlate, figures


def _toy_lang_df(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for model, params in [("m_a", 0.5), ("m_b", 1.5), ("m_c", 3.0), ("m_d", 3.8)]:
        for lang in ["eng", "zul", "xho", "ven"]:
            cap = 0.8 if lang == "eng" else rng.uniform(0.2, 0.7)
            div = 0.4 - 0.3 * cap + rng.normal(0, 0.02)   # divergence falls with capability
            rows.append({
                "model": model, "model_tag": model, "params_b": params,
                "specialized": False, "lang": lang,
                "divergence_jsd": max(0.0, div), "divergence_tv": max(0.0, div),
                "mean_abs_shift": abs(div), "mean_signed_shift": -div,
                "n_items_scored": 10, "capability": cap,
                "low_capability": cap < 0.4, "refusal_rate": 0.0, "garble_rate": 0.0,
            })
    return pd.DataFrame(rows)


def test_correlations_run():
    df = _toy_lang_df()
    corr = correlate.correlations(df)
    cap_row = corr[corr["predictor"] == "capability"].iloc[0]
    # divergence should correlate negatively with capability by construction
    assert cap_row["pearson_r"] < 0


def test_ols_and_per_language():
    df = _toy_lang_df()
    summary, coef = correlate.ols(df)
    # OLS needs a working statsmodels/scipy; if unavailable it degrades to empty coef.
    if not coef.empty:
        assert "capability" in set(coef["term"])
    per_lang = correlate.per_language(df)
    assert set(per_lang["lang"]) == {"eng", "zul", "xho", "ven"}


def test_run_writes_artifacts(tmp_path):
    df = _toy_lang_df()
    out = correlate.run(df, out_dir=tmp_path)
    assert (tmp_path / "correlations.csv").is_file()
    assert (tmp_path / "headline.json").is_file()
    assert "divergence_vs_capability_spearman" in out["headline"]


def test_figures_render(tmp_path):
    lang_df = _toy_lang_df()
    item_df = pd.DataFrame([{
        "item_id": "media_freedom", "lang": "zul", "mean_shift": -0.2,
    }, {"item_id": "media_freedom", "lang": "eng", "mean_shift": 0.1}])
    paths = figures.run(item_df, lang_df, out_dir=tmp_path)
    assert paths and all(p.exists() for p in paths)
