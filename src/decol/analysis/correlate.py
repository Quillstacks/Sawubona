"""Step 4: does divergence track language capability or model size?

Fits, on the per-(model, language) divergence table:
  * pairwise Spearman/Pearson correlations of divergence with capability and log(params);
  * an OLS ``jsd ~ capability + log10(params)`` to read partial effects;
  * per-language correlations of divergence vs capability across the model ladder.
Everything is returned as DataFrames and written to ``results/analysis/`` so the numbers
can be reused without re-running the pipeline. Low-capability cells are reported but, by
default, excluded from the size correlation (the proposal's "finding, not a forced score").
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

log = logging.getLogger(__name__)


def _prep(lang_df: pd.DataFrame, drop_low_capability: bool) -> pd.DataFrame:
    df = lang_df.copy()
    df = df[df["divergence_jsd"].notna()]
    if drop_low_capability and "low_capability" in df.columns:
        df = df[~df["low_capability"].fillna(False)]
    df["log10_params"] = np.log10(df["params_b"].astype(float))
    return df


def correlations(lang_df: pd.DataFrame, drop_low_capability: bool = True) -> pd.DataFrame:
    df = _prep(lang_df, drop_low_capability)
    rows = []
    for predictor in ("capability", "log10_params"):
        sub = df[[predictor, "divergence_jsd"]].dropna()
        if len(sub) >= 3 and sub[predictor].nunique() > 1:
            rows.append({
                "predictor": predictor, "n": len(sub),
                "pearson_r": sub[predictor].corr(sub["divergence_jsd"], "pearson"),
                "spearman_r": sub[predictor].corr(sub["divergence_jsd"], "spearman"),
            })
        else:
            rows.append({"predictor": predictor, "n": len(sub),
                         "pearson_r": np.nan, "spearman_r": np.nan})
    return pd.DataFrame(rows)


def ols(lang_df: pd.DataFrame, drop_low_capability: bool = True):
    """OLS divergence ~ capability + log10(params). Returns (summary_text, coef_df).

    Degrades gracefully (empty coefficients) if statsmodels is unavailable or the fit
    fails, so the rest of the analysis/plots still run.
    """
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:  # noqa: BLE001 - env/version issues shouldn't crash analysis
        log.warning("statsmodels unavailable (%s); skipping OLS.", exc)
        return f"statsmodels unavailable: {exc}", pd.DataFrame()

    df = _prep(lang_df, drop_low_capability).dropna(
        subset=["divergence_jsd", "capability", "log10_params"])
    if len(df) < 4:
        log.warning("Too few rows (%d) for OLS; skipping.", len(df))
        return "insufficient data", pd.DataFrame()
    try:
        model = smf.ols("divergence_jsd ~ capability + log10_params", data=df).fit()
    except Exception as exc:  # noqa: BLE001
        log.warning("OLS fit failed (%s); skipping.", exc)
        return f"OLS fit failed: {exc}", pd.DataFrame()
    coef = pd.DataFrame({
        "term": model.params.index, "coef": model.params.values,
        "std_err": model.bse.values, "t": model.tvalues.values,
        "p_value": model.pvalues.values,
    })
    return model.summary().as_text(), coef


def per_language(lang_df: pd.DataFrame) -> pd.DataFrame:
    """Correlation of divergence vs capability within each language (across models)."""
    rows = []
    for lang, grp in lang_df.dropna(subset=["divergence_jsd"]).groupby("lang"):
        sub = grp[["capability", "divergence_jsd"]].dropna()
        r = (sub["capability"].corr(sub["divergence_jsd"], "spearman")
             if len(sub) >= 3 and sub["capability"].nunique() > 1 else np.nan)
        rows.append({"lang": lang, "n_models": len(grp),
                     "mean_divergence_jsd": grp["divergence_jsd"].mean(),
                     "spearman_div_vs_capability": r})
    out = pd.DataFrame(rows).sort_values("mean_divergence_jsd", ascending=False)
    return out


def run(lang_df: pd.DataFrame, out_dir: Path | None = None,
        drop_low_capability: bool = True) -> dict:
    """Compute all analyses and persist them. Returns a dict of the artefacts."""
    out_dir = out_dir or (config.paths().results / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    corr = correlations(lang_df, drop_low_capability)
    summary_text, coef = ols(lang_df, drop_low_capability)
    per_lang = per_language(lang_df)

    corr.to_csv(out_dir / "correlations.csv", index=False)
    coef.to_csv(out_dir / "ols_coefficients.csv", index=False)
    per_lang.to_csv(out_dir / "per_language.csv", index=False)
    (out_dir / "ols_summary.txt").write_text(summary_text, encoding="utf-8")

    headline = _headline(corr, coef, per_lang)
    with open(out_dir / "headline.json", "w", encoding="utf-8") as fh:
        json.dump(headline, fh, indent=2)
    log.info("Wrote analysis -> %s", out_dir)
    return {"correlations": corr, "ols_coef": coef, "per_language": per_lang,
            "headline": headline}


def _headline(corr: pd.DataFrame, coef: pd.DataFrame, per_lang: pd.DataFrame) -> dict:
    def _r(pred):
        row = corr[corr["predictor"] == pred]
        return None if row.empty else _nan(row["spearman_r"].iloc[0])
    top = per_lang.head(3)[["lang", "mean_divergence_jsd"]].to_dict("records") \
        if not per_lang.empty else []
    return {
        "divergence_vs_capability_spearman": _r("capability"),
        "divergence_vs_logsize_spearman": _r("log10_params"),
        "ols_terms": coef.set_index("term")["coef"].to_dict() if not coef.empty else {},
        "most_divergent_languages": top,
    }


def _nan(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
