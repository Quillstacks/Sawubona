#!/usr/bin/env python3
"""Step 4: correlate divergence with capability and size, and render the deliverables.

Outputs:
  results/analysis/correlations.csv        divergence vs capability / log(params)
  results/analysis/ols_coefficients.csv    OLS partial effects
  results/analysis/ols_summary.txt         full regression summary
  results/analysis/per_language.csv        per-language divergence + within-language trend
  results/analysis/headline.json           one-glance answer to the research question
  results/figures/*.png|*.svg              per-language map, heatmaps, scatter
  results/run_manifest.json                provenance index (updated)
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from decol import config, manifest                          # noqa: E402
from decol.analysis import correlate, figures               # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep-low-capability", action="store_true",
                    help="include low-capability cells in the size correlation")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    div_dir = config.paths().results / "divergence"
    item_df = pd.read_csv(div_dir / "divergence_items.csv")
    lang_df = pd.read_csv(div_dir / "divergence_by_language.csv")

    out = correlate.run(lang_df, drop_low_capability=not args.keep_low_capability)
    fig_paths = figures.run(item_df, lang_df)

    manifest.update("analysis", {
        "analysis_dir": str(config.paths().results / "analysis"),
        "figures": [str(p) for p in fig_paths],
        "headline": out["headline"],
        "dropped_low_capability": not args.keep_low_capability,
    })

    h = out["headline"]
    logging.info("Research-question headline:")
    logging.info("  divergence vs capability (Spearman): %s",
                 h["divergence_vs_capability_spearman"])
    logging.info("  divergence vs log(size)  (Spearman): %s",
                 h["divergence_vs_logsize_spearman"])
    logging.info("  most divergent languages: %s", h["most_divergent_languages"])
    logging.info("Figures + tables written under %s", config.paths().results)


if __name__ == "__main__":
    main()
