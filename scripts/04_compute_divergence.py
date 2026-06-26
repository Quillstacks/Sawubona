#!/usr/bin/env python3
"""Step 3: compare model answer distributions to the human baselines.

Outputs:
  results/divergence/divergence_items.csv         per (model, language, item)
  results/divergence/divergence_by_language.csv   per (model, language)  [the master table]
  results/results_master.csv                      copy of the above for easy reuse
  results/run_manifest.json                       provenance index (updated)
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from decol import config, capability, manifest             # noqa: E402
from decol.data import baselines as baselines_mod           # noqa: E402
from decol.divergence import compare                        # noqa: E402


def select_models(spec: str | None):
    if not spec or spec == "all":
        return config.roster()
    wanted = {s.strip() for s in spec.split(",")}
    return [m for m in config.roster(include_oversized=True)
            if m.tag in wanted or m.name in wanted]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="all")
    ap.add_argument("--languages", default=None)
    ap.add_argument("--tau", type=float, default=compare.DEFAULT_TAU,
                    help="capability threshold for the low-capability regime")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    compare.DEFAULT_TAU = args.tau  # apply the chosen gate threshold

    models = select_models(args.models)
    langs = ([s.strip() for s in args.languages.split(",")]
             if args.languages else [l.code for l in config.load_languages()])
    bl = baselines_mod.load()
    caps = capability.load_scores()

    item_df, lang_df = compare.build_table(models=models, languages=langs,
                                           baselines=bl, capability_scores=caps)
    out_dir = compare.save_tables(item_df, lang_df)

    master = config.paths().results / "results_master.csv"
    lang_df.to_csv(master, index=False)

    manifest.update("divergence", {
        "tau": args.tau,
        "items_csv": str(out_dir / "divergence_items.csv"),
        "by_language_csv": str(out_dir / "divergence_by_language.csv"),
        "results_master_csv": str(master),
        "n_rows_items": int(len(item_df)),
        "n_rows_language": int(len(lang_df)),
    })
    if not lang_df.empty:
        top = (lang_df.dropna(subset=["divergence_jsd"])
               .groupby("lang")["divergence_jsd"].mean().sort_values(ascending=False))
        logging.info("Most divergent languages (mean JSD):\n%s", top.head(5).to_string())
    logging.info("Divergence tables -> %s and %s", out_dir, master)


if __name__ == "__main__":
    main()
