#!/usr/bin/env python3
"""Step 1: build human baselines and administer the survey to the edge models.

Resumable: re-running continues where it left off. Outputs:
  data/processed/baselines.json          human attitude marginals per (language, item)
  results/raw/<model>/<lang>.jsonl        every model response (one JSON object per line)
  results/run_manifest.json               provenance index (updated)
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from decol import config, manifest                         # noqa: E402
from decol.data import baselines as baselines_mod           # noqa: E402
from decol.elicit import runner                             # noqa: E402


def select_models(spec: str | None):
    if not spec or spec == "all":
        return config.roster()
    wanted = {s.strip() for s in spec.split(",")}
    return [m for m in config.roster(include_oversized=True)
            if m.tag in wanted or m.name in wanted]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="all", help="comma-separated tags, or 'all'")
    ap.add_argument("--languages", default=None, help="comma-separated ISO codes")
    ap.add_argument("--k-forced", type=int, default=0, help="samples/item (0=model default)")
    ap.add_argument("--k-open", type=int, default=2, help="open-ended cross-check samples")
    ap.add_argument("--no-paraphrases", action="store_true")
    ap.add_argument("--pull-missing", action="store_true", help="ollama pull on demand")
    ap.add_argument("--host", default=None, help="Ollama host (default local daemon)")
    ap.add_argument("--synthetic-baselines", action="store_true",
                    help="use random baselines (smoke only; no real microdata)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run: 1 small model, eng+zul, few samples")
    ap.add_argument("--simulate", action="store_true",
                    help="offline: fabricate responses (no Ollama / no microdata) so the "
                         "whole pipeline produces populated result files")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # 1. human baselines
    if args.synthetic_baselines or args.simulate:
        bl = baselines_mod.synthetic()
        baselines_mod.save_baselines(bl)
        logging.warning("Using SYNTHETIC baselines — for smoke/simulation only.")
    else:
        bl = baselines_mod.build(save=True)

    # 1b. offline simulation path — no Ollama required
    if args.simulate:
        from decol.elicit import simulate as sim
        from decol import capability as cap_pkg
        langs = ([s.strip() for s in args.languages.split(",")]
                 if args.languages else [l.code for l in config.load_languages()])
        models = select_models(args.models)
        cap_scores = cap_pkg.synthetic(models)
        cap_pkg.save_scores(cap_scores)        # so Step 3 can be skipped in simulation
        out_dir = sim.simulate(models=models, languages=langs, baselines=bl,
                               capability_scores=cap_scores)
        manifest.update("survey", {
            "simulated": True, "languages": langs,
            "models": [m.tag for m in models], "raw_dir": str(out_dir),
            "baselines_path": str(baselines_mod.default_path()),
            "config": manifest.config_snapshot(),
        })
        logging.info("SIMULATION complete. Raw responses + synthetic capability written.")
        return

    # 2. survey run config
    langs = ([s.strip() for s in args.languages.split(",")]
             if args.languages else [l.code for l in config.load_languages()])
    models = select_models(args.models)
    cfg = runner.RunConfig(
        languages=langs, k_forced=args.k_forced, k_open=args.k_open,
        include_paraphrases=not args.no_paraphrases,
        pull_missing=args.pull_missing, host=args.host,
    )
    if args.smoke:
        models = [m for m in config.roster(include_oversized=True)
                  if m.tag == "qwen2.5:0.5b"] or models[:1]
        cfg.languages = ["eng", "zul"]
        cfg.k_forced, cfg.k_open = 4, 1
        cfg.pull_missing = True

    cfg = cfg.resolved()
    runner.run(models=models, cfg=cfg)

    manifest.update("survey", {
        "synthetic_baselines": bool(args.synthetic_baselines),
        "languages": cfg.languages,
        "models": [m.tag for m in models],
        "k_forced": cfg.k_forced, "k_open": cfg.k_open,
        "include_paraphrases": cfg.include_paraphrases,
        "config": manifest.config_snapshot(),
        "raw_dir": str(cfg.out_dir),
        "baselines_path": str(baselines_mod.default_path()),
    })
    logging.info("Survey complete. Raw responses under %s", cfg.out_dir)


if __name__ == "__main__":
    main()
