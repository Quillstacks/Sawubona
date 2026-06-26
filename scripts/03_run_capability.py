#!/usr/bin/env python3
"""Step 2: score each model's per-language capability (AfroBench + extension).

Outputs:
  results/capability/scores.json          {model: {lang: score in [0,1]}}
  results/capability/afrobench/<model>/    raw lm-eval outputs
  results/run_manifest.json               provenance index (updated)
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from decol import config, capability, manifest             # noqa: E402
from decol.capability import afrobench, extend             # noqa: E402
from decol.elicit.ollama_client import OllamaClient        # noqa: E402


def select_models(spec: str | None):
    if not spec or spec == "all":
        return config.roster()
    wanted = {s.strip() for s in spec.split(",")}
    return [m for m in config.roster(include_oversized=True)
            if m.tag in wanted or m.name in wanted]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="all")
    ap.add_argument("--limit", type=int, default=None, help="cap items/task (debug)")
    ap.add_argument("--host", default=None)
    ap.add_argument("--no-afrobench", action="store_true", help="skip the lm-eval run")
    ap.add_argument("--no-extension", action="store_true",
                    help="skip the isiNdebele/Tshivenda proxy")
    ap.add_argument("--dry-run", action="store_true", help="print lm-eval command only")
    ap.add_argument("--synthetic", action="store_true",
                    help="write random capability scores (smoke only)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    models = select_models(args.models)

    if args.synthetic:
        scores = capability.synthetic(models)
        capability.save_scores(scores)
        manifest.update("capability", {"synthetic": True,
                                       "models": [m.tag for m in models]})
        logging.warning("Wrote SYNTHETIC capability scores — smoke only.")
        return

    client = OllamaClient(host=args.host)
    ab_scores: capability.Scores = {}
    ext_scores: capability.Scores = {}
    for m in models:
        if not args.no_afrobench:
            ab_scores[m.name] = afrobench.run_model(
                m.tag, m.name, limit=args.limit, dry_run=args.dry_run)
        if not args.no_extension and not args.dry_run:
            ext_scores[m.name] = extend.run_model(
                m.tag, m.name, client=client, limit=args.limit)

    scores = capability.merge_scores(ab_scores, ext_scores)
    if scores:
        capability.save_scores(scores)
    manifest.update("capability", {
        "synthetic": False, "afrobench": not args.no_afrobench,
        "extension": not args.no_extension, "models": [m.tag for m in models],
        "scores_path": str(capability.default_path()),
    })
    logging.info("Capability scoring complete.")


if __name__ == "__main__":
    main()
