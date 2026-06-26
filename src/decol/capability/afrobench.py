"""Run AfroBench (subsuming IrokoBench) per language via lm-evaluation-harness, pointed at
Ollama's OpenAI-compatible endpoint, and reduce the results to a per-(model, language)
capability score in [0, 1].

We drive lm-eval as a subprocess so the heavy harness stays an optional, install-time
dependency (``pip install -e ".[capability]"``). AfroBench's task↔language layout is read
from ``config/afrobench_tasks.yaml`` (which lm-eval task names map to which ISO codes and
how each is normalised), so we don't hardcode the harness's evolving task list.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from .. import config

log = logging.getLogger(__name__)

# Ollama exposes an OpenAI-compatible API here; lm-eval's local-completions talks to it.
OLLAMA_OPENAI_BASE = "http://localhost:11434/v1"


def _read_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def task_config() -> dict:
    """{lang_code: [{task, metric, random_baseline, n_classes}, ...]} from config."""
    path = config.paths().config / "afrobench_tasks.yaml"
    if not path.is_file():
        log.warning("config/afrobench_tasks.yaml missing — AfroBench scoring disabled. "
                    "See the file's template once it is created.")
        return {}
    return _read_yaml(path).get("languages", {})


def build_command(model_tag: str, tasks: list[str], output_path: Path,
                  limit: int | None = None) -> list[str]:
    """Construct the lm-eval invocation for an Ollama-served model."""
    model_args = (
        f"model={model_tag},base_url={OLLAMA_OPENAI_BASE},"
        "api_key=ollama,tokenized_requests=False"
    )
    cmd = [
        "lm_eval", "--model", "local-completions",
        "--model_args", model_args,
        "--tasks", ",".join(tasks),
        "--output_path", str(output_path),
        "--log_samples",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd


def run_model(model_tag: str, model_name: str, *, limit: int | None = None,
              dry_run: bool = False) -> dict[str, float]:
    """Run AfroBench tasks for one model; return {lang_code: capability_score}."""
    langs = task_config()
    if not langs:
        return {}
    out_root = config.paths().results / "capability" / "afrobench" / model_name
    out_root.mkdir(parents=True, exist_ok=True)

    # gather every task across languages, run once, then reduce per language
    all_tasks = sorted({t["task"] for specs in langs.values() for t in specs})
    out_path = out_root / "lm_eval.json"
    cmd = build_command(model_tag, all_tasks, out_path, limit=limit)
    log.info("AfroBench: %s", " ".join(cmd))
    if dry_run:
        return {}
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log.error("lm_eval failed (%s). Install with `pip install -e \".[capability]\"`.", exc)
        return {}

    raw = _load_lm_eval_results(out_path)
    return _reduce_scores(raw, langs)


def _load_lm_eval_results(out_path: Path) -> dict:
    # lm-eval writes results_*.json under the output dir
    candidates = sorted(out_path.glob("**/results*.json")) if out_path.is_dir() else []
    if not candidates and out_path.with_suffix(".json").is_file():
        candidates = [out_path.with_suffix(".json")]
    if not candidates:
        log.warning("No lm-eval results found under %s", out_path)
        return {}
    with open(candidates[-1], "r", encoding="utf-8") as fh:
        return json.load(fh).get("results", {})


def _reduce_scores(results: dict, langs: dict) -> dict[str, float]:
    """Normalise each task's metric to [0,1] (chance-corrected for accuracy) and average
    per language."""
    scores: dict[str, float] = {}
    for lang_code, specs in langs.items():
        vals = []
        for spec in specs:
            task, metric = spec["task"], spec.get("metric", "acc")
            res = results.get(task)
            if not res or metric not in res:
                continue
            val = float(res[metric])
            baseline = float(spec.get("random_baseline", 0.0))
            if baseline > 0 and baseline < 1:
                val = max(0.0, (val - baseline) / (1.0 - baseline))   # chance-correct
            vals.append(min(1.0, max(0.0, val)))
        if vals:
            scores[lang_code] = round(sum(vals) / len(vals), 3)
    return scores
