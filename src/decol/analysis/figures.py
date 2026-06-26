"""Deliverable figures, all written to ``results/figures/`` as PNG + SVG.

  1. per-language divergence "map" (ranked bars) — deliverable #1
  2. model × language divergence heatmap
  3. capability vs divergence scatter (point size = params) — deliverable #2/#3
  4. per-item signed mean-shift heatmap (where the model sits vs the people)

Pure matplotlib (Agg backend) so it runs headless on the Ubuntu box.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config  # noqa: E402

log = logging.getLogger(__name__)


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        paths.append(p)
    plt.close(fig)
    return paths


def _lang_name(code: str) -> str:
    try:
        return config.language(code).name
    except KeyError:
        return code


def language_divergence_bar(lang_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Mean per-language divergence across models — the headline per-language map."""
    df = lang_df.dropna(subset=["divergence_jsd"])
    if df.empty:
        return []
    agg = (df.groupby("lang")["divergence_jsd"].mean().sort_values(ascending=False))
    labels = [_lang_name(c) for c in agg.index]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, agg.values, color="#9b2226")
    ax.set_ylabel("Mean divergence (JSD)  model vs. speakers")
    ax.set_title("Where edge LLMs' politics diverge most from people's")
    ax.set_ylim(0, max(0.05, float(agg.max()) * 1.15))
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    return _save(fig, out_dir, "01_language_divergence_map")


def model_language_heatmap(lang_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    df = lang_df.dropna(subset=["divergence_jsd"])
    if df.empty:
        return []
    pivot = df.pivot_table(index="model", columns="lang", values="divergence_jsd")
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    fig, ax = plt.subplots(figsize=(1.1 * len(pivot.columns) + 3, 0.5 * len(pivot) + 2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(pivot.columns)),
                  [_lang_name(c) for c in pivot.columns], rotation=40, ha="right")
    ax.set_yticks(range(len(pivot.index)), list(pivot.index))
    fig.colorbar(im, ax=ax, label="Divergence (JSD)")
    ax.set_title("Divergence by model and language")
    return _save(fig, out_dir, "02_model_language_heatmap")


def capability_vs_divergence(lang_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    df = lang_df.dropna(subset=["divergence_jsd", "capability"])
    if df.empty:
        return []
    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = 20 + 120 * (df["params_b"].astype(float) /
                        max(df["params_b"].astype(float).max(), 1e-9))
    colors = np.where(df.get("specialized", False), "#005f73", "#bb3e03")
    ax.scatter(df["capability"], df["divergence_jsd"], s=sizes, c=colors,
               alpha=0.75, edgecolor="white", linewidth=0.5)
    if len(df) >= 3 and df["capability"].nunique() > 1:
        b, a = np.polyfit(df["capability"], df["divergence_jsd"], 1)
        xs = np.linspace(df["capability"].min(), df["capability"].max(), 50)
        ax.plot(xs, a + b * xs, "--", color="black", linewidth=1)
    ax.set_xlabel("Language capability (AfroBench, normalised)")
    ax.set_ylabel("Divergence (JSD)  model vs. speakers")
    ax.set_title("Does divergence track capability?  (point size = params; teal = SA-tuned)")
    return _save(fig, out_dir, "03_capability_vs_divergence")


def item_shift_heatmap(item_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    df = item_df.dropna(subset=["mean_shift"])
    if df.empty:
        return []
    pivot = df.pivot_table(index="item_id", columns="lang", values="mean_shift",
                           aggfunc="mean")
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    vmax = float(np.nanmax(np.abs(pivot.values))) or 1.0
    fig, ax = plt.subplots(figsize=(1.1 * len(pivot.columns) + 3, 0.45 * len(pivot) + 2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)),
                  [_lang_name(c) for c in pivot.columns], rotation=40, ha="right")
    ax.set_yticks(range(len(pivot.index)), list(pivot.index))
    fig.colorbar(im, ax=ax, label="Signed mean-shift (model − people)")
    ax.set_title("Direction of divergence per item (red = model higher on scale)")
    return _save(fig, out_dir, "04_item_mean_shift")


def run(item_df: pd.DataFrame, lang_df: pd.DataFrame,
        out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or (config.paths().results / "figures")
    paths: list[Path] = []
    paths += language_divergence_bar(lang_df, out_dir)
    paths += model_language_heatmap(lang_df, out_dir)
    paths += capability_vs_divergence(lang_df, out_dir)
    paths += item_shift_heatmap(item_df, out_dir)
    log.info("Wrote %d figure files -> %s", len(paths), out_dir)
    return paths
