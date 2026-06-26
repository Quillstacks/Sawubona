# Sawubona

> **Sawubona** — "I see you" (isiZulu). The question this repo asks: does a small edge
> model genuinely *see* each language's speakers and their real politics, or collapse them
> into an English-anchored default?

**Linguistic Decolonization: conditioned political bias of edge LLMs across South Africa's languages.**

Ask a small language model about South African politics in English and in isiZulu, and it
may answer differently. Prompted in English, frontier models tend libertarian-left; in
languages they saw little of during training, the same model drifts toward more
authoritarian framings — plausibly via transfer learning from English-centric data. This
repository **measures that gap**: across South Africa's official languages, how far does a
small edge LLM's elicited political position diverge from its speakers', and is that
divergence explained by the model's per-language **capability** or its **size**?

Companion to the proposal `LinguisticDecolonization.pdf` (Schutera & Dobbelstein, DHBW
Ravensburg). The full design lives in
[`docs` / the approved plan](#pipeline). Inference is **local, edge-scale (≤ ~6 GB GPU),
via Ollama (GGUF)** on a single Ubuntu GPU machine.

## Deliverables

1. A **public per-language map** of where edge LLMs' politics diverge from people's.
2. A **capability-informed explanation** of that divergence (AfroBench-scored).
3. **Evidence on whether divergence is explained by model size.**

## Pipeline

Four steps, per language, mirroring the proposal:

| Step | Script | What it does |
|------|--------|--------------|
| 1. Run the survey on the models | `scripts/02_run_survey.py` | Administer the Afrobarometer R9 SA political items to each edge model, prompted in the target language; sample *K* times → model answer distribution. |
| 2. Measure language capability | `scripts/03_run_capability.py` | Score each model's per-language capability with AfroBench (+ isiNdebele/Tshivenda extension). |
| 3. Compare model to people | `scripts/04_compute_divergence.py` | Jensen–Shannon divergence between the model's answer distribution and the speakers' real attitudes; capability-gated. |
| 4. Correlate | `scripts/05_correlate_and_plot.py` | `divergence ~ capability + log(params)`; emit the per-language map + tables. |

## Quickstart (Ubuntu GPU box)

```bash
# 0. Install
pip install -e .                       # core
pip install -e ".[capability,dev]"     # + AfroBench harness + tests
curl -fsSL https://ollama.com/install.sh | sh

# 1. Data (free, one-time registration for Afrobarometer; SASAS placed manually)
bash scripts/00_download_data.sh
#    ... then drop SASAS files into data/raw/sasas/  (see data/README.md)

# 2. Models (pulls the ≤6 GB roster from config/models.yaml)
bash scripts/01_pull_models.sh

# 3. Build human baselines + run the experiment
python scripts/02_run_survey.py        # resumable; safe to Ctrl-C / re-run
python scripts/03_run_capability.py
python scripts/04_compute_divergence.py
python scripts/05_correlate_and_plot.py
```

Long sweeps: launch under `tmux` or `nohup`. Every model response is written
incrementally, so a crash or a closed SSH session **resumes** instead of restarting.

### Try it now, offline (no GPU, no data)

```bash
pip install -e .
python scripts/02_run_survey.py --simulate    # fabricates responses + synthetic baselines
python scripts/04_compute_divergence.py
python scripts/05_correlate_and_plot.py
```

This populates the full `results/` tree (tables + figures) so downstream analysis can be
wired up before models/microdata are ready. Simulated runs are flagged `simulated: true`
in `results/run_manifest.json`, and `--smoke` / `--synthetic-baselines` are similarly
labelled — so nothing built on placeholder data is ever mistaken for the real thing.

## Outputs — everything is saved to disk

| File | Produced by | Contents |
|------|-------------|----------|
| `data/processed/baselines.json` | step 1 | human attitude marginals per (language, item) |
| `results/raw/<model>/<lang>.jsonl` | step 1 | every model response (1 JSON object/line; resumable) |
| `results/capability/scores.json` | step 2 | `{model: {lang: capability ∈ [0,1]}}` |
| `results/divergence/divergence_items.csv` | step 3 | divergence per (model, language, item) |
| `results/divergence/divergence_by_language.csv` | step 3 | divergence per (model, language) |
| `results/results_master.csv` | step 3 | **the master table** — divergence + capability + size + gating |
| `results/analysis/*.csv`, `headline.json` | step 4 | correlations, OLS coefficients, per-language trends |
| `results/figures/*.png|*.svg` | step 4 | per-language map, heatmaps, capability scatter |
| `results/run_manifest.json` | all steps | provenance index (config snapshot, real vs synthetic, timestamps) |

All tabular outputs are plain CSV/JSON for easy reuse in pandas/R/Excel.

## Methodology notes

- **Surviving Röttger** ([Political Compass or Spinning Arrow?](https://aclanthology.org/2024.acl-long.816/)):
  we report answer **distributions** over *K* samples and ≥3 paraphrases per item, run an
  **open-ended cross-check** on a subset, and compare against the **real human survey
  instrument** — not a synthetic political-compass test. Forced-choice is used only to make
  the model's answers commensurable with the human survey scales.
- **Human baseline per language group**: divergence compares "model prompted in language
  *L*" against "people who were surveyed in *L*", using the microdata's language variable.
- **Capability gating**: below an AfroBench threshold τ, a large gap is reported as a
  *finding* (low-capability regime), not forced into the size correlation.

## Layout

```
config/     languages, model roster, political items (+ axis mapping, paraphrases)
data/       acquisition instructions + (gitignored) microdata and built baselines
src/decol/  data · elicit · capability · divergence · analysis
scripts/    00_download_data … 05_correlate_and_plot
tests/      parsing, metrics, baseline-loader unit tests
results/    (gitignored) raw responses, tables, figures
```

See `data/README.md` for exactly how to obtain Afrobarometer R9 and SASAS microdata.
