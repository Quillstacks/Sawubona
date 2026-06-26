# Sawubona — project context for agents

**Read [`HANDOFF.md`](HANDOFF.md) first.** It is the authoritative onboarding for continuing
this work (status, setup, data acquisition, open questions, gotchas, definition of done).

**Sawubona** measures whether a small **edge LLM** (≤ ~6 GB GPU, via **Ollama**) reflects
South Africans' real politics in their own language or collapses to an English default. It
compares each model's answers on the **Afrobarometer R9 South Africa** political items —
prompted per language — against the speakers' actual attitudes, then relates the divergence
to per-language **capability** (AfroBench) and model **size**.

## Cardinal rule
**Always work from the real fetched source — never reconstruct an instrument, schema, or
codes from memory.** Items/scales/codes here are verbatim from the real R9 codebook + SA
summary in `data/raw/afrobarometer/`. When adding translations, AfroBench task names, or
eval sets, fetch and quote the authoritative source; flag anything provisional.

## Locked decisions
Edge ≤6 GB · Ollama (GGUF) · group by `Q102` language of interview · scale-native divergence
(JSD + signed mean-shift, no left/right projection) · capability-gated · Röttger-robust
(distributions over K samples + ≥3 paraphrases + open-ended cross-check). Don't relitigate.

## Run
```bash
pip install -e ".[capability,dev]"; pytest -q          # 36 tests
python scripts/02_run_survey.py --simulate && python scripts/04_compute_divergence.py \
  && python scripts/05_correlate_and_plot.py           # offline E2E (no data/GPU)
python scripts/validate_microdata.py                   # after dropping the real .sav in
```
Real data acquisition (two separate things: the `.sav` baselines AND the translated
questionnaire prompts) is in `HANDOFF.md` §6 and `data/README.md`.
