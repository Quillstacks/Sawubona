# Sawubona — Handoff for the Linux run

> **Sawubona** ("I see you", isiZulu). We measure whether a small **edge LLM** (≤ ~6 GB
> GPU) genuinely *sees* each of South Africa's languages' speakers and their real politics,
> or collapses them into an English-anchored default.

This file onboards the agent continuing the work on the **Ubuntu + GPU** machine. Read it
fully before running anything. The repo is **https://github.com/Quillstacks/Sawubona**.

---

## 1. Research question

Across South Africa's official languages, **where does a small edge LLM's elicited
political position diverge most from its speakers'**, and is that divergence explained by
the model's per-language **capability** or its **size**? Deliverables: a per-language
divergence map; a capability-informed explanation; evidence on the size effect.

The method (per language): (1) administer the **Afrobarometer R9 South Africa** political
items to each edge model, prompted in the target language; (2) score per-language
**capability** (AfroBench + extension); (3) **divergence** = distance between the model's
answer distribution and the speakers' real attitudes; (4) **correlate** divergence with
capability and size. Full design: `~/.claude/plans/...` is not on Linux — use `readme.md`
and this file.

---

## 2. THE CARDINAL RULE (do not break)

**Always work from the real fetched source — never reconstruct an instrument, schema, or
codes from memory.** The political items, scales, response codes, and Q102 language codes
in this repo were taken verbatim from the real R9 codebook + SA summary (see §10). When you
add anything (translations, AfroBench task names, capability eval sets), **fetch the
authoritative source and quote it**; mark anything provisional as such. This rule exists
because from-memory drafts were materially wrong (e.g. R9 Q23's numeric codes are
scrambled — see §10).

---

## 3. Current status

**Done & verified (on macOS, committed):**
- Full pipeline (config-driven): elicitation via Ollama, AfroBench capability, scale-native
  divergence, correlation + figures. 36 unit tests pass.
- `config/items_political.yaml`, `config/afrobarometer_recode.yaml`, `config/languages.yaml`
  rebuilt **verbatim from the real R9 codebook + SA summary** (numeric codes, Q102 map).
- All models in `config/models.yaml` are **auto-pullable** via Ollama (registry tags or
  `hf.co/...` GGUFs) — nothing to convert by hand.
- Offline `--simulate` reproduces the entire `results/` tree without data/GPU.
- `scripts/validate_microdata.py` checks a dropped-in `.sav` before a long run.

**Pending (your job on Linux):**
- [ ] Stand up Ollama on the GPU; pull the roster; run a `--simulate` smoke test.
- [ ] Acquire the **microdata `.sav`** → real human baselines (§6a).
- [ ] Acquire the **translated questionnaire wording** → real per-language prompts (§6b).
- [ ] Verify the **open empirical questions** against the real `.sav` (§8).
- [ ] Confirm **AfroBench lm-eval task names** in `config/afrobench_tasks.yaml` (§7).
- [ ] Run the full pipeline; sanity-check; iterate.

---

## 4. Locked design decisions (don't relitigate)

- **Edge scope:** models ≤ ~6 GB GPU (`config/models.yaml`); size axis lives *within* this
  range. (The DGX-Spark / larger-model extension was dropped.)
- **Inference:** **Ollama (GGUF)**, local daemon.
- **Grouping:** by **language of interview** (`Q102`), not home language — keeps the
  "model prompted in L vs. people interviewed in L" symmetry.
- **Divergence:** measured **directly on each Afrobarometer scale** (per-item
  Jensen–Shannon distance + signed ordinal mean-shift). **No** left/right or
  libertarian/authoritarian projection.
- **"Surviving Röttger"** (ref [6]): report **distributions** over K samples and ≥3
  paraphrases per item; open-ended cross-check on a subset; compare against the real human
  instrument. Forced-choice is only for commensurability with the survey scales.
- **Capability gating:** below an AfroBench threshold τ (default 0.40), a large gap is a
  *finding* (low-capability regime), not a forced score.
- **OK to lose a small language:** if Tshivenda/isiNdebele lack a clean baseline, report as
  a documented gap rather than hacking a noisy one.

---

## 5. Environment setup (Ubuntu + GPU)

```bash
git clone https://github.com/Quillstacks/Sawubona.git && cd Sawubona
python -m venv .venv && source .venv/bin/activate
pip install -e ".[capability,dev]"        # core + AfroBench harness + pytest
curl -fsSL https://ollama.com/install.sh | sh
bash scripts/01_pull_models.sh            # pulls the whole roster (registry + hf.co GGUFs)
pytest -q                                 # expect 36 passing
python scripts/02_run_survey.py --simulate && \
python scripts/04_compute_divergence.py && \
python scripts/05_correlate_and_plot.py   # proves the box works end-to-end (synthetic)
```
Long real runs: launch under `tmux`/`nohup`. Elicitation is **resumable** — re-running
continues from `results/raw/<model>/<lang>.jsonl`.

---

## 6. Data acquisition — TWO SEPARATE THINGS

> The `.sav` gives the **human baselines**. It does **not** contain the prompt wording in
> each language. The **translated questionnaire** is a separate artifact (§6b). You need
> both for a valid (not English-only) run.

### 6a. Microdata `.sav` → human baselines  (`data/README.md` has links)
- Register (free) at Afrobarometer; download the **merged R9 data set (SPSS `.sav`)** from
  the data portal (or DataFirst catalog 989). Drop it in `data/raw/afrobarometer/`.
- Then: `python scripts/validate_microdata.py` (must pass) → `python scripts/02_run_survey.py`
  builds `data/processed/baselines.json` from real responses (no code changes needed).

### 6b. Translated questionnaire wording → prompts  (`config/translations/<lang>.yaml`)
- The R9 questionnaire was professionally translated into the SA survey languages for
  fieldwork. Get that official translated wording from the
  **SA R9 questionnaire page** (https://www.afrobarometer.org/survey-resource/south-africa-round-9-questionnaire-2022/);
  if per-language versions aren't posted, **request them from Afrobarometer / the IJR**
  (do not invent or machine-translate silently — see the cardinal rule).
- Populate `config/translations/<code>.yaml` per `config/translations/_template.yaml`,
  keyed by the item ids in `config/items_political.yaml`. Missing translations fall back to
  English **with a warning**, so smoke runs work — but the real run needs them, or the
  language manipulation isn't happening.

### Capability benchmarks — no manual step
AfroBench / IrokoBench pulled from HF at runtime by `scripts/03_run_capability.py`.

---

## 7. Running the real pipeline

```bash
bash scripts/00_download_data.sh          # guidance + fetches the public codebook
# ... place the .sav (6a) and translations (6b) ...
python scripts/validate_microdata.py      # gate before long runs
python scripts/02_run_survey.py           # Step 1: survey the models (resumable)
python scripts/03_run_capability.py       # Step 2: AfroBench + extension
python scripts/04_compute_divergence.py   # Step 3: divergence tables
python scripts/05_correlate_and_plot.py   # Step 4: correlations + figures
```
Outputs (all CSV/JSON/PNG/SVG, indexed by `results/run_manifest.json`): see the "Outputs"
table in `readme.md`. `results/results_master.csv` is the one-stop table.

**AfroBench task names**: before Step 2, confirm the lm-eval task names per language in
`config/afrobench_tasks.yaml` against your installed harness
(`lm_eval --tasks list | grep -E 'afrimmlu|afrixnli|afrimgsm'`). zul/xho/sot are solid;
the rest are marked VERIFY.

---

## 8. Open empirical questions — verify against the real `.sav`

`validate_microdata.py` answers these automatically once the `.sav` is present:
- **Is `Q102 == 705` (Venda) actually used in SA R9?** If yes, Tshivenda has a clean
  Afrobarometer baseline and **SASAS is unnecessary**. If no, ven → SASAS or a gap.
- **Is `Q102 == 542` (Tsonga/Changana) the right code for Xitsonga in SA?** Adjust
  `config/languages.yaml` + `config/afrobarometer_recode.yaml` `language_map` if not.
- Confirm the weight var `withinwt_hh` (else `withinwt`/`withinwt_ea`) is present.
- Confirm every item variable (Q23, Q22A…) exists.

---

## 9. Architecture map

```
config/            languages, models, items (real R9 wording+codes), recode, afrobench tasks, translations
src/decol/
  config.py        loads all YAML into dataclasses; nothing is hardcoded
  dist.py          Distribution (shared categorical dist on a scale's ordered support)
  data/            afrobarometer.py (numeric .sav -> baselines), sasas.py, baselines.py
  elicit/          prompts.py, ollama_client.py, parse.py, runner.py (resumable), simulate.py
  capability/      afrobench.py (lm-eval -> Ollama OpenAI endpoint), extend.py (ven/nbl proxy)
  divergence/      metrics.py (JSD, mean-shift, gating), compare.py (-> tables)
  analysis/        correlate.py (div ~ capability + log(params)), figures.py
scripts/           00..05 + validate_microdata.py
tests/             36 tests (pure logic: parse, metrics, dist, config, compare, analysis)
```

---

## 10. Gotchas (learned from the real data)

- **Read `.sav` as NUMERIC codes** (`apply_value_formats=False`) — the loader maps integer
  codes, not label strings. Recode = `config/afrobarometer_recode.yaml`.
- **Q23 codes are scrambled:** `1=doesn't matter, 2=non-democratic preferable,
  3=democracy preferable`. Do not assume 1=Statement 1.
- **Two-statement items (Q15,Q18,Q19A,Q24,Q27A,Q27C,Q28,Q29A,Q29B,Q16):** codes
  `1=strongly Stmt1, 2=Stmt1, 3=Stmt2, 4=strongly Stmt2, 5=neither`. We place "neither" in
  the **middle** of the ordinal scale; `higher_means` documents what the top (Stmt2) is.
- **COUNTRY: South Africa = 33.** Weight = `withinwt_hh`.
- **The economic items I'd first drafted (tax/redistribution/jobs) do NOT exist in R9
  core** — they were removed. Don't re-add without a source.
- **statsmodels OLS** degrades gracefully if the env's scipy/statsmodels clash; correlations
  + figures still run.
- Local `results/` on macOS currently holds a **simulated** demo (flagged `simulated:true`
  in the manifest) and is gitignored — delete it / re-run for real.

---

## 11. Definition of done

A real run where: `validate_microdata.py` passes; `config/translations/` is populated from
official wording for the survey languages; Steps 1–4 complete; `results/results_master.csv`
holds per-(model, language) divergence + capability + size; and `results/figures/` shows the
per-language map and the capability/size scatter. Then interpret: which languages diverge
most, and whether capability or size explains it.
