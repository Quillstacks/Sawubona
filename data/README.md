# Human-baseline microdata

Two survey sources provide the **human baseline** that model answers are compared against.
Both are license-gated and **must not be committed** (`data/raw/` is gitignored).

```
data/
  raw/
    afrobarometer/        # Afrobarometer Round 9 (merged) + SA questionnaire + codebook
    sasas/                # SASAS microdata (manual; HSRC-gated) — Tshivenda cross-check
  processed/              # built baselines (generated; do not edit by hand)
```

---

## 1. Afrobarometer Round 9 (South Africa) — scripted

Nationally representative; nine survey languages (Afrikaans, English, isiZulu, isiXhosa,
Sepedi, Sesotho, Setswana, siSwati, Xitsonga). Free, one-time registration.

`scripts/00_download_data.sh` will guide/fetch these into `data/raw/afrobarometer/`:

1. **Merged Round 9 data set (SPSS `.sav`)** — needed for per-language attitude
   distributions. From the Afrobarometer data portal:
   <https://www.afrobarometer.org/data/data-sets/> (Round 9 merged), or DataFirst:
   <https://www.datafirst.uct.ac.za/dataportal/index.php/catalog/989>.
2. **South Africa Round 9 questionnaire (2022)** — the professionally translated item
   wording per language (used as the prompt text):
   <https://www.afrobarometer.org/survey-resource/south-africa-round-9-questionnaire-2022/>.
3. **Merged Round 9 codebook** — variable names + response codes, to confirm the item
   variables and the language-of-interview variable referenced in
   `config/items_political.yaml` and `src/decol/data/afrobarometer.py`:
   <https://www.afrobarometer.org/wp-content/uploads/2025/07/AB_R9.MergeCodebook_25Jun24.final_.pdf>.

> Registration may gate the direct download link. If the script cannot fetch a file, it
> prints the page URL; download manually and drop the file into
> `data/raw/afrobarometer/`. The loader auto-detects `.sav`.

Expected files (names may vary by release — the loader globs for `*.sav`):
```
data/raw/afrobarometer/
  R9.Merge_<date>.sav
  SA_R9_Questionnaire_2022.pdf
  AB_R9.MergeCodebook_<date>.pdf
```

## 2. SASAS — manual, and now OPTIONAL

The R9 codebook shows **705 = Venda** is a valid Q102 "language of interview" code, so
South Africa R9 *may* already provide a clean Tshivenda baseline directly (confirm by
checking whether Q102 == 705 appears in the SA `.sav`). If it does, SASAS is not needed.

SASAS (HSRC) remains available as a Tshivenda fallback / cross-survey validity check.
Access is gated (HSRC / DataFirst application). Once obtained, place the microdata in:
```
data/raw/sasas/
  sasas_<year>.sav
```
`src/decol/data/sasas.py` reads it to the same schema as the Afrobarometer loader. If
neither Q102==705 nor SASAS is available, the pipeline omits the Tshivenda baseline
(documented gap) — acceptable per the project's "okay to lose a small language" decision.

## 3. Capability benchmarks — no manual step

AfroBench / IrokoBench / SimBench and the MzansiText extension resources are pulled from
Hugging Face at runtime by `scripts/03_run_capability.py`.

---

## Coverage summary

| Language | Code | Q102 | Human baseline |
|----------|------|------|----------------|
| English | eng | 1 | **Afrobarometer R9** |
| Afrikaans | afr | 700 | **Afrobarometer R9** |
| isiXhosa | xho | 701 | **Afrobarometer R9** |
| Sepedi | nso | 702 | **Afrobarometer R9** |
| Sesotho | sot | 703 | **Afrobarometer R9** |
| Setswana | tsn | 704 | **Afrobarometer R9** |
| isiZulu | zul | 706 | **Afrobarometer R9** |
| siSwati | ssw | 1620 | **Afrobarometer R9** |
| Xitsonga | tso | 542 | **Afrobarometer R9** (verify code in .sav) |
| Tshivenda | ven | 705 | **Afrobarometer R9** if Q102==705 present, else SASAS, else gap |
| isiNdebele | nbl | — | *none* — capability measured, divergence reported as a gap |
