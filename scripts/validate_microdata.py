#!/usr/bin/env python3
"""Validate the Afrobarometer R9 .sav the moment you drop it in — before a long run.

Checks, against config/afrobarometer_recode.yaml + config/languages.yaml:
  - the .sav is found and South Africa (COUNTRY==33) rows exist;
  - the survey weight variable (or a fallback) is present;
  - Q102 "language of interview" is present, and WHICH language codes actually appear for
    SA (so you can confirm the open questions: is 705=Venda present? is 542=Tsonga?);
  - every recoded item variable (Q23, Q22A, ...) exists, with its per-language n.

Exit code 0 if all required checks pass, 1 otherwise. Run:  python scripts/validate_microdata.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from decol import config                              # noqa: E402
from decol.data import afrobarometer as ab            # noqa: E402

OK, WARN, FAIL = "  ok ", " warn", " FAIL"


def main() -> int:
    try:
        import pyreadstat
    except ImportError:
        print("pyreadstat not installed. `pip install pyreadstat`.")
        return 1

    try:
        sav = ab.find_sav()
    except FileNotFoundError as exc:
        print(f"{FAIL}  {exc}")
        return 1
    print(f"{OK}  found {sav.name}")

    df, _ = pyreadstat.read_sav(str(sav), apply_value_formats=False)
    recode = ab.load_recode()
    g = recode["global"]
    problems = 0

    # --- country filter ---
    import pandas as pd
    cvar, cval = g.get("country_var"), g.get("country_value")
    if cvar in df.columns:
        sa = df[pd.to_numeric(df[cvar], errors="coerce") == float(cval)]
        tag = OK if len(sa) else FAIL
        problems += 0 if len(sa) else 1
        print(f"{tag}  {cvar}=={cval} (South Africa): {len(sa)} rows")
    else:
        sa = df
        print(f"{WARN}  country var {cvar!r} absent; treating whole file as SA ({len(sa)} rows)")

    # --- weight ---
    wcands = [g.get("weight_var"), *(g.get("weight_var_fallbacks") or [])]
    wfound = next((w for w in wcands if w and w in df.columns), None)
    if wfound:
        print(f"{OK}  weight variable: {wfound}")
    else:
        problems += 1
        print(f"{FAIL}  no weight variable found (tried {wcands})")

    # --- Q102 language of interview: which codes actually appear for SA ---
    lang_var = g.get("language_var")
    lang_map = {int(k): v for k, v in (g.get("language_map") or {}).items()}
    if lang_var in sa.columns:
        codes = pd.to_numeric(sa[lang_var], errors="coerce").dropna().round().astype(int)
        vc = codes.value_counts().sort_index()
        print(f"{OK}  {lang_var} present; languages of interview seen in SA:")
        for code, n in vc.items():
            iso = lang_map.get(code, "??")
            flag = "" if code in lang_map else "   <-- not in language_map!"
            print(f"          {code:>5} -> {iso:<4} n={n}{flag}")
        for code, iso in [(705, "ven"), (542, "tso")]:
            present = code in set(vc.index)
            print(f"{(OK if present else WARN)}  open question: {code}={iso} "
                  f"{'present' if present else 'NOT present'} in SA")
    else:
        problems += 1
        print(f"{FAIL}  language var {lang_var!r} absent")

    # --- item variables ---
    print("       item variables:")
    for item_id, spec in (recode.get("items") or {}).items():
        var = spec["var"]
        present = var in df.columns
        if not present:
            problems += 1
        print(f"{(OK if present else FAIL)}    {item_id:<32} {var:<6} "
              f"{'present' if present else 'MISSING'}")

    print()
    if problems:
        print(f"{FAIL}  {problems} required check(s) failed — fix before running the survey.")
        return 1
    print(f"{OK}  all required checks passed — ready for scripts/02_run_survey.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
