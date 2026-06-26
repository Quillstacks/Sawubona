#!/usr/bin/env bash
# Fetch the Afrobarometer Round 9 (South Africa) materials into data/raw/afrobarometer/.
# SASAS is gated and placed manually (see data/README.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/raw/afrobarometer"
mkdir -p "$DEST"

echo "==> Afrobarometer R9 South Africa -> $DEST"
echo
echo "Afrobarometer requires a one-time (free) registration before data download."
echo "Open these pages, sign in, and save the files into:"
echo "    $DEST"
echo
echo "  1) Merged Round 9 data set (SPSS .sav):"
echo "     https://www.afrobarometer.org/data/data-sets/"
echo "     (mirror) https://www.datafirst.uct.ac.za/dataportal/index.php/catalog/989"
echo "  2) South Africa Round 9 questionnaire (2022, translated wording):"
echo "     https://www.afrobarometer.org/survey-resource/south-africa-round-9-questionnaire-2022/"
echo "  3) Merged Round 9 codebook (variable names + response codes):"
echo "     https://www.afrobarometer.org/wp-content/uploads/2025/07/AB_R9.MergeCodebook_25Jun24.final_.pdf"
echo

# Best-effort direct fetch of the public codebook PDF (no registration needed).
CODEBOOK_URL="https://www.afrobarometer.org/wp-content/uploads/2025/07/AB_R9.MergeCodebook_25Jun24.final_.pdf"
if command -v curl >/dev/null 2>&1; then
  echo "==> Attempting to fetch the public codebook PDF ..."
  curl -fL --retry 3 -o "$DEST/AB_R9.MergeCodebook.pdf" "$CODEBOOK_URL" \
    && echo "    saved $DEST/AB_R9.MergeCodebook.pdf" \
    || echo "    could not fetch automatically — download manually (see above)."
fi

echo
if ls "$DEST"/*.sav >/dev/null 2>&1; then
  echo "==> Found a .sav in $DEST — ready."
else
  echo "==> No .sav yet in $DEST. Download the merged R9 data set, then re-run the pipeline."
fi
echo
echo "Next: fill config/afrobarometer_recode.yaml from the codebook (item variables + codes)."
