#!/usr/bin/env bash
# Pull every model in the roster (config/models.yaml) into the local Ollama daemon.
# Tags are exact `ollama pull` identifiers — registry names or hf.co/<repo> GGUFs that
# Ollama fetches directly — so there is nothing to convert or place by hand.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama not found. Install: curl -fsSL https://ollama.com/install.sh | sh" >&2
  exit 1
fi

PY="${PYTHON:-python3}"

mapfile -t TAGS < <("$PY" - "$ROOT" <<'PYEOF'
import sys, pathlib
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / "src"))
from decol import config
for m in config.roster():                 # only models within the GPU budget
    print(m.tag)
PYEOF
)

echo "==> Pulling ${#TAGS[@]} models from the roster ..."
fail=0
for tag in "${TAGS[@]}"; do
  echo "  ollama pull $tag"
  if ! ollama pull "$tag"; then
    echo "    !! failed: $tag" >&2
    fail=$((fail + 1))
  fi
done

echo
if [ "$fail" -gt 0 ]; then
  echo "Done with $fail failure(s). Re-run to retry, or check the tag in config/models.yaml." >&2
else
  echo "All models pulled."
fi
