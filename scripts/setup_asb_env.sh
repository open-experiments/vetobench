#!/usr/bin/env bash
# Create ASB's own virtualenv at third_party/ASB/.venv (kept separate from vetobench's).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
ASB=third_party/ASB
if [ ! -d "$ASB" ]; then
  git clone https://github.com/agiresearch/ASB "$ASB"
  git -C "$ASB" checkout 1f561dccf92d55302368fa67679b4ba9d9c8fdc4
fi
if ! "$PY" -m venv "$ASB/.venv" >/dev/null 2>&1; then
  # Debian/Ubuntu Pythons without ensurepip: bootstrap pip manually.
  rm -rf "$ASB/.venv"
  "$PY" -m venv --without-pip "$ASB/.venv"
  curl -sSL https://bootstrap.pypa.io/get-pip.py | "$ASB/.venv/bin/python" - -q
fi
"$ASB/.venv/bin/pip" install -q -r scripts/requirements-asb.txt
"$ASB/.venv/bin/python" - <<'PY'
import os, sys
sys.path.insert(0, "third_party/ASB")
from aios.llm_core import llms  # noqa: F401  (import check)
print("ASB environment OK")
PY
