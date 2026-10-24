#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -z "${GENESIS_PYTHON:-}" ]]; then
    if python3 -c 'import importlib.util; raise SystemExit(importlib.util.find_spec("genesis") is None)' >/dev/null 2>&1; then
        GENESIS_PYTHON=python3
    elif [[ -x "$HOME/Desktop/genesis-local/.venv/bin/python" ]]; then
        # Preserve the original workstation setup without a fixed user path.
        GENESIS_PYTHON="$HOME/Desktop/genesis-local/.venv/bin/python"
    else
        echo 'Activate a Python environment with Genesis, or set GENESIS_PYTHON to its Python executable.' >&2
        exit 1
    fi
fi
exec "$GENESIS_PYTHON" run_simulation.py "$@"
