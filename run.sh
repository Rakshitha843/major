#!/usr/bin/env bash
# One-click launcher for macOS/Linux: activates the project's virtual environment (creating one
# on first run if it does not exist yet), installs dependencies if needed, starts the server, and
# opens the dashboard in the default browser.
#
# Usage:  ./run.sh   (run "chmod +x run.sh" once first if needed)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f ".venv/bin/activate" ]; then
    echo "[run.sh] No virtual environment found -- creating one in .venv ..."
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"

echo "[run.sh] Checking dependencies (this is quick if they are already installed) ..."
python -m pip install -r requirements.txt --quiet --disable-pip-version-check

echo "[run.sh] Starting the server at http://127.0.0.1:8000 ..."
echo "[run.sh] Press Ctrl+C to stop it."

# Open the dashboard in the background once the server actually answers on port 8000, instead
# of a fixed sleep -- a fixed delay can fire too early (slow first-time pip install) or leave an
# annoying gap (everything already warm), and firing before the server is ready shows
# "This site can't be reached" in the browser. Polls for up to 20 seconds.
( ok=""
  for _ in $(seq 1 40); do
    if curl -s -o /dev/null "http://127.0.0.1:8000"; then ok=1; break; fi
    sleep 0.5
  done
  if [ -n "$ok" ]; then
    if command -v open >/dev/null 2>&1; then open "http://127.0.0.1:8000"          # macOS
    elif command -v xdg-open >/dev/null 2>&1; then xdg-open "http://127.0.0.1:8000" # Linux
    fi
  fi
) &

uvicorn backend.main:app --reload
