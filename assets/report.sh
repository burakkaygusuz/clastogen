#!/usr/bin/env bash
# Makes assets/report.png. Run from the repository root: CHROME=/path/to/chrome bash assets/report.sh
set -euo pipefail
html="$(mktemp -d)/report.html"
uv run pytest -q --clastogen --clastogen-html="$html" examples/test_banking_eval.py
"${CHROME:-google-chrome}" --headless=new --hide-scrollbars --force-dark-mode \
  --blink-settings=preferredColorScheme=0 --force-device-scale-factor=1 --window-size=1100,1365 \
  --screenshot="$PWD/assets/report.png" "file://$html"
