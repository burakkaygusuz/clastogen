#!/usr/bin/env bash
# Makes assets/social.png. Run from the repository root: CHROME=/path/to/chrome bash assets/social.sh
set -euo pipefail
html="$(mktemp -d)/social.html"
cat > "$html" <<HTML
<!doctype html><meta charset="utf-8">
<style>
  html, body { margin: 0; width: 1280px; height: 640px; overflow: hidden; }
  body { background: radial-gradient(circle at 20% 0%, #313244 0%, #1e1e2e 55%, #181825 100%);
         color: #cdd6f4; font-family: -apple-system, "SF Pro Display", "Helvetica Neue", sans-serif;
         display: flex; flex-direction: column; justify-content: center; padding: 0 100px; box-sizing: border-box; }
  img { width: 820px; height: auto; margin: 0 0 24px -28px; }
  h1 { font-size: 96px; margin: 20px 0 8px; letter-spacing: -2px; color: #cba6f7; }
  .tag { font-size: 40px; white-space: nowrap; font-weight: 600; margin: 0 0 32px; }
  .sub { font-size: 24px; color: #a6adc8; margin: 0 0 28px; }
  code { display: inline-block; align-self: flex-start; font: 26px Menlo, "SF Mono", monospace;
         background: #11111b; border: 1px solid #45475a; border-radius: 10px; padding: 12px 22px; color: #f38ba8; }
  code b { color: #a6e3a1; font-weight: normal; }
</style>
<img src="file://$PWD/assets/logo.svg">
<p class="tag">Your LLM evals pass. Do they find a broken prompt?</p>
<code>Mutation Score: 60.0% <b>(3 of 5 killed)</b></code>
HTML
"${CHROME:-google-chrome}" --headless=new --hide-scrollbars --force-device-scale-factor=1 \
  --window-size=1280,640 --screenshot="$PWD/assets/social.png" "file://$html"
