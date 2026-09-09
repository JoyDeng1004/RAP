#!/usr/bin/env bash
# md2pdf.sh — convert a Markdown file (LaTeX math + images) to PDF.
#
# Usage:
#   tools/md2pdf/md2pdf.sh <input.md> [output.pdf]
#
# If output.pdf is omitted, writes alongside the input (same name, .pdf).
# First run installs node deps automatically; later runs reuse them.
#
# Override the browser with:  CHROME="/path/to/chrome" tools/md2pdf/md2pdf.sh ...

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <input.md> [output.pdf]" >&2
  exit 1
fi

IN="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
if [[ ! -f "$IN" ]]; then
  echo "error: input not found: $1" >&2
  exit 1
fi
OUT="${2:-${IN%.md}.pdf}"

# Resolve a Chrome/Chromium binary.
CHROME="${CHROME:-}"
if [[ -z "$CHROME" ]]; then
  for c in \
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    "/Applications/Chromium.app/Contents/MacOS/Chromium" \
    "$(command -v google-chrome || true)" \
    "$(command -v chromium || true)" \
    "$(command -v chromium-browser || true)"; do
    if [[ -n "$c" && -x "$c" ]]; then CHROME="$c"; break; fi
  done
fi
if [[ -z "$CHROME" || ! -x "$CHROME" ]]; then
  echo "error: no Chrome/Chromium found; set CHROME=/path/to/chrome" >&2
  exit 1
fi

# Install node deps on first use.
if [[ ! -d "$DIR/node_modules/mathjax" || ! -d "$DIR/node_modules/markdown-it" ]]; then
  echo "[md2pdf] installing node dependencies (first run)..." >&2
  ( cd "$DIR" && npm install --silent )
fi

TMP_HTML="$(mktemp -t md2pdf.XXXXXX).html"
trap 'rm -f "$TMP_HTML"' EXIT

node "$DIR/build.js" "$IN" "$TMP_HTML"

"$CHROME" --headless=new --disable-gpu --no-sandbox \
  --allow-file-access-from-files \
  --virtual-time-budget=30000 \
  --run-all-compositor-stages-before-draw \
  --print-to-pdf-no-header \
  --print-to-pdf="$OUT" \
  "file://$TMP_HTML" >/dev/null 2>&1

echo "[md2pdf] wrote: $OUT"
