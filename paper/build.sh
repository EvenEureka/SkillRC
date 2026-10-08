#!/usr/bin/env bash
# Build the manuscript (MemoHarness preprint style): pdflatex -> bibtex -> pdflatex x2.
# Needs libertine, newtx, inconsolata, tcolorbox, multirow, authblk, lastpage, wrapfig.
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error preprint.tex >/dev/null
bibtex preprint >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error preprint.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error preprint.tex | grep -E "Output written" || true
grep -E "Warning: (Citation|Reference).*undefined|Overfull \\\\hbox \([0-9]{2,}" preprint.log | sort -u || true
