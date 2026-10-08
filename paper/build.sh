#!/usr/bin/env bash
# Build the manuscript: pdflatex -> bibtex -> pdflatex x2.
# Needs libertine, newtx, inconsolata, tcolorbox, multirow, authblk, lastpage, wrapfig.
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex >/dev/null
bibtex manuscript >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex | grep -E "Output written" || true
grep -E "Warning: (Citation|Reference).*undefined|Overfull \\\\hbox \([0-9]{2,}" manuscript.log | sort -u || true
