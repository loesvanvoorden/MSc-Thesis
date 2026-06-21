#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

JOB="LoesVanVoorden_2055036_thesis_MSc_DSBE"
BUILD_DIR="build"

mkdir -p "$BUILD_DIR"

run_pdflatex() {
  pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" "$JOB.tex" || true
}

run_pdflatex
bibtex "$BUILD_DIR/$JOB"
run_pdflatex
run_pdflatex

cp "$BUILD_DIR/$JOB.pdf" "$ROOT/$JOB.pdf"
echo "Wrote $ROOT/$JOB.pdf (auxiliary files in $BUILD_DIR/)"
