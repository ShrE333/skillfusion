#!/usr/bin/env bash
# Build the doc corpus the code-gen agent retrieves from (RAG). Isaac Lab docs come from the repo;
# add Isaac Sim 6.0.1 docs by exporting them into the same folder (any .md/.rst/.txt is indexed).
set -euo pipefail
OUT=${1:-docs/isaac}; mkdir -p "$OUT"
TMP=$(mktemp -d)
git clone --depth 1 --branch "${ISAACLAB_TAG:-v3.0.0-beta2.patch1}" https://github.com/isaac-sim/IsaacLab "$TMP/IsaacLab"
cp -r "$TMP/IsaacLab/docs/source" "$OUT/isaaclab"
cp "$TMP/IsaacLab/AGENTS.md" "$OUT/" 2>/dev/null || true
echo "docs -> $OUT ($(find "$OUT" -type f | wc -l) files)"
