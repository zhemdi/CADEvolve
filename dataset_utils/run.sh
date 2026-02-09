#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------
# Project root (directory where this run.sh lives)
# ---------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ---------------------------------------------------------
# Paths to scripts + configs
# ---------------------------------------------------------
SAMPLING_PY="./canonicalization_run/sampling.py"
CANON_PY="./canonicalization_run/canonicalization.py"
ROT_PY="./rotation_run/mixed_rotation.py"

CFG_SAMPLING="./canonicalization_run/cfg_sampling.yaml"
CFG_CANON="./canonicalization_run/cfg_canonicalization.yaml"
CFG_ROT="./rotation_run/cfg_rotation.yaml"

# ---------------------------------------------------------
# Data root (single source of truth) (RELATIVE)
# ---------------------------------------------------------
OUT_DIR="./results"

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
ts()  { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

# ---------------------------------------------------------
# Ensure dirs exist
# ---------------------------------------------------------
mkdir -p "$OUT_DIR/logs"

# ---------------------------------------------------------
# 1) Sampling
# ---------------------------------------------------------
log "Step 1: sampling"
(
  cd "$OUT_DIR"
  python "$PROJECT_ROOT/$SAMPLING_PY" --config "$PROJECT_ROOT/$CFG_SAMPLING"
)

# ---------------------------------------------------------
# 2) Canonicalization
# ---------------------------------------------------------
log "Step 2: canonicalization"
(
  cd "$OUT_DIR"
  python "$PROJECT_ROOT/$CANON_PY" --config "$PROJECT_ROOT/$CFG_CANON"
)

# ---------------------------------------------------------
# 3) Flatten canonicalized/binarized -> canonicalized_flat
#    output: canonicalized_flat/<topdir>__<filename>.py   (NO SUBFOLDERS)
# ---------------------------------------------------------
log "Step 3: flatten canonicalized/binarized -> canonicalized_flat"

CANON_DIR="$OUT_DIR/canonicalized/binarized"
FLAT_DIR="$OUT_DIR/canonicalized_flat"

if find "$FLAT_DIR" -maxdepth 1 -type f -name "*.py" -print -quit >/dev/null 2>&1; then
  log "Step 3: flatten (SKIP) -> found existing flat scripts in $FLAT_DIR"
else
  rm -rf "$FLAT_DIR"
  mkdir -p "$FLAT_DIR"

  if ! find "$CANON_DIR" -type f -name "*.py" -print -quit >/dev/null 2>&1; then
    log "Flatten: ERROR -> no *.py found in $CANON_DIR (cannot proceed to Step 4)"
    exit 1
  fi

  while IFS= read -r -d '' f; do
    rel="${f#"$CANON_DIR"/}"
    top="${rel%%/*}"
    base="$(basename "$f")"

    if [[ "$top" == "$rel" ]]; then
      top="_root"
    fi

    dst="$FLAT_DIR/${top}__${base}"

    if [[ -e "$dst" ]]; then
      suf="$(python - <<'PY'
import hashlib, sys
s = sys.stdin.read().strip()
print(hashlib.md5(s.encode("utf-8")).hexdigest()[:8])
PY
<<< "$rel")"
      dst="$FLAT_DIR/${top}__${base%.py}__${suf}.py"
    fi

    cp -f "$f" "$dst"
  done < <(find "$CANON_DIR" -type f -name "*.py" -print0)

  log "Flatten done: $(find "$FLAT_DIR" -maxdepth 1 -type f -name "*.py" | wc -l) scripts in canonicalized_flat"
fi

# ---------------------------------------------------------
# 4) Rotation augmentation
# ---------------------------------------------------------
log "Step 4: rotation augmentation"
(
  cd "$OUT_DIR"
  python "$PROJECT_ROOT/$ROT_PY" --config "$PROJECT_ROOT/$CFG_ROT"
)

log "Pipeline finished OK"
