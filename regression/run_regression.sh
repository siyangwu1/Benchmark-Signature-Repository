#!/bin/bash
# Run signature extraction (Thrush pre-filter + Lasso regression).
# All settings are read from config/config.yaml.
# Usage: bash regression/run_regression.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cfg() { python3 "$REPO_ROOT/config/read_config.py" "$1"; }

export PYTHONUNBUFFERED=1

# Runtime parameters from config
PERF_DIR=$(cfg paths.performance_matrix_dir)
PERF_FILE=$(cfg regression.performance_matrix_file)
TASK_PREFIXES=$(cfg regression.task_prefixes)
INCLUDE=$(cfg regression.include)
EXCLUDE=$(cfg regression.exclude)
SIG_DIR=$(cfg paths.signature_dir)
MODEL_DIR=$(cfg paths.model_dir)
N=$(cfg regression.n_shards)
PART=$(cfg regression.shard_part)
PRESELECT_RATIO=$(cfg regression.preselect_ratio)
ALPHA=$(cfg regression.alpha)
ONLY_REMAINING=$(cfg regression.only_remaining)

# Resolve relative paths against repo root
if [[ "$PERF_DIR"  != /* ]]; then PERF_DIR="$REPO_ROOT/$PERF_DIR";   fi
if [[ "$SIG_DIR"   != /* ]]; then SIG_DIR="$REPO_ROOT/$SIG_DIR";    fi
if [[ "$MODEL_DIR" != /* ]]; then MODEL_DIR="$REPO_ROOT/$MODEL_DIR"; fi

PERF_CSV="$PERF_DIR/$PERF_FILE"
mkdir -p "$SIG_DIR" "$MODEL_DIR"

# Optional flags
EXTRA_FLAGS=""
if [ "$ONLY_REMAINING" = "true" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --only-remaining"
fi

cd "$SCRIPT_DIR"

python -u regression_run.py \
  --perf-csv        "$PERF_CSV" \
  --task-prefixes   "$TASK_PREFIXES" \
  --include         "$INCLUDE" \
  --exclude         "$EXCLUDE" \
  --sig-dir         "$SIG_DIR" \
  --model-dir       "$MODEL_DIR" \
  --N               "$N" \
  --part            "$PART" \
  --preselect-ratio "$PRESELECT_RATIO" \
  --alpha           "$ALPHA" \
  $EXTRA_FLAGS
