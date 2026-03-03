#!/bin/bash
# Run perplexity extraction.
# All settings are read from config/config.yaml.
# Usage: bash perplexity_eval/run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cfg() { python3 "$REPO_ROOT/config/read_config.py" "$1"; }

# Load cache environment from config
source "$SCRIPT_DIR/setup.sh"

# Runtime parameters from config
MODEL=$(cfg perplexity.model)
DP_SIZE=$(cfg perplexity.dp_size)
TP_SIZE=$(cfg perplexity.tp_size)
DATASET_FILE=$(cfg paths.chunked_dataset)
PROMPT_COLUMN=$(cfg perplexity.prompt_column)
DEBUG_LIMIT=$(cfg perplexity.debug_limit)

# Resolve relative dataset path against repo root
if [[ "$DATASET_FILE" != /* ]]; then
    DATASET_FILE="$REPO_ROOT/$DATASET_FILE"
fi

nvidia-smi || true

cd "$SCRIPT_DIR"

EXTRA_ARGS=""
if [ "$DEBUG_LIMIT" -gt 0 ]; then
    EXTRA_ARGS="--debug-limit $DEBUG_LIMIT"
fi

python wiki_eval.py \
    --model "$MODEL" \
    --dp-size "$DP_SIZE" \
    --tp-size "$TP_SIZE" \
    --dataset "$DATASET_FILE" \
    --prompt-column "$PROMPT_COLUMN" \
    $EXTRA_ARGS
