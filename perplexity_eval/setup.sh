#!/bin/bash
# Set up cache environment variables from config/config.yaml.
# Source this script before running wiki_eval.py or submitting jobs:
#   source setup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cfg() { python3 "$REPO_ROOT/config/read_config.py" "$1"; }

export HF_HOME=$(cfg environment.hf_home)
export HF_HUB_CACHE=$(cfg environment.hf_hub_cache)
export HF_XET_CACHE=$(cfg environment.hf_xet_cache)
export VLLM_CACHE_ROOT=$(cfg environment.vllm_cache_root)
export UV_CACHE_DIR=$(cfg environment.uv_cache_dir)

# Use HF_TOKEN from config if set, otherwise fall back to the shell environment
_HF_TOKEN_CFG=$(cfg environment.hf_token)
if [ -n "$_HF_TOKEN_CFG" ]; then
    export HF_TOKEN="$_HF_TOKEN_CFG"
fi
unset _HF_TOKEN_CFG
