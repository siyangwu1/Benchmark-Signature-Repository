#!/usr/bin/env python3
"""
Read a single value from config/config.yaml by dotted key path.

Usage (from any directory):
    python /path/to/repo/config/read_config.py <key.path>

Examples:
    python config/read_config.py perplexity.model
    python config/read_config.py paths.chunked_dataset
    python config/read_config.py benchmark.tasks        # prints space-separated list

Bash helper pattern used in submit/setup scripts:
    REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    cfg() { python3 "$REPO_ROOT/config/read_config.py" "$1"; }
    MODEL=$(cfg perplexity.model)
"""

import sys
from pathlib import Path

import yaml

CONFIG_FILE = Path(__file__).parent / "config.yaml"


def get_value(key_path: str):
    with open(CONFIG_FILE) as f:
        cfg = yaml.safe_load(f)
    keys = key_path.split(".")
    val = cfg
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            raise KeyError(f"Key '{k}' not found (full path: '{key_path}')")
        val = val[k]
    return val


def format_value(val) -> str:
    if isinstance(val, list):
        return " ".join(str(x) for x in val)
    if isinstance(val, bool):
        return str(val).lower()
    return str(val)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <key.path>", file=sys.stderr)
        sys.exit(1)
    try:
        value = get_value(sys.argv[1])
        print(format_value(value))
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: config file not found at {CONFIG_FILE}", file=sys.stderr)
        sys.exit(1)
