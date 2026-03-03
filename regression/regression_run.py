#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thrush-prefiltered Lasso signatures for benchmark analysis.

Pipeline:
  1. Load stacked feature matrix and chunk IDs.
  2. Load benchmark performance matrix (CSV).
  3. Filter target columns by prefix / include / exclude lists.
  4. For each benchmark, compute Thrush rank correlations to pre-select
     candidate features, then fit a fixed-alpha Lasso.
  5. Write per-benchmark signature CSVs and save fitted models (.joblib).

All paths and hyper-parameters default to values in config/config.yaml.
CLI arguments override those defaults.
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from joblib import dump
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


# ── Config helpers ────────────────────────────────────────────────────────────

def _find_repo_root() -> Path:
    """Walk up from this file until config/config.yaml is found."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "config" / "config.yaml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Cannot locate config/config.yaml")


def _load_config() -> tuple[dict, Path]:
    repo_root = _find_repo_root()
    with open(repo_root / "config" / "config.yaml") as f:
        return yaml.safe_load(f), repo_root


def _resolve(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else repo_root / p


# ── Load config at module level so function-signature defaults work ────────────

_CFG, _REPO_ROOT = _load_config()
_PATHS = _CFG["paths"]
_REG   = _CFG["regression"]

FEATURE_MATRIX_DIR = _resolve(_PATHS["feature_matrix_dir"],  _REPO_ROOT)
FILTERED_PARTS_DIR = _resolve(_PATHS["filtered_parts_dir"],  _REPO_ROOT)
DEFAULT_PERF_CSV   = (
    _resolve(_PATHS["performance_matrix_dir"], _REPO_ROOT)
    / _REG["performance_matrix_file"]
)
DEFAULT_SIG_DIR    = _resolve(_PATHS["signature_dir"], _REPO_ROOT)
DEFAULT_MODEL_DIR  = _resolve(_PATHS["model_dir"],     _REPO_ROOT)

DEFAULT_THRUSH_PCT    = _REG["preselect_ratio"]
DEFAULT_LASSO_ALPHA   = _REG["alpha"]
LASSO_MAX_ITER        = _REG["lasso_max_iter"]
LASSO_RANDOM_STATE    = _REG["lasso_random_state"]


# ═══════════════════════════════════════════════════════════════════════════════
# Thrush rank-correlation & candidate selection
# ═══════════════════════════════════════════════════════════════════════════════

def compute_thrush_rank_correlations(
    feature_df: pd.DataFrame,
    response: Dict[str, float],
) -> pd.Series:
    """
    Compute Thrush-style rank correlations between each feature row and
    the response vector across shared models.

    Parameters
    ----------
    feature_df : DataFrame (n_features × n_models)
    response   : dict mapping model name → benchmark score

    Returns
    -------
    Series indexed by feature row, values are Thrush correlation scores.
    """
    shared_models = sorted(set(feature_df.columns) & set(response.keys()))
    if len(shared_models) < 2:
        raise ValueError("Need ≥ 2 shared models between features and response.")

    X = feature_df[shared_models]
    y = pd.Series(response, dtype=float)[shared_models]

    n_models = y.size
    y_ranks = y.rank(method="first").to_numpy(dtype=np.float64)
    v = 2.0 * y_ranks - (n_models + 1.0)

    # R: (n_models × n_features), rank each feature across models
    R = X.T.rank(axis=0, method="first").to_numpy(dtype=np.float64)
    scores = 2.0 * (R.T @ v)  # (n_features,)

    return pd.Series(scores, index=feature_df.index, name="thrush_correlation")


def select_candidates_by_thrush(
    thrush_scores: pd.Series,
    pct: float = 0.01,
    min_candidates: int = 50,
) -> List[int]:
    """
    Select candidate feature indices from the top/bottom tails of
    the Thrush correlation distribution.

    Parameters
    ----------
    thrush_scores  : Series of Thrush scores (indexed by feature row).
    pct            : Fraction of features to take from each tail.
    min_candidates : Minimum total candidates to return.

    Returns
    -------
    List of integer feature-row indices.
    """
    scores = thrush_scores.dropna().astype(float)
    if scores.empty:
        return []

    pct = float(np.clip(pct, 0.0, 0.5))
    k = max(1, int(np.floor(pct * len(scores))))

    top_idx = scores.nlargest(k).index.tolist()
    bot_idx = scores.nsmallest(k).index.tolist()
    candidates = list(dict.fromkeys(bot_idx + top_idx))  # deduplicated, order-preserving

    # Pad with extra features if below the minimum
    if len(candidates) < min_candidates:
        remaining = scores[~scores.index.isin(candidates)]
        need = min_candidates - len(candidates)
        extra_top = remaining.nlargest((need + 1) // 2).index.tolist()
        extra_bot = remaining.nsmallest(need // 2).index.tolist()
        candidates = list(dict.fromkeys(bot_idx + extra_bot + top_idx + extra_top))

    return [int(i) for i in candidates]


# ═══════════════════════════════════════════════════════════════════════════════
# Regression helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_design_matrix(
    feature_df: pd.DataFrame,
    response: Dict[str, float],
    allowed_rows: Optional[Iterable[int]] = None,
):
    """
    Build centred/scaled design matrix X and centred response y.

    Parameters
    ----------
    feature_df   : DataFrame (n_features × n_models).
    response     : dict mapping model name → score.
    allowed_rows : optional subset of feature-row indices to include.

    Returns
    -------
    X_scaled     : ndarray (n_models × n_kept_features), standardised.
    y_centred    : ndarray (n_models,), zero-mean.
    kept_indices : ndarray of original feature-row positions that survived
                   NaN / zero-variance filtering.
    """
    shared_models = sorted(set(feature_df.columns) & set(response.keys()))
    if len(shared_models) < 2:
        raise ValueError("Need ≥ 2 shared models between features and response.")

    if allowed_rows is not None:
        row_idx = np.array(sorted(set(int(i) for i in allowed_rows)), dtype=int)
        sub = feature_df.reset_index(drop=True).iloc[row_idx]
    else:
        row_idx = np.arange(len(feature_df), dtype=int)
        sub = feature_df.reset_index(drop=True)

    values = sub[shared_models].to_numpy(dtype=np.float64)
    values = np.where(np.isfinite(values), values, np.nan)

    finite_mask = ~np.isnan(values).any(axis=1)
    values = values[finite_mask]
    row_idx = row_idx[finite_mask]

    variance = np.nanvar(values, axis=1)
    var_mask = variance > 0
    values = values[var_mask]
    row_idx = row_idx[var_mask]

    X = values.T
    y = pd.Series(response, dtype=float)[shared_models].to_numpy(np.float64)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X).astype(np.float64)
    y_centred = (y - y.mean()).astype(np.float64)

    return X_scaled, y_centred, row_idx


# ═══════════════════════════════════════════════════════════════════════════════
# Core: Thrush-prefiltered Lasso for a single benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def run_thrush_lasso(
    feature_df: pd.DataFrame,
    response: Dict[str, float],
    benchmark: str,
    thrush_pct: float = DEFAULT_THRUSH_PCT,
    alpha: float = DEFAULT_LASSO_ALPHA,
    text_lookup_df: Optional[pd.DataFrame] = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, Optional[Lasso]]:
    """
    Run the full Thrush → Lasso pipeline for one benchmark.

    Returns
    -------
    signatures : DataFrame with columns [benchmark, row_pos_full, coef, (chunk_text)].
    model      : Fitted Lasso object (or None if nothing selected).
    """
    empty = pd.DataFrame(columns=["benchmark", "row_pos_full", "coef", "chunk_text"])

    # Step 1 – Thrush pre-filtering
    thrush_scores = compute_thrush_rank_correlations(feature_df, response)
    n_models = len(set(feature_df.columns) & set(response.keys()))
    candidates = select_candidates_by_thrush(
        thrush_scores,
        pct=thrush_pct,
        min_candidates=max(2 * n_models, 50),
    )
    if not candidates:
        return empty, None

    # Step 2 – Build design matrix from candidates only
    X, y, kept_indices = build_design_matrix(feature_df, response, allowed_rows=candidates)
    if X.shape[1] == 0:
        return empty, None

    # Step 3 – Fit Lasso (intercept already handled by centring)
    lasso = Lasso(
        alpha=alpha,
        fit_intercept=False,
        max_iter=LASSO_MAX_ITER,
        random_state=LASSO_RANDOM_STATE,
    )
    lasso.fit(X, y)

    nonzero = lasso.coef_ != 0
    if not nonzero.any():
        return empty, None

    # Step 4 – Assemble results
    selected_rows = kept_indices[nonzero].astype(int)
    selected_coefs = lasso.coef_[nonzero]

    result = (
        pd.DataFrame({
            "benchmark": benchmark,
            "row_pos_full": selected_rows,
            "coef": selected_coefs,
        })
        .sort_values("coef", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )

    if text_lookup_df is not None and "chunk_text" in text_lookup_df.columns:
        result["chunk_text"] = (
            text_lookup_df.iloc[result["row_pos_full"]]["chunk_text"]
            .astype(str)
            .values
        )

    if verbose:
        print(
            f"  {benchmark}: candidates={len(candidates)}, "
            f"usable_features={X.shape[1]}, selected={len(result)}, alpha={alpha}"
        )

    return result, lasso


# ═══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_filtered_feature_matrix(parts_dir: Path = FILTERED_PARTS_DIR):
    """Load and concatenate parquet feature-matrix parts + chunk-ID arrays."""
    matrix_files = sorted(parts_dir.glob("part_*_matrix.filtered.parquet"))
    chunk_id_files = sorted(parts_dir.glob("part_*_chunk_ids.filtered.npy"))
    if not matrix_files or not chunk_id_files:
        raise FileNotFoundError(f"No filtered parts found in {parts_dir}")

    dfs = [pq.read_table(f).to_pandas() for f in matrix_files]
    combined = pd.concat(dfs, ignore_index=True)
    models = list(dfs[0].columns)
    matrix = combined.to_numpy(dtype=np.float32)
    chunk_ids = np.concatenate([np.load(f) for f in chunk_id_files])

    return matrix, models, chunk_ids


def load_chunk_text_map(feature_matrix_dir: Path = FEATURE_MATRIX_DIR):
    """Optionally load chunk_id → chunk_text mapping."""
    path = feature_matrix_dir / "chunkid2text.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if {"chunk_id", "chunk_text"}.issubset(df.columns):
        return df.set_index("chunk_id")["chunk_text"].to_dict()
    return None


def read_performance_matrix(csv_path: str) -> pd.DataFrame:
    """Read benchmark performance CSV; index by model name."""
    df = pd.read_csv(csv_path)
    if "model" in df.columns:
        return df.set_index("model")

    first_col = df.columns[0]
    if df[first_col].dtype == object:
        return df.set_index(first_col)

    raise ValueError("Could not identify a model-name column in the performance CSV.")


def make_signature_path(sig_dir: Path, benchmark: str, ratio: float) -> Path:
    """Deterministic output filename for a given benchmark + Thrush ratio."""
    ratio_str = str(ratio).lower().strip()

    if "e-" in ratio_str:
        exponent = "".join(ch for ch in ratio_str.split("e-")[1] if ch.isdigit())
        tag = f"e{int(exponent):02d}"
    elif "." in ratio_str:
        tag = ratio_str.replace(".", "").rstrip("0")[:12]
    else:
        tag = ratio_str

    return sig_dir / f"{benchmark}_signatures_ratio_{tag}.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run thrush-prefiltered Lasso signatures for benchmark analysis.",
    )
    parser.add_argument(
        "--perf-csv", type=str, default=str(DEFAULT_PERF_CSV),
        help="Path to the benchmark performance matrix CSV.",
    )
    parser.add_argument(
        "--task-prefixes", type=str, default=_REG.get("task_prefixes", "mmlu_"),
        help="Comma-separated column prefixes to select benchmarks (e.g. 'mmlu_,bbh_').",
    )
    parser.add_argument(
        "--include", type=str, default=_REG.get("include", ""),
        help="Comma-separated exact column names to include (overrides prefix filter).",
    )
    parser.add_argument(
        "--exclude", type=str, default=_REG.get("exclude", ""),
        help="Comma-separated exact column names to exclude.",
    )
    parser.add_argument(
        "--only-remaining", action="store_true",
        default=_REG.get("only_remaining", False),
        help="Skip benchmarks that already have a signature CSV.",
    )
    parser.add_argument(
        "--list-benchmarks", action="store_true",
        help="Print the resolved benchmark list and exit.",
    )
    parser.add_argument(
        "--sig-dir", type=str, default=str(DEFAULT_SIG_DIR),
        help="Output directory for signature CSVs.",
    )
    parser.add_argument(
        "--model-dir", type=str, default=str(DEFAULT_MODEL_DIR),
        help="Output directory for fitted Lasso .joblib files.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        default=_REG.get("verbose", False),
        help="Print detailed per-benchmark diagnostics.",
    )
    parser.add_argument(
        "--N", type=int, default=_REG.get("n_shards", 1),
        help="Total number of shards (for parallel runs).",
    )
    parser.add_argument(
        "--part", type=int, default=_REG.get("shard_part", 0),
        help="This shard's index (0 … N-1).",
    )
    parser.add_argument(
        "--preselect-ratio", type=float, default=DEFAULT_THRUSH_PCT,
        help="Fraction of features to pre-select from each Thrush tail.",
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_LASSO_ALPHA,
        help="Fixed regularisation strength for Lasso.",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_benchmarks(perf_df: pd.DataFrame, args) -> List[str]:
    """Turn CLI flags into a sorted list of benchmark column names."""
    prefixes = [p.strip() for p in args.task_prefixes.split(",") if p.strip()]
    if prefixes:
        by_prefix = [
            c for c in perf_df.columns
            if isinstance(c, str) and any(c.startswith(p) for p in prefixes)
        ]
    else:
        by_prefix = list(perf_df.columns)

    include = [s.strip() for s in args.include.split(",") if s.strip()]
    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}

    if include:
        targets = [b for b in include if b in perf_df.columns]
    else:
        targets = [b for b in by_prefix if b not in exclude]

    # Drop the aggregate 'bbh' column when running individual bbh_ tasks
    if any(p.startswith("bbh_") for p in prefixes):
        targets = [b for b in targets if b != "bbh"]

    return sorted(targets)


def main():
    args = parse_args()

    sig_dir = Path(args.sig_dir)
    model_dir = Path(args.model_dir)
    sig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    thrush_pct = args.preselect_ratio
    alpha = args.alpha

    print(f"Preselect ratio : {thrush_pct}")
    print(f"Lasso alpha     : {alpha}")
    print(f"Signature dir   : {sig_dir}")
    print(f"Model dir       : {model_dir}")

    # ── Load data ────────────────────────────────────────────────────────────
    matrix, models, chunk_ids = load_filtered_feature_matrix()
    feature_df = pd.DataFrame(matrix, columns=models)

    chunkid2text = load_chunk_text_map()
    perf_df = read_performance_matrix(args.perf_csv)

    # ── Resolve & shard benchmark list ───────────────────────────────────────
    targets = resolve_benchmarks(perf_df, args)

    if args.only_remaining:
        targets = [
            b for b in targets
            if not make_signature_path(sig_dir, b, thrush_pct).exists()
        ]

    n_shards = max(1, int(args.N))
    shard = int(args.part) % n_shards
    if n_shards > 1:
        targets = [b for i, b in enumerate(targets) if i % n_shards == shard]

    if args.list_benchmarks:
        print(",".join(targets))
        return

    if not targets:
        print("Nothing to run (empty target list).")
        return

    # ── Run per-benchmark ────────────────────────────────────────────────────
    for benchmark in targets:
        out_path = make_signature_path(sig_dir, benchmark, thrush_pct)
        if out_path.exists():
            print(f"  {benchmark}: already exists → {out_path}  (skipped)")
            continue

        resp_series = perf_df[benchmark].dropna()
        available = sorted(m for m in feature_df.columns if m in resp_series.index)
        print(f"\n[{benchmark}] overlapping models: {len(available)}")

        missing = sorted(set(resp_series.index) - set(feature_df.columns))
        if missing:
            print(f"  models in perf but not in features ({len(missing)}): {', '.join(missing)}")

        if len(available) < 2:
            print(f"  {benchmark}: skipped (need ≥ 2 overlapping models)")
            continue

        response = resp_series[available].astype(float).to_dict()

        sig_df, fitted_model = run_thrush_lasso(
            feature_df=feature_df,
            response=response,
            benchmark=benchmark,
            thrush_pct=thrush_pct,
            alpha=alpha,
            verbose=args.verbose,
        )

        if sig_df.empty:
            print(f"  {benchmark}: no features selected.")
            continue

        sig_df["chunk_id"] = sig_df["row_pos_full"].map(lambda i: int(chunk_ids[int(i)]))
        sig_df = sig_df.drop(columns=["row_pos_full"])

        if chunkid2text is not None:
            sig_df["chunk_text"] = sig_df["chunk_id"].map(
                lambda cid: chunkid2text.get(int(cid))
            )

        output_cols = ["benchmark", "chunk_id", "coef"]
        if "chunk_text" in sig_df.columns:
            output_cols.append("chunk_text")

        sig_df[output_cols].to_csv(out_path, index=False)
        print(f"  → saved signatures: {out_path}")

        if fitted_model is not None:
            model_path = model_dir / f"{benchmark}_model.joblib"
            dump(fitted_model, model_path)
            print(f"  → saved model:      {model_path}")


if __name__ == "__main__":
    main()
