"""
Evaluate the logprobs on the chunked-wiki dataset.

The script allows either data-level parallelism (DP) or tensor-level parallelism (TP).

Reference: https://docs.vllm.ai/en/latest/examples/offline_inference/data_parallel.html

Configuration is loaded from config/config.yaml at the repository root.
All CLI arguments override the values in that file.
"""
import os
import sys
import logging
from time import sleep
from multiprocessing import Process, Queue
from pathlib import Path
import pandas as pd
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
from vllm import LLM, SamplingParams
from vllm.sequence import Logprob
import yaml


# ── Config helpers ────────────────────────────────────────────────────────────

def _find_repo_root() -> Path:
    """Walk up from this file until config/config.yaml is found."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "config" / "config.yaml").exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        "Cannot locate config/config.yaml. "
        "Make sure you run from within the repository."
    )


def _load_config() -> tuple[dict, Path]:
    """Return (config_dict, repo_root)."""
    repo_root = _find_repo_root()
    with open(repo_root / "config" / "config.yaml") as f:
        return yaml.safe_load(f), repo_root


def _resolve(path_str: str, repo_root: Path) -> Path:
    """Resolve a config path string against the repo root if relative."""
    p = Path(path_str)
    return p if p.is_absolute() else repo_root / p


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args(cfg: dict, repo_root: Path):
    import argparse

    ppl = cfg.get("perplexity", {})
    paths = cfg.get("paths", {})

    # Default output file: <perplexity_output_dir>/<model_safe_name>.parquet
    # Set to None here; computed from model name in __main__ if not supplied.
    parser = argparse.ArgumentParser(description="Data Parallel Inference")
    parser.add_argument(
        "--model",
        type=str,
        default=ppl.get("model", "Qwen/Qwen3-0.6B"),
        help="Model name or path",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=ppl.get("dp_size", 2),
        help="Data parallel size",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=ppl.get("tp_size", 1),
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(_resolve(paths.get("chunked_dataset", "data/chunked_texts_df.parquet"), repo_root)),
        help="Path to the dataset Parquet file",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default=ppl.get("prompt_column", "chunk_text"),
        help="Column name of the dataset to use as prompts",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help=(
            "Path for the output Parquet file. "
            "Defaults to <paths.perplexity_output_dir>/<model_name>.parquet"
        ),
    )
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=ppl.get("debug_limit", 0),
        help="Limit the number of prompts to process (0 = full dataset)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=ppl.get("verbose", False),
        help="Enable verbose vLLM output",
    )
    return parser.parse_args()


# ── Core logic ────────────────────────────────────────────────────────────────

def process_logprobs(raw_logprobs: list[Optional[dict[int, Logprob]]]) -> list[float]:
    logprobs = []
    # the first one is always None, so we skip it
    for logprob in raw_logprobs[1:]:
        if logprob is not None:
            logprobs.append(next(iter(logprob.values())).logprob)
    return logprobs


def main(
    model: str,
    dp_size: int,
    tp_size: int,
    local_dp_rank: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    vllm_config: dict,
    dataset: pd.DataFrame,
    prompt_column: str,
    results_queue: Queue,
    verbose: bool = False,
):
    # Configure logging to control vLLM verbosity
    if not verbose:
        logging.getLogger("vllm").setLevel(logging.WARNING)
        logging.getLogger("vllm.engine").setLevel(logging.WARNING)
        logging.getLogger("vllm.worker").setLevel(logging.WARNING)
        logging.getLogger("vllm.distributed").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)

    sys.stdout.flush()

    try:
        os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
        os.environ["VLLM_DP_SIZE"] = str(dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
        os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

        # Distribute prompts to each rank
        floor = len(dataset) // dp_size
        remainder = len(dataset) % dp_size

        def start(rank):
            return rank * floor + min(rank, remainder)

        beg_idx, end_idx = start(global_dp_rank), start(global_dp_rank + 1)
        prompt_dicts = dataset.iloc[beg_idx:end_idx].to_dict(orient="records")
        prompts = [prompt_dict[prompt_column] for prompt_dict in prompt_dicts]

        assert len(prompt_dicts) > 0, f"DP rank {global_dp_rank} has no prompts"
        print(f"DP rank {global_dp_rank} processing {len(prompt_dicts)} prompts", flush=True)

        sampling_params = SamplingParams(
            temperature=0.0, prompt_logprobs=0, max_tokens=1
        )

        print(f"DP rank {global_dp_rank}: Initializing LLM...", flush=True)
        llm = LLM(
            model=model,
            tensor_parallel_size=tp_size,
            **vllm_config,
        )

        print(f"DP rank {global_dp_rank}: Generating outputs...", flush=True)
        outputs = llm.generate(prompts, sampling_params)

        print(f"DP rank {global_dp_rank}: Processing {len(outputs)} outputs...", flush=True)
        for i, output in enumerate(outputs):
            text_id, chunk_id = int(prompt_dicts[i]["text_id"]), int(prompt_dicts[i]["chunk_id"])
            logprobs = process_logprobs(output.prompt_logprobs)
            results_queue.put({"text_id": text_id, "chunk_id": chunk_id, "logprobs": logprobs})

        results_queue.put(None)
        print(f"DP rank {global_dp_rank}: Completed processing", flush=True)

        sleep(1)

    except Exception as e:
        print(f"DP rank {global_dp_rank}: ERROR - {str(e)}", flush=True)
        raise


if __name__ == "__main__":
    cfg, repo_root = _load_config()

    # Apply cache / token env vars from config unless already set in the shell.
    # This lets users run `python wiki_eval.py` without sourcing any setup script.
    _env_cfg = cfg.get("environment", {})
    for _var, _key in [
        ("HF_HOME",        "hf_home"),
        ("HF_HUB_CACHE",   "hf_hub_cache"),
        ("HF_XET_CACHE",   "hf_xet_cache"),
        ("VLLM_CACHE_ROOT", "vllm_cache_root"),
        ("UV_CACHE_DIR",   "uv_cache_dir"),
    ]:
        _val = str(_env_cfg.get(_key, "")).strip()
        if _val and _val != "/path/to/cache" and not os.environ.get(_var):
            os.environ[_var] = _val
    _hf_token = str(_env_cfg.get("hf_token", "")).strip()
    if _hf_token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = _hf_token
    if _env_cfg.get("hf_allow_code_eval"):
        os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

    args = parse_args(cfg, repo_root)

    # Resolve output file path
    if args.output_file is None:
        model_safe = args.model.replace("/", "_")
        out_dir = _resolve(cfg["paths"]["perplexity_output_dir"], repo_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_file = str(out_dir / f"{model_safe}.parquet")

    dp_size = args.dp_size
    tp_size = args.tp_size
    model = args.model
    output_filename = args.output_file

    # vLLM settings come from config (perplexity.vllm section)
    vllm_config = cfg["perplexity"]["vllm"]

    dp_master_ip = "127.0.0.1"
    dp_master_port = 8000

    results_queue = Queue()

    wiki_dataset = pd.read_parquet(args.dataset)

    print(f"Total number of chunks/requests: {len(wiki_dataset)}")
    if args.debug_limit > 0:
        wiki_dataset = wiki_dataset.head(args.debug_limit)
        print(f"Debug mode: only processing {args.debug_limit} prompts")

    wiki_prompt_column = args.prompt_column

    print(f"Starting {dp_size} data parallel processes...", flush=True)
    procs = []
    for local_dp_rank, global_dp_rank in enumerate(range(dp_size)):
        proc = Process(
            target=main,
            args=(
                model,
                dp_size,
                tp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                vllm_config,
                wiki_dataset,
                wiki_prompt_column,
                results_queue,
                args.verbose,
            ),
        )
        proc.start()
        procs.append(proc)
        print(f"Started process {proc.pid} for DP rank {global_dp_rank}", flush=True)

    # Collect results and stream to Parquet
    finished_workers = 0
    parquet_writer = None

    schema = pa.schema([
        pa.field('text_id', pa.int32()),
        pa.field('chunk_id', pa.int32()),
        pa.field('logprobs', pa.list_(pa.float32()))
    ])

    results_batch = []
    batch_size = 100

    try:
        parquet_writer = pq.ParquetWriter(output_filename, schema)
        print(f"Opened Parquet writer for {output_filename}", flush=True)

        while finished_workers < len(procs):
            result = results_queue.get()

            if result is None:
                finished_workers += 1
                print(f"Worker {finished_workers}/{len(procs)} completed", flush=True)
            else:
                results_batch.append(result)

            if len(results_batch) >= batch_size or (finished_workers == len(procs) and results_batch):
                table = pa.Table.from_pydict({
                    'text_id': [r['text_id'] for r in results_batch],
                    'chunk_id': [r['chunk_id'] for r in results_batch],
                    'logprobs': [r['logprobs'] for r in results_batch]
                }, schema=schema)
                parquet_writer.write_table(table)
                print(f"Wrote a batch of {len(results_batch)} results to {output_filename}", flush=True)
                results_batch = []
    finally:
        if parquet_writer:
            parquet_writer.close()
        print(f"Finished writing all results to {output_filename}.", flush=True)

    print("Waiting for all processes to complete...", flush=True)
    exit_code = 0
    for i, proc in enumerate(procs):
        proc.join(timeout=300)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} (DP rank {i}) that didn't stop within 5 minutes.", flush=True)
            proc.kill()
            exit_code = 1
        elif proc.exitcode != 0:
            print(f"Process {proc.pid} (DP rank {i}) exited with code {proc.exitcode}", flush=True)
            exit_code = proc.exitcode
        else:
            print(f"Process {proc.pid} (DP rank {i}) completed successfully", flush=True)

    if exit_code == 0:
        print("All processes completed successfully!", flush=True)
    else:
        print(f"Some processes failed with exit code {exit_code}", flush=True)

    exit(exit_code)
