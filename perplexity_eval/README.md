# Perplexity Evaluation with vLLM

A high-performance perplexity evaluation framework using vLLM for efficient LLM inference with data and tensor parallelism support.

## Installation

### Option 1: Using `uv` (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Option 2: Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Inference

All settings (model, paths, parallelism, vLLM engine parameters) are configured in **`config/config.yaml`** at the repository root. Edit that file before running.

### Prerequisites

- One or more GPUs with enough VRAM for the model
- Dataset in Parquet format with columns `text_id`, `chunk_id`, `chunk_text`
- Model available locally or via HuggingFace

### Run from the repository root

```bash
python perplexity_eval/wiki_eval.py
```

All defaults come from `config/config.yaml`. Override any setting at the command line:

```bash
python perplexity_eval/wiki_eval.py \
    --model Qwen/Qwen3-8B \
    --dp-size 2 \
    --tp-size 1 \
    --dataset data/chunked_texts_df.parquet \
    --output-file output/perplexity/Qwen3-8B.parquet
```

### Key parameters

| Argument | Config key | Description |
|----------|-----------|-------------|
| `--model` | `perplexity.model` | HuggingFace model name or local path |
| `--dp-size` | `perplexity.dp_size` | Data-parallel workers (one per GPU) |
| `--tp-size` | `perplexity.tp_size` | Tensor-parallel GPUs per worker |
| `--dataset` | `paths.chunked_dataset` | Path to the chunked Parquet dataset |
| `--prompt-column` | `perplexity.prompt_column` | Column name containing text prompts |
| `--output-file` | `paths.perplexity_output_dir` | Output Parquet path (auto-named if omitted) |
| `--debug-limit` | `perplexity.debug_limit` | Process only N prompts for testing (0 = full) |

## Configuration

All vLLM engine settings live under `perplexity.vllm` in `config/config.yaml`:

```yaml
perplexity:
  model: "Qwen/Qwen3-8B"
  dp_size: 2
  tp_size: 1
  vllm:
    enforce_eager: true           # deterministic execution for evaluation
    trust_remote_code: false
    max_num_seqs: 512             # max concurrent sequences (reduce if OOM)
    gpu_memory_utilization: 0.8   # fraction of GPU memory to use
    dtype: "bfloat16"
```

See the [vLLM documentation](https://docs.vllm.ai/en/v0.7.2/api/offline_inference/llm.html) for the full list of supported engine settings.

## Parallelism

This project supports two types of parallelism for efficient large-scale inference:

### Data Parallelism (DP)
- Distributes different data samples across multiple GPUs
- Use when the model fits on a single GPU but multiple GPUs are available
- Scales linearly with the number of GPUs
- Set `perplexity.dp_size` in config (or `--dp-size` on the command line)

### Tensor Parallelism (TP)
- Splits the model weights across multiple GPUs
- Use when the model is too large to fit on a single GPU
- Set `perplexity.tp_size` in config (or `--tp-size` on the command line)

### Hybrid

Both can be combined: use TP to fit a large model across GPUs, and DP to process multiple batches simultaneously.

**Resources:**
- [vLLM Data Parallel Documentation](https://docs.vllm.ai/en/latest/examples/offline_inference/data_parallel.html)
- [vLLM Tensor Parallel Guide](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#running-vllm-on-multiple-nodes)

## Output Format

Each output Parquet file contains one row per text chunk:

| Column | Type | Description |
|--------|------|-------------|
| `text_id` | int32 | Source document identifier |
| `chunk_id` | int32 | Chunk identifier within the document |
| `logprobs` | list[float32] | Per-token log-probabilities |

## Debugging

Process only a small subset of data to verify the setup:

```bash
python perplexity_eval/wiki_eval.py --debug-limit 100
```

Inspect any output Parquet file:

```bash
python perplexity_eval/script/read_result_parquet.py
```
