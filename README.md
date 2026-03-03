# Mapping Overlaps in Benchmarks through Perplexity in the Wild

> **Authors:** Siyang Wu†, Honglin Bao†, Sida Li†, Ari Holtzman, James A. Evans  
> †Co-leads on the project with equal contribution  
> **Institution:** University of Chicago  Data Science Institute
> **Published in:** ICLR 2026  
> **ArXiv:** [arXiv:2509.23488](https://arxiv.org/abs/2509.23488)  
> **PDF:** [View PDF](https://arxiv.org/pdf/2509.23488)  
> **License:** [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)

## Abstract

We introduce benchmark signatures to characterize the capacity demands of LLM
benchmarks and their overlaps. Signatures are sets of salient tokens from in-thewild corpora whose model token perplexity, reflecting training exposure, predicts
benchmark performance. We extract them via stepwise forward selection with linear regression in a meta-evaluation spanning 32 LLMs and 89 benchmarks across
diverse domains. We then analyze how these signatures relate to both the semantic
similarity of benchmark questions and the correlation structure of model performance. While performance correlations are uniformly high and semantic overlaps
stay in a narrow mid-range, benchmark signatures reveal more nuanced structure.
For instance, they uncover substantial overlap between benchmarks in knowledge
and reasoning tasks, whereas benchmarks in culture- and humanity-oriented domains show low similarity with each other. Unlike raw performance correlations,
which are influenced by benchmark-orthogonal factors such as question formats,
signatures are robust to such confounds. We further identify cross-functional
overlaps between logic, math, language, instruction following, and cultural/world
modeling, with coding emerging as the most isolated function, interacting only
moderately with the ability of detecting missing information. Qualitative analysis
shows that only the knowledge signature aligns with actual knowledge, suggesting that LLM semantic organization may differ from human conceptual structure.
Together, these findings offer insights into benchmark validity, LLM sensitivities,
and the landscape of interconnected LLM capacities. 

## Overview

This paper introduces **benchmark signatures** - a novel framework for understanding what LLM benchmarks actually measure and how they relate to each other. The approach works at characterizing benchmarks by token-level perplexity patterns on large-scale, in-the-wild corpora.

<!-- ### Key Findings

- Performance-level overlaps between benchmarks are universally high, while semantic overlaps stay in a narrow mid-range.
- Benchmark signatures are highly informative in capturing variation, overlap, and divergence between benchmarks.
- Knowledge and reasoning subtasks show significant overlap, whereas multilingual and cultural benchmarks exhibit less similarity — even less than cross-task overlap.
- Performance-level results are strongly influenced by benchmark-orthogonal factors (e.g., question format), but signatures remain robust to such confounds.
- Cross-functional overlaps exist among logic, math, language, instruction following, and world modeling.
- **Coding emerges as the least overlapping domain.** -->

## Getting Started

### Installation

```bash
git clone https://github.com/siyangwu1/Benchmark-Signature-Repository.git
cd Benchmark-Signature-Repository
pip install -r requirements.txt
```

### Data

The benchmark signatures are extracted from the [RedPajama](https://github.com/togethercomputer/RedPajama-Data) dataset, which contains large-scale textual data across multiple domains:

- CommonCrawl
- C4
- GitHub
- arXiv
- Books
- Wikipedia
- StackExchange

## Methodology of Signature Extraction

1. **Perplexity Extraction:** Compute token-level perplexity for 32 LLMs on in-the-wild corpora using vLLM.
2. **Benchmark Evaluation:** Evaluate all 32 models across 89 benchmarks using lm-evaluation-harness under consistent conditions.
3. **Signature Extraction:** Apply stepwise forward selection with linear regression to identify salient tokens whose perplexity is predictive of benchmark performance.


All three steps are configured through **`config/config.yaml`** at the repository root. Edit that file once before running anything. All commands below are run from the **repository root**.

### 1. **Perplexity Extraction:**

Token-level log-probabilities are computed for each LLM over chunked in-the-wild corpora using [vLLM](https://github.com/vllm-project/vllm).

**Configure `config/config.yaml`**

```yaml
paths:
  chunked_dataset: "data/chunked_texts_df.parquet"   # pre-processed chunked input
  perplexity_output_dir: "output/perplexity"          # one Parquet file per model

environment:
  hf_home: "/path/to/cache/hf_home"         # HuggingFace model cache
  vllm_cache_root: "/path/to/cache/vllm"    # vLLM compilation cache
  hf_token: ""                               # HF access token (or set HF_TOKEN env var)

perplexity:
  model: "Qwen/Qwen3-8B"
  dp_size: 2        # data-parallel workers — one process per GPU
  tp_size: 1        # tensor-parallel GPUs per worker (increase for large models)
  debug_limit: 0    # set > 0 to process only N chunks for testing
  vllm:
    gpu_memory_utilization: 0.8
    max_num_seqs: 512
    dtype: "bfloat16"
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 1a — Download the corpus**

```bash
python perplexity_eval/script/download_redpajama_data.py
```

Downloads all RedPajama-1T-Sample Parquet shards from HuggingFace into `paths.redpajama_download_dir`. After downloading, pre-process the data into a chunked Parquet file (columns: `text_id`, `chunk_id`, `chunk_text`) at the path set in `paths.chunked_dataset`.

**Step 1b — Run perplexity extraction**

All settings are read from `config/config.yaml`. Cache and token environment variables are applied automatically from the `environment` section.

```bash
python perplexity_eval/wiki_eval.py
```

Override any setting at the command line (CLI values take priority over config):

```bash
python perplexity_eval/wiki_eval.py \
    --model Qwen/Qwen3-8B \
    --dp-size 2 \
    --tp-size 1 \
    --dataset data/chunked_texts_df.parquet \
    --output-file output/perplexity/Qwen3-8B.parquet
```

Test on a small subset before a full run:

```bash
python perplexity_eval/wiki_eval.py --debug-limit 100
```

Run this step **once per LLM**. The output file is named automatically as `<perplexity_output_dir>/<model_name>.parquet` when `--output-file` is not specified.

**Output format** — one row per text chunk:

| Column | Type | Description |
|--------|------|-------------|
| `text_id` | int32 | Source document identifier |
| `chunk_id` | int32 | Chunk identifier within the document |
| `logprobs` | list[float32] | Per-token log-probabilities |

Inspect any output file:

```bash
python perplexity_eval/script/read_result_parquet.py
```

### 2. **Benchmark Evaluation:**

Benchmark scores are collected using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) under consistent conditions across all models.

**Configure `config/config.yaml`**

```yaml
paths:
  benchmark_output_dir: "output/benchmark_eval"

benchmark:
  model: "meta-llama/Llama-3.2-1B-Instruct"
  tasks: ["asdiv", "mmlu", "bbh", "mbpp", "ifeval"]
  tensor_parallel_size: 2
  data_parallel_size: 1
  gpu_memory_utilization: 0.7
  tokenizer_mode: "auto"
  batch_size: "auto"
```

**Install lm-evaluation-harness**

```bash
pip install lm_eval
```

**Set required environment variables**

```bash
export HF_TOKEN="<your_huggingface_token>"
export HF_ALLOW_CODE_EVAL=1          # required for code benchmarks (mbpp, etc.)
export HF_HOME="<path/to/model/cache>"
```

**Run a single benchmark task**

```bash
lm_eval --model vllm \
    --model_args "pretrained=meta-llama/Llama-3.2-1B-Instruct,\
tensor_parallel_size=2,data_parallel_size=1,\
gpu_memory_utilization=0.7,tokenizer_mode=auto" \
    --tasks mmlu \
    --batch_size auto \
    --confirm_run_unsafe_code \
    --verbosity INFO \
    --output_path output/benchmark_eval/mmlu
```

Key `--model_args` fields:

| Field | Description |
|-------|-------------|
| `pretrained` | HuggingFace model name or local path |
| `tensor_parallel_size` | GPUs to split the model across |
| `data_parallel_size` | Parallel evaluation instances |
| `gpu_memory_utilization` | Fraction of GPU memory to use |

For OOM-prone tasks (long-context benchmarks), reduce batch size:

```bash
lm_eval --model vllm \
    --model_args "pretrained=meta-llama/Llama-3.2-1B-Instruct,tensor_parallel_size=2,gpu_memory_utilization=0.7" \
    --tasks bbh \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path output/benchmark_eval/bbh
```

**Output:** JSON result files under `output/benchmark_eval/<task>/`. After running all models and tasks, aggregate the scores into a performance matrix CSV (rows = models, columns = benchmarks) at `output/benchmark_performance/performance_matrix.csv` for use in Step 3.

### 3. **Signature Extraction:**

Signatures are extracted via a two-stage pipeline: Thrush rank-correlation pre-filtering followed by Lasso regression.

**Configure `config/config.yaml`**

```yaml
paths:
  filtered_parts_dir: "output/feature_matrix/parts_filtered"  # logprob feature matrix parts
  performance_matrix_dir: "output/benchmark_performance"
  signature_dir: "output/benchmark_signature"
  model_dir: "output/benchmark_models"

regression:
  performance_matrix_file: "performance_matrix.csv"
  task_prefixes: "mmlu_"     # comma-separated benchmark column prefixes
  include: ""                 # exact benchmark names to force-include
  exclude: ""                 # exact benchmark names to exclude
  preselect_ratio: 0.000001   # Thrush tail fraction for pre-filtering
  alpha: 0.01                 # Lasso regularisation strength
  only_remaining: false
  n_shards: 1
  shard_part: 0
```

**Expected input layout** (built from Step 1 outputs):

```
output/
├── feature_matrix/
│   ├── parts_filtered/
│   │   ├── part_*_matrix.filtered.parquet   # token × model logprob matrix
│   │   └── part_*_chunk_ids.filtered.npy    # corresponding chunk IDs
│   └── chunkid2text.parquet                 # optional: chunk_id → text
└── benchmark_performance/
    └── performance_matrix.csv               # model × benchmark score matrix
```

**Run signature extraction** (all defaults from config):

```bash
python regression/regression_run.py
```

Override any setting at the command line:

```bash
python regression/regression_run.py \
    --perf-csv output/benchmark_performance/performance_matrix.csv \
    --task-prefixes mmlu_ \
    --preselect-ratio 0.000001 \
    --alpha 0.01 \
    --sig-dir output/benchmark_signature/mmlu \
    --model-dir output/benchmark_models/mmlu
```

List all resolved benchmark targets without running:

```bash
python regression/regression_run.py --list-benchmarks
```

Resume an interrupted run (skip already-completed benchmarks):

```bash
python regression/regression_run.py --only-remaining
```

Run multiple benchmarks in parallel by sharding the target list:

```bash
# Launch each in a separate terminal (or as separate jobs)
python regression/regression_run.py --N 4 --part 0
python regression/regression_run.py --N 4 --part 1
python regression/regression_run.py --N 4 --part 2
python regression/regression_run.py --N 4 --part 3
```

**Algorithm:**

1. **Thrush pre-filtering** — For each benchmark, compute a rank-correlation score between every token's per-model logprob vector and the benchmark's per-model scores. Select the top and bottom `preselect_ratio` fraction of tokens as candidates.
2. **Lasso regression** — Fit a Lasso on the standardised candidate features to predict benchmark scores. Non-zero coefficients identify the signature tokens.

**Output** — `output/benchmark_signature/<benchmark>_signatures_ratio_<r>.csv`:

| Column | Description |
|--------|-------------|
| `benchmark` | Benchmark name |
| `chunk_id` | Identifier of the signature token chunk |
| `coef` | Lasso coefficient (magnitude = importance, sign = direction) |
| `chunk_text` | Text of the token chunk (when lookup table is available) |

Fitted Lasso models are saved as `output/benchmark_models/<benchmark>_model.joblib`.


<!-- ## Benchmarks

The 88 benchmarks span diverse categories:

| Category | Examples |
|----------|----------|
| Knowledge | Business, Humanities, Social Sciences, Science & Engineering, Medicine |
| Coding | Code generation and understanding tasks |
| Logic | Logical reasoning benchmarks |
| Instruction Following | Instruction adherence tasks |
| Math | Mathematical reasoning benchmarks |
| Language | Multilingual and linguistic tasks |
| Reasoning | General reasoning benchmarks |
| World Modeling | Cultural knowledge and world understanding |

## Models

The study evaluates 32 widely-used language models. Please refer to the paper for the complete list of models used. -->

## Citation

```bibtex
@inproceedings{
wu2026mapping,
title={Mapping Overlaps in Benchmarks through Perplexity in the Wild},
author={Siyang Wu and Honglin Bao and Sida Li and Ari Holtzman and James Evans},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=QD0cuAmi9z}
}
```

## Acknowledgements

Please refer to the paper for full acknowledgements and funding information.

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
