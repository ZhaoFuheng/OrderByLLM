# OrderByLLM

Tools for comparing LLM-based ranking algorithms and budget-aware optimizer policies on:

- Dev datasets: `nba`, `dl19`
- Test datasets: `population`, `dl20`, `sembench_movie`, `hellaswag`, `nfcorpus`

The experiments use three models, referred to everywhere by a provider-agnostic short name:

| Model | Short name | Provider used |
| --- | --- | --- |
| Llama-3.1 70B | `llama3.1-70b` | `fireworks` |
| Claude Haiku 4.5 | `claude-haiku-4-5` | `anthropic` |
| GPT-5 mini | `openai-gpt-5-mini` | `openai` |

## Setup

Clone with [Git LFS](https://git-lfs.com) installed — the SembenchMovie reviews under `data/movie/` and the response-cache export `sort_cache_export.jsonl.gz` are stored in LFS (run `git lfs pull` if you cloned without it). Then create the conda environment:

```bash
conda env create -f environment.yml
conda activate llm_order_by
```

## Reproducing the Results from the Response Cache

The LLM responses behind the published results are cached, so the experiments can be re-run without any API key and without making a single API call.

1. Import the two cache exports that ship with the repository: `sort_cache_export.jsonl.gz` (the LLM responses; 1.7 GB, about 8.1 million entries) and `wiki_cache_export.jsonl.gz` (the Wikipedia lookups used by the `*_with_search` algorithms):

```bash
python cache_tools.py import sort_cache_export.jsonl.gz
python cache_tools.py import wiki_cache_export.jsonl.gz --cache wiki
```

   The first import takes roughly 40 minutes and the resulting `sort_cache/` uses about 10 GB of disk.

2. Re-run everything from the cache (about an hour):

```bash
python run_all.py --cache-only
```

`--cache-only` runs every experiment with `--provider cache`, a client that answers only from the cache and refuses to call any API. Each run rewrites its `results_<model>.json` / `optimizer_<model>.json` and the figures are redrawn under `figures/`. A run that needs a response missing from the cache stops without writing anything and is listed at the end.

What to expect:

- The regenerated files are identical to the committed ones apart from their `generated_at` timestamps; the committed files were themselves produced this way. (Token counts, and therefore the reported prices, of a cached response are those of the response that was recorded; scores do not depend on them.)
- The first DL19 / DL20 run downloads the MS MARCO passage collection through `ir_datasets`.

A single experiment can be replayed the same way, for example:

```bash
python test/run_experiment.py --dataset dl20 --models claude-haiku-4-5 --provider cache
```

## Running with an LLM Provider

To run experiments that are not in the cache, set API credentials in a local `.env` file. Only the providers you run need credentials:

```bash
# --provider openai
OPENAI_API_KEY=...
# --provider anthropic
ANTHROPIC_API_KEY=...
# --provider fireworks  (FIREWORKS_LLAMA_70B_MODEL selects your own Llama-3.1-70B deployment)
FIREWORKS_API_KEY=...
FIREWORKS_LLAMA_70B_MODEL=...
# --provider hf
HF_TOKEN=...
# --provider cortex (the default): Snowflake Cortex REST API
OPENAI_BASE_URL=https://<account>.snowflakecomputing.com/api/v2/cortex/v1
OPENAI_API_KEY=<programmatic access token>
```

`.env` is gitignored and is only used to populate environment variables when they are not already set in your shell.

For Cortex you can instead authenticate with a session token: set `SNOWFLAKE_CONNECTION` to a profile in `~/.snowflake/connections.toml` (requires `pip install snowflake-connector-python`). Each provider's remaining options (concurrency limits, base URLs) are documented in `order_by/clients.py`.

Run the full experiment + optimizer + plotting pipeline:

```bash
python run_all.py
```

Optional skips:

```bash
python run_all.py --skip plot
python run_all.py --skip dev
python run_all.py --skip optimizer
```

Responses already in the cache are reused; everything else calls the LLM APIs, which is expensive — the optimizer budgets alone are listed in `OPTIMIZER_RUNS` in `run_all.py`.

## LLM Response Cache

All LLM API responses are cached in `sort_cache/` (a [diskcache](https://grantjenks.com/docs/diskcache/) SQLite database). A response is keyed by the model's short name and the exact prompt, so it is reused across providers, algorithms and scripts.

`cache_tools.py import` (above) fills the cache from an export. To keep existing local entries and only add missing ones, pass `--no-overwrite`. To use cache directories that live elsewhere (for example ones shared between checkouts), set `SORT_CACHE_DIR` and `WIKI_CACHE_DIR` in your shell.

To create a shareable snapshot of your current cache:

```bash
python cache_tools.py export                # -> sort_cache_export.jsonl.gz
python cache_tools.py export --cache wiki   # -> wiki_cache_export.jsonl.gz
```

To check that a model's DL20 runs are fully cached without running the whole pipeline:

```bash
python latency/check_cache_coverage.py --model claude-haiku-4-5 --max-queries 54
```

## Main Scripts

- `dev/run_experiment.py`: run dev-set experiments
- `dev/plot_experiment.py`: plot dev-set results
- `test/run_experiment.py`: run test-set experiments
- `test/run_optimizer.py`: run budget-aware optimizer experiments
- `test/plot_experiment.py`: plot test-set results
- `run_all.py`: orchestrate the default workflow
- `cache_tools.py`: export / import the response cache and the wiki cache
- `make_figures.sh`: redraw all figures from the result files (`--ci` adds per-query confidence intervals)
- `make_results_table.py`: print the main results table (LaTeX)

Each runner takes `--models` and `--provider` (`cache`, or a real provider), for example:

```bash
python test/run_experiment.py --dataset dl20 --models claude-haiku-4-5 --provider anthropic
python test/run_optimizer.py --dataset dl20 --models claude-haiku-4-5 --provider anthropic \
    --budgets 14,28,42,56 --proxy-policies rrf_ensemble,llm_judge
```

Optimizer policies: `rrf_ensemble` (self-consistency: fuse the best affordable algorithms with reciprocal rank fusion) and `llm_judge` (an LLM picks among the candidate rankings); `borda`, `rrf`, `borda_ensemble` and `ideal` are also available.

## Additional Studies

- `test/dl20_ablation/`: judge-model ablation (`run_ablation.py`) and optimizer sample-size ablation (`run_samplesize_ablation.py`) on DL20, with their analysis and plot scripts
- `inquiry_prompt_ablation/`: accuracy of the inquiry prompt at separating parametric from situation-dependent questions (NQ-open + SituatedQA)
- `latency/`: per-request latency calibration and the estimated DL20 wall-clock latency of each algorithm and of the optimizer (Haiku-4.5 and GPT-5 mini)
- `verify_num_calls_and_cost/`: validation of the optimizer's cost model — `optimizer_call_count_error.py` compares the predicted number of LLM calls with the calls a full run issues, and `optimizer_cost_estimation_error.py` does the same for the estimated monetary cost

Each script's docstring describes its usage. The ablation, inquiry-prompt and cost-verification result files also replay from the cache: those scripts take `--provider cache` (the judge ablation takes it as the ranker's and each judge's provider). The latency study is different: `calibrate_latency.py` measures live request latency and needs a real provider, and the other latency scripts replay from the cache but report simulated wall-clock times, so their JSON files are measurements rather than cache replays.

## Data

- `nba`, `population`, `sembench_movie`, `hellaswag` and `nfcorpus` data files are under `data/` (`data/movie/` via Git LFS).
- NFCorpus runs use three test queries by default (`PLAIN-1018`, `PLAIN-102`, `PLAIN-1050`), each ranking the full 3,633-document corpus; `--nfcorpus-queries` selects other ids and `--nfcorpus-queries ''` with `--nfcorpus-limit N` takes the first N queries instead.
- `dl19` / `dl20` passages are downloaded on first use by `ir_datasets`; the BM25 first-stage runs are under `data/`.
- `data/hellaswag/prepare.py` regenerates the HellaSwag files (requires `pip install datasets`).

## Outputs

Results are written under:

- `dev/<dataset>/`
- `test/<dataset>/`

Typical files include:

- `results_<model>.json`
- `optimizer_<model>.json`

Figures are written to `figures/` (or `figures_with_confidence_interval/`).
