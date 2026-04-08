# OrderByLLM

Tools for comparing LLM-based ranking algorithms and budget-aware optimizer policies on:

- Dev datasets: `nba`, `dl19`
- Test datasets: `population`, `dl20`, `sembench_movie`

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate llm_order_by
```

Set API credentials in a local `.env` file:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
```

`.env` is gitignored and is only used to populate environment variables when they are not already set in your shell.

## Run Everything

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

## Main Scripts

- `dev/run_experiment.py`: run dev-set experiments
- `dev/plot_experiment.py`: plot dev-set results
- `test/run_experiment.py`: run test-set experiments
- `test/run_optimizer.py`: run budget-aware optimizer experiments
- `test/plot_experiment.py`: plot test-set results
- `run_all.py`: orchestrate the default workflow

## Outputs

Results and figures are written under:

- `dev/<dataset>/`
- `test/<dataset>/`

Typical files include:

- `results_<model>.json`
- `optimizer_<model>.json`
- `<dataset>_<model>_<metric>.png`