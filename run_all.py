#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Models under study.
HAIKU = "claude-haiku-4-5"
LLAMA = "llama3.1-70b"
GPT5MINI = "openai-gpt-5-mini"

# Which provider serves each model (see order_by/clients.py). The response cache
# is keyed on the model name alone, so cached calls are reused across providers.
PROVIDER_BY_MODEL = {
    LLAMA: "fireworks",
    GPT5MINI: "openai",
    HAIKU: "anthropic",
}

# Order models are processed in (llama first). Applied to the optimizer phase and
# the experiment model lists.
MODEL_ORDER = [LLAMA, HAIKU, GPT5MINI]
_MODEL_PRIORITY = {m: i for i, m in enumerate(MODEL_ORDER)}

OPTIMIZER_SAMPLE_SIZES = (16,18,20)
VARY_SAMPLE_DATASETS = {"dl20"}

DEV_DATASETS = ("nba", "dl19")
TEST_DATASETS = ("population", "dl20", "sembench_movie", "hellaswag", "nfcorpus")

_PROXY = "rrf_ensemble,llm_judge"

# Total dollar budgets swept per (dataset, model); each budget is split evenly
# across the dataset's queries.
OPTIMIZER_RUNS = (
    # ── HellaSwag ──
    {"dataset": "hellaswag",      "model": HAIKU,    "budgets": "110,330,550,770,990", "proxy_policies": _PROXY},
    {"dataset": "hellaswag",      "model": LLAMA,    "budgets": "4,12,20,28,36",       "proxy_policies": _PROXY},
    {"dataset": "hellaswag",      "model": GPT5MINI, "budgets": "7,23,39,55,71",       "proxy_policies": _PROXY},

    # ── NFCorpus ──
    {"dataset": "nfcorpus",       "model": HAIKU,    "budgets": "10,40,70,100,160",    "proxy_policies": _PROXY},
    {"dataset": "nfcorpus",       "model": LLAMA,    "budgets": "5,10,18,26",          "proxy_policies": _PROXY},
    {"dataset": "nfcorpus",       "model": GPT5MINI, "budgets": "10,16,22,28,34",      "proxy_policies": _PROXY},

    # ── DL20 ──
    {"dataset": "dl20",           "model": HAIKU,    "budgets": "14,28,42,56",         "proxy_policies": _PROXY},
    {"dataset": "dl20",           "model": LLAMA,    "budgets": "1,2,4,6",             "proxy_policies": _PROXY},
    {"dataset": "dl20",           "model": GPT5MINI, "budgets": "3,6,9,12",            "proxy_policies": _PROXY},

    # ── SembenchMovie ──
    {"dataset": "sembench_movie", "model": HAIKU,    "budgets": "2,6,12,18",           "proxy_policies": _PROXY},
    {"dataset": "sembench_movie", "model": LLAMA,    "budgets": ".1,.25,.5,.75",       "proxy_policies": _PROXY},
    {"dataset": "sembench_movie", "model": GPT5MINI, "budgets": ".5,1,1.5,2",          "proxy_policies": _PROXY},

    # ── Population ──
    {"dataset": "population",     "model": HAIKU,    "budgets": "1",                   "proxy_policies": _PROXY},
    {"dataset": "population",     "model": LLAMA,    "budgets": "0.1",                 "proxy_policies": _PROXY},
    {"dataset": "population",     "model": GPT5MINI, "budgets": ".5",                  "proxy_policies": _PROXY},
)


def _print_header(title: str) -> None:
    print(f"\n\033[1;36m== {title} ==\033[0m", flush=True)


# With --cache-only every run uses the `cache` provider (no API keys, no API
# calls); a run whose responses are not all cached fails without writing
# anything, is recorded here, and the pipeline moves on.
CACHE_ONLY = False
NOT_REPRODUCED: list[str] = []


def _provider(model: str) -> str:
    return "cache" if CACHE_ONLY else PROVIDER_BY_MODEL.get(model, "cortex")


def _run(cmd: list[str], env: dict | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if result.returncode != 0:
        if not CACHE_ONLY:
            raise subprocess.CalledProcessError(result.returncode, cmd)
        NOT_REPRODUCED.append(" ".join(cmd[1:]))


def _run_dev_experiments() -> None:
    _print_header("Dev Experiments")
    for model in MODEL_ORDER:          # llama first, then the next model, ...
        provider = _provider(model)
        for dataset in DEV_DATASETS:
            _run(
                [
                    sys.executable,
                    "dev/run_experiment.py",
                    "--dataset",
                    dataset,
                    "--models",
                    model,
                    "--provider",
                    provider,
                ]
            )


def _run_test_experiments() -> None:
    _print_header("Test Experiments")
    for model in MODEL_ORDER:          # llama first, then the next model, ...
        provider = _provider(model)
        for dataset in TEST_DATASETS:
            _run(
                [
                    sys.executable,
                    "test/run_experiment.py",
                    "--dataset",
                    dataset,
                    "--models",
                    model,
                    "--provider",
                    provider,
                ]
            )


def _run_test_optimizers(run_vary_samples: bool = False, only_models: set[str] | None = None) -> None:
    _print_header("Test Optimizers")
    # Process models in MODEL_ORDER (llama first); stable sort keeps dataset order.
    ordered_runs = sorted(OPTIMIZER_RUNS, key=lambda s: _MODEL_PRIORITY.get(s["model"], 99))
    for spec in ordered_runs:
        if only_models and spec["model"] not in only_models:
            continue
        safe_model = spec["model"].replace("/", "-")
        base_cmd = [
            sys.executable,
            "test/run_optimizer.py",
            "--dataset",
            spec["dataset"],
            "--models",
            spec["model"],
            "--provider",
            _provider(spec["model"]),
            "--budgets",
            spec["budgets"],
            "--proxy-policies",
            spec["proxy_policies"],
        ]

        if spec["dataset"] in VARY_SAMPLE_DATASETS and run_vary_samples:
            for sample_size in OPTIMIZER_SAMPLE_SIZES:
                _run(base_cmd + [
                    "--sample-size",
                    str(sample_size),
                    "--output",
                    f"test/vary_samples/optimizer_{spec['dataset']}_{safe_model}_sample{sample_size}.json",
                ])
        else:
            _run(base_cmd)

        print("\n\n", flush=True)


def _run_plots(scope: str) -> None:
    """Draw figures via make_figures.sh (handles retired-model filtering and the
    figures/{devFigures,testFigures} output dirs). scope: dev | test | all."""
    _print_header(f"Figures ({scope})")
    _run(["bash", "make_figures.sh", scope], env={**os.environ, "PYTHON": sys.executable})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all dev/test experiments, optimizers, and plots."
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        choices=["dev", "test", "optimizer", "plot"],
        default=[],
        help="Optional phases to skip.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Reproduce from the LLM response cache: no API keys are needed and no "
             "API call is made. A run whose responses are not all cached is skipped "
             "(its results file is left untouched) and listed at the end.",
    )
    parser.add_argument(
        "--run-vary-samples",
        action="store_true",
        help="Run the vary-sample-size optimizer runs.",
    )
    parser.add_argument(
        "--optimizer-only-models",
        default=None,
        help="Comma-separated model names to restrict the optimizer phase to "
             "(e.g. 'openai-gpt-5-mini,claude-haiku-4-5'). Default: all models.",
    )
    args = parser.parse_args()

    global CACHE_ONLY
    CACHE_ONLY = args.cache_only

    skip = set(args.skip)
    only_models = (
        {m.strip() for m in args.optimizer_only_models.split(",") if m.strip()}
        if args.optimizer_only_models else None
    )

    if "dev" not in skip:
        _run_dev_experiments()

    if "test" not in skip:
        _run_test_experiments()

    if "optimizer" not in skip:
        _run_test_optimizers(run_vary_samples=args.run_vary_samples, only_models=only_models)

    if "plot" not in skip:
        # Figures are drawn from existing result files, so they don't depend on the
        # dev/test experiment phases having run this session. Draw everything unless
        # --skip plot. (Use ./make_figures.sh {dev,test} for a narrower redraw.)
        _run_plots("all")

    _print_header("Done")
    if NOT_REPRODUCED:
        print("Not reproduced from the cache (results files left untouched):")
        for cmd in NOT_REPRODUCED:
            print("  ", cmd)
        sys.exit(1)


if __name__ == "__main__":
    main()
