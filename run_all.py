#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DEV_MODELS = "llama3.1-70b,llama3.1-405b,openai-gpt-4.1"
TEST_MODELS = "llama3.1-70b,openai-gpt-4.1"
OPTIMIZER_SAMPLE_SIZES = (16,18,20)
VARY_SAMPLE_DATASETS = {"dl20"}

DEV_DATASETS = ("nba", "dl19")
TEST_DATASETS = ("population", "dl20", "sembench_movie")

OPTIMIZER_RUNS = (
    {
        "dataset": "dl20",
        "model": "openai-gpt-4.1",
        "budgets": "10,20,40,80",
        "proxy_policies": "borda,llm_judge",
    },
    {
        "dataset": "dl20",
        "model": "llama3.1-70b",
        "budgets": "1,3,5,7",
        "proxy_policies": "borda,llm_judge",
    },
    {
        "dataset": "sembench_movie",
        "model": "openai-gpt-4.1",
        "budgets": "3,6,12,24",
        "proxy_policies": "borda,llm_judge",
    },
    {
        "dataset": "sembench_movie",
        "model": "llama3.1-70b",
        "budgets": "0.2,0.4,0.8,1.6",
        "proxy_policies": "borda,llm_judge",
    },
    {
        "dataset": "population",
        "model": "openai-gpt-4.1",
        "budgets": "1",
        "proxy_policies": "borda,llm_judge",
    },
    {
        "dataset": "population",
        "model": "llama3.1-70b",
        "budgets": "0.1",
        "proxy_policies": "borda,llm_judge",
    },
)


def _print_header(title: str) -> None:
    print(f"\n\033[1;36m== {title} ==\033[0m", flush=True)


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def _run_dev_experiments() -> None:
    _print_header("Dev Experiments")
    for dataset in DEV_DATASETS:
        _run(
            [
                sys.executable,
                "dev/run_experiment.py",
                "--dataset",
                dataset,
                "--models",
                DEV_MODELS,
            ]
        )


def _run_test_experiments() -> None:
    _print_header("Test Experiments")
    for dataset in TEST_DATASETS:
        _run(
            [
                sys.executable,
                "test/run_experiment.py",
                "--dataset",
                dataset,
                "--models",
                TEST_MODELS,
            ]
        )


def _run_test_optimizers(run_vary_samples: bool = False) -> None:
    _print_header("Test Optimizers")
    for spec in OPTIMIZER_RUNS:
        safe_model = spec["model"].replace("/", "-")
        budget_tokens = [b.strip() for b in spec["budgets"].split(",") if b.strip()]
        max_budget = max(budget_tokens, key=lambda b: float(b))
        base_cmd = [
            sys.executable,
            "test/run_optimizer.py",
            "--dataset",
            spec["dataset"],
            "--models",
            spec["model"],
            "--budgets",
            spec["budgets"],
            "--proxy-policies",
            spec["proxy_policies"],
        ]

        if spec["dataset"] in VARY_SAMPLE_DATASETS and run_vary_samples:
            # _run(base_cmd + ["--sample-size", "20"])
            for sample_size in OPTIMIZER_SAMPLE_SIZES:
                vary_budgets = spec["budgets"]
                vary_proxy_policies = "borda,llm_judge"
                _run(
                    [
                        sys.executable,
                        "test/run_optimizer.py",
                        "--dataset",
                        spec["dataset"],
                        "--models",
                        spec["model"],
                        "--budgets",
                        vary_budgets,
                        "--proxy-policies",
                        vary_proxy_policies,
                        "--sample-size",
                        str(sample_size),
                        "--output",
                        f"test/vary_samples/optimizer_{spec['dataset']}_{safe_model}_sample{sample_size}.json",
                    ]
                )
        else:
            _run(base_cmd)

        print("\n\n", flush=True)


def _run_dev_plots() -> None:
    _print_header("Dev Plots")
    for dataset in DEV_DATASETS:
        _run(
            [
                sys.executable,
                "dev/plot_experiment.py",
                "--input-dir",
                f"dev/{dataset}",
            ]
        )


def _run_test_plots() -> None:
    _print_header("Test Plots")
    for dataset in TEST_DATASETS:
        _run(
            [
                sys.executable,
                "test/plot_experiment.py",
                "--input-dir",
                f"test/{dataset}",
            ]
        )


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
        "--run-vary-samples",
        action="store_true",
        help="Run the vary-sample-size optimizer runs.",
    )
    args = parser.parse_args()

    skip = set(args.skip)

    if "dev" not in skip:
        _run_dev_experiments()

    if "test" not in skip:
        _run_test_experiments()

    if "optimizer" not in skip:
        _run_test_optimizers(run_vary_samples=args.run_vary_samples)

    if "plot" not in skip:
        if "dev" not in skip:
            _run_dev_plots()
        if "test" not in skip:
            _run_test_plots()

    _print_header("Done")


if __name__ == "__main__":
    main()
