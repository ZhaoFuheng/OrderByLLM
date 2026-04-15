#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_ROOT = PROJECT_ROOT / "test"
VARY_SAMPLES_ROOT = TEST_ROOT / "vary_samples"
DATASETS = ("dl20",)
TARGET_SAMPLE_SIZES = (16,18,20,22)
POLICY_LABELS = {
    "borda": "Opt(self-cons)",
    "llm_judge": "Opt(judge)",
}
LINE_COLORS = {
    ("borda", 0): "tab:blue",
    ("llm_judge", 0): "tab:orange",
}
MODEL_LABELS = {
    "llama3.1-70b": "llama3.1-70b",
    "openai-gpt-4.1": "GPT 4.1",
}
BASELINE_COLORS = {
    "llama3.1-70b": "tab:purple",
    "openai-gpt-4.1": "tab:green",
}
POLICY_LINESTYLES = {
    "borda": "-",
    "llm_judge": "--",
}
POLICY_MARKERS = {
    "borda": "X",
    "llm_judge": "P",
}


def _short_label(alg_name: str) -> str:
    return (
        alg_name
        .replace("external_pointwise_4", "ext_point_4")
        .replace("external_pointwise", "ext_point")
        .replace("external_bubble_sort_4", "ext_bubble_4")
        .replace("external_bubble_sort", "ext_bubble")
        .replace("external_merge_sort_4", "ext_merge_4")
        .replace("external_merge_sort", "ext_merge")
        .replace("quick_sort3", "quick_3")
        .replace("quick_sort", "quick")
        .replace("pointwise_with_search", "point_search")
        .replace("pointwise", "point")
        .replace("_with_search", "_search")
    )


def _collect_scores(dataset: str) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    scores_by_model: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )

    for path in sorted(VARY_SAMPLES_ROOT.glob("optimizer_*_sample*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dataset") != dataset:
            continue
        for model, model_results in payload.get("results_by_model", {}).items():
            for policy in POLICY_LABELS:
                budgets = model_results.get(policy, {})
                if not budgets:
                    continue

                for budget_str, rec in budgets.items():
                    settings = rec.get("settings", {})
                    sample_size = int(settings.get("sample_size", 20))
                    if sample_size not in TARGET_SAMPLE_SIZES:
                        continue
                    score = rec.get("score_mean")
                    if score is None:
                        continue
                    sample_key = str(sample_size)
                    prev = scores_by_model[model][policy][budget_str].get(sample_key)
                    score = float(score)
                    if prev is None or score > prev:
                        scores_by_model[model][policy][budget_str][sample_key] = score

    return scores_by_model


def _load_best_single_algorithm_scores(dataset_dir: Path) -> dict[str, tuple[float, str]]:
    baselines: dict[str, tuple[float, str]] = {}

    for path in sorted(dataset_dir.glob("results_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = str(payload.get("settings", {}).get("model", ""))
        if not model:
            continue

        best_score = None
        best_alg = None
        for metric in payload.get("metrics", []):
            score = metric.get("score_mean")
            if score is None:
                continue
            score = float(score)
            if best_score is None or score > best_score:
                best_score = score
                best_alg = str(metric.get("algorithm", "alg"))

        if best_score is not None and best_alg is not None:
            baselines[model] = (best_score, best_alg)

    return baselines


def _safe_model_name(model: str) -> str:
    return model.replace("/", "-")


def _plot_dataset_model(dataset: str, model: str, output_dir: Path) -> Path:
    dataset_dir = TEST_ROOT / dataset
    scores_by_model = _collect_scores(dataset)
    baseline_scores = _load_best_single_algorithm_scores(dataset_dir)
    policy_map = scores_by_model.get(model, {})

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False

    all_sample_sizes = set()
    for policy, label in POLICY_LABELS.items():
        budget_map = policy_map.get(policy, {})
        if not budget_map:
            continue

        best_sample_scores: dict[str, float] = {}
        for sample_size in TARGET_SAMPLE_SIZES:
            sample_key = str(sample_size)
            best_score = None
            for sample_map in budget_map.values():
                score = sample_map.get(sample_key)
                if score is None:
                    continue
                score = float(score)
                if best_score is None or score > best_score:
                    best_score = score
            if best_score is not None:
                best_sample_scores[sample_key] = best_score

        xs = [sample_size for sample_size in TARGET_SAMPLE_SIZES if str(sample_size) in best_sample_scores]
        if not xs:
            continue
        ys = [best_sample_scores[str(sample_size)] for sample_size in xs]
        all_sample_sizes.update(xs)
        ax.plot(
            xs,
            ys,
            marker=POLICY_MARKERS.get(policy, "o"),
            linewidth=2.5,
            markersize=8,
            linestyle=POLICY_LINESTYLES[policy],
            color=LINE_COLORS.get((policy, 0), "tab:gray"),
            label=label,
        )
        plotted = True

    if not plotted:
        raise ValueError(f"No optimizer data found for dataset={dataset}, model={model}")

    if all_sample_sizes:
        xmin = min(TARGET_SAMPLE_SIZES)
        xmax = max(TARGET_SAMPLE_SIZES)
        baseline = baseline_scores.get(model)
        if baseline is not None:
            score, alg_name = baseline
            ax.hlines(
                y=score,
                xmin=xmin,
                xmax=xmax,
                colors=BASELINE_COLORS.get(model, "tab:gray"),
                linestyles=":",
                linewidth=2.0,
                label=f"{_short_label(alg_name)}(best)",
            )

    ax.set_xlabel("Sample size")
    ax.set_ylabel("Mean nDCG@10")
    ax.set_xticks(list(TARGET_SAMPLE_SIZES))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_{_safe_model_name(model)}_vary_samplesize_ndcg@10.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot optimizer performance vs sample size for dl20 and sembench_movie."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to write figures to. Defaults to test/vary_samples.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir) if args.output_dir else TEST_ROOT / "vary_samples"
    generated = []
    for dataset in DATASETS:
        scores_by_model = _collect_scores(dataset)
        for model in sorted(scores_by_model):
            generated.append(_plot_dataset_model(dataset, model, output_root))

    for path in generated:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
