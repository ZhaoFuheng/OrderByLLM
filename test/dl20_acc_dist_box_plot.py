#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DL20_DIR = SCRIPT_DIR / "dl20"

FAMILY_COLORS = {
    "bm25": "tab:gray",
    "pointwise": "tab:blue",
    "ext_pointwise": "tab:green",
    "quick": "tab:purple",
    "quick3": "mediumpurple",
    "bubble": "tab:red",
    "merge": "tab:brown",
    "optimizer_judge": "tab:cyan",
    "optimizer_self_cons": "tab:orange",
}

PLOT_ORDER = {
    "bm25": 0,
    "point": 1,
    "ext_point_4": 2,
    "quick": 3,
    "quick_3": 4,
    "ext_merge_4": 5,
    "ext_bubble_4": 6,
    "Opt(judge)": 7,
    "Opt(self-cons)": 8,
}


def _family(alg_name: str) -> str:
    name = alg_name.lower()
    if "bm25" in name:
        return "bm25"
    if "quick_sort3" in name:
        return "quick3"
    if "quick" in name:
        return "quick"
    if "bubble" in name:
        return "bubble"
    if "merge" in name:
        return "merge"
    if "ext" in name or "external" in name:
        return "ext_pointwise"
    return "pointwise"


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


def _extract_query_scores(per_query_scores: dict) -> list[float]:
    if not per_query_scores:
        raise ValueError("Missing per_query_scores")
    first_seed_scores = next(iter(per_query_scores.values()))
    return [float(score) for score in first_seed_scores.values()]


def _load_single_algorithm_entries(results_path: Path) -> tuple[str, list[dict]]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    model = str(payload["settings"]["model"])
    entries = []

    for metric in payload.get("metrics", []):
        scores = _extract_query_scores(metric.get("per_query_scores", {}))
        alg_name = str(metric["algorithm"])
        entries.append(
            {
                "label": _short_label(alg_name),
                "scores": scores,
                "color": FAMILY_COLORS[_family(alg_name)],
            }
        )

    return model, entries


def _max_budget_optimizer_record(optimizer_payload: dict, model: str, policy: str) -> dict:
    policy_runs = optimizer_payload["results_by_model"][model].get(policy, {})
    if not policy_runs:
        return {}
    _, best_record = max(
        policy_runs.items(),
        key=lambda item: float(item[0]),
    )
    return best_record


def _load_optimizer_entries(optimizer_path: Path, model: str) -> list[dict]:
    payload = json.loads(optimizer_path.read_text(encoding="utf-8"))
    policy_specs = [
        ("llm_judge", "Opt(judge)", FAMILY_COLORS["optimizer_judge"]),
        ("borda", "Opt(self-cons)", FAMILY_COLORS["optimizer_self_cons"]),
    ]
    entries = []

    for policy, label, color in policy_specs:
        budget_record = _max_budget_optimizer_record(payload, model, policy)
        if not budget_record:
            continue
        scores = [float(score) for score in budget_record.get("per_query_scores", {}).values()]
        entries.append(
            {
                "label": label,
                "scores": scores,
                "color": color,
            }
        )

    return entries


def plot_model(results_path: Path, output_dir: Path) -> Path:
    model, entries = _load_single_algorithm_entries(results_path)
    optimizer_path = results_path.with_name(f"optimizer_{model.replace('/', '-')}.json")
    entries.extend(_load_optimizer_entries(optimizer_path, model))
    entries.sort(key=lambda entry: (PLOT_ORDER.get(entry["label"], 999), entry["label"]))

    labels = [entry["label"] for entry in entries]
    score_lists = [entry["scores"] for entry in entries]
    colors = [entry["color"] for entry in entries]
    self_cons_mean = None
    for entry in entries:
        if entry["label"] == "Opt(self-cons)" and entry["scores"]:
            self_cons_mean = sum(entry["scores"]) / len(entry["scores"])
            break

    fig, ax = plt.subplots(figsize=(11, 6))
    box = ax.boxplot(
        score_lists,
        patch_artist=True,
        tick_labels=labels,
        widths=0.65,
        showfliers=False,
        showmeans=True,
        meanline=True,
        meanprops={"color": "black", "linewidth": 1.8, "linestyle": "-"},
        medianprops={"color": "black", "linewidth": 0.0},
        whiskerprops={"color": "black", "linewidth": 1.2},
        capprops={"color": "black", "linewidth": 1.2},
        boxprops={"edgecolor": "black", "linewidth": 1.2},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    if self_cons_mean is not None:
        ax.axhline(
            y=self_cons_mean,
            color=FAMILY_COLORS["optimizer_self_cons"],
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
        )

    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"dl20_{model.replace('/', '-')}_acc_dist_box.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot DL20 per-query nDCG@10 box plots for algorithms and optimizers."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DL20_DIR),
        help="Directory containing DL20 results_*.json and optimizer_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to the input directory.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    generated_paths = []
    for results_path in sorted(input_dir.glob("results_*.json")):
        generated_paths.append(plot_model(results_path, output_dir))

    for path in generated_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
