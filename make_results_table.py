#!/usr/bin/env python3
"""Fill in the main results table (Table~\\ref{tab:main-results-table}).

For each dataset x model it reports:
  - the top-3 standalone algorithms by score_mean (excluding bm25 and the retired
    _6/_8 batch variants), and
  - the best optimizer score for the Judge (llm_judge) and Self-Consistency
    (rrf_ensemble) policies, taken as the max score_mean across budgets.

Reads test/<dataset>/results_<model>.json and optimizer_<model>.json.
Prints the LaTeX table body; missing values are shown as '--'.

Usage:  python make_results_table.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (display name, dataset dir under test/)
DATASETS = [
    ("World",     "population"),
    ("Sembench",  "sembench_movie"),
    ("DL20",      "dl20"),
    ("NfCorpus",  "nfcorpus"),
    ("HellaSwag", "hellaswag"),
]
# (display name, filename/model key)
MODELS = [
    ("Llama-3.1 70B", "llama3.1-70b"),
    ("GPT-5 mini",    "openai-gpt-5-mini"),
    ("Haiku-4.5",     "claude-haiku-4-5"),
]

EXCLUDE_ALGS = {
    "bm25",
    "external_merge_sort_6", "external_merge_sort_8",
    "external_bubble_sort_6", "external_bubble_sort_8",
}
JUDGE_POLICY = "llm_judge"
SELFCONS_POLICY = "rrf_ensemble"
NA = "--"


def _short(alg: str) -> str:
    s = (alg
         .replace("external_pointwise_4", "ext_point_4")
         .replace("external_pointwise", "ext_point")
         .replace("external_bubble_sort_4", "ext_bubble_4")
         .replace("external_merge_sort_4", "ext_merge_4")
         .replace("quick_sort3", "quick_3")
         .replace("quick_sort", "quick")
         .replace("pointwise_with_search", "point_search")
         .replace("pointwise", "point")
         .replace("_with_search", "_search"))
    return s.replace("_", r"\_")           # LaTeX-safe underscores


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else NA


def top3_standalone(dataset_dir: Path, model: str):
    """Return [(short_alg, score), ...] top-3 by score_mean; pads to 3 with (--,--)."""
    p = dataset_dir / f"results_{model}.json"
    if not p.exists():
        return [(NA, NA)] * 3
    d = json.loads(p.read_text())
    rows = []
    for m in d.get("metrics", []):
        alg = str(m.get("algorithm", ""))
        if alg in EXCLUDE_ALGS:
            continue
        sc = m.get("score_mean", m.get("score"))
        if sc is None:
            continue
        rows.append((alg, float(sc)))
    rows.sort(key=lambda r: -r[1])
    top = [(_short(a), s) for a, s in rows[:3]]
    top += [(NA, NA)] * (3 - len(top))
    return top


def optimizer_best(dataset_dir: Path, model: str):
    """Return (judge_best, selfcons_best) = max score_mean across budgets, or NA."""
    p = dataset_dir / f"optimizer_{model}.json"
    if not p.exists():
        return NA, NA
    d = json.loads(p.read_text())
    by_model = d.get("results_by_model", {}).get(model, {})

    def best(policy):
        budgets = by_model.get(policy, {})
        scores = [rec.get("score_mean") for rec in budgets.values()
                  if rec.get("score_mean") is not None]
        return max(scores) if scores else NA

    return best(JUDGE_POLICY), best(SELFCONS_POLICY)


def main():
    print("% ==== auto-generated table body (paste between \\midrule ... \\bottomrule) ====")
    for di, (dset_disp, dset_dir) in enumerate(DATASETS):
        ddir = ROOT / "test" / dset_dir
        if di > 0:
            print(r"\midrule")
        print(rf"\multirow{{3}}{{*}}{{{dset_disp}}}")
        for mi, (mdl_disp, mdl_key) in enumerate(MODELS):
            (a1, s1), (a2, s2), (a3, s3) = top3_standalone(ddir, mdl_key)
            judge, selfc = optimizer_best(ddir, mdl_key)
            print(
                f"& {mdl_disp} "
                f"& {a1} & {_fmt(s1)} & {a2} & {_fmt(s2)} & {a3} & {_fmt(s3)} "
                f"& {_fmt(judge)} & {_fmt(selfc)} \\\\"
            )
    print("% ==== end ====")


if __name__ == "__main__":
    main()
