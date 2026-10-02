#!/usr/bin/env python3
"""How often does each selection policy match or beat the best static access path?

For every query we take the ground-truth-best STATIC access path -- the argmax ndcg
over the six standalone paths {ext_merge_4, ext_bubble_4, quick, quick_3, point,
ext_point} (test/dl20/results_<model>.json). We then report, per budget and per
selection policy (per-model judge, or RRF self-consistency), the percentage of
queries whose selected/ fused ranking achieves ndcg >= that best static path. A
single-path judge can at best tie this oracle; RRF can exceed it by fusing paths.

Reads the judge-ablation results (test/dl20_ablation/results_<model>.json), which
store per_query_scores for each policy x budget.

  --pick-dist   also print the judge pick-vs-true-best family distribution.
  --latex       also emit a LaTeX table.

Usage:
    python test/dl20_ablation/judge_pick_distribution.py --model llama3.1-70b --latex
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "test"))
import run_optimizer as RO  # noqa: E402

HERE = Path(__file__).resolve().parent
_EPS = 1e-9
_CONFIG_ORDER = ["llama_judge", "haiku_judge", "sonnet_judge", "gpt5nano_judge", "rrf"]
_LABEL = {"llama_judge": "Llama", "haiku_judge": "Haiku", "sonnet_judge": "Sonnet",
          "gpt5nano_judge": "GPT5-nano", "rrf": "RRF"}
_FAMILIES = ["ext_point", "point", "ext_merge", "ext_bubble", "quick", "quick_3"]


def _family(alg: str) -> str:
    a = str(alg).lower()
    if "quick_sort3" in a or "quick_3" in a:          return "quick_3"
    if "quick" in a:                                  return "quick"
    if "bubble" in a:                                 return "ext_bubble"
    if "merge" in a:                                  return "ext_merge"
    if "external_pointwise" in a or "ext_point" in a: return "ext_point"
    if "point" in a:                                  return "point"
    if "bm25" in a:                                   return "bm25"
    return a


def _best_static(model):
    """Per-query best static ndcg (max over the six standalone paths, bm25 excluded)."""
    _b, alg_scores, _c = RO._load_oracle_data(model, "dl20")
    best_ndcg, best_fam = {}, {}
    for qid, scores in alg_scores.items():
        cand = {a: s for a, s in scores.items() if "bm25" not in a.lower()}
        if not cand:
            continue
        ba = max(cand, key=cand.get)
        best_ndcg[str(qid)] = cand[ba]
        best_fam[str(qid)] = _family(ba)
    return best_ndcg, best_fam


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama3.1-70b")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--pick-dist", action="store_true",
                    help="Also print judge pick-vs-true-best family distribution (at the top budget).")
    args = ap.parse_args()

    d = json.loads((HERE / f"results_{args.model}.json").read_text())
    configs = d["configs"]
    budgets = d["budgets"]
    cfgs = [c for c in _CONFIG_ORDER if c in configs]

    best_ndcg, best_fam = _best_static(args.model)

    # ── main table: % queries >= best static, per budget x policy ─────────────
    def _beat_pct(cfg, b):
        pqs = configs[cfg][b].get("per_query_scores", {})
        qids = [q for q in pqs if str(q) in best_ndcg]
        if not qids:
            return float("nan")
        hit = sum(1 for q in qids if float(pqs[q]) >= best_ndcg[str(q)] - _EPS)
        return 100.0 * hit / len(qids)

    table = {b: {c: _beat_pct(c, b) for c in cfgs} for b in budgets}

    print(f"\nDL20: % of queries matching/beating the best static access path  "
          f"(ranker={args.model})\n")
    hdr = f"{'rank budget':<12s}" + "".join(f"  {_LABEL[c]:>10s}" for c in cfgs)
    print(hdr); print("-" * len(hdr))
    for b in budgets:
        print(f"${b:<11s}" + "".join(f"  {table[b][c]:>9.1f}%" for c in cfgs))

    if args.latex:
        print("\n% ---- LaTeX ----")
        print(f"\\begin{{tabular}}{{l{'r'*len(cfgs)}}}")
        print("\\toprule")
        print("Ranking budget & " + " & ".join(_LABEL[c] for c in cfgs) + " \\\\")
        print("\\midrule")
        for b in budgets:
            print(f"\\${b} & " + " & ".join(f"{table[b][c]:.1f}\\%" for c in cfgs) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")

    # ── optional: pick distribution vs true best (top budget) ─────────────────
    if args.pick_dist:
        b = budgets[-1]
        judges = [c for c in cfgs if c != "rrf"]
        qids = sorted(configs[judges[0]][b]["per_query_chosen_alg"].keys())
        n = len(qids)
        tb = {f: 0 for f in _FAMILIES}
        for q in qids:
            f = best_fam.get(str(q))
            if f in tb:
                tb[f] += 1
        print(f"\nJudge pick distribution vs. true best  (budget=${b}, n={n})\n")
        hdr = f"{'family':<12s}  {'true-best%':>10s}" + "".join(f"  {_LABEL[c]+'%':>10s}" for c in judges)
        print(hdr); print("-" * len(hdr))
        for f in _FAMILIES:
            row = f"{f:<12s}  {100.0*tb[f]/n:>10.1f}"
            for c in judges:
                pqc = configs[c][b]["per_query_chosen_alg"]
                pct = 100.0 * sum(1 for q in qids if _family(pqc[q]) == f) / n
                row += f"  {pct:>10.1f}"
            print(row)


if __name__ == "__main__":
    main()
