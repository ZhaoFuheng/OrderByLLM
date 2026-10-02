#!/usr/bin/env bash
# Regenerate the paper figures from the results_*.json / optimizer_*.json files.
#   ./make_figures.sh              # dev + test/optimizer plots -> figures/
#   ./make_figures.sh dev          # dev plots only
#   ./make_figures.sh test         # test + optimizer plots only
#   ./make_figures.sh all --ci     # same plots, with a per-query 80% confidence
#                                  # interval as the y error bar
#                                  # -> figures_with_confidence_interval/
#
# Test plots auto-overlay the optimizer curves (Self-Cons + Judge) from
# optimizer_<model>.json in each dataset dir.
#
# Models that are no longer part of the study (gpt-4.1, gpt-5-nano, llama-405b,
# sonnet) are skipped even if their results_*.json files are still present.
#
# Uses `python` from the active environment; set PYTHON to override.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"

DEV_DATASETS=(nba dl19)
TEST_DATASETS=(population dl20 sembench_movie hellaswag nfcorpus)

# Models no longer part of the study — never plot these.
EXCLUDE_RE='(openai-gpt-4\.1|openai-gpt-5-nano|llama3\.1-405b|claude-sonnet)'

WHICH="${1:-all}"
CI=0
if [ "${2:-}" = "--ci" ]; then
    CI=1
elif [ -n "${2:-}" ]; then
    echo "usage: $0 [dev|test|all] [--ci]" >&2; exit 2
fi

if [ "$CI" = 1 ]; then
    OUT=figures_with_confidence_interval
    DEV_PLOT=plot_experiment_ci.py
    TEST_PLOT=plot_experiment_ci.py
else
    OUT=figures
    DEV_PLOT=dev/plot_experiment.py
    TEST_PLOT=test/plot_experiment.py
fi

# _plot <script> <results.json> <output_dir> [extra args...]  — skip retired models.
_plot() {
    local script="$1" f="$2" out="$3"; shift 3
    local model; model="$(basename "$f" .json)"; model="${model#results_}"
    if [[ "$model" =~ $EXCLUDE_RE ]]; then
        echo "  [skip model] $model"
        return 0
    fi
    "$PYTHON" "$script" --input "$f" --output-dir "$out" "$@"
}

run_dev() {
    echo "== Dev plots -> $OUT/devFigures =="
    for d in "${DEV_DATASETS[@]}"; do
        [ -d "dev/$d" ] || { echo "  [skip] dev/$d (missing)"; continue; }
        echo "-- dev/$d"
        for f in dev/"$d"/results_*.json; do
            [ -e "$f" ] || continue
            # dl19 also gets the companion bar chart.
            if [ "$d" = dl19 ] && [ "$CI" = 0 ]; then
                _plot "$DEV_PLOT" "$f" "$OUT/devFigures" --include-dl19-bar
            else
                _plot "$DEV_PLOT" "$f" "$OUT/devFigures"
            fi
        done
    done
}

run_test() {
    echo "== Test + optimizer plots -> $OUT/testFigures =="
    for d in "${TEST_DATASETS[@]}"; do
        [ -d "test/$d" ] || { echo "  [skip] test/$d (missing)"; continue; }
        echo "-- test/$d"
        for f in test/"$d"/results_*.json; do
            [ -e "$f" ] || continue
            _plot "$TEST_PLOT" "$f" "$OUT/testFigures"
        done
    done
}

case "$WHICH" in
    dev)  run_dev ;;
    test) run_test ;;
    all)  run_dev; run_test ;;
    *)    echo "usage: $0 [dev|test|all] [--ci]" >&2; exit 2 ;;
esac

echo "Done."
