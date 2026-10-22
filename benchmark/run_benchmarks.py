"""Benchmark NumCompute against plain-Python loops and write benchmark/results.md.

Run from the repository root:

    python -m benchmark.run_benchmarks            # full sizes (about a minute)
    python -m benchmark.run_benchmarks --quick    # small sizes, a few seconds

All inputs come from np.random.default_rng(SEED) with fixed sizes, so every run
times the same data. Each case checks that both versions agree before timing.
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from benchmark import loops
from numcompute import io, metrics, preprocessing, rank, sort_search, stats
from numcompute.benchmarking import compare, environment_info, format_table

SEED = 42
DEFAULT_OUTPUT = Path(__file__).with_name("results.md")

# Input sizes and repeats; --quick is for smoke tests, not for reporting
FULL = {"n": 100_000, "n_auc": 2_000, "n_rank": 2_000, "csv_shape": (20_000, 5),
        "scale_shape": (20_000, 10), "bins": 20, "k": 10, "repeat": 7}
QUICK = {"n": 1_000, "n_auc": 200, "n_rank": 200, "csv_shape": (500, 5),
         "scale_shape": (500, 5), "bins": 20, "k": 10, "repeat": 2}


def build_cases(cfg, rng, tmp_dir):
    """Return the cases, each (operation, loop_func, vec_func, notes).

    Python lists for the loop side are made here, so converting arrays to lists
    is not part of the timing.
    """
    n, n_auc, n_rank = cfg["n"], cfg["n_auc"], cfg["n_rank"]

    # Classification labels: 5 classes, about 70% predicted correctly
    y_true = rng.integers(0, 5, n)
    y_pred = np.where(rng.random(n) < 0.7, y_true, rng.integers(0, 5, n))
    true_list, pred_list = y_true.tolist(), y_pred.tolist()

    # Regression values
    v_true, v_pred = rng.normal(size=n), rng.normal(size=n)
    v_true_list, v_pred_list = v_true.tolist(), v_pred.tolist()

    # Binary labels with scores rounded to 2 decimals, so there are ties
    yb = rng.integers(0, 2, n_auc)
    score = np.round(rng.random(n_auc) + 0.3 * yb, 2)
    yb_list, score_list = yb.tolist(), score.tolist()

    csv_path = Path(tmp_dir) / "bench.csv"
    _write_csv(csv_path, rng, cfg["csv_shape"])
    
    X = rng.normal(5, 2, size=cfg["scale_shape"])
    X_list = X.tolist()
    
    x = rng.normal(size=n)
    x_list = x.tolist()
    
    bins, k = cfg["bins"], cfg["k"]
    ties = rng.integers(0, n_rank // 4, n_rank)          # many repeated values
    ties_list = ties.tolist()

    cases = [
        (f"io.load_csv dtype=float {cfg['csv_shape'][0]:,} x {cfg['csv_shape'][1]}",
         lambda: loops.load_csv_loop(csv_path),
         lambda: io.load_csv(csv_path, dtype=float),
         "Both parse text line by line; parsing dominates"),
        (f"metrics.accuracy n={n:,}",
         lambda: loops.accuracy_loop(true_list, pred_list),
         lambda: metrics.accuracy(y_true, y_pred),
         "One comparison per element, done in C"),
        (f"metrics.confusion_matrix n={n:,}",
         lambda: loops.confusion_matrix_loop(true_list, pred_list),
         lambda: metrics.confusion_matrix(y_true, y_pred),
         "Sorts the labels (np.union1d) to accept any label type; the loop uses a dict"),
        (f"metrics.mse n={n:,}",
         lambda: loops.mse_loop(v_true_list, v_pred_list),
         lambda: metrics.mse(v_true, v_pred),
         "Element-wise O(n) in C instead of the interpreter"),
        (f"metrics.roc_curve + auc n={n_auc:,}",
         lambda: loops.auc_loop(yb_list, score_list),
         lambda: metrics.auc(*metrics.roc_curve(yb, score)[:2]),
         "The loop compares every positive-negative pair, O(P*N); NumPy sorts once"),
        (f"preprocessing.StandardScaler {cfg['scale_shape'][0]:,} x {cfg['scale_shape'][1]}",
         lambda: loops.standard_scale_loop(X_list),
         lambda: preprocessing.StandardScaler().fit_transform(X),
         "Column means and stds over axis 0, one pass each"),
        (f"stats.histogram n={n:,} bins={bins}",
         lambda: loops.histogram_loop(x_list, bins),
         lambda: stats.histogram(x, bins=bins)[0],
         "Bins all values at once"),
        (f"sort_search.stable_sort n={n:,}",
         lambda: loops.stable_sort_loop(x_list),
         lambda: sort_search.stable_sort(x),
         "Merge sort in Python vs np.sort(kind='stable')"),
        # Both return (values, indices); the check compares them as one (2, k) array
        (f"sort_search.topk n={n:,} k={k}",
         lambda: loops.topk_loop(x_list, k),
         lambda: sort_search.topk(x, k),
         "k passes over the data vs np.argpartition"),
        (f"rank.rank method='average' n={n_rank:,}",
         lambda: loops.average_rank_loop(ties_list),
         lambda: rank.rank(ties, method="average"),
         "The loop compares every pair, O(n^2); NumPy sorts once"),
    ]

    return cases


def run(quick=False):
    """Run every case; return the Markdown report."""
    cfg = QUICK if quick else FULL
    rng = np.random.default_rng(SEED)

    start = time.perf_counter()
    rows = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        cases = build_cases(cfg, rng, tmp_dir)
        for case, loop_func, vec_func, notes in cases:
            print(f"  {case} ...", flush=True)
            row = compare(case, loop_func, vec_func, repeat=cfg["repeat"], rtol=1e-6)
            row["notes"] = notes                          # shown in the table's Notes column
            rows.append(row)
    elapsed = time.perf_counter() - start

    command = "python -m benchmark.run_benchmarks" + (" --quick" if quick else "")
    lines = [
        "# Benchmark results: Python loops vs NumCompute",
        "",
        f"Generated by `{command}` (seed {SEED}, {cfg['repeat']} repeats per version, "
        f"{elapsed:.1f} s in total). Times are the median per call. Baseline/result "
        "gives the Python-loop time and, in parentheses, the speed-up "
        "(loop time / NumCompute time). Both versions are checked to give the same "
        "result before they are timed.",
        "",
        "## Environment",
        "",
    ]
    
    lines += [f"- {key}: {value}" for key, value in environment_info().items()]
    lines += ["", "## Results", "", format_table(rows)]
    
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark NumCompute against Python loops.")
    parser.add_argument("--quick", action="store_true", help="small sizes for a fast check")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Markdown file to write (default: benchmark/results.md)")
    args = parser.parse_args(argv)

    report = run(quick=args.quick)
    args.output.write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"Saved to {args.output}")


def _write_csv(path, rng, shape):
    """Numeric CSV with a header row and about 2% empty (missing) cells."""
    data = rng.normal(size=shape).round(6).astype(str)
    data[rng.random(shape) < 0.02] = ""
    header = ",".join(f"x{j}" for j in range(shape[1]))
    body = "\n".join(",".join(row) for row in data.tolist())
    path.write_text(header + "\n" + body + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
