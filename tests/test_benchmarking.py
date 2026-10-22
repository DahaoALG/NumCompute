"""Unit tests for :mod:`numcompute.benchmarking` and the benchmark/ runner.

Timings depend on the machine, so no test asserts how fast anything is: they
check shapes, call counts, the result check, validation and the report format.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from numcompute.benchmarking import compare, environment_info, format_table, time_function

REPO_ROOT = Path(__file__).resolve().parents[1]


# time_function
def test_time_function_shape_and_values():
    times = time_function(lambda: sum(range(1000)), repeat=3)
    assert isinstance(times, np.ndarray)
    assert times.shape == (3,)
    assert np.all(times > 0)


def test_time_function_call_count():
    calls = []
    time_function(lambda: calls.append(1), repeat=3, number=4, warmup=2)
    assert len(calls) == 2 + 3 * 4                      # warmup + repeat * number


def test_time_function_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        time_function(42)


@pytest.mark.parametrize("bad", [{"repeat": 0}, {"number": 0}, {"warmup": -1}])
def test_time_function_rejects_bad_counts(bad):
    with pytest.raises(ValueError):
        time_function(lambda: None, **bad)


# compare
def test_compare_keys_and_speedup():
    row = compare("sum", lambda: sum(range(1000)), lambda: np.arange(1000).sum(), repeat=3)
    assert set(row) == {"case", "loop_s", "vec_s", "speedup"}
    assert row["case"] == "sum"
    assert row["loop_s"] > 0 and row["vec_s"] > 0
    assert row["speedup"] == pytest.approx(row["loop_s"] / row["vec_s"])


def test_compare_rejects_different_results():
    with pytest.raises(ValueError, match="differ"):
        compare("wrong value", lambda: [1.0, 2.0], lambda: np.array([1.0, 2.5]), repeat=1)
    with pytest.raises(ValueError, match="differ"):
        compare("wrong shape", lambda: [1.0, 2.0], lambda: np.array([[1.0, 2.0]]), repeat=1)


def test_compare_accepts_rounding_and_nan():
    row = compare("float", lambda: [0.1 + 0.2, float("nan")],
                  lambda: np.array([0.3, np.nan]), repeat=1)
    assert row["speedup"] > 0


def test_compare_text_must_match_exactly():
    compare("text", lambda: ["a", "b"], lambda: np.array(["a", "b"]), repeat=1)
    with pytest.raises(ValueError, match="differ"):
        compare("text", lambda: ["a", "b"], lambda: np.array(["a", "c"]), repeat=1)


def test_compare_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        compare("bad", lambda: 1, 1)


# environment_info
def test_environment_info_keys_and_values():
    info = environment_info()
    assert set(info) == {"python", "numpy", "os", "cpu", "cpu_count"}
    assert info["numpy"] == np.__version__
    assert all(isinstance(value, str) and value for value in info.values())


# format_table
def test_format_table_known_rows():
    rows = [{"case": "mse", "loop_s": 0.01, "vec_s": 0.0002, "speedup": 50.0,
             "notes": "Element-wise in C"},
            {"case": "accuracy", "loop_s": 1.5, "vec_s": 0.001, "speedup": 1500.0}]
    lines = format_table(rows).splitlines()
    assert lines[0] == "| Operation | NumCompute result | Baseline/result | Notes |"
    assert lines[2] == "| mse | 0.200 ms | 10.000 ms (50.0x) | Element-wise in C |"
    assert lines[3] == "| accuracy | 1.000 ms | 1,500.000 ms (1,500.0x) |  |"   # no notes


def test_format_table_no_rows_gives_header_only():
    assert len(format_table([]).splitlines()) == 2      # header and separator


# benchmark/run_benchmarks.py
def test_runner_quick_writes_report(tmp_path):
    out = tmp_path / "results.md"
    done = subprocess.run([sys.executable, "-m", "benchmark.run_benchmarks", "--quick",
                           "--output", str(out)],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    report = out.read_text(encoding="utf-8")
    assert "## Environment" in report
    for case in ("io.load_csv", "metrics.accuracy", "metrics.confusion_matrix", "metrics.mse",
                 "metrics.roc_curve + auc", "preprocessing.StandardScaler", "stats.histogram",
                 "sort_search.stable_sort", "sort_search.topk", "rank.rank"):
        assert case in report
