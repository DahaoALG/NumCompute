# NumCompute

P_AI Assignment 2026 S2

.......

## Required structure

```text
NumCompute/
├── .gitignore
├── README.md
├── pyproject.toml
├── numcompute/
│   ├── init.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── stats.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── utils.py
│   └── benchmarking.py
├── tests/
│   └── README.md
├── benchmark/
│   ├── loops.py              # plain-Python versions (the "loop" side)
│   ├── run_benchmarks.py     # seeded runner, writes results.md
│   └── results.md            # latest full run
└── demo/
    └── quickstart.ipynb
```

## Task allocation

| Module                                         | Owner                                                                |
| ---------------------------------------------- | -------------------------------------------------------------------- |
| `io.py`, `metrics.py`, `benchmarking.py` | Jose                                                                 |
| `preprocessing.py`, `stats.py`             | Dat                                                                  |
| `sort_search.py`, `rank.py`                | Faris                                                                |
| `optim.py`, `pipeline.py`                  | Shuhao                                                               |
| `utils.py`, tests, packaging                 | Shared; each member owns their functions and 3--4 tests per function |

## Installation

Requires Python 3.10+ and NumPy (no other runtime dependencies).

```bash
git clone https://github.com/DahaoALG/NumCompute.git
cd NumCompute
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[test]"        # editable install of numcompute, plus pytest
python -m pytest                # run the test suite
```

To run the demo, open `demo/quickstart.ipynb` and select the `.venv` interpreter as the kernel.

## API overview and examples

### `numcompute.io`: reading CSV files

| Function | Returns |
| -------- | ------- |
| `read_header(path, delimiter=",")` | Column names from the first line, `list[str]` |
| `load_csv(path, ...)` | The whole file as a 2-D array, shape `(n_rows, n_cols)` |
| `iter_csv_chunks(path, chunk_size, ...)` | The file streamed as 2-D arrays of at most `chunk_size` rows |

Main options (shared by `load_csv` and `iter_csv_chunks`):

- `delimiter`: field separator, e.g. `","` or `"\t"`; `None` splits on any whitespace.
- `dtype`: `str` (default), float, int or unsigned int.
- `usecols`: columns to keep, by name or position.
- `missing_values`: cell contents treated as missing (default: `""`, `"NA"`, `"N/A"`, `"NaN"`, `"nan"`, `"null"`, `"None"`, `"?"`, `"NULL"`).
- `on_missing`: `"fill"` (default) writes `fill_value` into missing cells (NaN for floats, `""` for text), `"skip"` drops rows with a missing cell, `"raise"` raises on the first one.
- `comments`: text that starts a comment, e.g. `"#"`; off by default.

```python
import numpy as np
from numcompute import io

X, names = io.load_csv("demo/sample.csv", usecols=["age", "income", "height"],
                       dtype=float, return_header=True)                     # blank cells -> NaN
city = io.load_csv("demo/sample.csv", usecols="city")                      # text, blank cells -> ""
complete = io.load_csv("demo/sample.csv", usecols=names, dtype=float,
                       on_missing="skip")                                   # only rows with no missing cell

total = 0.0
for chunk in io.iter_csv_chunks("demo/sample.csv", chunk_size=3, usecols="income", dtype=float):
    total += np.nansum(chunk)                                               # memory bounded by chunk_size
```

Quoted fields that contain the delimiter (e.g. `"Smith, J"`) are not supported.

### `numcompute.sort_search`: sorting, selection, and search

These functions accept one-dimensional real numeric sequences or NumPy arrays.
They do not modify the input. NaNs, where supported, sort after all numeric
values.

| Function | Purpose |
| -------- | ------- |
| `stable_sort(values)` | Return a stable ascending sort using NumPy. |
| `lexsort(keys)` | Return indices that sort by multiple keys; the last key is the primary key. |
| `topk(values, k, largest=True, return_indices=True)` | Use NumPy's `argpartition` to return the `k` largest values descending or smallest ascending, with their input indices by default. |
| `top_k(values, k, largest=True, return_indices=False)` | Backward-compatible spelling that returns only values by default. |
| `binary_search(sorted_array, x)` | Return `(insertion_index, found)` using binary search. Input must already be sorted ascending, with NaNs at the end. |
| `quickselect(values, kth)` | Return the value at zero-based sorted position `kth` without full sorting, using the hand-written selection algorithm. |
| `quicksort(values)` | Additional hand-written, unstable sorting implementation. |

`kth` is zero-based and must identify an existing element. `topk` and `top_k` accept
`k` from zero through the input length; if indices are requested, the function
returns `(values, indices)` with matching order. NaNs appear after non-NaN
results in either direction. `binary_search` does not check whether its input
is sorted; when the value is absent, it still returns the insertion position.
When searching for NaN, the insertion position is the first NaN index.

```python
from numcompute.sort_search import (
    binary_search, lexsort, quickselect, stable_sort, topk,
)

values = [8, 1, 4, 2, 7, 3]
ordered = stable_sort(values)          # [1, 2, 3, 4, 7, 8]
median_upper = quickselect(values, 3)  # 4; kth is a zero-based position
largest_three, indices = topk(values, 3)
position, found = binary_search(ordered, 4)  # 3, True
by_two_keys = lexsort(([2, 1, 1], [1, 0, 1]))
```

### `numcompute.rank`: ranks and percentiles

| Function | Purpose |
| -------- | ------- |
| `rank(values, method="average", ascending=True)` | Return one-based ranks in the original input order. |
| `percentile_rank(values, scores, kind="mean")` | Return the percentile rank (0–100) for one or more scores relative to a sample. |
| `percentile(values, q, interpolation="linear")` | Return percentile value(s) for `q` in the range 0–100. |

`rank` supports tie methods `"average"` (default), `"min"`, `"max"`,
`"dense"`, and `"ordinal"`. Set `ascending=False` to assign smaller ranks to
larger values. NaNs receive NaN ranks. `rankdata` remains available as a
compatibility alias.

`percentile_rank` accepts `"strict"` (sample values below the score),
`"weak"` (values less than or equal to the score), `"mean"` (the average of
those two), and `"rank"` (the average one-based rank of matching sample values
as a percentage of the sample size). NaNs are omitted from the sample and a
NaN score produces a NaN result. If the sample contains no non-NaN values, the
result is NaN.

`percentile` omits NaNs from the sample and uses NumPy percentile interpolation
methods, including `"linear"` (default), `"lower"`, `"higher"`, `"midpoint"`,
and `"nearest"`. At least one non-NaN sample value is required. A scalar score
or percentile returns a scalar; a sequence returns an array.

```python
from numcompute.rank import percentile, percentile_rank, rank

scores = [30, 10, 20, 20]
ranks = rank(scores)                               # [4.0, 1.0, 2.5, 2.5]
rank_of_20 = percentile_rank(scores, 20)           # 50.0
quartiles = percentile([0, 10, 20, 30], [25, 50, 75])  # [7.5, 15.0, 22.5]
```

### `numcompute.metrics`: evaluation metrics

| Function | Returns |
| -------- | ------- |
| `accuracy(y_true, y_pred)` | Fraction of matching labels, `float` |
| `confusion_matrix(y_true, y_pred, labels=None)` | Counts, shape `(k, k)`: rows = true label, columns = predicted label |
| `precision`, `recall`, `f1(y_true, y_pred, average="binary", pos_label=1, zero_division=0.0)` | `float`, or shape `(k,)` when `average=None` |
| `mse(y_true, y_pred)` | Mean squared error, `float` |
| `roc_curve(y_true, y_score, pos_label=1)` | `fpr, tpr, thresholds`, each shape `(m + 1,)` for `m` distinct scores |
| `auc(x, y)` | Area under any monotonic curve (trapezoid rule), `float` |

Main options:

- `average` (precision, recall, F1): `"binary"` scores only `pos_label` (at most 2 classes); `"macro"` is the plain mean over classes; `"micro"` uses the counts pooled over classes (equal to accuracy); `"weighted"` weights each class by its number of true samples; `None` returns one value per class, in sorted label order.
- `pos_label`: the positive class, e.g. `1` or `"spam"`.
- `zero_division`: the value returned when a denominator is 0 (e.g. nothing predicted positive), instead of NaN.
- `labels` (confusion matrix): row/column order; may include labels that never appear.

```python
from numcompute import metrics

y_true, y_pred = [0, 1, 1, 0, 1], [0, 1, 0, 0, 1]
metrics.confusion_matrix(y_true, y_pred)        # [[2, 0], [1, 2]]: one positive missed
metrics.accuracy(y_true, y_pred)                # 0.8
metrics.precision(y_true, y_pred)               # 1.0  (2 of 2 predicted positives are right)
metrics.recall(y_true, y_pred)                  # 0.667 (2 of 3 true positives found)
metrics.precision(y_true, y_pred, average=None) # [0.667, 1.0], for labels 0 and 1

metrics.f1(["spam", "ham", "spam", "spam"], ["spam", "spam", "ham", "spam"], pos_label="spam")  # 0.667
metrics.mse([3.0, 5.0, 2.0], [2.5, 5.0, 4.0])   # 1.417

fpr, tpr, thresholds = metrics.roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
metrics.auc(fpr, tpr)                           # 0.75: a positive outscores a negative in 3 of 4 pairs
```

Inputs are 1-D. Labels can be int, float, bool or str, but numbers and text are never mixed: `accuracy(["0", "1"], [0, 1])` raises `TypeError` instead of scoring 0, which matters for labels read as text by `io.load_csv`. NaN in any input raises `ValueError`. Samples with tied scores form one point on the ROC curve.

### `numcompute.benchmarking`: loop vs vectorised timing

| Function | Returns |
| -------- | ------- |
| `time_function(func, repeat=7, number=1, warmup=1)` | Seconds per call for each repeat, shape `(repeat,)` |
| `compare(case, loop_func, vec_func, repeat=7, number=1, rtol=1e-7, atol=1e-9)` | `{"case", "loop_s", "vec_s", "speedup"}`: median seconds per call; `speedup = loop_s / vec_s` |
| `environment_info()` | Python, NumPy, OS, CPU and CPU count, `dict[str, str]` |
| `format_table(rows)` | `compare` results as a Markdown table (the columns of the performance table below); an optional `"notes"` entry in a row fills the Notes column |

The functions passed in take no arguments, so inputs are bound with a lambda. `compare` first checks that both versions return the same result (numbers within tolerance, text exactly) and raises `ValueError` if they differ, so a fast but wrong version never reaches the table. The median is reported because one slow repeat (OS noise) does not move it.

```python
import numpy as np
from numcompute import metrics
from numcompute.benchmarking import compare, format_table
from benchmark import loops                       # run from the repository root

a, b = np.random.default_rng(0).normal(size=(2, 100_000))
a_list, b_list = a.tolist(), b.tolist()           # converted outside the timing
row = compare("mse", lambda: loops.mse_loop(a_list, b_list), lambda: metrics.mse(a, b))
print(format_table([row]))
```

.........

## Performance table

Python loops (`benchmark/loops.py`) against NumCompute on the same inputs; both versions are checked to give the same result before they are timed. Reproduce from the repository root with:

```bash
python -m benchmark.run_benchmarks
```

Inputs come from `np.random.default_rng(42)` at fixed sizes. Times are the median of 7 calls. Baseline/result gives the Python-loop time and, in parentheses, the speed-up (loop time / NumCompute time). Environment: Python 3.14.7, NumPy 2.5.4, Windows 11, Intel CPU with 12 logical cores. Timings vary between machines and runs; the full report is in `benchmark/results.md`.

| Operation | NumCompute result | Baseline/result | Notes |
|---|---:|---:|---|
| io.load_csv dtype=float 20,000 x 5 | 20.943 ms | 33.622 ms (1.6x) | Both parse text line by line; parsing dominates |
| metrics.accuracy n=100,000 | 0.128 ms | 3.807 ms (29.8x) | One comparison per element, done in C |
| metrics.confusion_matrix n=100,000 | 4.304 ms | 8.719 ms (2.0x) | Sorts the labels (np.union1d) to accept any label type; the loop uses a dict |
| metrics.mse n=100,000 | 0.525 ms | 7.994 ms (15.2x) | Element-wise O(n) in C instead of the interpreter |
| metrics.roc_curve + auc n=2,000 | 0.158 ms | 26.276 ms (166.1x) | The loop compares every positive-negative pair, O(P*N); NumPy sorts once |
| preprocessing.StandardScaler 20,000 x 10 | 3.479 ms | 52.368 ms (15.1x) | Column means and stds over axis 0, one pass each |
| stats.histogram n=100,000 bins=20 | 1.749 ms | 10.451 ms (6.0x) | Bins all values at once |
| sort_search.stable_sort n=100,000 | 12.906 ms | 203.833 ms (15.8x) | Merge sort in Python vs np.sort(kind='stable') |
| sort_search.topk n=100,000 k=10 | 1.806 ms | 95.117 ms (52.7x) | k passes over the data vs np.argpartition |
| rank.rank method='average' n=2,000 | 0.447 ms | 272.715 ms (610.6x) | The loop compares every pair, O(n^2); NumPy sorts once |

## Test requirement
