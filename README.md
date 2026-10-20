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
| `quicksort(values)` | Return a sorted copy in ascending order; average `O(n log n)`, worst-case `O(n²)`. |
| `merge_sort(values)` | Return a stably sorted copy in ascending order; `O(n log n)` time and `O(n)` auxiliary space. |
| `argpartition(values, kth)` | Return indices partitioned around the value at zero-based sorted position `kth`; average `O(n)`. Order within either partition is unspecified. |
| `quickselect(values, kth)` | Return the value at zero-based sorted position `kth` without fully sorting; average `O(n)`. |
| `top_k(values, k, largest=True)` | Return the `k` largest values in descending order, or the `k` smallest in ascending order when `largest=False`; average `O(n + k log k)`. |
| `binary_search(sorted_values, target)` | Return the first matching index, or `-1` if absent; `O(log n)`. Input must already be sorted ascending, with NaNs at the end. |

`kth` is zero-based and must identify an existing element. `top_k` accepts
`k` from zero through the input length. For `top_k`, NaNs appear after
non-NaN results in either direction; ties are not guaranteed to retain their
input order. `binary_search` does not check whether its input is sorted. When
searching for NaN, it returns the first NaN index.

```python
from numcompute.sort_search import binary_search, merge_sort, quickselect, top_k

values = [8, 1, 4, 2, 7, 3]
ordered = merge_sort(values)           # [1, 2, 3, 4, 7, 8]
median_upper = quickselect(values, 3)  # 4; kth is a zero-based position
largest_three = top_k(values, 3)      # [8, 7, 4]
position = binary_search(ordered, 4)  # 3
```

### `numcompute.rank`: ranks and percentiles

| Function | Purpose |
| -------- | ------- |
| `rankdata(values, method="average", ascending=True)` | Return one-based ranks in the original input order. |
| `percentile_rank(values, scores, kind="mean")` | Return the percentile rank (0–100) for one or more scores relative to a sample. |
| `percentile(values, q, method="linear")` | Return percentile value(s) for `q` in the range 0–100. |

`rankdata` supports tie methods `"average"` (default), `"min"`, `"max"`,
`"dense"`, and `"ordinal"`. Set `ascending=False` to assign smaller ranks to
larger values. NaNs receive NaN ranks.

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
from numcompute.rank import percentile, percentile_rank, rankdata

scores = [30, 10, 20, 20]
ranks = rankdata(scores)                           # [4.0, 1.0, 2.5, 2.5]
rank_of_20 = percentile_rank(scores, 20)           # 50.0
quartiles = percentile([0, 10, 20, 30], [25, 50, 75])  # [7.5, 15.0, 22.5]
```

.........

## Performance table

| Operation | NumCompute result | Baseline/result | Notes                                    |
| --------- | ----------------- | --------------- | ---------------------------------------- |
| ......    | ?                 | ?               | Benchmark harness pending implementation |

## Test requirement
