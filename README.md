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

.........

## Performance table

| Operation | NumCompute result | Baseline/result | Notes                                    |
| --------- | ----------------- | --------------- | ---------------------------------------- |
| ......    | ?                 | ?               | Benchmark harness pending implementation |

## Test requirement
