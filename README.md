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
│   ├── __init__.py
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
- `missing_values`: cell contents treated as missing (default: `""`, `"NA"`, `"N/A"`, `"NaN"`, `"nan"`, `"null"`, `"None"`, `"?"`).
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

.........

## Performance table

| Operation | NumCompute result | Baseline/result | Notes                                    |
| --------- | ----------------- | --------------- | ---------------------------------------- |
| ......    | ?                 | ?               | Benchmark harness pending implementation |

## Test requirement
