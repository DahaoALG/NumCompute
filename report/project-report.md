# NumCompute — Project Report

**Project:** NumCompute, a NumPy-based scientific computing toolkit  
**Version:** 0.1.0  
**Project state reviewed:** 22 October 2026; repository commit `b301779`  
**Purpose:** Summary of design decisions, testing, benchmark evidence, and teamwork

> **Layout note:** The `\newpage` markers divide this draft into four A4-sized sections. Final pagination will depend on the font, margins, and Markdown-to-Word conversion settings.

## 1. Project and design choices

NumCompute is a small Python package for common numerical and machine-learning tasks: reading tabular data, preprocessing, descriptive statistics, sorting and ranking, evaluation metrics, numerical differentiation, and simple pipelines. The package targets Python 3.10 and newer and has NumPy as its only runtime dependency. The design therefore uses NumPy as the common data representation and computation layer rather than depending on a larger scientific-computing framework.

### Vectorisation strategy

The central performance choice is to express element-wise and array-wide work with NumPy operations. Comparisons, reductions, broadcasting, axis-based calculations, sorting, and selection can run in compiled code rather than repeatedly dispatching Python operations for each value. This is most useful for workloads with many elements, where interpreter overhead is a substantial part of a straightforward loop implementation.

Examples include accuracy and mean-squared error calculations, column-wise scaling, histogram counting, and ranking. Sorting and partial selection use NumPy facilities where appropriate: stable sorting delegates to NumPy, and top-*k* uses `argpartition` rather than fully sorting the input. Ranking groups equal values after sorting. These choices keep the common numeric paths concise while preserving a clear relationship between the public operation and its result.

The project also retains hand-written algorithms where the algorithm itself is part of the library’s learning and demonstration value. `quicksort` and `quickselect` are examples; binary search is explicitly implemented as a search operation. The benchmark’s plain-Python loop implementations provide a reference point for comparing vectorised operations. Vectorisation is not forced onto tasks for which it would obscure the algorithm or offer little benefit: CSV parsing, for example, still processes textual records and the measured parsing cost limits the gain.

### Input and mutation principles

Several numeric APIs make their input expectations explicit: one-dimensional sequences for sorting, searching, and ranking; arrays with suitable dimensions for preprocessing; and compatible, equal-length arrays for metrics. Sorting and selection return results without modifying caller input. At the interface level, edge conditions such as invalid dimensions, unsupported options, and inconsistent lengths are generally reported as errors rather than silently coerced into misleading outputs. NaN behavior is specified for relevant operations instead of being left to incidental comparison behavior.

This combination—NumPy for routine array work, explicit algorithms for selected educational operations, and defined input contracts—aims to make the package both useful in small analyses and understandable to students reading its implementation.

\newpage

## 2. API design and package surfaces

The public API is grouped by task in small modules, so callers can import a focused module or use the package’s documented examples. Function names describe the operation, and parameters expose meaningful policy choices instead of requiring users to reproduce common variations themselves.

### Data and transformation APIs

`numcompute.io` reads headers, loads a CSV file into a two-dimensional array, or yields bounded-size chunks. Callers may select columns by name or position, choose a data type and delimiter, and decide how missing values are treated: fill, skip the affected row, or raise an error. Chunk iteration is useful when a caller wants to process a file without holding every row in memory. The documented limitation is that quoted CSV fields containing the delimiter are not supported.

`numcompute.preprocessing` provides estimator-style transformers, including standard and min-max scaling, simple imputation, and one-hot encoding. The fit/transform style keeps learned parameters (for example, scaling statistics or categories) associated with a transformer and makes it usable as a pipeline step. `numcompute.pipeline` composes compatible transformers and estimators, while `numcompute.optim` offers finite-difference gradient and Jacobian calculations with selectable difference methods.

### Numeric and evaluation APIs

`numcompute.stats` provides familiar summaries—mean, median, standard deviation, minimum, maximum, quantiles, and histograms—plus `RunningStats` for incremental accumulation. Several summary functions expose axis and NaN-handling choices. The sorting and selection module distinguishes stable full sorting, lexicographic index ordering, partial top-*k* selection, zero-based quick selection, and binary search. The top-*k* interface supports largest or smallest results and optional input indices; NaNs are placed after numeric results where supported. Search returns both an insertion position and whether the value was found.

`numcompute.rank` supplies one-based ranks with average, minimum, maximum, dense, or ordinal tie policies, as well as percentile ranks and percentile values. NaN treatment and interpolation choices are specified in the API. In `numcompute.metrics`, callers can calculate accuracy, a confusion matrix, precision, recall, F1, mean-squared error, an ROC curve, and area under a curve. Classification metrics support binary or multi-class averaging policies, an explicit positive label, and a configurable zero-division result.

### Consistency and usability

The API favors explicit choices that affect interpretation: tie methods, missing-value policies, label order, averaging mode, and percentile interpolation. It also documents shape and result conventions, such as confusion-matrix rows representing true labels and columns representing predictions. This matters because numerical results can look plausible even when axes, labels, or rank conventions are misunderstood.

There is a deliberate balance between convenience and predictable behavior. Scalar inputs produce scalar outputs in percentile functions, while sequences produce array results. The `rankdata` compatibility alias is retained alongside the primary `rank` name. Validation is part of the interface: for example, metric inputs must have compatible lengths and label types, rather than being paired through truncating iteration.

\newpage

## 3. Testing, edge cases, and benchmark results

### Testing approach and current result

The repository has focused test modules for I/O, preprocessing, sorting/searching, ranking, statistics, metrics, optimization, pipelines, utilities, and benchmarking. The test plan calls for normal, boundary, invalid-input, and numerical-stability coverage for implemented functions. In the current checkout, I ran:

```text
python -m pytest -q
199 passed in 1.06s
```

The tests exercise important edge conditions rather than only ordinary examples. Examples visible in the suite include empty and invalid inputs; NaNs in sorting, ranking, statistics, and metrics; ties in ranking and ROC scores; constant columns in scaling; missing and skipped CSV rows; chunk sizes and final partial batches; zero-division behavior; and rejection of inconsistent shapes or unsupported options. Tests also check non-mutation and agreement with expected numerical results. The benchmark comparison tests cover result matching, numeric tolerance, NaN comparisons, shape differences, and invalid timing arguments.

### Recorded benchmark results

The table reproduces the latest checked-in benchmark report. It compares NumCompute with plain-Python loop baselines using the same generated inputs and checks result agreement before timing. Results are median time per call; speed-up is baseline time divided by NumCompute time.

| Operation and input | NumCompute | Python-loop baseline | Speed-up |
|---|---:|---:|---:|
| CSV load, float, 20,000 × 5 | 20.943 ms | 33.622 ms | 1.6× |
| Accuracy, 100,000 items | 0.128 ms | 3.807 ms | 29.8× |
| Confusion matrix, 100,000 items | 4.304 ms | 8.719 ms | 2.0× |
| Mean-squared error, 100,000 items | 0.525 ms | 7.994 ms | 15.2× |
| ROC curve + AUC, 2,000 items | 0.158 ms | 26.276 ms | 166.1× |
| StandardScaler, 20,000 × 10 | 3.479 ms | 52.368 ms | 15.1× |
| Histogram, 100,000 items, 20 bins | 1.749 ms | 10.451 ms | 6.0× |
| Stable sort, 100,000 items | 12.906 ms | 203.833 ms | 15.8× |
| Top-*k*, 100,000 items, *k* = 10 | 1.806 ms | 95.117 ms | 52.7× |
| Average ranks, 2,000 items | 0.447 ms | 272.715 ms | 610.6× |

The benchmark used seed 42 and seven repeats, reporting medians. Its recorded environment was Python 3.14.7, NumPy 2.5.4, Windows 11, and an Intel CPU with 12 logical cores. These are the project’s recorded measurements, not a fresh run on the current macOS workstation.

\newpage

## 4. Interpreting results and teamwork reflections

### What the measurements suggest

The largest gains occur where the reference implementation does substantially more work: the rank baseline compares values pairwise (quadratic work), and the ROC/AUC baseline compares positive-negative score pairs. NumPy-based sorting avoids those pairwise loops. Top-*k* also benefits from partial selection: when *k* is small relative to the full input, partitioning avoids repeated scans or a complete ordering. For accuracy and MSE, both approaches are linear, but NumPy’s compiled element-wise operations reduce per-item interpreter overhead.

The smaller improvement for CSV loading is consistent with the benchmark note: both implementations must parse textual records, so parsing limits how much array vectorisation can accelerate the full operation. Confusion-matrix construction also has label-resolution overhead, which helps explain why its result is less dramatic than a simple element-wise metric. These comparisons should be read as evidence for the tested inputs, not as universal speed guarantees. Results can change with data shape, label cardinality, missing-value frequency, hardware, runtime version, and background system load. The benchmark harness is reproducible and checks outputs, but performance claims should be rerun on the target deployment environment before being used for capacity planning.

### Additional quickstart results

The quickstart notebook includes a second, smaller benchmark run on macOS with Python 3.14.6 and NumPy 2.5.1 (ARM, 10 logical CPUs). It uses different inputs from the full benchmark above, so these measurements are supplemental rather than directly comparable:

| Operation | Input | NumCompute | Python-loop baseline | Speed-up |
|---|---:|---:|---:|---:|
| Mean-squared error | 100,000 values | 0.238 ms | 7.359 ms | 31.0× |
| Accuracy | 3,000 city labels | 0.025 ms | 0.115 ms | 4.7× |
| Average ranking | 1,000 incomes | 0.059 ms | 19.764 ms | 332.9× |

The same synthetic-data walkthrough reports a regression RMSE of **9,986** for a noisy income estimate, compared with **42,079** for the mean-income baseline. The simulated six-city classifier reaches **72.3% accuracy** (macro F1 **0.631**); Sydney-versus-rest scoring gives precision **0.771**, recall **0.792**, F1 **0.781**, and income-score ROC AUC **0.934**. These are illustrative results on generated data, not evidence of real-world predictive performance. The pipeline also transforms 900 training rows and 2,100 test rows, and encodes the city column into seven categories.

### Teamwork reflections

The project documentation records a module-based allocation: Jose owned I/O, metrics, and benchmarking; Dat owned preprocessing and statistics; Faris owned sorting/search and ranking; and Shuhao owned optimization and pipelines. Utilities, tests, and packaging were shared, with each member responsible for tests around their own functions. This division gives contributors clear implementation areas while keeping integration concerns—shared helpers, packaging, and test conventions—visible to the whole team.

That structure is well suited to parallel work because many modules have distinct responsibilities. It also creates natural integration points: APIs must agree on array shapes and numeric types; NaN and missing-value conventions must remain coherent across modules; and pipeline steps must work with the transformers and estimators they compose. The current test organization and explicit behavior contracts help manage these boundaries. The merged project state and its single package configuration indicate that the independently owned work has been brought together into one installable toolkit.

For a subsequent iteration, the team could make cross-module integration review an explicit final activity: agree on shared validation and missing-data conventions, review public signatures together, then run the complete suite and benchmark smoke test against the integrated branch. A short written record of decisions that cross ownership boundaries would also help future contributors understand why particular conventions were chosen. These are improvement opportunities suggested by the project structure, not claims that the team experienced a specific coordination problem.

### Overall assessment

NumCompute combines a broad educational API with a practical performance goal. The strongest evidence in the current state is a passing 199-test suite, benchmark runs that validate result agreement before measuring speed, and a quickstart notebook that demonstrates the APIs end to end. Its main strengths are a lightweight dependency footprint, explicit behavior choices, and measurable gains for array-heavy operations. The main caveats are the documented CSV quoting limitation and the environment-specific nature of the benchmark and synthetic-demo results. Continued attention to integration contracts, reproducible measurements, and documentation will help keep the package approachable as it grows.
