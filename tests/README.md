# Unit-test plan

> Required coverage: **at least 20 unit tests**, including edge cases.

Task allocation: testing is shared. Each member writes **3--4 tests for every
function they implement**, including normal, boundary, invalid-input, and
numerical-stability cases where applicable.

| Module               | Implementation owner | Test responsibility                                |
| -------------------- | -------------------- | -------------------------------------------------- |
| `io.py`            | Jose                 | Jose writes 3--4 tests per implemented function.   |
| `preprocessing.py` | Dat                  | Dat writes 3--4 tests per implemented function.    |
| `sort_search.py`   | Faris                | Faris writes 3--4 tests per implemented function.  |
| `rank.py`          | Faris                | Faris writes 3--4 tests per implemented function.  |
| `stats.py`         | Dat                  | Dat writes 3--4 tests per implemented function.    |
| `metrics.py`       | Jose                 | Jose writes 3--4 tests per implemented function.   |
| `optim.py`         | Shuhao               | Shuhao writes 3--4 tests per implemented function. |
| `pipeline.py`      | Shuhao               | Shuhao writes 3--4 tests per implemented function. |
| `utils.py`         | All members          | Each member tests the utilities they own.          |
| `benchmarking.py`  | Jose                 | Jose tests deterministic benchmark configuration.  |

Add test files as `test_<module>.py`.
