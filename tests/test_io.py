"""Unit tests for :mod:`numcompute.io`: header reading, CSV loading and chunked streaming."""

import numpy as np
import pytest

from numcompute.io import DEFAULT_MISSING, iter_csv_chunks, load_csv, read_header

# Seven data rows with missing cells spread over several chunks
GAPPY_CSV = (
    "a,b,c\n"
    "1,2,3\n"
    "4,NA,6\n"
    "7,8,9\n"
    "10,11,\n"
    "13,14,15\n"
    "16,17,18\n"
    ",20,21\n"
)


@pytest.fixture
def write_csv(tmp_path):
    """Return a function that writes ``text`` to a file in ``tmp_path`` and gives back its path."""

    def _write(text: str, name: str = "data.csv", encoding: str = "utf-8"):
        path = tmp_path / name
        path.write_text(text, encoding=encoding, newline="")  # keep "\n" as written, on every OS

        return path

    return _write


# ---- read_header ----

def test_read_header_strips_names_and_bom(write_csv):
    """Names lose surrounding spaces and the byte order mark that Excel adds."""
    path = write_csv("﻿ age , income \n34,52000.5\n")

    assert read_header(path) == ["age", "income"]


def test_read_header_tab_and_whitespace_delimiters(write_csv):
    """A custom delimiter splits on it; ``None`` splits on any run of whitespace."""
    tsv = write_csv("age\tincome\tcity\n34\t52000.5\tAdelaide\n", name="data.tsv")
    spaced = write_csv("age   income  city\n34 52000.5 Adelaide\n", name="spaced.txt")

    assert read_header(tsv, delimiter="\t") == ["age", "income", "city"]
    assert read_header(spaced, delimiter=None) == ["age", "income", "city"]


def test_read_header_empty_file_and_missing_path(write_csv, tmp_path):
    """An empty file has no names; a missing file raises."""
    assert read_header(write_csv("", name="empty.csv")) == []

    with pytest.raises(FileNotFoundError):
        read_header(tmp_path / "missing.csv")


# ---- load_csv ----

def test_load_csv_default_returns_stripped_text_and_header(write_csv):
    """Default output is text, with spaces around each cell removed."""
    path = write_csv("age,income,city\n34, 52000.5 ,Adelaide\n28,61000,Perth\n")

    data, header = load_csv(path, return_header=True)

    assert data.dtype.kind == "U"
    assert data.shape == (2, 3)
    assert header == ["age", "income", "city"]
    np.testing.assert_array_equal(data[0], ["34", "52000.5", "Adelaide"])


def test_load_csv_float_turns_every_missing_marker_into_nan(write_csv):
    """Each default missing marker becomes NaN in float output."""
    # one data row per marker, the marker sits in column "b"
    rows = "".join(f"{i},{marker}\n" for i, marker in enumerate(DEFAULT_MISSING))
    path = write_csv("a,b\n" + rows)

    data = load_csv(path, dtype=float)

    assert data.dtype == np.float64
    assert data.shape == (len(DEFAULT_MISSING), 2)
    assert np.isnan(data[:, 1]).all()
    np.testing.assert_array_equal(data[:, 0], np.arange(len(DEFAULT_MISSING)))


def test_load_csv_on_missing_skip_and_raise(write_csv):
    """With "skip" rows with a missing cell are dropped; with "raise" the first one is named."""
    path = write_csv("a,b,c\n1,2,3\n4,NA,6\n7,8,9\n,11,12\n")

    kept = load_csv(path, dtype=float, on_missing="skip")

    np.testing.assert_array_equal(kept, [[1, 2, 3], [7, 8, 9]])

    with pytest.raises(ValueError, match="data row 2, column 1"):
        load_csv(path, dtype=float, on_missing="raise")


def test_load_csv_int_needs_fill_value(write_csv):
    """Integers cannot hold NaN, so missing cells need a fill_value or "skip"."""
    path = write_csv("a,b\n1,2\n3,NA\n5,6\n")

    with pytest.raises(ValueError, match="fill_value"):
        load_csv(path, dtype=int)

    filled = load_csv(path, dtype=int, fill_value=-1)

    assert filled.dtype.kind == "i"
    np.testing.assert_array_equal(filled, [[1, 2], [3, -1], [5, 6]])
    np.testing.assert_array_equal(load_csv(path, dtype=int, on_missing="skip"), [[1, 2], [5, 6]])


def test_load_csv_fill_value_is_checked_against_dtype(write_csv):
    """fill_value is converted for text output and rejected when it cannot fit."""
    path = write_csv("a,b\n1,2\n3,NA\n")

    np.testing.assert_array_equal(load_csv(path, fill_value=0), [["1", "2"], ["3", "0"]])  # 0 -> "0"
    np.testing.assert_array_equal(load_csv(path, dtype=int, fill_value=2.0), [[1, 2], [3, 2]])  # whole float is fine

    with pytest.raises(TypeError, match="pass a number"):
        load_csv(path, dtype=float, fill_value="x")

    with pytest.raises(TypeError, match="not a whole number"):
        load_csv(path, dtype=int, fill_value=0.7)  # would be truncated to 0

    with pytest.raises(ValueError, match="out of range"):
        load_csv(path, dtype=np.uint8, fill_value=-1)

    with pytest.raises(TypeError, match="not a whole number"):
        load_csv(path, dtype=int, on_missing="skip", fill_value=0.7)  # checked even when not used


def test_load_csv_usecols_by_name_index_and_negative(write_csv):
    """Columns are picked by name or position, in the order given."""
    path = write_csv("age,income,city\n34,52000.5,Adelaide\n28,61000,Perth\n")

    data, header = load_csv(path, usecols=["city", 0], return_header=True)

    assert header == ["city", "age"]
    np.testing.assert_array_equal(data, [["Adelaide", "34"], ["Perth", "28"]])

    data, header = load_csv(path, usecols=-1, return_header=True)

    assert header == ["city"]
    assert data.shape == (2, 1)

    with pytest.raises(ValueError, match="not found"):
        load_csv(path, usecols="country")

    with pytest.raises(ValueError, match="out of range"):
        load_csv(path, usecols=3)

    with pytest.raises(ValueError, match="empty"):
        load_csv(path, usecols=[])

    with pytest.raises(ValueError, match="no header"):
        load_csv(path, has_header=False, usecols="age")


@pytest.mark.parametrize("dtype", [str, float])
def test_load_csv_fast_and_general_paths_agree(write_csv, monkeypatch, dtype):
    """np.loadtxt (fast) and np.genfromtxt (general) give the same array."""
    path = write_csv("a,b,c\n1.5, NA ,7\n,2,?\n3,4e2,  5 \n")

    fast = load_csv(path, dtype=dtype)

    monkeypatch.setattr("numcompute.io._read_fast", lambda *args, **kwargs: None)  # force np.genfromtxt
    general = load_csv(path, dtype=dtype)

    assert fast.dtype.kind == general.dtype.kind
    np.testing.assert_array_equal(fast, general)  # NaN == NaN here


def test_load_csv_tab_delimiter_and_comments(write_csv):
    """Tab-separated files work, and "#" is only a comment when asked for."""
    tsv = write_csv("a\tb\n1\t\n\t2\n", name="data.tsv")

    np.testing.assert_array_equal(load_csv(tsv, delimiter="\t", dtype=float), [[1, np.nan], [np.nan, 2]])

    langs = write_csv("year,lang\n2000,C#\n1991,Python\n", name="langs.csv")

    assert load_csv(langs)[0, 1] == "C#"  # comments=None keeps "#" as data
    assert load_csv(langs, comments="#")[0, 1] == "C"  # the rest of the line is cut off

    commented = write_csv("a,b\n# a full comment line\n1,2\n3,4 # trailing comment\n", name="commented.csv")

    np.testing.assert_array_equal(load_csv(commented, dtype=float, comments="#"), [[1, 2], [3, 4]])


def test_load_csv_empty_header_only_and_all_skipped(write_csv):
    """No data rows still gives a 2-D array with the right number of columns."""
    assert load_csv(write_csv("", name="empty.csv")).shape == (0, 0)
    assert load_csv(write_csv("a,b,c\n", name="header.csv"), dtype=float).shape == (0, 3)

    gaps = write_csv("a,b,c\n1,NA,3\n,5,6\n", name="gaps.csv")

    assert load_csv(gaps, dtype=float, on_missing="skip").shape == (0, 3)


def test_load_csv_rejects_bad_dtype_strategy_and_ragged_rows(write_csv):
    """Unsupported options and badly shaped files raise clear errors."""
    path = write_csv("a,b\n1,2\n")

    with pytest.raises(TypeError, match="dtype"):
        load_csv(path, dtype=bool)

    with pytest.raises(ValueError, match="on_missing"):
        load_csv(path, on_missing="drop")

    with pytest.raises(ValueError, match="columns"):
        load_csv(write_csv("1,2,3\n4,5\n", name="ragged.csv"), has_header=False, dtype=float)

    with pytest.raises(ValueError, match="header has 3 columns"):
        load_csv(write_csv("a,b,c\n1,2\n3,4\n", name="narrow.csv"), dtype=float)


# ---- iter_csv_chunks ----

@pytest.mark.parametrize("dtype", [str, float])
@pytest.mark.parametrize("chunk_size", [1, 3, 100])
def test_iter_csv_chunks_vstack_matches_load_csv(write_csv, chunk_size, dtype):
    """Stacking every chunk gives the same array as loading the file at once."""
    path = write_csv(GAPPY_CSV)

    chunks = list(iter_csv_chunks(path, chunk_size, dtype=dtype))

    assert all(1 <= chunk.shape[0] <= chunk_size for chunk in chunks)
    np.testing.assert_array_equal(np.vstack(chunks), load_csv(path, dtype=dtype))


def test_iter_csv_chunks_skip_does_not_yield_empty_chunks(write_csv):
    """A chunk whose rows are all skipped is left out."""
    path = write_csv("a,b\n1,2\n3,4\nNA,6\n7,\n9,10\n")

    chunks = list(iter_csv_chunks(path, 2, dtype=float, on_missing="skip"))

    assert len(chunks) == 2
    np.testing.assert_array_equal(chunks[0], [[1, 2], [3, 4]])
    np.testing.assert_array_equal(chunks[1], [[9, 10]])


def test_iter_csv_chunks_raise_reports_row_in_whole_file(write_csv):
    """The error gives the row number in the file, not inside the chunk."""
    path = write_csv("a,b\n1,2\n3,4\n5,6\n7,8\n9,NA\n")

    chunks = iter_csv_chunks(path, 2, dtype=float, on_missing="raise")

    np.testing.assert_array_equal(next(chunks), [[1, 2], [3, 4]])  # rows before the bad one still arrive

    with pytest.raises(ValueError, match="data row 5, column 1"):
        list(chunks)


@pytest.mark.parametrize("chunk_size, error", [(0, ValueError), (2.5, TypeError)])
def test_iter_csv_chunks_rejects_bad_chunk_size(write_csv, chunk_size, error):
    """chunk_size must be a positive integer."""
    path = write_csv("a,b\n1,2\n")

    with pytest.raises(error, match="chunk_size"):
        iter_csv_chunks(path, chunk_size)  # raises on the call, no need to iterate


def test_iter_csv_chunks_checks_arguments_when_called(write_csv, tmp_path):
    """Bad arguments and missing files raise on the call, before any chunk is read."""
    path = write_csv("a,b\n1,2\n3,NA\n")

    with pytest.raises(FileNotFoundError):
        iter_csv_chunks(tmp_path / "missing.csv", 2, has_header=False)

    with pytest.raises(ValueError, match="on_missing"):
        iter_csv_chunks(path, 2, on_missing="drop")

    with pytest.raises(TypeError, match="dtype"):
        iter_csv_chunks(path, 2, dtype=bool)

    with pytest.raises(TypeError, match="pass a number"):
        iter_csv_chunks(path, 2, dtype=float, fill_value="x")

    with pytest.raises(ValueError, match="not found"):
        iter_csv_chunks(path, 2, usecols="c")

    chunks = list(iter_csv_chunks(path, 2, fill_value=0))  # text output, 0 -> "0"

    np.testing.assert_array_equal(np.vstack(chunks), [["1", "2"], ["3", "0"]])
