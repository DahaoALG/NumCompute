"""CSV reader (streaming/chunking) and dtype handling.

Reads delimited text files into NumPy arrays, either all at once
(:func:`load_csv`) or in fixed-size chunks for files that do not fit in memory
(:func:`iter_csv_chunks`).
"""

import errno
import os
import re
import warnings
from collections.abc import Iterator, Sequence
from io import StringIO

import numpy as np

from .utils import iter_line_batches

# All that can be imported from this module
__all__ = ["DEFAULT_MISSING", "read_header", "load_csv", "iter_csv_chunks"]

# Cell contents treated as missing unless the caller passes another set
DEFAULT_MISSING: tuple[str, ...] = ("", "NA", "N/A", "NaN", "nan", "null", "None", "?")

# Possible values for on_missing
_ON_MISSING = ("fill", "skip", "raise")


def read_header(
    path: str | os.PathLike,
    delimiter: str | None = ",",
    encoding: str = "utf-8-sig",
) -> list[str]:
    """Return the column names stored in the first line of a CSV file.

    Parameters
    ----------
    path : str or os.PathLike
        File to read.
    delimiter : str or None, default ","
        Field separator. ``None`` splits on any run of whitespace.
    encoding : str, default "utf-8-sig"
        Text encoding of the file. "utf-8-sig" is safer than "utf-8" because
        it handles the byte order mark (BOM).

    Returns
    -------
    list[str]
        Stripped column names, length ``n_cols``. Empty list for an empty file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.

    Notes
    -----
    Time Complexity: O(L: length of the first line).
    Space Complexity: O(c: number of columns).
    Only the first line is read.

    Examples
    --------
    >>> read_header("data.csv")
    ['age', 'income', 'city']
    """
    with open(path, encoding=encoding) as f:
        first = f.readline()

    if not first.strip():
        return []

    return [name.strip() for name in first.rstrip("\r\n").split(delimiter)]


def load_csv(
    path: str | os.PathLike,
    delimiter: str | None = ",",
    has_header: bool = True,
    usecols: int | str | Sequence[int | str] | None = None,
    dtype: type | str | np.dtype = str,
    missing_values: Sequence[str] = DEFAULT_MISSING,
    fill_value: float | int | str | None = None,
    on_missing: str = "fill",
    return_header: bool = False,
    comments: str | None = None,
    encoding: str = "utf-8-sig",
) -> np.ndarray | tuple[np.ndarray, list[str]]:
    """Load a delimited text file into a 2-D NumPy array (text by default).

    By default every cell is returned as text. Pass ``dtype=float`` for purely
    numeric files.

    Text and float data take a fast path parsed in C by :func:`numpy.loadtxt`;
    integers (or anything the fast path cannot read) use :func:`numpy.genfromtxt`
    with a mask. Both give identical results and are followed by one vectorised
    fill / drop / check step.

    Parameters
    ----------
    path : str or os.PathLike
        CSV file to read.
    delimiter : str or None, default ","
        Field separator, e.g. ``","`` or ``"\\t"``. ``None`` = any whitespace.
    has_header : bool, default True
        If True the first line holds column names and is not parsed as data.
    usecols : int, str or sequence of int/str, optional
        Columns to keep, by position (negative allowed) or by header name
        (names need ``has_header=True``). Default: all columns.
    dtype : type, str or numpy.dtype, default str
        Output dtype. Supported kinds: ``str``, float, int, unsigned int.
        With ``str`` numbers stay as text (e.g. ``"1.5"``), with surrounding
        spaces removed.
    missing_values : sequence of str, default :data:`DEFAULT_MISSING`
        Cell contents treated as missing. For float output any cell that cannot
        be parsed as a number (it becomes NaN) also counts as missing.
    fill_value : scalar, optional
        Value written into missing cells when ``on_missing="fill"``.
        Default: ``nan`` for floats, ``""`` for strings. Integer output has no
        default (an int cannot hold NaN), so it must be given if cells are missing.
        Text output stores it as text (``0`` -> ``"0"``); numeric output needs a
        number, and integer output a whole number within the dtype's range.
    on_missing : {"fill", "skip", "raise"}, default "fill"
        ``"fill"`` replaces missing cells, ``"skip"`` drops every row that has at
        least one missing cell, ``"raise"`` raises on the first missing cell.
    return_header : bool, default False
        If True the header is returned as a list of strings.
    comments : str or None, default None
        Text that starts a comment, e.g. ``"#"``: the rest of that line is
        ignored. ``None`` (default) means no comments, so ``#`` inside a value
        such as ``"C#"`` is kept.
    encoding : str, default "utf-8-sig"
        Text encoding of the file. "utf-8-sig" reads plain UTF-8 too, and also
        drops the byte-order mark that Excel puts at the start of CSV files.

    Returns
    -------
    data : numpy.ndarray, shape (n_rows, n_cols)
        Parsed values. ``n_rows`` may be 0 (empty file, or every row skipped).
    header : list[str], length n_cols
        Only if ``return_header=True``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    TypeError
        If ``dtype`` is not a supported kind; or ``fill_value`` is not a number
        for numeric output, or not a whole number for integer output.
    ValueError
        If ``on_missing`` is unknown; ``usecols`` is empty, names a column that
        is not in the header (or there is no header) or is out of range;
        rows have different numbers of columns, or a different number than the
        header; ``on_missing="raise"`` and a cell is missing; integer output
        has missing cells and no ``fill_value``; or ``fill_value`` is out of
        range for an integer dtype.

    Notes
    -----
    Time Complexity: O(n: number of cells, rows * columns in the file).
    Space Complexity: O(n: number of cells, rows * columns) for the result plus
    a boolean mask of the same shape. Use :func:`iter_csv_chunks` to bound memory.

    Examples
    --------
    >>> S, header = load_csv("data.csv", return_header=True)
    >>> S[0]
    array(['34', '52000.5', 'Adelaide'], dtype='<U8')
    >>> data = load_csv("data.csv", usecols=["age", "income"], dtype=float)
    """
    dt = _check_dtype(dtype)
    _check_on_missing(on_missing)
    fill_value = _check_fill_value(fill_value, dt)

    header = read_header(path, delimiter, encoding) if has_header else []
    cols = _resolve_cols(usecols, header)
    n_cols = len(cols) if cols is not None else len(header)

    data, _ = _parse(
        path,
        skip_header=int(has_header),
        delimiter=delimiter,
        usecols=cols,
        dtype=dt,
        missing_values=missing_values,
        fill_value=fill_value,
        on_missing=on_missing,
        comments=comments,
        encoding=encoding,
        n_cols=n_cols,
        row_offset=0,
    )

    # A header and rows of different widths would give names that don't match the data
    if has_header and data.shape[0] > 0 and data.shape[1] != n_cols:
        raise ValueError(f"header has {n_cols} columns but the rows have {data.shape[1]}")

    if return_header:
        selected = [header[c] for c in cols] if (cols and header) else header

        return data, selected

    return data


def iter_csv_chunks(
    path: str | os.PathLike,
    chunk_size: int,
    delimiter: str | None = ",",
    has_header: bool = True,
    usecols: int | str | Sequence[int | str] | None = None,
    dtype: type | str | np.dtype = str,
    missing_values: Sequence[str] = DEFAULT_MISSING,
    fill_value: float | int | str | None = None,
    on_missing: str = "fill",
    comments: str | None = None,
    encoding: str = "utf-8-sig",
) -> Iterator[np.ndarray]:
    """Stream a CSV file as a sequence of 2-D arrays of at most ``chunk_size`` rows.

    The file is read lazily, ``chunk_size`` lines at a time, so memory use
    depends on the chunk size, not the file size. Concatenating all chunks with
    ``np.vstack`` gives the same array as :func:`load_csv` with the same options.

    Parameters
    ----------
    path : str or os.PathLike
        CSV file to read.
    chunk_size : int
        Maximum number of file lines per chunk (blank lines and lines starting
        with ``comments`` count towards it, so a chunk can hold fewer data
        rows). Must be >= 1.
    delimiter : str or None, default ","
        Field separator, e.g. ``","`` or ``"\\t"``. ``None`` = any whitespace.
    has_header : bool, default True
        If True the first line holds column names and is not parsed as data.
    usecols : int, str or sequence of int/str, optional
        Columns to keep, by position (negative allowed) or by header name
        (names need ``has_header=True``). Default: all columns.
    dtype : type, str or numpy.dtype, default str
        Output dtype. Supported kinds: ``str``, float, int, unsigned int.
        With ``str`` numbers stay as text (e.g. ``"1.5"``), with surrounding
        spaces removed.
    missing_values : sequence of str, default :data:`DEFAULT_MISSING`
        Cell contents treated as missing. For float output any cell that cannot
        be parsed as a number (it becomes NaN) also counts as missing.
    fill_value : scalar, optional
        Value written into missing cells when ``on_missing="fill"``.
        Default: ``nan`` for floats, ``""`` for strings. Integer output has no
        default (an int cannot hold NaN), so it must be given if cells are missing.
        Text output stores it as text (``0`` -> ``"0"``); numeric output needs a
        number, and integer output a whole number within the dtype's range.
    on_missing : {"fill", "skip", "raise"}, default "fill"
        ``"fill"`` replaces missing cells, ``"skip"`` drops every row that has at
        least one missing cell, ``"raise"`` raises on the first missing cell.
    comments : str or None, default None
        Text that starts a comment, e.g. ``"#"``: the rest of that line is
        ignored. ``None`` (default) means no comments, so ``#`` inside a value
        such as ``"C#"`` is kept.
    encoding : str, default "utf-8-sig"
        Text encoding of the file. "utf-8-sig" reads plain UTF-8 too, and also
        drops the byte-order mark that Excel puts at the start of CSV files.

    Returns
    -------
    Iterator[numpy.ndarray]
        Yields one chunk at a time, each of shape ``(n_chunk_rows, n_cols)``
        with ``1 <= n_chunk_rows <= chunk_size``. Chunks that end up empty
        (e.g. every row skipped) are not yielded.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    TypeError
        If ``dtype`` is not a supported kind; ``fill_value`` is not a number for
        numeric output, or not a whole number for integer output; or
        ``chunk_size`` is not an integer.
    ValueError
        If ``on_missing`` is unknown; ``usecols`` is empty, names a column that
        is not in the header (or there is no header) or is out of range;
        ``fill_value`` is out of range for an integer dtype; ``chunk_size < 1``;
        rows have different numbers of columns, or a different number than the
        header; ``on_missing="raise"`` and a cell is missing; or integer output
        has missing cells and no ``fill_value``.

    Notes
    -----
    Time Complexity: O(n: number of cells, rows * columns in the file).
    Space Complexity: O(k * c: chunk_size * columns), whatever the file size.
    The loop over chunks is the only Python-level loop: each chunk is parsed
    and cleaned with vectorised NumPy calls.
    Errors in the arguments, a missing file or unknown columns raise when the
    function is called; errors in the data raise while iterating, when the
    chunk that holds them is read.

    Examples
    --------
    >>> total = 0.0
    >>> for chunk in iter_csv_chunks("big.csv", chunk_size=10_000):
    ...     total += np.nansum(chunk[:, 0])
    """
    dt = _check_dtype(dtype)
    _check_on_missing(on_missing)
    fill_value = _check_fill_value(fill_value, dt)
    _check_chunk_size(chunk_size)

    # Without a header nothing below opens the file, so check it exists here
    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.fspath(path))

    header = read_header(path, delimiter, encoding) if has_header else []
    cols = _resolve_cols(usecols, header)
    n_cols = len(cols) if cols is not None else len(header)

    # All checks above run now; the file itself is read lazily by the generator
    return _iter_chunks(
        path,
        chunk_size,
        delimiter=delimiter,
        has_header=has_header,
        usecols=cols,
        dtype=dt,
        missing_values=missing_values,
        fill_value=fill_value,
        on_missing=on_missing,
        comments=comments,
        encoding=encoding,
        n_cols=n_cols,
    )


def _iter_chunks(
    path: str | os.PathLike,
    chunk_size: int,
    *,
    delimiter: str | None,
    has_header: bool,
    usecols: tuple[int, ...] | None,
    dtype: np.dtype,
    missing_values: Sequence[str],
    fill_value: float | int | str | None,
    on_missing: str,
    comments: str | None,
    encoding: str,
    n_cols: int,
) -> Iterator[np.ndarray]:
    """Generator behind :func:`iter_csv_chunks`; receives arguments already checked."""
    with open(path, encoding=encoding) as f:
        if has_header:
            next(f, None)  # skip the header line if there is one

        rows_parsed = 0  # data rows seen so far

        for lines in iter_line_batches(f, chunk_size):
            chunk, n_parsed = _parse(
                lines,
                skip_header=0,
                delimiter=delimiter,
                usecols=usecols,
                dtype=dtype,
                missing_values=missing_values,
                fill_value=fill_value,
                on_missing=on_missing,
                comments=comments,
                encoding=encoding,
                n_cols=n_cols,
                row_offset=rows_parsed,
            )

            rows_parsed += n_parsed  # to move the offset for the next chunk

            if has_header and chunk.shape[0] > 0 and chunk.shape[1] != n_cols:
                raise ValueError(f"header has {n_cols} columns but the rows have {chunk.shape[1]}")

            if chunk.shape[0] > 0:
                yield chunk


def _check_dtype(dtype: type | str | np.dtype) -> np.dtype:
    """Normalise ``dtype`` and reject kinds the parser does not support."""
    try:
        dt = np.dtype(dtype)
    except TypeError as e:
        raise TypeError(f"dtype {dtype!r} is not understood by NumPy") from e

    if dt.kind not in ("f", "i", "u", "U"):  # float, int, unsigned int, str
        raise TypeError(f"dtype must be a float, int, unsigned int or str type, got {dt}")

    return dt


def _check_on_missing(on_missing: str) -> None:
    """Check if a valid missing strategy is provided."""
    if on_missing not in _ON_MISSING:
        raise ValueError(f"on_missing must be one of {_ON_MISSING}, got {on_missing!r}")


def _check_fill_value(
    fill_value: float | int | str | None,
    dtype: np.dtype,
) -> float | int | str | None:
    """Check ``fill_value`` fits the output dtype and return it converted (None stays None)."""
    if fill_value is None:
        return None

    # Text output stores every cell as text, so 0 -> "0"
    if dtype.kind == "U":
        return str(fill_value)

    if isinstance(fill_value, (str, bytes)):
        raise TypeError(f"fill_value {fill_value!r} cannot be stored in {dtype}, pass a number")

    try:
        value = float(fill_value)
    except (TypeError, ValueError):
        raise TypeError(f"fill_value {fill_value!r} cannot be stored in {dtype}, pass a number") from None

    if dtype.kind == "f":
        return value

    # Integer output: no silent truncation (0.7 -> 0) and no wrap-around (-1 -> 255)
    if not value.is_integer():
        raise TypeError(f"fill_value {fill_value!r} is not a whole number, cannot be stored in {dtype}")

    whole = int(fill_value) if isinstance(fill_value, (int, np.integer)) else int(value)
    info = np.iinfo(dtype)

    if not info.min <= whole <= info.max:
        raise ValueError(f"fill_value {fill_value!r} is out of range for {dtype} [{info.min}, {info.max}]")

    return whole


def _check_chunk_size(chunk_size: int) -> None:
    """Check ``chunk_size`` is a positive integer."""
    if not isinstance(chunk_size, (int, np.integer)):
        raise TypeError(f"chunk_size must be an int, got {type(chunk_size).__name__}")

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be greater or equal than 1, got {chunk_size}")


def _resolve_cols(
    usecols: int | str | Sequence[int | str] | None,
    header: list[str],
) -> tuple[int, ...] | None:
    """Turn ``usecols`` (positions and/or names) into a tuple of non-negative positions."""
    if usecols is None:
        return None

    if isinstance(usecols, (int, str)):
        usecols = [usecols]  # a single column -> list of one

    if len(usecols) == 0:
        raise ValueError("usecols is an empty sequence, pass None to keep all columns")

    cols = []
    for c in usecols:  # tiny list of column specs, not data
        if isinstance(c, str):
            if not header:
                raise ValueError(f"column name {c!r} given but the file has no header")

            if c not in header:
                raise ValueError(f"column {c!r} not found in header {header}")

            cols.append(header.index(c))

        elif isinstance(c, (int, np.integer)):
            idx = int(c)

            if header:
                if not -len(header) <= idx < len(header):
                    raise ValueError(f"column index {idx} out of range for {len(header)} columns")

                idx %= len(header)  # convert negative indexes to positive

            cols.append(idx)

        else:
            raise TypeError(f"usecols entries must be int or str, got {type(c).__name__}")

    return tuple(cols)


def _parse(
    source: str | os.PathLike | list[str],
    *,
    skip_header: int,
    delimiter: str | None,
    usecols: tuple[int, ...] | None,
    dtype: np.dtype,
    missing_values: Sequence[str],
    fill_value: float | int | str | None,
    on_missing: str,
    comments: str | None,
    encoding: str,
    n_cols: int,
    row_offset: int,
) -> tuple[np.ndarray, int]:
    """Parse a file path or list of lines; return (clean 2-D array, rows parsed)."""

    options = dict(
        skip_header=skip_header,
        delimiter=delimiter,
        usecols=usecols,
        dtype=dtype,
        missing_values=missing_values,
        comments=comments,
        encoding=encoding,
    )

    parsed = _read_fast(source, **options)

    if parsed is None:
        parsed = _read_general(source, **options)

    values, missing = parsed

    if values.size == 0:  # empty file/chunk -> consistent (0, n_cols) shape
        return np.empty((0, n_cols), dtype=dtype), 0

    n_parsed = values.shape[0]

    # No missing cells, happy path
    if not missing.any():
        return values, n_parsed

    # Missing cells raise error
    elif on_missing == "raise":
        r, c = np.argwhere(missing)[0]  # first missing cell in row-major order
        raise ValueError(f"missing value at data row {row_offset + r + 1}, column {c}")

    # Missing cells are skipped
    elif on_missing == "skip":
        return values[~missing.any(axis=1)], n_parsed  # keep rows with no missing cell

    # Missing cells are filled
    else:
        # Check if fill_value has been provided
        if fill_value is None:
            if values.dtype.kind in "iu":
                raise ValueError("integer columns cannot hold NaN: pass fill_value or use "
                                 "on_missing='skip' (or load with dtype=float)")

            fill_value = "" if values.dtype.kind == "U" else np.nan

        # vectorised replace
        filled = np.where(missing, fill_value, values)

        if values.dtype.kind != "U":
            filled = filled.astype(values.dtype, copy=False)  # keep the requested numeric dtype

        return filled, n_parsed


def _read_general(
    source: str | os.PathLike | list[str],
    *,
    skip_header: int,
    delimiter: str | None,
    usecols: tuple[int, ...] | None,
    dtype: np.dtype,
    missing_values: Sequence[str],
    comments: str | None,
    encoding: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Any dtype, any missing markers: ``np.genfromtxt`` with a mask. Returns (values, missing)."""

    with warnings.catch_warnings():
        # genfromtxt warns on empty input; the caller handles that case
        warnings.simplefilter("ignore", UserWarning)
        raw = np.genfromtxt(
            source,
            delimiter=delimiter,
            skip_header=skip_header,
            usecols=usecols,
            dtype=dtype,
            missing_values={None: list(missing_values)},  # None key = every column
            usemask=True,
            autostrip=True,
            ndmin=2,
            comments=comments,
            encoding=encoding,
        )

    values = np.ma.getdata(raw)

    missing = np.ma.getmaskarray(raw)  # True where the cell matched missing_values / was empty

    if values.dtype.kind == "f":
        # "nan" text or unparseable numbers also count as missing
        missing = missing | np.isnan(values)

    return values, missing


def _read_fast(
    source: str | os.PathLike | list[str],
    *,
    skip_header: int,
    delimiter: str | None,
    usecols: tuple[int, ...] | None,
    dtype: np.dtype,
    missing_values: Sequence[str],
    comments: str | None,
    encoding: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fast path for text and float data, parsed in C by ``np.loadtxt``."""
    if delimiter is None:
        return None

    if dtype.kind == "U":
        return _read_fast_text(
            source,
            skip_header=skip_header,
            delimiter=delimiter,
            usecols=usecols,
            missing_values=missing_values,
            comments=comments,
            encoding=encoding,
        )

    if dtype.kind != "f":
        return None

    markers = []

    for m in missing_values:  # tiny list of markers
        try:
            if not np.isnan(float(m)):
                return None
        except ValueError:
            markers.append(re.escape(m.strip()))

    if isinstance(source, list):
        text = "".join(source)

    else:
        with open(source, encoding=encoding) as f:
            text = f.read()

    d = re.escape(delimiter)

    # longest first, so "N/A" is tried before "N"
    tokens = "|".join(sorted(set(markers), key=len, reverse=True))
    cell = rf"[ ]*(?:{tokens})[ ]*"

    # Pass 1: missing cells after a delimiter. Pass 2: missing first cells of a line.
    # A blank line is not a cell (no delimiter next to it), so it stays blank.
    # Every line now starts after "\n" and ends with "\n"
    text = "\n" + text + ("" if text.endswith("\n") else "\n")
    repl = (delimiter + "nan").replace("\\", "\\\\")  # re.sub treats "\" as special
    text = re.sub(rf"{d}{cell}(?={d}|\n)", repl, text)
    text = re.sub(rf"\n{cell}(?={d})", "\nnan", text)[1:]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # loadtxt warns on empty input
            values = np.loadtxt(
                StringIO(text),
                delimiter=delimiter,
                skiprows=skip_header,
                usecols=usecols,
                dtype=dtype,
                ndmin=2,
                comments=comments,
            )
    except ValueError:
        return None
    return values, np.isnan(values)


def _read_fast_text(
    source: str | os.PathLike | list[str],
    *,
    skip_header: int,
    delimiter: str,
    usecols: tuple[int, ...] | None,
    missing_values: Sequence[str],
    comments: str | None,
    encoding: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Text fast path: ``np.loadtxt(dtype=str)`` reads every cell as it is."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # loadtxt warns on empty input
            values = np.loadtxt(
                source,
                delimiter=delimiter,
                skiprows=skip_header,
                usecols=usecols,
                dtype=str,
                ndmin=2,
                comments=comments,
                encoding=encoding,
            )
    except ValueError:
        return None

    values = np.char.strip(values)
    missing = np.isin(values, [m.strip() for m in missing_values])  # (n_rows, n_cols) bool

    return values, missing
