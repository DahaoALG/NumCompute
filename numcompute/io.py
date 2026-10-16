"""Simple CSV loading with NumPy."""

import numpy as np


def load_csv(path, delimiter=",", skip_header=0, missing="fill"):
    """Read CSV data, keeping text and representing missing values with np.nan.

    Parameters
    ----------
    path : str or Path
        CSV file path.
    delimiter : str
        Field separator, e.g. ',' or '\t'.
    skip_header : int
        Number of header lines to skip.
    missing : {'fill', 'skip'}
        Fill missing cells with np.nan, or skip rows containing missing cells.

    Returns
    -------
    numpy.ndarray
        Object array of shape (rows, columns), containing strings and np.nan.
        Convert numeric columns with .astype(float); np.nan stays unchanged.

    Raises
    ------
    ValueError
        Invalid missing policy or inconsistent column counts.

    """
    if missing not in ("fill", "skip"):
        raise ValueError("missing must be 'fill' or 'skip'")

    data = np.genfromtxt(
        path, delimiter=delimiter, skip_header=skip_header,
        dtype=str, autostrip=True, ndmin=2,
    )
    if data.size == 0: # Empty file
        return np.empty((0, 0), dtype=object)

    missing_mask = np.isin(data, ["", "NA", "N/A", "NaN", "nan", "NULL", "null"])
    data = data.astype(object)
    if missing == "skip":
        # Check whether each row contains a missing value.
        has_missing = missing_mask.any(axis=1)

        # Keep rows without missing values.
        keep_rows = (has_missing == False)
        return data[keep_rows]

    data[missing_mask] = np.nan
    return data
