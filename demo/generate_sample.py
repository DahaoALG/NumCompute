"""Regenerate sample.csv: 1,200 synthetic records, fixed seed 42.

Run from any directory: python path/to/demo/generate_sample.py
Only Python and NumPy are used. The first five records preserve the scaffold's
example. This synthetic dataset is for software demonstration, not inference
about real people or cities.
"""

from pathlib import Path

import numpy as np


def main():
    """Write deterministic sample data next to this script."""
    rng = np.random.default_rng(42)
    n_rows = 1200
    ages = rng.integers(18, 71, size=n_rows).astype(str)
    incomes = np.char.mod("%.2f", np.round(rng.uniform(25000, 150000, size=n_rows), 2))
    cities = rng.choice(["Adelaide", "Melbourne", "Sydney", "Brisbane", "Perth", "Hobart"], n_rows)
    data = np.column_stack((ages, incomes, cities)).astype("U32")
    data[:5] = [
        ["20", "30000", "Adelaide"],
        ["", "40000", "Melbourne"],
        ["40", "", "Adelaide"],
        ["30", "50000", "Sydney"],
        ["25", "35000", ""],
    ]
    # Exact column missing counts, including one in each of the initial rows:
    # age: 72 (6%), income: 96 (8%), city: 60 (5%).
    for column, additional_missing in enumerate((71, 95, 59)):
        indices = rng.choice(np.arange(5, n_rows), size=additional_missing, replace=False)
        tokens = np.array(["", "NA", "NaN"]) if column < 2 else np.array(["", "NULL", "N/A"])
        data[indices, column] = np.resize(tokens, additional_missing)
    destination = Path(__file__).resolve().parent / "data" / "sample.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, data, fmt="%s", delimiter=",", header="age,income,city", comments="", encoding="utf-8")
    print(f"Generated {n_rows} synthetic records: {destination}")


if __name__ == "__main__":
    main()
