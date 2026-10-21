"""Generate 3,000 synthetic rows: age, income, city, city_predict.

Each city has 500 rows. Its age/income profile is entirely fictional and does
not describe real Australian cities. Overlapping distributions create a
classification exercise rather than a perfectly predictable mapping.

Age has 6% missing values; income has 8%. True city is blank in 80 rows;
city_predict stays complete. Income includes deliberate ties and outliers.
city_predict is a feature-based simulation, not a trained model. Its predictor
sees only the observable age/income values, with missing values median-filled.
Use age/income as X, city as y_true, and city_predict as y_pred for evaluation.
Exclude blank true cities when calculating classification metrics.

Run: python demo/generate_sample.py
Output: demo/data/sample.csv. Fixed seed 42 makes the file reproducible.
"""

from pathlib import Path

import numpy as np


N_ROWS = 3000
CITIES = np.array(["Adelaide", "Melbourne", "Sydney", "Brisbane", "Perth", "Hobart"])
AGE_CENTERS = np.array([30, 35, 40, 45, 50, 55])
INCOME_BASES = np.array([60000, 90000, 115000, 75000, 105000, 65000])
# Smaller standard deviations reduce overlap between cities,
# making city predictions easier on this synthetic dataset.

AGE_STD = 7
INCOME_STD = 8000

AGE_INCOME_EFFECT = 900

def simulate_city_predictions(ages, incomes):
    """Return one simulated city per row using only its age and income.

    Inputs are 1-D numeric arrays with NaNs. Missing features are replaced by
    their column medians from this generated dataset. The nearest fictional
    city profile wins; overlap and missing values naturally create errors.
    No true city labels are supplied to this function. This is a demo rule,
    not a fitted predictor or a held-out model-performance experiment.
    """
    ages = np.where(np.isnan(ages), np.nanmedian(ages), ages)
    incomes = np.where(np.isnan(incomes), np.nanmedian(incomes), incomes)

    # Compare every row against the six city profiles at once.
    age_difference = ages[:, None] - AGE_CENTERS
    expected_income = INCOME_BASES + AGE_INCOME_EFFECT * age_difference
    age_distance = age_difference / AGE_STD
    income_distance = (incomes[:, None] - expected_income) / INCOME_STD
    distances = age_distance ** 2 + income_distance ** 2

    nearest_city = np.argmin(distances, axis=1)
    return CITIES[nearest_city]

def generate_data(seed=42):
    """Return a (3000, 4) string array, ready to save as CSV."""
    rng = np.random.default_rng(seed)

    # Start with exactly 500 examples of each true city.
    city_indices = np.repeat(np.arange(CITIES.size), N_ROWS // CITIES.size)
    ages = rng.normal(AGE_CENTERS[city_indices], AGE_STD)
    ages = np.clip(np.rint(ages), 18, 70)
    incomes = (
        INCOME_BASES[city_indices]
        + AGE_INCOME_EFFECT * (ages - AGE_CENTERS[city_indices])
        + rng.normal(0, INCOME_STD, N_ROWS)
    )
    incomes = np.round(np.maximum(incomes, 20000), 2)

    # Shuffle so train/test slices contain a mixture of cities.
    order = rng.permutation(N_ROWS)
    ages = ages[order]
    incomes = incomes[order]
    cities = CITIES[city_indices[order]]

    # Add three extreme incomes per city to show scaling effects and a long tail.
    # A loop over six cities keeps this small data-generation step easy to read.
    special_income = np.zeros(N_ROWS, dtype=bool)
    for city, base in zip(CITIES, INCOME_BASES):
        rows = rng.choice(np.flatnonzero(cities == city), size=3, replace=False)
        incomes[rows] = base * 6
        special_income[rows] = True

    # Equal incomes create ties: average and dense ranking give different ranks.
    # Keep these rows separate from the outliers.
    for city, count, income in [("Sydney", 14, 115000), ("Melbourne", 21, 90000)]:
        available = np.flatnonzero((cities == city) & ~special_income)
        rows = rng.choice(available, size=count, replace=False)
        incomes[rows] = income
        special_income[rows] = True
    # Preserve the demonstration incomes when adding missing feature values.
    missing_age = rng.choice(N_ROWS, size=180, replace=False)
    available = np.flatnonzero(~special_income)
    missing_income = rng.choice(available, size=240, replace=False)
    ages[missing_age] = np.nan
    incomes[missing_income] = np.nan
    predictions = simulate_city_predictions(ages, incomes)

    # Blank cities demonstrate zero encoding and imputation with "Unknown".
    # Keep the labelled tie/outlier rows so their city groups remain visible.
    missing_city = rng.choice(available, size=80, replace=False)
    cities[missing_city] = ""
    # Too many blank true labels reduce the rows available for reliable metrics.
    # Exclude these rows from metrics; counting blanks as classes distorts scores.

    # Exercise every default CSV missing token in both numerical columns.
    age_text = np.char.mod("%.0f", ages)
    income_text = np.char.mod("%.2f", incomes)
    age_text = age_text.astype("U32")
    income_text = income_text.astype("U32")
    missing_tokens = ["", "NA", "NaN", "NULL", "N/A"]
    age_text[missing_age] = np.resize(missing_tokens, missing_age.size)
    income_text[missing_income] = np.resize(missing_tokens, missing_income.size)

    return np.column_stack((age_text, income_text, cities, predictions))

def main():
    """Write the generated dataset next to the demo notebook's data folder."""
    data = generate_data()
    destination = Path(__file__).resolve().parent / "data" / "sample.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        destination, data, fmt="%s", delimiter=",",
        header="age,income,city,city_predict", comments="", encoding="utf-8",
    )
    print(f"Generated {len(data)} rows: {destination}")
    print("Generated 500 rows per city before hiding 80 city labels")
    print("Missing values: age=180, income=240, city=80")
    labelled = data[:, 2] != ""
    accuracy = np.mean(data[labelled, 2] == data[labelled, 3])
    print(f"Simulated city accuracy (labelled rows only): {accuracy:.3f}")


if __name__ == "__main__":
    main()