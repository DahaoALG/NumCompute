"""Generate 3,000 synthetic rows: age, income, city, city_predict.

Each city has 500 rows. Its age/income profile is entirely fictional and does
not describe real Australian cities. Overlapping distributions create a
classification exercise rather than a perfectly predictable mapping.

Age has 6% missing values; income has 8%. Both city columns are complete.
city_predict is a feature-based simulation, not a trained model. Its predictor
sees only the observable age/income values, with missing values median-filled.
Use age/income as X, city as y_true, and city_predict as y_pred for evaluation.

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

    # Hide features before generating predictions; city labels stay complete.
    missing_age = rng.choice(N_ROWS, size=180, replace=False)
    missing_income = rng.choice(N_ROWS, size=240, replace=False)
    ages[missing_age] = np.nan
    incomes[missing_income] = np.nan
    predictions = simulate_city_predictions(ages, incomes)

    # Different missing tokens exercise the CSV reader and preprocessing.
    age_text = np.char.mod("%.0f", ages)
    income_text = np.char.mod("%.2f", incomes)
    age_text = age_text.astype("U32")
    income_text = income_text.astype("U32")
    age_text[missing_age] = np.resize(["", "NA", "NaN"], missing_age.size)
    income_text[missing_income] = np.resize(["", "NA", "NaN"], missing_income.size)

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
    print("True city counts: 500 per city")
    print("Missing values: age=180, income=240")
    print(f"Simulated city accuracy: {np.mean(data[:, 2] == data[:, 3]):.3f}")


if __name__ == "__main__":
    main()
