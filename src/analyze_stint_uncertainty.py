import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/stint_dataset.csv"

TARGET = "SmoothedSlope"

NUMERIC_FEATURES = [
    "StartLap",
    "StartTyreLife",
]

CATEGORICAL_FEATURES = [
    "Compound",
]

FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ============================================================
# LOAD DATA
# ============================================================

print(f"Loading dataset: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)


# ============================================================
# AVAILABLE RACES
# ============================================================

races = (
    df[
        [
            "Year",
            "GrandPrix",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "Year",
            "GrandPrix",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# LEAVE-ONE-RACE-OUT PREDICTIONS
# ============================================================

prediction_rows = []


for _, race in races.iterrows():

    test_year = int(
        race["Year"]
    )

    test_race = race[
        "GrandPrix"
    ]


    test_mask = (
        (df["Year"] == test_year)
        & (df["GrandPrix"] == test_race)
    )


    train_df = df[
        ~test_mask
    ].copy()


    test_df = df[
        test_mask
    ].copy()


    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "compound",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=3,
                    min_samples_leaf=3,
                    random_state=42,
                ),
            ),
        ]
    )


    model.fit(
        train_df[FEATURES],
        train_df[TARGET]
    )


    predictions = model.predict(
        test_df[FEATURES]
    )


    for index, prediction in zip(
        test_df.index,
        predictions
    ):

        actual = df.loc[
            index,
            TARGET
        ]

        prediction_rows.append({
            "Year": df.loc[index, "Year"],
            "GrandPrix": df.loc[index, "GrandPrix"],
            "Driver": df.loc[index, "Driver"],
            "Compound": df.loc[index, "Compound"],
            "ActualSlope": actual,
            "PredictedSlope": prediction,
            "Residual": actual - prediction,
            "AbsoluteError": abs(
                actual - prediction
            ),
        })


# ============================================================
# CREATE RESULT DATASET
# ============================================================

results = pd.DataFrame(
    prediction_rows
)


# ============================================================
# OVERALL UNCERTAINTY
# ============================================================

print("\n========================================")
print("RANDOM FOREST UNCERTAINTY")
print("========================================")

print(
    f"Predictions: "
    f"{len(results)}"
)

print(
    f"Mean residual: "
    f"{results['Residual'].mean():+.4f} s/lap"
)

print(
    f"Residual standard deviation: "
    f"{results['Residual'].std():.4f} s/lap"
)

print(
    f"Mean absolute error: "
    f"{results['AbsoluteError'].mean():.4f} s/lap"
)

print(
    f"Median absolute error: "
    f"{results['AbsoluteError'].median():.4f} s/lap"
)

print(
    f"90th percentile absolute error: "
    f"{results['AbsoluteError'].quantile(0.90):.4f} s/lap"
)


# ============================================================
# UNCERTAINTY BY COMPOUND
# ============================================================

print("\n========================================")
print("UNCERTAINTY BY COMPOUND")
print("========================================")

compound_summary = (
    results.groupby("Compound")
    .agg(
        Stints=("Residual", "size"),
        MeanResidual=("Residual", "mean"),
        ResidualStd=("Residual", "std"),
        MAE=("AbsoluteError", "mean"),
        MedianAE=("AbsoluteError", "median"),
    )
    .reset_index()
)


print(
    compound_summary.to_string(
        index=False,
        formatters={
            "MeanResidual": (
                lambda x: f"{x:+.4f}"
            ),
            "ResidualStd": (
                lambda x: f"{x:.4f}"
            ),
            "MAE": (
                lambda x: f"{x:.4f}"
            ),
            "MedianAE": (
                lambda x: f"{x:.4f}"
            ),
        }
    )
)


# ============================================================
# LARGEST ERRORS
# ============================================================

print("\n========================================")
print("10 LARGEST PREDICTION ERRORS")
print("========================================")

print(
    results.sort_values(
        "AbsoluteError",
        ascending=False
    )
    .head(10)
    .to_string(
        index=False
    )
)