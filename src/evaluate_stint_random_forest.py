import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


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


print("\n========================================")
print("DATASET")
print("========================================")

print(f"Stints: {len(df)}")

print(
    f"Races: "
    f"{df[['Year', 'GrandPrix']].drop_duplicates().shape[0]}"
)

print(
    f"Drivers: "
    f"{df['Driver'].nunique()}"
)


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
# PREPROCESSING
# ============================================================

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


# ============================================================
# RANDOM FOREST
# ============================================================
#
# Intentionally conservative settings because the dataset
# currently contains only 38 stints.
# ============================================================

random_forest = Pipeline(
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


# ============================================================
# LEAVE-ONE-RACE-OUT
# ============================================================

results = []

total_baseline_error = 0.0
total_rf_error = 0.0
total_rows = 0


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


    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        TARGET
    ]


    X_test = test_df[
        FEATURES
    ]

    y_test = test_df[
        TARGET
    ].values


    # ========================================================
    # COMPOUND-MEDIAN BASELINE
    # ========================================================

    global_median = (
        train_df[
            TARGET
        ].median()
    )


    compound_medians = (
        train_df.groupby(
            "Compound"
        )[TARGET]
        .median()
        .to_dict()
    )


    baseline_predictions = (
        test_df["Compound"]
        .map(compound_medians)
        .fillna(global_median)
        .values
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    random_forest.fit(
        X_train,
        y_train
    )


    rf_predictions = (
        random_forest.predict(
            X_test
        )
    )


    # ========================================================
    # METRICS
    # ========================================================

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )


    rf_mae = mean_absolute_error(
        y_test,
        rf_predictions
    )


    improvement = (
        baseline_mae
        - rf_mae
    )


    print("\n========================================")
    print(
        f"TESTING: "
        f"{test_year} {test_race}"
    )
    print("========================================")

    print(
        f"Training stints: "
        f"{len(train_df)}"
    )

    print(
        f"Testing stints: "
        f"{len(test_df)}"
    )

    print(
        f"Compound baseline MAE: "
        f"{baseline_mae:.4f} s/lap"
    )

    print(
        f"Random Forest MAE: "
        f"{rf_mae:.4f} s/lap"
    )

    print(
        f"Difference vs baseline: "
        f"{improvement:+.4f} s/lap"
    )


    results.append({
        "TestYear": test_year,
        "TestRace": test_race,
        "TestingStints": len(test_df),
        "BaselineMAE": baseline_mae,
        "RandomForestMAE": rf_mae,
        "Improvement": improvement,
    })


    total_baseline_error += (
        np.abs(
            y_test
            - baseline_predictions
        ).sum()
    )


    total_rf_error += (
        np.abs(
            y_test
            - rf_predictions
        ).sum()
    )


    total_rows += len(
        test_df
    )


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n========================================")
print("LEAVE-ONE-RACE-OUT SUMMARY")
print("========================================")

print(
    results_df.to_string(
        index=False,
        formatters={
            "BaselineMAE": (
                lambda x: f"{x:.4f}"
            ),
            "RandomForestMAE": (
                lambda x: f"{x:.4f}"
            ),
            "Improvement": (
                lambda x: f"{x:+.4f}"
            ),
        }
    )
)


# ============================================================
# OVERALL RESULT
# ============================================================

overall_baseline_mae = (
    total_baseline_error
    / total_rows
)


overall_rf_mae = (
    total_rf_error
    / total_rows
)


print("\n========================================")
print("OVERALL RESULT")
print("========================================")

print(
    f"Total evaluated stints: "
    f"{total_rows}"
)

print(
    f"Compound baseline MAE: "
    f"{overall_baseline_mae:.4f} s/lap"
)

print(
    f"Random Forest MAE: "
    f"{overall_rf_mae:.4f} s/lap"
)

print(
    f"Difference vs baseline: "
    f"{overall_baseline_mae - overall_rf_mae:+.4f} s/lap"
)