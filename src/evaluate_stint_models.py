import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV
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
            StandardScaler(),
            NUMERIC_FEATURES,
        ),
        (
            "compound",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False,
            ),
            CATEGORICAL_FEATURES,
        ),
    ]
)


# ============================================================
# MODELS
# ============================================================

linear_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            LinearRegression(),
        ),
    ]
)


ridge_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            RidgeCV(
                alphas=[
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                    100.0,
                ]
            ),
        ),
    ]
)


FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ============================================================
# LEAVE-ONE-RACE-OUT
# ============================================================

results = []

total_baseline_error = 0.0
total_linear_error = 0.0
total_ridge_error = 0.0
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
    # LINEAR REGRESSION
    # ========================================================

    linear_model.fit(
        X_train,
        y_train
    )


    linear_predictions = (
        linear_model.predict(
            X_test
        )
    )


    # ========================================================
    # RIDGE REGRESSION
    # ========================================================

    ridge_model.fit(
        X_train,
        y_train
    )


    ridge_predictions = (
        ridge_model.predict(
            X_test
        )
    )


    selected_alpha = (
        ridge_model
        .named_steps["model"]
        .alpha_
    )


    # ========================================================
    # METRICS
    # ========================================================

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )


    linear_mae = mean_absolute_error(
        y_test,
        linear_predictions
    )


    ridge_mae = mean_absolute_error(
        y_test,
        ridge_predictions
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
        f"Linear regression MAE: "
        f"{linear_mae:.4f} s/lap"
    )

    print(
        f"Ridge regression MAE: "
        f"{ridge_mae:.4f} s/lap"
    )

    print(
        f"Ridge alpha: "
        f"{selected_alpha}"
    )


    results.append({
        "TestYear": test_year,
        "TestRace": test_race,
        "TestingStints": len(test_df),
        "BaselineMAE": baseline_mae,
        "LinearMAE": linear_mae,
        "RidgeMAE": ridge_mae,
        "RidgeAlpha": selected_alpha,
    })


    total_baseline_error += (
        np.abs(
            y_test
            - baseline_predictions
        ).sum()
    )


    total_linear_error += (
        np.abs(
            y_test
            - linear_predictions
        ).sum()
    )


    total_ridge_error += (
        np.abs(
            y_test
            - ridge_predictions
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
            "LinearMAE": (
                lambda x: f"{x:.4f}"
            ),
            "RidgeMAE": (
                lambda x: f"{x:.4f}"
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

overall_linear_mae = (
    total_linear_error
    / total_rows
)

overall_ridge_mae = (
    total_ridge_error
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
    f"Linear regression MAE: "
    f"{overall_linear_mae:.4f} s/lap"
)

print(
    f"Ridge regression MAE: "
    f"{overall_ridge_mae:.4f} s/lap"
)


print("\nDifference vs compound baseline:")

print(
    f"Linear: "
    f"{overall_baseline_mae - overall_linear_mae:+.4f} s/lap"
)

print(
    f"Ridge: "
    f"{overall_baseline_mae - overall_ridge_mae:+.4f} s/lap"
)