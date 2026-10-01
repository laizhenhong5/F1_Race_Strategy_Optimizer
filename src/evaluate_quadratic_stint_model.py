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

INPUT_FILE = "data/quadratic_stint_dataset.csv"

FEATURES = [
    "StartLap",
    "StartTyreLife",
    "Compound",
]

NUMERIC_FEATURES = [
    "StartLap",
    "StartTyreLife",
]

CATEGORICAL_FEATURES = [
    "Compound",
]

LINEAR_TARGET = "QuadraticLinearTerm"
CURVATURE_TARGET = "Curvature"


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
    "Races: "
    f"{df[['Year', 'GrandPrix']].drop_duplicates().shape[0]}"
)

print(
    f"Drivers: {df['Driver'].nunique()}"
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
# CREATE RANDOM FOREST MODEL
# ============================================================

def create_model():

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

    return model


# ============================================================
# CALCULATE CURVE ERROR
# ============================================================

def calculate_curve_mae(
    actual_linear,
    actual_curvature,
    predicted_linear,
    predicted_curvature,
    max_age,
):

    ages = np.arange(
        0,
        int(round(max_age)) + 1,
    )


    actual_curve = (
        actual_linear * ages
        + actual_curvature * (ages ** 2)
    )


    predicted_curve = (
        predicted_linear * ages
        + predicted_curvature * (ages ** 2)
    )


    return np.mean(
        np.abs(
            actual_curve
            - predicted_curve
        )
    )


# ============================================================
# LEAVE-ONE-RACE-OUT
# ============================================================

results = []

all_actual_linear = []
all_rf_linear = []
all_baseline_linear = []

all_actual_curvature = []
all_rf_curvature = []
all_baseline_curvature = []

all_rf_curve_errors = []
all_baseline_curve_errors = []


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


    # ========================================================
    # COMPOUND-MEDIAN BASELINE
    # ========================================================

    global_linear_median = (
        train_df[
            LINEAR_TARGET
        ].median()
    )


    global_curvature_median = (
        train_df[
            CURVATURE_TARGET
        ].median()
    )


    compound_linear_medians = (
        train_df.groupby(
            "Compound"
        )[LINEAR_TARGET]
        .median()
        .to_dict()
    )


    compound_curvature_medians = (
        train_df.groupby(
            "Compound"
        )[CURVATURE_TARGET]
        .median()
        .to_dict()
    )


    baseline_linear = (
        test_df["Compound"]
        .map(
            compound_linear_medians
        )
        .fillna(
            global_linear_median
        )
        .values
    )


    baseline_curvature = (
        test_df["Compound"]
        .map(
            compound_curvature_medians
        )
        .fillna(
            global_curvature_median
        )
        .values
    )


    # ========================================================
    # RANDOM FOREST: LINEAR TERM
    # ========================================================

    linear_model = create_model()

    linear_model.fit(
        train_df[FEATURES],
        train_df[LINEAR_TARGET],
    )


    rf_linear = linear_model.predict(
        test_df[FEATURES]
    )


    # ========================================================
    # RANDOM FOREST: CURVATURE
    # ========================================================

    curvature_model = create_model()

    curvature_model.fit(
        train_df[FEATURES],
        train_df[CURVATURE_TARGET],
    )


    rf_curvature = (
        curvature_model.predict(
            test_df[FEATURES]
        )
    )


    # ========================================================
    # COEFFICIENT ERRORS
    # ========================================================

    actual_linear = (
        test_df[
            LINEAR_TARGET
        ].values
    )


    actual_curvature = (
        test_df[
            CURVATURE_TARGET
        ].values
    )


    baseline_linear_mae = (
        mean_absolute_error(
            actual_linear,
            baseline_linear,
        )
    )


    rf_linear_mae = (
        mean_absolute_error(
            actual_linear,
            rf_linear,
        )
    )


    baseline_curvature_mae = (
        mean_absolute_error(
            actual_curvature,
            baseline_curvature,
        )
    )


    rf_curvature_mae = (
        mean_absolute_error(
            actual_curvature,
            rf_curvature,
        )
    )


    # ========================================================
    # CURVE ERRORS
    # ========================================================

    race_baseline_curve_errors = []
    race_rf_curve_errors = []


    for i in range(
        len(test_df)
    ):

        max_age = (
            test_df[
                "MaxAgeFromStart"
            ].iloc[i]
        )


        baseline_curve_error = (
            calculate_curve_mae(
                actual_linear[i],
                actual_curvature[i],
                baseline_linear[i],
                baseline_curvature[i],
                max_age,
            )
        )


        rf_curve_error = (
            calculate_curve_mae(
                actual_linear[i],
                actual_curvature[i],
                rf_linear[i],
                rf_curvature[i],
                max_age,
            )
        )


        race_baseline_curve_errors.append(
            baseline_curve_error
        )

        race_rf_curve_errors.append(
            rf_curve_error
        )


    baseline_curve_mae = np.mean(
        race_baseline_curve_errors
    )


    rf_curve_mae = np.mean(
        race_rf_curve_errors
    )


    # ========================================================
    # PRINT RACE RESULT
    # ========================================================

    print("\n========================================")
    print(
        f"TESTING: "
        f"{test_year} {test_race}"
    )
    print("========================================")

    print(
        f"Testing stints: "
        f"{len(test_df)}"
    )

    print(
        f"Baseline curve MAE: "
        f"{baseline_curve_mae:.3f} s"
    )

    print(
        f"Random Forest curve MAE: "
        f"{rf_curve_mae:.3f} s"
    )

    print(
        f"Difference vs baseline: "
        f"{baseline_curve_mae - rf_curve_mae:+.3f} s"
    )


    results.append(
        {
            "TestYear": test_year,
            "TestRace": test_race,
            "TestingStints": len(test_df),

            "BaselineLinearMAE": (
                baseline_linear_mae
            ),

            "RFLinearMAE": (
                rf_linear_mae
            ),

            "BaselineCurvatureMAE": (
                baseline_curvature_mae
            ),

            "RFCurvatureMAE": (
                rf_curvature_mae
            ),

            "BaselineCurveMAE": (
                baseline_curve_mae
            ),

            "RFCurveMAE": (
                rf_curve_mae
            ),

            "CurveImprovement": (
                baseline_curve_mae
                - rf_curve_mae
            ),
        }
    )


    # ========================================================
    # STORE OVERALL RESULTS
    # ========================================================

    all_actual_linear.extend(
        actual_linear
    )

    all_rf_linear.extend(
        rf_linear
    )

    all_baseline_linear.extend(
        baseline_linear
    )


    all_actual_curvature.extend(
        actual_curvature
    )

    all_rf_curvature.extend(
        rf_curvature
    )

    all_baseline_curvature.extend(
        baseline_curvature
    )


    all_rf_curve_errors.extend(
        race_rf_curve_errors
    )

    all_baseline_curve_errors.extend(
        race_baseline_curve_errors
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
    results_df[
        [
            "TestYear",
            "TestRace",
            "TestingStints",
            "BaselineCurveMAE",
            "RFCurveMAE",
            "CurveImprovement",
        ]
    ]
    .to_string(
        index=False,
        formatters={
            "BaselineCurveMAE": (
                lambda x: f"{x:.3f}"
            ),
            "RFCurveMAE": (
                lambda x: f"{x:.3f}"
            ),
            "CurveImprovement": (
                lambda x: f"{x:+.3f}"
            ),
        }
    )
)


# ============================================================
# OVERALL RESULT
# ============================================================

overall_baseline_linear_mae = (
    mean_absolute_error(
        all_actual_linear,
        all_baseline_linear,
    )
)


overall_rf_linear_mae = (
    mean_absolute_error(
        all_actual_linear,
        all_rf_linear,
    )
)


overall_baseline_curvature_mae = (
    mean_absolute_error(
        all_actual_curvature,
        all_baseline_curvature,
    )
)


overall_rf_curvature_mae = (
    mean_absolute_error(
        all_actual_curvature,
        all_rf_curvature,
    )
)


overall_baseline_curve_mae = (
    np.mean(
        all_baseline_curve_errors
    )
)


overall_rf_curve_mae = (
    np.mean(
        all_rf_curve_errors
    )
)


print("\n========================================")
print("OVERALL RESULT")
print("========================================")

print(
    f"Total evaluated stints: "
    f"{len(df)}"
)


print("\nLinear coefficient:")

print(
    f"  Baseline MAE: "
    f"{overall_baseline_linear_mae:.5f}"
)

print(
    f"  Random Forest MAE: "
    f"{overall_rf_linear_mae:.5f}"
)


print("\nCurvature coefficient:")

print(
    f"  Baseline MAE: "
    f"{overall_baseline_curvature_mae:.5f}"
)

print(
    f"  Random Forest MAE: "
    f"{overall_rf_curvature_mae:.5f}"
)


print("\nReconstructed stint curve:")

print(
    f"  Baseline curve MAE: "
    f"{overall_baseline_curve_mae:.3f} s"
)

print(
    f"  Random Forest curve MAE: "
    f"{overall_rf_curve_mae:.3f} s"
)

print(
    f"  Difference vs baseline: "
    f"{overall_baseline_curve_mae - overall_rf_curve_mae:+.3f} s"
)