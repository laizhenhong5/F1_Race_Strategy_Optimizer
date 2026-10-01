import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# SETTINGS
# ============================================================

LINEAR_FILE = "data/stint_dataset.csv"
QUADRATIC_FILE = "data/quadratic_stint_dataset.csv"

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

LINEAR_TARGET = "SmoothedSlope"

QUADRATIC_LINEAR_TARGET = "QuadraticLinearTerm"
CURVATURE_TARGET = "Curvature"


# ============================================================
# LOAD DATA
# ============================================================

print(f"Loading: {LINEAR_FILE}")
linear_df = pd.read_csv(LINEAR_FILE)

print(f"Loading: {QUADRATIC_FILE}")
quadratic_df = pd.read_csv(QUADRATIC_FILE)


# ============================================================
# MERGE MATCHING STINTS
# ============================================================

KEYS = [
    "Year",
    "GrandPrix",
    "Driver",
    "Stint",
    "Compound",
]


linear_columns = (
    KEYS
    + [
        "SmoothedSlope",
    ]
)


df = quadratic_df.merge(
    linear_df[
        linear_columns
    ],
    on=KEYS,
    how="inner",
)


print("\n========================================")
print("DATASET")
print("========================================")

print(f"Matched stints: {len(df)}")

print(
    f"Races: "
    f"{df[['Year', 'GrandPrix']].drop_duplicates().shape[0]}"
)

print(
    f"Drivers: "
    f"{df['Driver'].nunique()}"
)


# ============================================================
# RANDOM FOREST FACTORY
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


    return Pipeline(
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
# CURVE ERROR
# ============================================================

def curve_mae(
    actual_b,
    actual_c,
    predicted_b,
    predicted_c,
    max_age,
):

    ages = np.arange(
        0,
        int(round(max_age)) + 1,
    )


    actual_curve = (
        actual_b * ages
        + actual_c * ages ** 2
    )


    predicted_curve = (
        predicted_b * ages
        + predicted_c * ages ** 2
    )


    return np.mean(
        np.abs(
            actual_curve
            - predicted_curve
        )
    )


# ============================================================
# RACES
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


results = []

overall_linear_baseline_errors = []
overall_linear_rf_errors = []

overall_quadratic_baseline_errors = []
overall_quadratic_rf_errors = []


# ============================================================
# LEAVE-ONE-RACE-OUT
# ============================================================

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
    # LINEAR BASELINE
    # ========================================================

    global_slope_median = (
        train_df[
            LINEAR_TARGET
        ].median()
    )


    compound_slope_medians = (
        train_df.groupby(
            "Compound"
        )[LINEAR_TARGET]
        .median()
        .to_dict()
    )


    linear_baseline_predictions = (
        test_df["Compound"]
        .map(
            compound_slope_medians
        )
        .fillna(
            global_slope_median
        )
        .values
    )


    # ========================================================
    # LINEAR RANDOM FOREST
    # ========================================================

    linear_rf = create_model()

    linear_rf.fit(
        train_df[FEATURES],
        train_df[LINEAR_TARGET],
    )


    linear_rf_predictions = (
        linear_rf.predict(
            test_df[FEATURES]
        )
    )


    # ========================================================
    # QUADRATIC BASELINE
    # ========================================================

    global_b_median = (
        train_df[
            QUADRATIC_LINEAR_TARGET
        ].median()
    )

    global_c_median = (
        train_df[
            CURVATURE_TARGET
        ].median()
    )


    compound_b_medians = (
        train_df.groupby(
            "Compound"
        )[QUADRATIC_LINEAR_TARGET]
        .median()
        .to_dict()
    )


    compound_c_medians = (
        train_df.groupby(
            "Compound"
        )[CURVATURE_TARGET]
        .median()
        .to_dict()
    )


    quadratic_baseline_b = (
        test_df["Compound"]
        .map(
            compound_b_medians
        )
        .fillna(
            global_b_median
        )
        .values
    )


    quadratic_baseline_c = (
        test_df["Compound"]
        .map(
            compound_c_medians
        )
        .fillna(
            global_c_median
        )
        .values
    )


    # ========================================================
    # QUADRATIC RANDOM FORESTS
    # ========================================================

    quadratic_b_rf = create_model()

    quadratic_b_rf.fit(
        train_df[FEATURES],
        train_df[
            QUADRATIC_LINEAR_TARGET
        ],
    )


    quadratic_rf_b = (
        quadratic_b_rf.predict(
            test_df[FEATURES]
        )
    )


    quadratic_c_rf = create_model()

    quadratic_c_rf.fit(
        train_df[FEATURES],
        train_df[
            CURVATURE_TARGET
        ],
    )


    quadratic_rf_c = (
        quadratic_c_rf.predict(
            test_df[FEATURES]
        )
    )


    # ========================================================
    # COMPARE ALL FOUR CURVES
    # ========================================================

    race_linear_baseline_errors = []
    race_linear_rf_errors = []

    race_quadratic_baseline_errors = []
    race_quadratic_rf_errors = []


    for i in range(
        len(test_df)
    ):

        actual_b = (
            test_df[
                QUADRATIC_LINEAR_TARGET
            ].iloc[i]
        )

        actual_c = (
            test_df[
                CURVATURE_TARGET
            ].iloc[i]
        )

        max_age = (
            test_df[
                "MaxAgeFromStart"
            ].iloc[i]
        )


        # Linear curves have c = 0.
        linear_baseline_error = curve_mae(
            actual_b,
            actual_c,
            linear_baseline_predictions[i],
            0.0,
            max_age,
        )


        linear_rf_error = curve_mae(
            actual_b,
            actual_c,
            linear_rf_predictions[i],
            0.0,
            max_age,
        )


        quadratic_baseline_error = curve_mae(
            actual_b,
            actual_c,
            quadratic_baseline_b[i],
            quadratic_baseline_c[i],
            max_age,
        )


        quadratic_rf_error = curve_mae(
            actual_b,
            actual_c,
            quadratic_rf_b[i],
            quadratic_rf_c[i],
            max_age,
        )


        race_linear_baseline_errors.append(
            linear_baseline_error
        )

        race_linear_rf_errors.append(
            linear_rf_error
        )

        race_quadratic_baseline_errors.append(
            quadratic_baseline_error
        )

        race_quadratic_rf_errors.append(
            quadratic_rf_error
        )


    # ========================================================
    # RACE SUMMARY
    # ========================================================

    linear_baseline_mae = np.mean(
        race_linear_baseline_errors
    )

    linear_rf_mae = np.mean(
        race_linear_rf_errors
    )

    quadratic_baseline_mae = np.mean(
        race_quadratic_baseline_errors
    )

    quadratic_rf_mae = np.mean(
        race_quadratic_rf_errors
    )


    results.append(
        {
            "Year": test_year,
            "Race": test_race,
            "Stints": len(test_df),

            "LinearBaseline": (
                linear_baseline_mae
            ),

            "LinearRF": (
                linear_rf_mae
            ),

            "QuadraticBaseline": (
                quadratic_baseline_mae
            ),

            "QuadraticRF": (
                quadratic_rf_mae
            ),
        }
    )


    overall_linear_baseline_errors.extend(
        race_linear_baseline_errors
    )

    overall_linear_rf_errors.extend(
        race_linear_rf_errors
    )

    overall_quadratic_baseline_errors.extend(
        race_quadratic_baseline_errors
    )

    overall_quadratic_rf_errors.extend(
        race_quadratic_rf_errors
    )


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n========================================")
print("LEAVE-ONE-RACE-OUT CURVE COMPARISON")
print("========================================")


print(
    results_df.to_string(
        index=False,
        formatters={
            "LinearBaseline": (
                lambda x: f"{x:.3f}"
            ),
            "LinearRF": (
                lambda x: f"{x:.3f}"
            ),
            "QuadraticBaseline": (
                lambda x: f"{x:.3f}"
            ),
            "QuadraticRF": (
                lambda x: f"{x:.3f}"
            ),
        }
    )
)


# ============================================================
# OVERALL
# ============================================================

print("\n========================================")
print("OVERALL CURVE MAE")
print("========================================")


print(
    f"Linear compound baseline:    "
    f"{np.mean(overall_linear_baseline_errors):.3f} s"
)

print(
    f"Linear Random Forest:        "
    f"{np.mean(overall_linear_rf_errors):.3f} s"
)

print(
    f"Quadratic compound baseline: "
    f"{np.mean(overall_quadratic_baseline_errors):.3f} s"
)

print(
    f"Quadratic Random Forest:     "
    f"{np.mean(overall_quadratic_rf_errors):.3f} s"
)