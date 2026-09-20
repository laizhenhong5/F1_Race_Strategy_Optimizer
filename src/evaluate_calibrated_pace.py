import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILES = [
    # Russell
    "data/normal_racing_laps_las_vegas_2024.csv",
    "data/normal_racing_laps_bahrain_2025.csv",
    "data/normal_racing_laps_japan_2025.csv",
    "data/normal_racing_laps_miami_2025_repaired.csv",
    "data/normal_racing_laps_emilia_romagna_2025.csv",
    "data/normal_racing_laps_barcelona_catalunya_2025.csv",
    "data/normal_racing_laps_canada_2025.csv",
    "data/normal_racing_laps_singapore_2025.csv",

    # Antonelli
    "data/normal_racing_laps_antonelli_bahrain_2025.csv",
    "data/normal_racing_laps_antonelli_japan_2025.csv",
    "data/normal_racing_laps_antonelli_canada_2025.csv",
]

CALIBRATION_LAPS = 5


# ============================================================
# LOAD DATA
# ============================================================

datasets = []

for file in INPUT_FILES:

    print(f"Loading: {file}")

    dataset = pd.read_csv(file)

    datasets.append(dataset)


df = pd.concat(
    datasets,
    ignore_index=True
)


# Ensure important columns are numeric.
numeric_columns = [
    "LapNumber",
    "LapTimeSeconds",
    "TyreLife",
    "Stint",
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df = df.sort_values(
    [
        "Year",
        "GrandPrix",
        "Driver",
        "LapNumber",
    ]
).reset_index(drop=True)


print("\n========================================")
print("DATASET")
print("========================================")

print(
    f"Total normal-racing laps: "
    f"{len(df)}"
)

print(
    f"Number of races: "
    f"{df[['Year', 'GrandPrix']].drop_duplicates().shape[0]}"
)

print(
    f"Number of drivers: "
    f"{df['Driver'].nunique()}"
)


# ============================================================
# CREATE RACE-SPECIFIC CALIBRATION BASELINE
# ============================================================
#
# For each driver/race:
#
# Use ONLY the first few clean racing laps to estimate
# that driver's basic pace for that particular race.
#
# This avoids using the future of the race.
# ============================================================

df["CalibrationBaseline"] = np.nan

df["IsCalibrationLap"] = False


group_columns = [
    "Year",
    "GrandPrix",
    "Driver",
]


for _, indices in df.groupby(
    group_columns
).groups.items():

    group_df = (
        df.loc[indices]
        .sort_values("LapNumber")
    )

    calibration_indices = (
        group_df.index[
            :CALIBRATION_LAPS
        ]
    )

    calibration_baseline = (
        df.loc[
            calibration_indices,
            "LapTimeSeconds"
        ]
        .median()
    )

    df.loc[
        indices,
        "CalibrationBaseline"
    ] = calibration_baseline

    df.loc[
        calibration_indices,
        "IsCalibrationLap"
    ] = True


# Target:
#
# How much faster/slower is the current lap compared
# with the early-race calibration pace?
df["RelativePace"] = (
    df["LapTimeSeconds"]
    - df["CalibrationBaseline"]
)


# ============================================================
# CREATE PREVIOUS-LAP BASELINE
# ============================================================
#
# Previous-lap comparison is only valid when:
#
# - same race
# - same driver
# - same stint
# - immediately consecutive lap
# ============================================================

previous_lap_groups = [
    "Year",
    "GrandPrix",
    "Driver",
    "Stint",
]


df["PreviousLapNumber"] = (
    df.groupby(
        previous_lap_groups
    )["LapNumber"]
    .shift(1)
)


df["PreviousLapTime"] = (
    df.groupby(
        previous_lap_groups
    )["LapTimeSeconds"]
    .shift(1)
)


df["HasConsecutivePreviousLap"] = (
    (
        df["LapNumber"]
        - df["PreviousLapNumber"]
    )
    == 1
)


# ============================================================
# ENCODE COMPOUNDS
# ============================================================

compound_dummies = pd.get_dummies(
    df["Compound"],
    prefix="Compound",
    dtype=int
)


df = pd.concat(
    [
        df,
        compound_dummies,
    ],
    axis=1
)


# HARD is treated as the reference compound.
for column in [
    "Compound_MEDIUM",
    "Compound_SOFT",
]:

    if column not in df.columns:
        df[column] = 0


FEATURES = [
    "LapNumber",
    "TyreLife",
    "Compound_MEDIUM",
    "Compound_SOFT",
]


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


print("\n========================================")
print("AVAILABLE RACES")
print("========================================")

print(
    races.to_string(
        index=False
    )
)


# ============================================================
# LEAVE-ONE-RACE-OUT EVALUATION
# ============================================================

results = []

total_model_error = 0.0
total_baseline_error = 0.0
total_test_rows = 0


for _, race in races.iterrows():

    test_year = int(
        race["Year"]
    )

    test_race = (
        race["GrandPrix"]
    )


    test_race_mask = (
        (df["Year"] == test_year)
        & (df["GrandPrix"] == test_race)
    )


    # --------------------------------------------------------
    # TRAINING DATA
    # --------------------------------------------------------
    #
    # Completely exclude the held-out race.
    #
    # Calibration laps are also excluded because our model
    # is meant to predict behaviour AFTER calibration.
    # --------------------------------------------------------

    train_df = df[
        (~test_race_mask)
        & (~df["IsCalibrationLap"])
    ].copy()


    # --------------------------------------------------------
    # TEST DATA
    # --------------------------------------------------------
    #
    # Exclude the five laps used to calibrate this race.
    #
    # Also require a genuine consecutive previous lap so the
    # ML model and previous-lap baseline are compared on the
    # exact same observations.
    # --------------------------------------------------------

    test_df = df[
        test_race_mask
        & (~df["IsCalibrationLap"])
        & (df["HasConsecutivePreviousLap"])
    ].copy()


    if len(test_df) == 0:
        continue


    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        "RelativePace"
    ]


    X_test = test_df[
        FEATURES
    ]


    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICT RELATIVE PACE
    # --------------------------------------------------------

    predicted_relative_pace = (
        model.predict(
            X_test
        )
    )


    # Reconstruct actual lap time.
    predicted_lap_time = (
        test_df[
            "CalibrationBaseline"
        ].values
        + predicted_relative_pace
    )


    actual_lap_time = (
        test_df[
            "LapTimeSeconds"
        ].values
    )


    # --------------------------------------------------------
    # PREVIOUS-LAP BASELINE
    # --------------------------------------------------------

    baseline_prediction = (
        test_df[
            "PreviousLapTime"
        ].values
    )


    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    model_mae = mean_absolute_error(
        actual_lap_time,
        predicted_lap_time
    )


    baseline_mae = mean_absolute_error(
        actual_lap_time,
        baseline_prediction
    )


    improvement = (
        baseline_mae
        - model_mae
    )


    print("\n========================================")
    print(
        f"TESTING: "
        f"{test_year} {test_race}"
    )
    print("========================================")

    print(
        f"Training rows: "
        f"{len(train_df)}"
    )

    print(
        f"Testing rows: "
        f"{len(test_df)}"
    )

    print(
        f"Calibrated pace model MAE: "
        f"{model_mae:.3f} s"
    )

    print(
        f"Previous-lap baseline MAE: "
        f"{baseline_mae:.3f} s"
    )

    print(
        f"Difference vs baseline: "
        f"{improvement:+.3f} s"
    )


    results.append({
        "TestYear": test_year,
        "TestRace": test_race,
        "TestingRows": len(test_df),
        "ModelMAE": model_mae,
        "BaselineMAE": baseline_mae,
        "Improvement": improvement,
    })


    total_model_error += (
        np.abs(
            actual_lap_time
            - predicted_lap_time
        ).sum()
    )

    total_baseline_error += (
        np.abs(
            actual_lap_time
            - baseline_prediction
        ).sum()
    )

    total_test_rows += (
        len(test_df)
    )

# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n========================================")
print("SUMMARY")
print("========================================")

print(
    results_df.to_string(
        index=False,
        formatters={
            "ModelMAE": (
                lambda x: f"{x:.3f}"
            ),
            "BaselineMAE": (
                lambda x: f"{x:.3f}"
            ),
            "Improvement": (
                lambda x: f"{x:+.3f}"
            ),
        }
    )
)

# ============================================================
# OVERALL RESULT
# ============================================================

overall_model_mae = (
    total_model_error
    / total_test_rows
)

overall_baseline_mae = (
    total_baseline_error
    / total_test_rows
)


print("\n========================================")
print("OVERALL RESULT")
print("========================================")

print(
    f"Total evaluated laps: "
    f"{total_test_rows}"
)

print(
    f"Calibrated pace model MAE: "
    f"{overall_model_mae:.3f} s"
)

print(
    f"Previous-lap baseline MAE: "
    f"{overall_baseline_mae:.3f} s"
)

print(
    f"Difference vs baseline: "
    f"{overall_baseline_mae - overall_model_mae:+.3f} s"
)