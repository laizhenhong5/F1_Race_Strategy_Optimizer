import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
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
    "data/normal_racing_laps_antonelli_barcelona_catalunya_2025.csv",
    "data/normal_racing_laps_antonelli_emilia_romagna_2025.csv",
    "data/normal_racing_laps_antonelli_miami_2025_repaired.csv",
    "data/normal_racing_laps_antonelli_singapore_2025.csv",
]

ROLLING_WINDOW = 3
MIN_STINT_LAPS = 5

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

df = (
    df.sort_values(
        [
            "Year",
            "GrandPrix",
            "Driver",
            "Stint",
            "LapNumber",
        ]
    )
    .reset_index(drop=True)
)


print("\n========================================")
print("DATASET")
print("========================================")

print(
    f"Normal-racing laps: "
    f"{len(df)}"
)

print(
    f"Drivers: "
    f"{df['Driver'].nunique()}"
)

print(
    f"Races: "
    f"{df[['Year', 'GrandPrix']].drop_duplicates().shape[0]}"
)

# ============================================================
# STINT GROUPS
# ============================================================

stint_groups = [
    "Year",
    "GrandPrix",
    "Driver",
    "Stint",
]


# ============================================================
# IDENTIFY CONSECUTIVE-LAP SEGMENTS
# ============================================================

df["LapDifference"] = (
    df.groupby(stint_groups)[
        "LapNumber"
    ]
    .diff()
)


df["NewSegment"] = (
    (
        df["LapDifference"].isna()
        | (df["LapDifference"] != 1)
    )
    .astype(int)
)

df["ConsecutiveSegment"] = (
    df.groupby(stint_groups)[
        "NewSegment"
    ]
    .cumsum()
)


# ============================================================
# CREATE SMOOTHED PACE TARGET
# ============================================================
# This is an OFFLINE training/evaluation target.
# center=True is acceptable here because SmoothedLapTime
# will NOT be used as an input feature during prediction.
# ============================================================

smoothing_groups = (
    stint_groups
    + ["ConsecutiveSegment"]
)

df["SmoothedLapTime"] = (
    df.groupby(smoothing_groups)[
        "LapTimeSeconds"
    ]
    .transform(
        lambda series:
        series.rolling(
            window=ROLLING_WINDOW,
            center=True,
            min_periods=1
        ).median()
    )
)

# ============================================================
# KEEP USABLE STINTS
# ============================================================

df["StintRows"] = (
    df.groupby(stint_groups)[
        "LapNumber"
    ]
    .transform("size")
)


df = df[
    df["StintRows"] >= MIN_STINT_LAPS
].copy()

print(
    f"Usable stints: "
    f"{df[stint_groups].drop_duplicates().shape[0]}"
)


# ============================================================
# CREATE STINT-EVOLUTION TARGET
# ============================================================

df["StartLapNumber"] = (
    df.groupby(stint_groups)[
        "LapNumber"
    ]
    .transform("first")
)


df["StartTyreLife"] = (
    df.groupby(stint_groups)[
        "TyreLife"
    ]
    .transform("first")
)


df["StintStartPace"] = (
    df.groupby(stint_groups)[
        "SmoothedLapTime"
    ]
    .transform("first")
)


df["LapsIntoStint"] = (
    df["LapNumber"]
    - df["StartLapNumber"]
)


# Positive value:
# pace has become slower than at stint start.
#
# Negative value:
# pace has become faster than at stint start.
df["StintEvolution"] = (
    df["SmoothedLapTime"]
    - df["StintStartPace"]
)


# ============================================================
# COMPOUND FEATURES
# ============================================================

df["IsMedium"] = (
    df["Compound"] == "MEDIUM"
).astype(int)


df["IsSoft"] = (
    df["Compound"] == "SOFT"
).astype(int)


# ============================================================
# INTERACTION FEATURES
# ============================================================
#
# These all become zero when LapsIntoStint == 0.
#
# That lets the model naturally anchor the beginning
# of every stint at zero pace evolution.
# ============================================================

df["Progress_x_StartLap"] = (
    df["LapsIntoStint"]
    * df["StartLapNumber"]
)


df["Progress_x_StartTyreLife"] = (
    df["LapsIntoStint"]
    * df["StartTyreLife"]
)


df["Progress_x_Medium"] = (
    df["LapsIntoStint"]
    * df["IsMedium"]
)


df["Progress_x_Soft"] = (
    df["LapsIntoStint"]
    * df["IsSoft"]
)


FEATURES = [
    "LapsIntoStint",
    "Progress_x_StartLap",
    "Progress_x_StartTyreLife",
    "Progress_x_Medium",
    "Progress_x_Soft",
]


TARGET = "StintEvolution"


# ============================================================
# REMOVE STINT-START ROWS FROM EVALUATION
# ============================================================
#
# At LapsIntoStint = 0, StintEvolution is automatically zero,
# which would make both models look artificially better.
# ============================================================

df = df[
    df["LapsIntoStint"] > 0
].copy()


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
# LEAVE-ONE-RACE-OUT
# ============================================================

results = []

total_model_error = 0.0
total_baseline_error = 0.0
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


    if len(test_df) == 0:
        continue


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
    ]


    # No intercept:
    #
    # At zero stint progress, predicted evolution
    # should also equal zero.
    model = LinearRegression(
        fit_intercept=False
    )


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    # ========================================================
    # ZERO-EVOLUTION BASELINE
    # ========================================================
    #
    # Baseline assumption:
    #
    # "The underlying stint pace never changes from
    #  its starting value."
    # ========================================================

    baseline_predictions = np.zeros(
        len(test_df)
    )


    model_mae = mean_absolute_error(
        y_test,
        predictions
    )


    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )


    model_r2 = r2_score(
        y_test,
        predictions
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
        f"Stint-evolution MAE: "
        f"{model_mae:.3f} s"
    )

    print(
        f"Zero-evolution baseline MAE: "
        f"{baseline_mae:.3f} s"
    )

    print(
        f"Difference vs baseline: "
        f"{improvement:+.3f} s"
    )

    print(
        f"R2: "
        f"{model_r2:.3f}"
    )


    results.append({
        "TestYear": test_year,
        "TestRace": test_race,
        "TestingRows": len(test_df),
        "ModelMAE": model_mae,
        "BaselineMAE": baseline_mae,
        "Improvement": improvement,
        "R2": model_r2,
    })


    total_model_error += (
        np.abs(
            y_test.values
            - predictions
        ).sum()
    )


    total_baseline_error += (
        np.abs(
            y_test.values
            - baseline_predictions
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
            "ModelMAE": (
                lambda x: f"{x:.3f}"
            ),
            "BaselineMAE": (
                lambda x: f"{x:.3f}"
            ),
            "Improvement": (
                lambda x: f"{x:+.3f}"
            ),
            "R2": (
                lambda x: f"{x:.3f}"
            ),
        }
    )
)


# ============================================================
# OVERALL RESULT
# ============================================================

overall_model_mae = (
    total_model_error
    / total_rows
)


overall_baseline_mae = (
    total_baseline_error
    / total_rows
)


print("\n========================================")
print("OVERALL RESULT")
print("========================================")

print(
    f"Total evaluated observations: "
    f"{total_rows}"
)

print(
    f"Stint-evolution model MAE: "
    f"{overall_model_mae:.3f} s"
)

print(
    f"Zero-evolution baseline MAE: "
    f"{overall_baseline_mae:.3f} s"
)

print(
    f"Difference vs baseline: "
    f"{overall_baseline_mae - overall_model_mae:+.3f} s"
)