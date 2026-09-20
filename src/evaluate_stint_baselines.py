import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/stint_dataset.csv"
OUTPUT_FILE = "data/stint_baseline_results.csv"

TARGET = "SmoothedSlope"


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


print("\n========================================")
print("AVAILABLE RACES")
print("========================================")

print(
    races.to_string(
        index=False
    )
)


# ============================================================
# BASELINES
# ============================================================
#
# We evaluate three very simple assumptions:
#
# 1. ZERO SLOPE
#    The stint pace does not change.
#
# 2. GLOBAL MEDIAN
#    Predict the median slope observed in all TRAINING stints.
#
# 3. COMPOUND MEDIAN
#    Predict the median TRAINING slope for that compound.
#
# Importantly, the held-out race is never used to calculate
# the baseline values.
# ============================================================

results = []

total_zero_error = 0.0
total_global_error = 0.0
total_compound_error = 0.0
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


    y_test = test_df[
        TARGET
    ].values


    # ========================================================
    # BASELINE 1: ZERO SLOPE
    # ========================================================

    zero_predictions = np.zeros(
        len(test_df)
    )


    # ========================================================
    # BASELINE 2: GLOBAL TRAINING MEDIAN
    # ========================================================

    global_median = (
        train_df[
            TARGET
        ].median()
    )


    global_predictions = np.full(
        len(test_df),
        global_median
    )


    # ========================================================
    # BASELINE 3: COMPOUND-SPECIFIC TRAINING MEDIAN
    # ========================================================

    compound_medians = (
        train_df.groupby(
            "Compound"
        )[TARGET]
        .median()
        .to_dict()
    )


    compound_predictions = (
        test_df["Compound"]
        .map(compound_medians)
        .fillna(global_median)
        .values
    )


    # ========================================================
    # EVALUATION
    # ========================================================

    zero_mae = mean_absolute_error(
        y_test,
        zero_predictions
    )


    global_mae = mean_absolute_error(
        y_test,
        global_predictions
    )


    compound_mae = mean_absolute_error(
        y_test,
        compound_predictions
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
        f"Zero-slope MAE: "
        f"{zero_mae:.4f} s/lap"
    )

    print(
        f"Global-median MAE: "
        f"{global_mae:.4f} s/lap"
    )

    print(
        f"Compound-median MAE: "
        f"{compound_mae:.4f} s/lap"
    )


    results.append({
        "TestYear": test_year,
        "TestRace": test_race,
        "TrainingStints": len(train_df),
        "TestingStints": len(test_df),
        "ZeroSlopeMAE": zero_mae,
        "GlobalMedianMAE": global_mae,
        "CompoundMedianMAE": compound_mae,
    })


    total_zero_error += (
        np.abs(
            y_test
            - zero_predictions
        ).sum()
    )


    total_global_error += (
        np.abs(
            y_test
            - global_predictions
        ).sum()
    )


    total_compound_error += (
        np.abs(
            y_test
            - compound_predictions
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
            "ZeroSlopeMAE": (
                lambda x: f"{x:.4f}"
            ),
            "GlobalMedianMAE": (
                lambda x: f"{x:.4f}"
            ),
            "CompoundMedianMAE": (
                lambda x: f"{x:.4f}"
            ),
        }
    )
)


# ============================================================
# OVERALL RESULTS
# ============================================================

overall_zero_mae = (
    total_zero_error
    / total_rows
)


overall_global_mae = (
    total_global_error
    / total_rows
)


overall_compound_mae = (
    total_compound_error
    / total_rows
)


print("\n========================================")
print("OVERALL RESULTS")
print("========================================")

print(
    f"Total evaluated stints: "
    f"{total_rows}"
)

print(
    f"Zero-slope MAE: "
    f"{overall_zero_mae:.4f} s/lap"
)

print(
    f"Global-median MAE: "
    f"{overall_global_mae:.4f} s/lap"
)

print(
    f"Compound-median MAE: "
    f"{overall_compound_mae:.4f} s/lap"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nSaved results to: "
    f"{OUTPUT_FILE}"
)