import joblib
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "models/stint_slope_random_forest.pkl"
STINT_DATASET_FILE = "data/stint_dataset.csv"

RACE_LAPS = 57

FIRST_COMPOUND = "MEDIUM"
SECOND_COMPOUND = "HARD"

FIRST_START_TYRE_LIFE = 1
SECOND_START_TYRE_LIFE = 1

BASE_LAP_TIME = 90.0
PIT_LOSS_SECONDS = 20.0

EARLIEST_PIT_LAP = 10
LATEST_PIT_LAP = 40


COMPOUND_BASE_OFFSETS = {
    "SOFT": -0.2467,
    "MEDIUM": 0.0000,
    "HARD": +0.2580,
}


# ============================================================
# LOAD DATA / MODEL
# ============================================================

print(f"Loading model: {MODEL_FILE}")

rf_model = joblib.load(
    MODEL_FILE
)


print(f"Loading dataset: {STINT_DATASET_FILE}")

stint_df = pd.read_csv(
    STINT_DATASET_FILE
)


# ============================================================
# COMPOUND-MEDIAN SLOPES
# ============================================================

compound_median_slopes = (
    stint_df
    .groupby("Compound")[
        "SmoothedSlope"
    ]
    .median()
    .to_dict()
)


print("\nCompound-median slopes:")

for compound, slope in (
    compound_median_slopes.items()
):

    print(
        f"  {compound}: "
        f"{slope:+.4f} s/lap"
    )


# ============================================================
# DATA-SUPPORTED TYRE LIMITS
# ============================================================

max_supported_tyre_age = (
    stint_df
    .groupby("Compound")[
        "EndTyreLife"
    ]
    .max()
    .to_dict()
)


# ============================================================
# SIMULATE ONE STINT
# ============================================================

def simulate_stint(
    compound,
    start_lap,
    start_tyre_life,
    stint_length,
    slope,
):

    compound_offset = (
        COMPOUND_BASE_OFFSETS[
            compound
        ]
    )


    total_time = 0.0


    for index in range(
        stint_length
    ):

        relative_change = (
            slope
            * index
        )


        lap_time = (
            BASE_LAP_TIME
            + compound_offset
            + relative_change
        )


        total_time += lap_time


    return total_time


# ============================================================
# OPTIMIZATION
# ============================================================

results = []


for pit_lap in range(
    EARLIEST_PIT_LAP,
    LATEST_PIT_LAP + 1,
):

    first_length = pit_lap

    second_start = (
        pit_lap + 1
    )

    second_length = (
        RACE_LAPS
        - pit_lap
    )


    first_end_age = (
        FIRST_START_TYRE_LIFE
        + first_length
        - 1
    )


    second_end_age = (
        SECOND_START_TYRE_LIFE
        + second_length
        - 1
    )


    # Reject extrapolation beyond observed tyre age.
    if (
        first_end_age
        > max_supported_tyre_age[
            FIRST_COMPOUND
        ]
    ):
        continue


    if (
        second_end_age
        > max_supported_tyre_age[
            SECOND_COMPOUND
        ]
    ):
        continue


    # ========================================================
    # COMPOUND BASELINE
    # ========================================================

    baseline_first_slope = (
        compound_median_slopes[
            FIRST_COMPOUND
        ]
    )


    baseline_second_slope = (
        compound_median_slopes[
            SECOND_COMPOUND
        ]
    )


    baseline_first_time = simulate_stint(
        FIRST_COMPOUND,
        1,
        FIRST_START_TYRE_LIFE,
        first_length,
        baseline_first_slope,
    )


    baseline_second_time = simulate_stint(
        SECOND_COMPOUND,
        second_start,
        SECOND_START_TYRE_LIFE,
        second_length,
        baseline_second_slope,
    )


    baseline_total = (
        baseline_first_time
        + baseline_second_time
        + PIT_LOSS_SECONDS
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    first_input = pd.DataFrame(
        [
            {
                "StartLap": 1,
                "StartTyreLife": (
                    FIRST_START_TYRE_LIFE
                ),
                "Compound": (
                    FIRST_COMPOUND
                ),
            }
        ]
    )


    second_input = pd.DataFrame(
        [
            {
                "StartLap": second_start,
                "StartTyreLife": (
                    SECOND_START_TYRE_LIFE
                ),
                "Compound": (
                    SECOND_COMPOUND
                ),
            }
        ]
    )


    rf_first_slope = (
        rf_model.predict(
            first_input
        )[0]
    )


    rf_second_slope = (
        rf_model.predict(
            second_input
        )[0]
    )


    rf_first_time = simulate_stint(
        FIRST_COMPOUND,
        1,
        FIRST_START_TYRE_LIFE,
        first_length,
        rf_first_slope,
    )


    rf_second_time = simulate_stint(
        SECOND_COMPOUND,
        second_start,
        SECOND_START_TYRE_LIFE,
        second_length,
        rf_second_slope,
    )


    rf_total = (
        rf_first_time
        + rf_second_time
        + PIT_LOSS_SECONDS
    )


    results.append(
        {
            "PitLap": pit_lap,

            "BaselineFirstSlope": (
                baseline_first_slope
            ),

            "BaselineSecondSlope": (
                baseline_second_slope
            ),

            "BaselineTotalTime": (
                baseline_total
            ),

            "RFFirstSlope": (
                rf_first_slope
            ),

            "RFSecondSlope": (
                rf_second_slope
            ),

            "RFTotalTime": (
                rf_total
            ),
        }
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


baseline_ranked = (
    results_df
    .sort_values(
        "BaselineTotalTime"
    )
    .reset_index(drop=True)
)


rf_ranked = (
    results_df
    .sort_values(
        "RFTotalTime"
    )
    .reset_index(drop=True)
)


print("\n========================================")
print("COMPOUND-BASELINE OPTIMIZER")
print("========================================")

print(
    baseline_ranked[
        [
            "PitLap",
            "BaselineFirstSlope",
            "BaselineSecondSlope",
            "BaselineTotalTime",
        ]
    ]
    .head(5)
    .to_string(
        index=False,
        formatters={
            "BaselineFirstSlope": (
                lambda x: f"{x:+.4f}"
            ),
            "BaselineSecondSlope": (
                lambda x: f"{x:+.4f}"
            ),
            "BaselineTotalTime": (
                lambda x: f"{x:.3f}"
            ),
        }
    )
)


print("\n========================================")
print("RANDOM-FOREST OPTIMIZER")
print("========================================")

print(
    rf_ranked[
        [
            "PitLap",
            "RFFirstSlope",
            "RFSecondSlope",
            "RFTotalTime",
        ]
    ]
    .head(5)
    .to_string(
        index=False,
        formatters={
            "RFFirstSlope": (
                lambda x: f"{x:+.4f}"
            ),
            "RFSecondSlope": (
                lambda x: f"{x:+.4f}"
            ),
            "RFTotalTime": (
                lambda x: f"{x:.3f}"
            ),
        }
    )
)


# ============================================================
# BEST STRATEGIES
# ============================================================

baseline_best = (
    baseline_ranked.iloc[0]
)

rf_best = (
    rf_ranked.iloc[0]
)


print("\n========================================")
print("MODEL SENSITIVITY")
print("========================================")

print(
    f"Compound baseline best pit lap: "
    f"{int(baseline_best['PitLap'])}"
)

print(
    f"Random Forest best pit lap: "
    f"{int(rf_best['PitLap'])}"
)


if (
    int(baseline_best["PitLap"])
    == int(rf_best["PitLap"])
):

    print(
        "Both models select the same pit timing."
    )

else:

    print(
        "The models select different pit timings."
    )