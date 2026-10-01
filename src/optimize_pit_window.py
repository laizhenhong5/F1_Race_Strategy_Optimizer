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
FIRST_STINT_START_TYRE_LIFE = 1
SECOND_STINT_START_TYRE_LIFE = 1

# Still artificial for now.
BASE_LAP_TIME = 90.0
PIT_LOSS_SECONDS = 20.0
# Search range.
EARLIEST_PIT_LAP = 10
LATEST_PIT_LAP = 40

# ============================================================
# COMPOUND STARTING-PACE OFFSETS
# ============================================================

COMPOUND_BASE_OFFSETS = {
    "SOFT": -0.2467,
    "MEDIUM": 0.0000,
    "HARD": +0.2580,
}


# ============================================================
# LOAD MODEL
# ============================================================

print(f"Loading model: {MODEL_FILE}")

model = joblib.load(
    MODEL_FILE
)

# ============================================================
# LOAD DATA-SUPPORTED TYRE-AGE LIMITS
# ============================================================
#
# These are NOT physical tyre limits.
#
# They represent the maximum tyre ages observed in the
# current stint training dataset. We use them to prevent
# the optimizer from extrapolating beyond available data.
# ============================================================

stint_dataset = pd.read_csv(
    STINT_DATASET_FILE
)


MAX_SUPPORTED_TYRE_AGE = (
    stint_dataset
    .groupby("Compound")["EndTyreLife"]
    .max()
    .to_dict()
)


print("\nData-supported maximum tyre ages:")

for compound, max_age in (
    MAX_SUPPORTED_TYRE_AGE.items()
):

    print(
        f"  {compound}: "
        f"{max_age:.0f} laps"
    )

# ============================================================
# PREDICT ONE STINT
# ============================================================

def simulate_stint(
    compound,
    start_lap,
    start_tyre_life,
    stint_length,
):

    model_input = pd.DataFrame(
        [
            {
                "StartLap": start_lap,
                "StartTyreLife": start_tyre_life,
                "Compound": compound,
            }
        ]
    )


    predicted_slope = model.predict(
        model_input
    )[0]


    compound_offset = (
        COMPOUND_BASE_OFFSETS[
            compound
        ]
    )


    total_time = 0.0


    for index in range(
        stint_length
    ):

        relative_pace_change = (
            predicted_slope
            * index
        )


        lap_time = (
            BASE_LAP_TIME
            + compound_offset
            + relative_pace_change
        )


        total_time += lap_time


    return (
        total_time,
        predicted_slope,
    )

# ============================================================
# SEARCH PIT WINDOW
# ============================================================
results = []


for pit_lap in range(
    EARLIEST_PIT_LAP,
    LATEST_PIT_LAP + 1,
):

    first_stint_length = (
        pit_lap
    )


    second_stint_start = (
        pit_lap + 1
    )


    second_stint_length = (
        RACE_LAPS
        - pit_lap
    )
    # ========================================================
    # CHECK TYRE-AGE SUPPORT
    # ========================================================

    first_end_tyre_life = (
        FIRST_STINT_START_TYRE_LIFE
        + first_stint_length
        - 1
    )


    second_end_tyre_life = (
        SECOND_STINT_START_TYRE_LIFE
        + second_stint_length
        - 1
    )


    first_supported = (
        first_end_tyre_life
        <= MAX_SUPPORTED_TYRE_AGE[
            FIRST_COMPOUND
        ]
    )


    second_supported = (
        second_end_tyre_life
        <= MAX_SUPPORTED_TYRE_AGE[
            SECOND_COMPOUND
        ]
    )

    if not (
        first_supported
        and second_supported
    ):
        continue

    first_time, first_slope = (
        simulate_stint(
            compound=FIRST_COMPOUND,
            start_lap=1,
            start_tyre_life=(
                FIRST_STINT_START_TYRE_LIFE
            ),
            stint_length=first_stint_length,
        )
    )


    second_time, second_slope = (
        simulate_stint(
            compound=SECOND_COMPOUND,
            start_lap=second_stint_start,
            start_tyre_life=(
                SECOND_STINT_START_TYRE_LIFE
            ),
            stint_length=second_stint_length,
        )
    )


    driving_time = (
        first_time
        + second_time
    )


    total_time = (
        driving_time
        + PIT_LOSS_SECONDS
    )


    results.append(
        {
            "PitLap": pit_lap,

            "FirstStintLaps": (
                first_stint_length
            ),

            "SecondStintLaps": (
                second_stint_length
            ),

            "FirstSlope": (
                first_slope
            ),

            "SecondSlope": (
                second_slope
            ),

            "DrivingTime": (
                driving_time
            ),

            "PitLoss": (
                PIT_LOSS_SECONDS
            ),

            "TotalTime": (
                total_time
            ),
        }
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = (
    results_df.sort_values(
        "TotalTime"
    )
    .reset_index(drop=True)
)


best_time = (
    results_df[
        "TotalTime"
    ].iloc[0]
)


results_df[
    "TimeDifference"
] = (
    results_df[
        "TotalTime"
    ]
    - best_time
)

print("\n========================================")
print("PIT WINDOW OPTIMIZATION")
print("========================================")

print(
    f"Strategy: "
    f"{FIRST_COMPOUND} -> "
    f"{SECOND_COMPOUND}"
)

print(
    f"Pit window tested: "
    f"Lap {EARLIEST_PIT_LAP} "
    f"to Lap {LATEST_PIT_LAP}"
)

print(
    f"Valid data-supported pit timings: "
    f"{len(results_df)}"
)
print("\nTop 10 pit timings:")

print(
    results_df.head(10).to_string(
        index=False,
        formatters={
            "FirstSlope": (
                lambda x: f"{x:+.4f}"
            ),
            "SecondSlope": (
                lambda x: f"{x:+.4f}"
            ),
            "DrivingTime": (
                lambda x: f"{x:.3f}"
            ),
            "PitLoss": (
                lambda x: f"{x:.3f}"
            ),
            "TotalTime": (
                lambda x: f"{x:.3f}"
            ),
            "TimeDifference": (
                lambda x: f"{x:+.3f}"
            ),
        }
    )
)

# ============================================================
# BEST RESULT
# ============================================================

best = results_df.iloc[0]

print("\n========================================")
print("BEST PIT TIMING")
print("========================================")

print(
    f"Pit after Lap: "
    f"{int(best['PitLap'])}"
)

print(
    f"First stint: "
    f"{int(best['FirstStintLaps'])} laps "
    f"on {FIRST_COMPOUND}"
)

print(
    f"Second stint: "
    f"{int(best['SecondStintLaps'])} laps "
    f"on {SECOND_COMPOUND}"
)

print(
    f"Predicted strategy time: "
    f"{best['TotalTime']:.3f} s"
)