import joblib
import pandas as pd
# ============================================================
# SETTINGS
# ============================================================
MODEL_FILE = "models/stint_slope_random_forest.pkl"
RACE_LAPS = 57
PIT_LOSS_SECONDS = 20.0
# Artificial common starting pace for now.
#
# We deliberately use the same base pace so this experiment
# tests the strategy engine and stint-evolution model rather
# than pretending we already have an absolute-pace model.
BASE_LAP_TIME = 90.0
# ============================================================
# DATA-DERIVED COMPOUND STARTING-PACE OFFSETS
# ============================================================
# Derived from the median early-stint adjusted pace in the
# current 38-stint Mercedes dataset.
# MEDIUM is used as the reference compound.
# Negative = faster than Medium.
# Positive = slower than Medium.
# ============================================================
COMPOUND_BASE_OFFSETS = {
    "SOFT": -0.2467,
    "MEDIUM": 0.0000,
    "HARD": +0.2580,
}
# ============================================================
# STRATEGIES
# ============================================================

STRATEGIES = {
    "Strategy A - Pit Lap 20": [
        {
            "compound": "MEDIUM",
            "start_lap": 1,
            "start_tyre_life": 1,
            "stint_length": 20,
        },
        {
            "compound": "HARD",
            "start_lap": 21,
            "start_tyre_life": 1,
            "stint_length": 37,
        },
    ],

    "Strategy B - Pit Lap 25": [
        {
            "compound": "MEDIUM",
            "start_lap": 1,
            "start_tyre_life": 1,
            "stint_length": 25,
        },
        {
            "compound": "HARD",
            "start_lap": 26,
            "start_tyre_life": 1,
            "stint_length": 32,
        },
    ],
}


# ============================================================
# LOAD MODEL
# ============================================================

print(f"Loading model: {MODEL_FILE}")

model = joblib.load(
    MODEL_FILE
)


# ============================================================
# SIMULATE ONE STINT
# ============================================================

def simulate_stint(stint_number, stint):

    model_input = pd.DataFrame(
        [
            {
                "StartLap": stint["start_lap"],
                "StartTyreLife": stint["start_tyre_life"],
                "Compound": stint["compound"],
            }
        ]
    )

    predicted_slope = model.predict(
        model_input
    )[0]

    compound_offset = (
    COMPOUND_BASE_OFFSETS[
        stint["compound"]
    ]
)

    rows = []

    for index in range(
        stint["stint_length"]
    ):

        race_lap = (
            stint["start_lap"]
            + index
        )

        tyre_life = (
            stint["start_tyre_life"]
            + index
        )

        relative_pace_change = (
            predicted_slope
            * index
        )

        predicted_lap_time = (
            BASE_LAP_TIME
            + relative_pace_change + compound_offset
        )

        rows.append(
            {
                "Stint": stint_number,
                "RaceLap": race_lap,
                "Compound": stint["compound"],
                "TyreLife": tyre_life,
                "PredictedSlope": predicted_slope,
                "RelativePaceChange": relative_pace_change,
                "PredictedLapTime": predicted_lap_time,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SIMULATE COMPLETE STRATEGY
# ============================================================

def simulate_strategy(
    strategy_name,
    strategy,
):

    stint_results = []

    for stint_number, stint in enumerate(
        strategy,
        start=1,
    ):

        result = simulate_stint(
            stint_number,
            stint,
        )

        stint_results.append(
            result
        )

    race_df = pd.concat(
        stint_results,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    expected_laps = list(
        range(
            1,
            RACE_LAPS + 1,
        )
    )
    actual_laps = (
        race_df["RaceLap"]
        .astype(int)
        .tolist()
    )

    if actual_laps != expected_laps:

        raise ValueError(
            f"{strategy_name} does not cover "
            f"Race Laps 1-{RACE_LAPS} correctly."
        )

    # --------------------------------------------------------
    # TIMES
    # --------------------------------------------------------

    driving_time = (
        race_df[
            "PredictedLapTime"
        ].sum()
    )

    pit_stops = (
        len(strategy)
        - 1
    )

    pit_loss = (
        pit_stops
        * PIT_LOSS_SECONDS
    )

    total_time = (
        driving_time
        + pit_loss
    )

    return {
        "Strategy": strategy_name,
        "PitStops": pit_stops,
        "DrivingTime": driving_time,
        "PitLoss": pit_loss,
        "TotalTime": total_time,
        "RaceData": race_df,
    }


# ============================================================
# RUN ALL STRATEGIES
# ============================================================

results = []

for strategy_name, strategy in (
    STRATEGIES.items()
):

    result = simulate_strategy(
        strategy_name,
        strategy,
    )

    results.append(
        result
    )

# ============================================================
# PRINT STINT PREDICTIONS
# ============================================================
print("\n========================================")
print("STINT MODEL PREDICTIONS")
print("========================================")

for result in results:

    print(
        f"\n{result['Strategy']}"
    )

    race_df = result[
        "RaceData"
    ]

    for stint_number in (
        race_df["Stint"].unique()
    ):

        stint_df = race_df[
            race_df["Stint"]
            == stint_number
        ]

        print(
            f"  Stint {stint_number} "
            f"{stint_df['Compound'].iloc[0]}: "
            f"{stint_df['PredictedSlope'].iloc[0]:+.4f} s/lap"
        )

# ============================================================
# COMPARISON TABLE
# ============================================================

summary_rows = []

for result in results:

    summary_rows.append(
        {
            "Strategy": result[
                "Strategy"
            ],
            "PitStops": result[
                "PitStops"
            ],
            "DrivingTime": result[
                "DrivingTime"
            ],
            "PitLoss": result[
                "PitLoss"
            ],
            "TotalTime": result[
                "TotalTime"
            ],
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)

best_time = (
    summary_df[
        "TotalTime"
    ].min()
)

summary_df[
    "TimeDifference"
] = (
    summary_df["TotalTime"]
    - best_time
)

print("\n========================================")
print("STRATEGY COMPARISON")
print("========================================")

print(
    summary_df.to_string(
        index=False,
        formatters={
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