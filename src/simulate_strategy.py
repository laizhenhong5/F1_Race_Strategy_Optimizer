import joblib
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "models/stint_slope_random_forest.pkl"


# Artificial test strategy.
#
# base_lap_time is currently supplied manually because our
# ML model predicts stint evolution, not absolute race pace.
STRATEGY = [
    {
        "compound": "MEDIUM",
        "start_lap": 1,
        "start_tyre_life": 1,
        "stint_length": 20,
        "base_lap_time": 90.000,
    },
    {
        "compound": "HARD",
        "start_lap": 21,
        "start_tyre_life": 1,
        "stint_length": 37,
        "base_lap_time": 89.500,
    },
]


# Artificial pit-loss value for now.
#
# Later this will become circuit-specific.
PIT_LOSS_SECONDS = 20.0


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


    rows = []


    for stint_index in range(
        stint["stint_length"]
    ):

        race_lap = (
            stint["start_lap"]
            + stint_index
        )

        stint_lap = (
            stint_index
            + 1
        )

        tyre_life = (
            stint["start_tyre_life"]
            + stint_index
        )

        relative_pace_change = (
            predicted_slope
            * stint_index
        )

        predicted_lap_time = (
            stint["base_lap_time"]
            + relative_pace_change
        )


        rows.append(
            {
                "Stint": stint_number,
                "RaceLap": race_lap,
                "StintLap": stint_lap,
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
# SIMULATE FULL STRATEGY
# ============================================================

stint_results = []


for stint_number, stint in enumerate(
    STRATEGY,
    start=1
):

    result = simulate_stint(
        stint_number,
        stint
    )

    stint_results.append(
        result
    )


race_df = pd.concat(
    stint_results,
    ignore_index=True
)


# ============================================================
# PIT STOPS
# ============================================================

number_of_pit_stops = (
    len(STRATEGY)
    - 1
)


total_pit_loss = (
    number_of_pit_stops
    * PIT_LOSS_SECONDS
)


# ============================================================
# RACE SUMMARY
# ============================================================

total_driving_time = (
    race_df[
        "PredictedLapTime"
    ].sum()
)


total_strategy_time = (
    total_driving_time
    + total_pit_loss
)


print("\n========================================")
print("STRATEGY")
print("========================================")


for stint_number, stint in enumerate(
    STRATEGY,
    start=1
):

    print(
        f"Stint {stint_number}: "
        f"{stint['compound']} | "
        f"Lap {stint['start_lap']} -> "
        f"{stint['start_lap'] + stint['stint_length'] - 1} | "
        f"Start tyre age "
        f"{stint['start_tyre_life']}"
    )


print("\n========================================")
print("MODEL STINT PREDICTIONS")
print("========================================")


for stint_number in race_df[
    "Stint"
].unique():

    stint_df = race_df[
        race_df["Stint"]
        == stint_number
    ]

    print(
        f"Stint {stint_number} "
        f"({stint_df['Compound'].iloc[0]}): "
        f"{stint_df['PredictedSlope'].iloc[0]:+.4f} s/lap"
    )


print("\n========================================")
print("SIMULATED RACE")
print("========================================")


print(
    race_df[
        [
            "RaceLap",
            "Stint",
            "Compound",
            "TyreLife",
            "RelativePaceChange",
            "PredictedLapTime",
        ]
    ].to_string(
        index=False,
        formatters={
            "TyreLife": (
                lambda x: f"{x:.1f}"
            ),
            "RelativePaceChange": (
                lambda x: f"{x:+.3f}"
            ),
            "PredictedLapTime": (
                lambda x: f"{x:.3f}"
            ),
        }
    )
)


print("\n========================================")
print("STRATEGY SUMMARY")
print("========================================")

print(
    f"Race laps simulated: "
    f"{len(race_df)}"
)

print(
    f"Pit stops: "
    f"{number_of_pit_stops}"
)

print(
    f"Driving time: "
    f"{total_driving_time:.3f} s"
)

print(
    f"Pit-loss time: "
    f"{total_pit_loss:.3f} s"
)

print(
    f"Total strategy time: "
    f"{total_strategy_time:.3f} s"
)