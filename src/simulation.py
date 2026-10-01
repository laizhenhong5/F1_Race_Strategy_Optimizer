import argparse
import os

import joblib
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "models/stint_slope_random_forest.pkl"

VALID_COMPOUNDS = [
    "SOFT",
    "MEDIUM",
    "HARD",
]


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Simulate an F1 tyre stint using the trained stint model."
)

parser.add_argument(
    "--compound",
    required=True,
    choices=VALID_COMPOUNDS,
    help="Tyre compound for the stint."
)

parser.add_argument(
    "--start-lap",
    required=True,
    type=int,
    help="Race lap on which the stint begins."
)

parser.add_argument(
    "--start-tyre-life",
    required=True,
    type=float,
    help="Tyre age at the beginning of the stint."
)

parser.add_argument(
    "--stint-length",
    required=True,
    type=int,
    help="Number of laps to simulate."
)

parser.add_argument(
    "--base-lap-time",
    required=True,
    type=float,
    help="Expected lap time at the beginning of the stint, in seconds."
)

args = parser.parse_args()


# ============================================================
# BASIC VALIDATION
# ============================================================

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model not found: {MODEL_FILE}\n"
        "Run src/train_stint_model.py first."
    )


if args.start_lap < 1:
    raise ValueError(
        "start-lap must be at least 1."
    )


if args.start_tyre_life < 0:
    raise ValueError(
        "start-tyre-life cannot be negative."
    )


if args.stint_length < 1:
    raise ValueError(
        "stint-length must be at least 1."
    )


# ============================================================
# LOAD MODEL
# ============================================================

print(f"Loading model: {MODEL_FILE}")

model = joblib.load(
    MODEL_FILE
)


# ============================================================
# CREATE MODEL INPUT
# ============================================================

model_input = pd.DataFrame(
    [
        {
            "StartLap": args.start_lap,
            "StartTyreLife": args.start_tyre_life,
            "Compound": args.compound,
        }
    ]
)


# ============================================================
# PREDICT STINT SLOPE
# ============================================================

predicted_slope = model.predict(
    model_input
)[0]


print("\n========================================")
print("STINT INPUT")
print("========================================")

print(
    f"Compound:        "
    f"{args.compound}"
)

print(
    f"Start lap:       "
    f"{args.start_lap}"
)

print(
    f"Starting tyre age: "
    f"{args.start_tyre_life:.1f}"
)

print(
    f"Stint length:    "
    f"{args.stint_length} laps"
)

print(
    f"Base lap time:   "
    f"{args.base_lap_time:.3f} s"
)


print("\n========================================")
print("MODEL PREDICTION")
print("========================================")

print(
    f"Predicted stint slope: "
    f"{predicted_slope:+.4f} s/lap"
)


# ============================================================
# SIMULATE STINT
# ============================================================

simulation_rows = []


for stint_index in range(
    args.stint_length
):

    race_lap = (
        args.start_lap
        + stint_index
    )

    stint_lap = (
        stint_index
        + 1
    )

    tyre_life = (
        args.start_tyre_life
        + stint_index
    )

    relative_pace_change = (
        predicted_slope
        * stint_index
    )

    predicted_lap_time = (
        args.base_lap_time
        + relative_pace_change
    )


    simulation_rows.append(
        {
            "RaceLap": race_lap,
            "StintLap": stint_lap,
            "TyreLife": tyre_life,
            "RelativePaceChange": relative_pace_change,
            "PredictedLapTime": predicted_lap_time,
        }
    )


simulation_df = pd.DataFrame(
    simulation_rows
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("SIMULATED STINT")
print("========================================")

print(
    simulation_df.to_string(
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


# ============================================================
# SUMMARY
# ============================================================

total_stint_time = (
    simulation_df[
        "PredictedLapTime"
    ].sum()
)


average_lap_time = (
    simulation_df[
        "PredictedLapTime"
    ].mean()
)


total_pace_change = (
    simulation_df[
        "RelativePaceChange"
    ].iloc[-1]
)


print("\n========================================")
print("STINT SUMMARY")
print("========================================")

print(
    f"Predicted total stint time: "
    f"{total_stint_time:.3f} s"
)

print(
    f"Predicted average lap time: "
    f"{average_lap_time:.3f} s"
)

print(
    f"Pace change from first to last lap: "
    f"{total_pace_change:+.3f} s"
)