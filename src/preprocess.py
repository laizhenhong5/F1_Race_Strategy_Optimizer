#preprocess first then inspect
import argparse

import pandas as pd


# ============================================================
# COMMAND-LINE SETTINGS
# ============================================================

parser = argparse.ArgumentParser(
    description="Create a normal-racing-lap dataset."
)

parser.add_argument(
    "input_file",
    help="Path to the cleaned F1 lap dataset."
)

parser.add_argument(
    "output_file",
    help="Path where the normal-lap dataset will be saved."
)

args = parser.parse_args() #example :python src/preprocess.py data/cleaned_russell_las_vegas_2024.csv data/normal_racing_laps_las_vegas_2024.csv


# ============================================================
# LOAD DATA
# ============================================================

print(f"Loading dataset: {args.input_file}")

df = pd.read_csv(args.input_file)

# ============================================================
# PREPARE TRACK STATUS
# ============================================================

# Convert TrackStatus safely to numeric.
# Normal green-track running is represented by status 1.
df["TrackStatus"] = pd.to_numeric(
    df["TrackStatus"],
    errors="coerce"
)

# ============================================================
# IDENTIFY SPECIAL LAPS
# ============================================================

# Opening lap
opening_lap = (
    df["LapNumber"]
    == df["LapNumber"].min()
)

# Pit-in lap
pit_in_lap = df["PitInTime"].notna()

# Pit-out lap
pit_out_lap = df["PitOutTime"].notna()

# Normal track running (green track status)
normal_track_status = (
    df["TrackStatus"] == 1
)


# ============================================================
# KEEP NORMAL RACING LAPS
# ============================================================

normal_laps = df[
    (~opening_lap)
    & (~pit_in_lap)
    & (~pit_out_lap)
    & normal_track_status
].copy()


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("PREPROCESSING RESULTS")
print("========================================")

print(f"Original rows: {len(df)}")

print(
    f"Opening laps removed: "
    f"{opening_lap.sum()}"
)

print(
    f"Pit-lane entry-marked laps removed: "
    f"{pit_in_lap.sum()}"
)

print(
    f"Pit-lane exit-marked laps removed: "
    f"{pit_out_lap.sum()}"
)

print(
    f"Non-normal track-status laps removed: "
    f"{(~normal_track_status).sum()}"
)

print(
    f"Normal racing laps: "
    f"{len(normal_laps)}"
)

print(
    f"Total removed rows: "
    f"{len(df) - len(normal_laps)}"
)

# print(f"Normal racing laps: {len(normal_laps)}")

# print(
#     f"Removed rows: "
#     f"{len(df) - len(normal_laps)}"
# )

# ============================================================
# SAVE DATASET
# ============================================================

normal_laps.to_csv(
    args.output_file,
    index=False
)

print(
    f"\nSaved normal-racing dataset to: "
    f"{args.output_file}"
)