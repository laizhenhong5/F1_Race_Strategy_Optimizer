#SECOND STEP
from pathlib import Path
import pandas as pd
import argparse
# ============================================================
# COMMAND-LINE SETTINGS
# ============================================================

parser = argparse.ArgumentParser(
    description="Create engineered F1 lap-time features."
)

parser.add_argument(
    "input_file",
    help="Path to the cleaned CSV dataset."
)

parser.add_argument(
    "output_file",
    help="Path where the feature-engineered CSV will be saved."
)

args = parser.parse_args()
INPUT_FILE = args.input_file
OUTPUT_FILE = args.output_file

# ============================================================
# LOAD DATA
# ============================================================

print(f"Loading dataset: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================
numeric_columns = [
    "LapNumber",
    "TyreLife",
    "Stint",
    "TrackStatus",
    "LapTimeSeconds",
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

# ============================================================
# SORT DATA
# ============================================================

df = df.sort_values(
    "LapNumber"
).reset_index(drop=True)

# ============================================================
# FEATURE 1: PIT LAP
# ============================================================

df["IsPitLap"] = (
    df["PitInTime"]
    .notna()
    .astype(int)
)

# ============================================================
# FEATURE 2: LAP TIME DIFFERENCE
# ============================================================

# ============================================================
# PREVIOUS CONSECUTIVE LAP INFORMATION
# ============================================================

group_columns = [
    "Year",
    "GrandPrix",
    "Driver",
    "Stint",
]

# Previous lap number within the same stint
df["PreviousLapNumber"] = (
    df.groupby(group_columns)["LapNumber"]
    .shift(1)
)

# Previous lap time within the same stint
df["PreviousLapTime"] = (
    df.groupby(group_columns)["LapTimeSeconds"]
    .shift(1)
)

# Only accept the previous lap if it is actually consecutive.
consecutive_lap = (
    (df["LapNumber"] - df["PreviousLapNumber"])
    == 1
)

# If a lap was skipped/removed, do not pretend that the older
# observation is the immediately previous lap.
df.loc[
    ~consecutive_lap,
    "PreviousLapTime"
] = pd.NA

# Lap-time change relative to the genuine previous lap.
df["LapTimeDelta"] = (
    df["LapTimeSeconds"]
    - df["PreviousLapTime"]
)

# ============================================================
# FEATURE 3: LAP NUMBER WITHIN STINT
# ============================================================

df["StintLap"] = (
    df.groupby("Stint")
    .cumcount()
    + 1
)

# ============================================================
# FEATURE 4: TYRE COMPOUND ENCODING
# ============================================================

compound_map = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "INTERMEDIATE": 3,
    "WET": 4,
}

df["CompoundEncoded"] = (
    df["Compound"].map(compound_map)
)

# ============================================================
# FEATURE 5: NORMALIZED TYRE AGE
# ============================================================

df["TyreLifeNormalized"] = (
    df["TyreLife"] / df["TyreLife"].max()
)

# ============================================================
# SELECT FEATURES
# ============================================================

feature_columns = [
    # Race / driver identity
    "Year",
    "GrandPrix",
    "Session",
    "Driver",

    # Lap information
    "LapNumber",
    "LapTimeSeconds",
    "PreviousLapTime",

    # Tyre information
    "Compound",
    "CompoundEncoded",
    "TyreLife",
    "TyreLifeNormalized",

    # Stint information
    "Stint",
    "StintLap",

    # Race-state information
    "TrackStatus",
    "IsPitLap",
    "Position",

    # Sector information
    "Sector1TimeSeconds",
    "Sector2TimeSeconds",
    "Sector3TimeSeconds",

    # Difference from previous lap
    "LapTimeDelta",
]

feature_df = df[feature_columns].copy()
# ============================================================
# REMOVE MISSING VALUES
# ============================================================
# feature_df = feature_df.dropna()

# ============================================================
# REMOVE ROWS MISSING ESSENTIAL ML VALUES
# ============================================================

required_columns = [
    "LapNumber",
    "LapTimeSeconds",
    "PreviousLapTime",
    "CompoundEncoded",
    "TyreLife",
    "StintLap",
    "TrackStatus",
    "Position",
]

feature_df = feature_df.dropna(
    subset=required_columns
)
# ============================================================
# SAVE
# ============================================================

Path(OUTPUT_FILE).parent.mkdir(
    parents=True,
    exist_ok=True
)
feature_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# DISPLAY RESULT
# ============================================================

print("\nFeature engineering completed successfully.")

print("\nFirst 10 rows:")
print(feature_df.head(10))

print("\nRows after feature engineering:")
print(len(feature_df))

print(f"\nSaved to: {OUTPUT_FILE}")