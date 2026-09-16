#hardcoded to inspect dataset
# import pandas as pd
# df = pd.read_csv("data/f1_laps_raw.csv")
# print("Rows:", len(df))
# print("Columns:", df.columns.tolist())
# print("\nNumber of drivers:", df["Driver"].nunique())
# print("Drivers:", df["Driver"].unique())
# print("\nNumber of races:", df["GrandPrix"].nunique())
# print("Races:", df["GrandPrix"].unique())
# print("\nCompound counts:")
# print(df["Compound"].value_counts(dropna=False))
# print("\nMissing values:")
# print(df.isnull().sum())
import argparse

import pandas as pd


# ============================================================
# COMMAND-LINE SETTINGS
# ============================================================

parser = argparse.ArgumentParser(
    description="Inspect an F1 CSV dataset."
)

parser.add_argument(
    "input_file",
    help="Path to the CSV dataset to inspect."
)

args = parser.parse_args()


# ============================================================
# LOAD DATA
# ============================================================

print(f"Loading dataset: {args.input_file}")
df = pd.read_csv(args.input_file)


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")
print(f"Rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")


# ============================================================
# DRIVER INFORMATION
# ============================================================

if "Driver" in df.columns:

    print("\nNumber of drivers:")
    print(df["Driver"].nunique())
    print("\nDrivers:")
    print(df["Driver"].unique())


# ============================================================
# RACE INFORMATION
# ============================================================

if "GrandPrix" in df.columns:
    print("\nNumber of races:")
    print(df["GrandPrix"].nunique())

    print("\nRaces:")
    print(df["GrandPrix"].unique())


# ============================================================
# TYRE COMPOUNDS
# ============================================================

if "Compound" in df.columns:

    print("\nCompound counts:")
    print(
        df["Compound"]
        .value_counts(dropna=False)
    )


# ============================================================
# TRACK STATUS
# ============================================================

if "TrackStatus" in df.columns:
    print("\nTrack Status counts:")
    print(
        df["TrackStatus"]
        .value_counts(dropna=False)
        .sort_index()
    )

# ============================================================
# PIT LAPS
# ============================================================
if "PitInTime" in df.columns:
    pit_laps = df["PitInTime"].notna()
    print("\nPit-in observations:")
    print(pit_laps.sum())

# ============================================================
# MISSING VALUES
# ============================================================
print("\nMissing values:")
print(df.isnull().sum())
# ============================================================
# DATA PREVIEW
# ============================================================
print("\nFirst 10 rows:")
print(df.head(10))
#new addition
# ============================================================
# IDENTIFY UNUSUALLY SLOW LAPS
# ============================================================
print("\n========================================")
print("POTENTIALLY UNUSUAL LAPS(Opening Lap, Safety Car, Red Flag, etc.)")
print("========================================")
slow_laps = df[
    df["LapTimeSeconds"] > 100
].copy()

slow_columns = [
    "LapNumber",
    "LapTimeSeconds",
    "Compound",
    "TyreLife",
    "Stint",
    "TrackStatus",
]

# Add pit columns only if they exist in this dataset.
for column in ["PitInTime", "PitOutTime"]:
    if column in df.columns:
        slow_columns.append(column)

print(
    slow_laps[slow_columns]
)
# slow_laps = df[
#     df["LapTimeSeconds"] > 100
# ]
# print(
#     slow_laps[
#         [
#             "LapNumber",
#             "LapTimeSeconds",
#             "Compound",
#             "TyreLife",
#             "Stint",
#             "TrackStatus",
#             "PitInTime",
#             "PitOutTime",
#         ]
#     ]
# )



# ============================================================
# CLASSIFY UNUSUAL LAPS
# ============================================================
print("\n========================================")
print("UNUSUAL LAP CLASSIFICATION")
print("========================================")
unusual = df[
    df["LapTimeSeconds"] > 100
].copy()
unusual["LapType"] = "Normal"
#check does pit in time exist or not
# Opening lap
unusual.loc[
    unusual["LapNumber"] == unusual["LapNumber"].min(),
    "LapType"
] = "Opening Lap"


# Pit-in lap
if "PitInTime" in unusual.columns:
    unusual.loc[
        unusual["PitInTime"].notna(),
        "LapType"
    ] = "Pit-In Lap"


# Pit-out lap
if "PitOutTime" in unusual.columns:
    unusual.loc[
        unusual["PitOutTime"].notna(),
        "LapType"
    ] = "Pit-Out Lap"
print(
    unusual[
        [
            "LapNumber",
            "LapTimeSeconds",
            "Compound",
            "TyreLife",
            "Stint",
            "TrackStatus",
            "LapType",
        ]
    ]
)
# ============================================================
# INSPECT NON-NORMAL TRACK STATUS
# ============================================================

print("\n========================================")
print("NON-NORMAL TRACK STATUS LAPS")
print("========================================")

non_normal_status = df[
    df["TrackStatus"] != 1
]
print(
    non_normal_status[
        [
            "LapNumber",
            "LapTimeSeconds",
            "Compound",
            "TyreLife",
            "Stint",
            "TrackStatus",
        ]
    ]
)