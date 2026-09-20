import pandas as pd


INPUT_FILE = "data/cleaned_antonelli_miami_2025.csv"
OUTPUT_FILE = "data/cleaned_antonelli_miami_2025_repaired.csv"


print(f"Loading dataset: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)


# ============================================================
# ANTONELLI - MIAMI 2025 TYRE DATA CORRECTION
# ============================================================
#
# FastF1 tyre/stint information is incomplete and misaligned
# for this session.
#
# Verified race strategy:
#
# Laps 1-25  -> MEDIUM, Stint 1
# Laps 26-57 -> HARD,   Stint 2
#
# Antonelli entered the pit lane at the end of Lap 25 and
# exited on Lap 26.
# ============================================================


# First stint: MEDIUM, Laps 1-25
first_stint = (
    df["LapNumber"].between(1, 25)
)

df.loc[
    first_stint,
    "Compound"
] = "MEDIUM"

df.loc[
    first_stint,
    "Stint"
] = 1

df.loc[
    first_stint,
    "TyreLife"
] = df.loc[
    first_stint,
    "LapNumber"
]


# Second stint: HARD, Laps 26-57
second_stint = (
    df["LapNumber"] >= 26
)

df.loc[
    second_stint,
    "Compound"
] = "HARD"

df.loc[
    second_stint,
    "Stint"
] = 2

df.loc[
    second_stint,
    "TyreLife"
] = (
    df.loc[
        second_stint,
        "LapNumber"
    ]
    - 25
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nTyre data after repair:")

print(
    df[
        [
            "LapNumber",
            "Compound",
            "TyreLife",
            "Stint",
            "TrackStatus",
            "PitInTime",
            "PitOutTime",
        ]
    ].to_string(index=False)
)


print(
    f"\nSaved repaired dataset to: "
    f"{OUTPUT_FILE}"
)