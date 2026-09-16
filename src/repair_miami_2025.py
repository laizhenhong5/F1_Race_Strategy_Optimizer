import pandas as pd


INPUT_FILE = "data/cleaned_russell_miami_2025.csv"
OUTPUT_FILE = "data/cleaned_russell_miami_2025_repaired.csv"


print(f"Loading dataset: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)


# ============================================================
# MIAMI 2025 TYRE DATA CORRECTION
# ============================================================
#
# FastF1 tyre/stint information for Russell is incomplete and
# misaligned in this session.
#
# Russell started on HARD tyres and pitted at the end of Lap 29.
# From Lap 30 onward he ran the MEDIUM compound.
#
# The correction below reconstructs Compound, Stint and TyreLife
# using the observed pit-stop boundary.
# ============================================================


# First stint: HARD, Laps 1-29
first_stint = (
    df["LapNumber"].between(1, 29)
)

df.loc[
    first_stint,
    "Compound"
] = "HARD"

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


# Second stint: MEDIUM, Laps 30-57
second_stint = (
    df["LapNumber"] >= 30
)

df.loc[
    second_stint,
    "Compound"
] = "MEDIUM"

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
    - 29
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
            "PitInTime",
            "PitOutTime",
        ]
    ].to_string(index=False)
)

print(
    f"\nSaved repaired dataset to: "
    f"{OUTPUT_FILE}"
)