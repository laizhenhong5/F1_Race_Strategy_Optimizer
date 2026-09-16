import pandas as pd


# ============================================================
# INPUT DATASETS
# ============================================================

INPUT_FILES = [
    "data/feature_engineered_normal_singapore_2025.csv",
    "data/feature_engineered_normal_las_vegas_2024.csv",
    "data/feature_engineered_normal_bahrain_2025.csv",
    "data/feature_engineered_normal_japan_2025.csv",
    "data/feature_engineered_normal_miami_2025.csv",
    "data/feature_engineered_normal_emilia_romagna_2025.csv",
    "data/feature_engineered_normal_barcelona_catalunya_2025.csv",
    "data/feature_engineered_normal_canada_2025.csv",
]


# ============================================================
# OUTPUT DATASET
# ============================================================

OUTPUT_FILE = (
    "data/combined_feature_engineered.csv"
)


# ============================================================
# LOAD DATASETS
# ============================================================

dataframes = []

for file in INPUT_FILES:

    print(f"Loading: {file}")

    df = pd.read_csv(file)

    dataframes.append(df)


# ============================================================
# COMBINE
# ============================================================

combined_df = pd.concat(
    dataframes,
    ignore_index=True
)


# ============================================================
# SORT
# ============================================================

combined_df = combined_df.sort_values(
    [
        "Year",
        "GrandPrix",
        "Driver",
        "LapNumber",
    ]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

combined_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("COMBINED DATASET")
print("========================================")

print(
    f"Total rows: {len(combined_df)}"
)

print(
    f"Number of races: "
    f"{combined_df['GrandPrix'].nunique()}"
)

print(
    f"Number of drivers: "
    f"{combined_df['Driver'].nunique()}"
)

print("\nRows by race:")

print(
    combined_df.groupby(
        ["Year", "GrandPrix"]
    ).size()
)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)