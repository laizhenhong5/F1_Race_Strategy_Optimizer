import pandas as pd
from sklearn.linear_model import LinearRegression
# ============================================================
# SETTINGS
# ============================================================

INPUT_FILES = [
    "data/normal_racing_laps_las_vegas_2024.csv",
    "data/normal_racing_laps_bahrain_2025.csv",
    "data/normal_racing_laps_japan_2025.csv",
    "data/normal_racing_laps_miami_2025_repaired.csv",
    "data/normal_racing_laps_emilia_romagna_2025.csv",
    "data/normal_racing_laps_barcelona_catalunya_2025.csv",
    "data/normal_racing_laps_canada_2025.csv",
    "data/normal_racing_laps_singapore_2025.csv",
    "data/normal_racing_laps_antonelli_bahrain_2025.csv",
    "data/normal_racing_laps_antonelli_japan_2025.csv",
    "data/normal_racing_laps_antonelli_canada_2025.csv",
]

MIN_STINT_LAPS = 5

# ============================================================
# LOAD DATA
# ============================================================

datasets = []

for file in INPUT_FILES:
    print(f"Loading: {file}")
    df = pd.read_csv(file)
    datasets.append(df)


combined_df = pd.concat(
    datasets,
    ignore_index=True
)


print("\n========================================")
print("DATASET")
print("========================================")
print(
    f"Total normal-racing laps: "
    f"{len(combined_df)}"
)

print(
    f"Number of races: "
    f"{combined_df[['Year', 'GrandPrix']].drop_duplicates().shape[0]}"
)


# ============================================================
# CALCULATE STINT DEGRADATION
# ============================================================

results = []

group_columns = [
    "Year",
    "GrandPrix",
    "Driver",
    "Stint",
    "Compound",
]


for group_values, stint_df in combined_df.groupby(
    group_columns
):

    year, grand_prix, driver, stint, compound = (
        group_values
    )

    stint_df = (
        stint_df[
            [
                "TyreLife",
                "LapTimeSeconds",
            ]
        ]
        .dropna()
        .sort_values("TyreLife")
    )


    # Ignore very short stints.
    if len(stint_df) < MIN_STINT_LAPS:
        continue


    # Need variation in tyre age.
    if stint_df["TyreLife"].nunique() < 2:
        continue


    X = stint_df[
        ["TyreLife"]
    ]

    y = stint_df[
        "LapTimeSeconds"
    ]


    model = LinearRegression()

    model.fit(
        X,
        y
    )


    slope = model.coef_[0]

    intercept = model.intercept_

    r2 = model.score(
        X,
        y
    )


    results.append({
        "Year": year,
        "GrandPrix": grand_prix,
        "Driver": driver,
        "Stint": int(stint),
        "Compound": compound,
        "Laps": len(stint_df),
        "StartTyreLife": stint_df["TyreLife"].min(),
        "EndTyreLife": stint_df["TyreLife"].max(),
        "DegradationSlope": slope,
        "Intercept": intercept,
        "R2": r2,
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n========================================")
print("STINT DEGRADATION RESULTS")
print("========================================")

print(
    results_df[
        [
            "Year",
            "GrandPrix",
            "Driver",
            "Stint",
            "Compound",
            "Laps",
            "StartTyreLife",
            "EndTyreLife",
            "DegradationSlope",
            "R2",
        ]
    ].to_string(
        index=False,
        formatters={
            "DegradationSlope": (
                lambda x: f"{x:+.4f}"
            ),
            "R2": (
                lambda x: f"{x:.3f}"
            ),
        }
    )
)


# ============================================================
# SUMMARY BY COMPOUND
# ============================================================

print("\n========================================")
print("AVERAGE SLOPE BY COMPOUND")
print("========================================")

compound_summary = (
    results_df
    .groupby("Compound")
    .agg(
        Stints=("DegradationSlope", "size"),
        MeanSlope=("DegradationSlope", "mean"),
        MedianSlope=("DegradationSlope", "median"),
        MeanR2=("R2", "mean"),
    )
    .reset_index()
)


print(
    compound_summary.to_string(
        index=False,
        formatters={
            "MeanSlope": (
                lambda x: f"{x:+.4f}"
            ),
            "MedianSlope": (
                lambda x: f"{x:+.4f}"
            ),
            "MeanR2": (
                lambda x: f"{x:.3f}"
            ),
        }
    )
)