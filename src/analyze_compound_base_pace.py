import pandas as pd

from sklearn.linear_model import LinearRegression


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILES = [
    # Russell
    "data/normal_racing_laps_las_vegas_2024.csv",
    "data/normal_racing_laps_bahrain_2025.csv",
    "data/normal_racing_laps_japan_2025.csv",
    "data/normal_racing_laps_miami_2025_repaired.csv",
    "data/normal_racing_laps_emilia_romagna_2025.csv",
    "data/normal_racing_laps_barcelona_catalunya_2025.csv",
    "data/normal_racing_laps_canada_2025.csv",
    "data/normal_racing_laps_singapore_2025.csv",

    # Antonelli
    "data/normal_racing_laps_antonelli_bahrain_2025.csv",
    "data/normal_racing_laps_antonelli_japan_2025.csv",
    "data/normal_racing_laps_antonelli_canada_2025.csv",
    "data/normal_racing_laps_antonelli_barcelona_catalunya_2025.csv",
    "data/normal_racing_laps_antonelli_emilia_romagna_2025.csv",
    "data/normal_racing_laps_antonelli_miami_2025_repaired.csv",
    "data/normal_racing_laps_antonelli_singapore_2025.csv",
]

ROLLING_WINDOW = 3
EARLY_STINT_LAPS = 3
MIN_STINT_LAPS = 5


# ============================================================
# LOAD DATA
# ============================================================

datasets = []

for file in INPUT_FILES:

    print(f"Loading: {file}")

    dataset = pd.read_csv(file)

    datasets.append(dataset)


df = pd.concat(
    datasets,
    ignore_index=True
)


numeric_columns = [
    "LapNumber",
    "LapTimeSeconds",
    "TyreLife",
    "Stint",
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


df = (
    df.sort_values(
        [
            "Year",
            "GrandPrix",
            "Driver",
            "Stint",
            "LapNumber",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# IDENTIFY CONSECUTIVE SEGMENTS
# ============================================================

stint_groups = [
    "Year",
    "GrandPrix",
    "Driver",
    "Stint",
]


df["LapDifference"] = (
    df.groupby(stint_groups)["LapNumber"]
    .diff()
)


df["NewSegment"] = (
    (
        df["LapDifference"].isna()
        | (df["LapDifference"] != 1)
    )
    .astype(int)
)


df["ConsecutiveSegment"] = (
    df.groupby(stint_groups)["NewSegment"]
    .cumsum()
)


# ============================================================
# SMOOTH LAP TIMES
# ============================================================

smoothing_groups = (
    stint_groups
    + ["ConsecutiveSegment"]
)


df["SmoothedLapTime"] = (
    df.groupby(smoothing_groups)["LapTimeSeconds"]
    .transform(
        lambda series:
        series.rolling(
            window=ROLLING_WINDOW,
            center=True,
            min_periods=1,
        ).median()
    )
)


# ============================================================
# REMOVE EACH DRIVER'S BROAD RACE-LAP TREND
# ============================================================
#
# Different circuits have very different absolute lap times.
#
# Cars also generally become faster as fuel burns off.
#
# We fit a simple lap-number trend separately for every
# driver/race and measure pace relative to that trend.
#
# Negative adjusted pace = faster than expected for that
# point in the race.
#
# Positive adjusted pace = slower than expected.
# ============================================================

race_driver_groups = [
    "Year",
    "GrandPrix",
    "Driver",
]


df["ExpectedRaceTrendPace"] = 0.0


for _, group_df in df.groupby(
    race_driver_groups
):

    X = group_df[
        ["LapNumber"]
    ]

    y = group_df[
        "SmoothedLapTime"
    ]


    trend_model = LinearRegression()

    trend_model.fit(
        X,
        y,
    )


    predictions = trend_model.predict(
        X
    )


    df.loc[
        group_df.index,
        "ExpectedRaceTrendPace",
    ] = predictions


df["AdjustedPace"] = (
    df["SmoothedLapTime"]
    - df["ExpectedRaceTrendPace"]
)


# ============================================================
# MEASURE EARLY-STINT PACE
# ============================================================

stint_results = []


group_columns = [
    "Year",
    "GrandPrix",
    "Driver",
    "Stint",
    "Compound",
]


for group_values, stint_df in df.groupby(
    group_columns
):

    (
        year,
        grand_prix,
        driver,
        stint,
        compound,
    ) = group_values


    stint_df = (
        stint_df
        .sort_values("LapNumber")
        .copy()
    )

    if len(stint_df) < MIN_STINT_LAPS:
        continue


    early_df = stint_df.head(
        EARLY_STINT_LAPS
    )


    if len(early_df) == 0:
        continue


    stint_results.append({
        "Year": int(year),
        "GrandPrix": grand_prix,
        "Driver": driver,
        "Stint": int(stint),
        "Compound": compound,

        "StartLap": (
            stint_df["LapNumber"].iloc[0]
        ),

        "StartTyreLife": (
            stint_df["TyreLife"].iloc[0]
        ),

        "EarlyLapsUsed": (
            len(early_df)
        ),

        "EarlyAdjustedPace": (
            early_df["AdjustedPace"].median()
        ),
    })


stint_df = pd.DataFrame(
    stint_results
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("EARLY-STINT BASE PACE")
print("========================================")

print(
    f"Stints analysed: "
    f"{len(stint_df)}"
)


print("\nAdjusted early pace by compound:")

compound_summary = (
    stint_df.groupby("Compound")
    ["EarlyAdjustedPace"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ]
    )
    .round(4)
)


print(
    compound_summary
)


# ============================================================
# INTERPRETATION HELP
# ============================================================

print("\n========================================")
print("INTERPRETATION")
print("========================================")

print(
    "Negative value = faster than the driver's "
    "expected race pace at that lap."
)

print(
    "Positive value = slower than the driver's "
    "expected race pace at that lap."
)


# ============================================================
# INDIVIDUAL STINTS
# ============================================================

print("\n========================================")
print("INDIVIDUAL STINTS")
print("========================================")

print(
    stint_df.sort_values(
        [
            "Compound",
            "Year",
            "GrandPrix",
            "Driver",
            "Stint",
        ]
    ).to_string(
        index=False,
        formatters={
            "EarlyAdjustedPace": (
                lambda x: f"{x:+.3f}"
            ),
        },
    )
)