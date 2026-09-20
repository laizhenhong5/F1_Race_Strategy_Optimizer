#reduce lap to lap noises
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

# ============================================================
# LOAD DATA
# ============================================================
datasets = []
for file in INPUT_FILES:

    print(f"Loading: {file}")

    df = pd.read_csv(file)

    datasets.append(df)

df = pd.concat(
    datasets,
    ignore_index=True
)


df = df.sort_values(
    [
        "Year",
        "GrandPrix",
        "Driver",
        "Stint",
        "LapNumber",
    ]
).reset_index(drop=True)


print("\n========================================")
print("DATASET")
print("========================================")

print(
    f"Total normal-racing laps: "
    f"{len(df)}"
)

print(
    f"Number of drivers: "
    f"{df['Driver'].nunique()}"
)

print(
    f"Number of races: "
    f"{df[['Year', 'GrandPrix']].drop_duplicates().shape[0]}"
)


# ============================================================
# SMOOTH LAP TIMES WITHIN EACH STINT
# ============================================================

group_columns = [
    "Year",
    "GrandPrix",
    "Driver",
    "Stint",
]

# ============================================================
# IDENTIFY CONSECUTIVE-LAP SEGMENTS
# ============================================================

df["LapDifference"] = (
    df.groupby(group_columns)["LapNumber"]
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
    df.groupby(group_columns)["NewSegment"]
    .cumsum()
)

df["SmoothedLapTime"] = (
    df.groupby(group_columns + ["ConsecutiveSegment"])[
        "LapTimeSeconds"
    ]
    .transform(
        lambda series:
        series.rolling(
            window=ROLLING_WINDOW,
            center=True,
            min_periods=1
        ).median()
    )
)


# Difference between actual lap time and underlying
# smoothed pace.
df["NoiseResidual"] = (
    df["LapTimeSeconds"]
    - df["SmoothedLapTime"]
)


# ============================================================
# OVERALL NOISE STATISTICS
# ============================================================

print("\n========================================")
print("RAW VS SMOOTHED PACE")
print("========================================")

print(
    f"Mean absolute noise residual: "
    f"{df['NoiseResidual'].abs().mean():.3f} s"
)

print(
    f"Median absolute noise residual: "
    f"{df['NoiseResidual'].abs().median():.3f} s"
)

print(
    f"Maximum absolute noise residual: "
    f"{df['NoiseResidual'].abs().max():.3f} s"
)


# ============================================================
# CORRELATION WITH SMOOTHED PACE
# ============================================================

# Remove each driver/race's typical pace so different
# circuit lap lengths do not dominate the result.
df["RaceDriverMedian"] = (
    df.groupby(
        [
            "Year",
            "GrandPrix",
            "Driver",
        ]
    )["SmoothedLapTime"]
    .transform("median")
)


df["SmoothedRelativePace"] = (
    df["SmoothedLapTime"]
    - df["RaceDriverMedian"]
)


print("\n========================================")
print("CORRELATION WITH SMOOTHED RELATIVE PACE")
print("========================================")

for feature in [
    "LapNumber",
    "TyreLife",
]:

    pearson = df[
        [
            feature,
            "SmoothedRelativePace",
        ]
    ].corr(
        method="pearson"
    ).iloc[0, 1]

    spearman = df[
        [
            feature,
            "SmoothedRelativePace",
        ]
    ].corr(
        method="spearman"
    ).iloc[0, 1]

    print(
        f"{feature}: "
        f"Pearson = {pearson:+.3f}, "
        f"Spearman = {spearman:+.3f}"
    )


# ============================================================
# LARGEST NOISE SPIKES
# ============================================================

print("\n========================================")
print("10 LARGEST RAW-LAP DEVIATIONS")
print("========================================")

largest_noise = (
    df.assign(
        AbsoluteNoise=df[
            "NoiseResidual"
        ].abs()
    )
    .sort_values(
        "AbsoluteNoise",
        ascending=False
    )
    [
        [
            "Year",
            "GrandPrix",
            "Driver",
            "LapNumber",
            "Compound",
            "TyreLife",
            "LapTimeSeconds",
            "SmoothedLapTime",
            "NoiseResidual",
        ]
    ]
    .head(10)
)


print(
    largest_noise.to_string(
        index=False
    )
)

# ============================================================
# RAW VS SMOOTHED STINT TREND COMPARISON
# ============================================================

print("\n========================================")
print("RAW VS SMOOTHED STINT TRENDS")
print("========================================")

stint_group_columns = [
    "Year",
    "GrandPrix",
    "Driver",
    "Stint",
    "Compound",
]

comparison_results = []


for group_values, stint_df in df.groupby(
    stint_group_columns
):

    (
        year,
        grand_prix,
        driver,
        stint,
        compound,
    ) = group_values


    stint_df = (
        stint_df[
            [
                "TyreLife",
                "LapTimeSeconds",
                "SmoothedLapTime",
            ]
        ]
        .dropna()
        .sort_values("TyreLife")
    )


    # Ignore very short stints.
    if len(stint_df) < 5:
        continue


    # Need tyre-age variation.
    if stint_df["TyreLife"].nunique() < 2:
        continue


    X = stint_df[
        ["TyreLife"]
    ]


    # ----------------------------------------
    # RAW LAP-TIME TREND
    # ----------------------------------------

    raw_model = LinearRegression()

    raw_model.fit(
        X,
        stint_df["LapTimeSeconds"]
    )

    raw_slope = raw_model.coef_[0]

    raw_r2 = raw_model.score(
        X,
        stint_df["LapTimeSeconds"]
    )

    # ----------------------------------------
    # SMOOTHED LAP-TIME TREND
    # ----------------------------------------

    smooth_model = LinearRegression()

    smooth_model.fit(
        X,
        stint_df["SmoothedLapTime"]
    )

    smoothed_slope = (
        smooth_model.coef_[0]
    )

    smoothed_r2 = smooth_model.score(
        X,
        stint_df["SmoothedLapTime"]
    )


    comparison_results.append({
        "Year": year,
        "GrandPrix": grand_prix,
        "Driver": driver,
        "Stint": int(stint),
        "Compound": compound,
        "RawSlope": raw_slope,
        "SmoothedSlope": smoothed_slope,
        "RawR2": raw_r2,
        "SmoothedR2": smoothed_r2,
    })


comparison_df = pd.DataFrame(
    comparison_results
)


comparison_df["R2Improvement"] = (
    comparison_df["SmoothedR2"]
    - comparison_df["RawR2"]
)


comparison_df["SlopeChange"] = (
    comparison_df["SmoothedSlope"]
    - comparison_df["RawSlope"]
).abs()

# ============================================================
# SUMMARY
# ============================================================

print(
    f"Usable stints: "
    f"{len(comparison_df)}"
)

print(
    f"Average raw R2: "
    f"{comparison_df['RawR2'].mean():.3f}"
)

print(
    f"Average smoothed R2: "
    f"{comparison_df['SmoothedR2'].mean():.3f}"
)

print(
    f"Median raw R2: "
    f"{comparison_df['RawR2'].median():.3f}"
)

print(
    f"Median smoothed R2: "
    f"{comparison_df['SmoothedR2'].median():.3f}"
)

improved_stints = (
    comparison_df["R2Improvement"] > 0
).sum()

print(
    f"Stints with improved R2: "
    f"{improved_stints} / "
    f"{len(comparison_df)}"
)

print(
    f"Mean absolute slope change: "
    f"{comparison_df['SlopeChange'].mean():.4f} s/lap"
)
print(
    f"Median absolute slope change: "
    f"{comparison_df['SlopeChange'].median():.4f} s/lap"
)

# ============================================================
# SLOPE DIRECTION CHECK
# ============================================================

comparison_df["SameSlopeDirection"] = (
    (
        (comparison_df["RawSlope"] >= 0)
        & (comparison_df["SmoothedSlope"] >= 0)
    )
    |
    (
        (comparison_df["RawSlope"] < 0)
        & (comparison_df["SmoothedSlope"] < 0)
    )
)

same_direction = (
    comparison_df["SameSlopeDirection"].sum()
)

print(
    f"Stints keeping same slope direction: "
    f"{same_direction} / {len(comparison_df)}"
)