import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


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
]


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


print("\n========================================")
print("DATASET")
print("========================================")

print(
    f"Total normal-racing laps: "
    f"{len(df)}"
)

print(
    f"Number of races: "
    f"{df[['Year', 'GrandPrix']].drop_duplicates().shape[0]}"
)

print(
    f"Number of drivers: "
    f"{df['Driver'].nunique()}"
)


# ============================================================
# CALCULATE RELATIVE PACE
# ============================================================

group_columns = [
    "Year",
    "GrandPrix",
    "Driver",
]


# Median clean lap time for each driver in each race.
df["RaceDriverMedianLapTime"] = (
    df.groupby(group_columns)[
        "LapTimeSeconds"
    ]
    .transform("median")
)


# Positive RelativePace = slower than that driver's
# typical clean race pace.
#
# Negative RelativePace = faster than typical pace.
df["RelativePace"] = (
    df["LapTimeSeconds"]
    - df["RaceDriverMedianLapTime"]
)


print("\n========================================")
print("RELATIVE PACE STATISTICS")
print("========================================")

print(
    f"Mean relative pace: "
    f"{df['RelativePace'].mean():+.3f} s"
)

print(
    f"Median relative pace: "
    f"{df['RelativePace'].median():+.3f} s"
)

print(
    f"Standard deviation: "
    f"{df['RelativePace'].std():.3f} s"
)


# ============================================================
# SIMPLE CORRELATIONS
# ============================================================

print("\n========================================")
print("CORRELATION WITH RELATIVE PACE")
print("========================================")

for feature in [
    "LapNumber",
    "TyreLife",
]:

    pearson = df[
        [feature, "RelativePace"]
    ].corr(
        method="pearson"
    ).iloc[0, 1]

    spearman = df[
        [feature, "RelativePace"]
    ].corr(
        method="spearman"
    ).iloc[0, 1]

    print(
        f"{feature}: "
        f"Pearson = {pearson:+.3f}, "
        f"Spearman = {spearman:+.3f}"
    )


# ============================================================
# RELATIVE PACE BY COMPOUND
# ============================================================

print("\n========================================")
print("RELATIVE PACE BY COMPOUND")
print("========================================")

compound_stats = (
    df.groupby("Compound")
    .agg(
        Laps=("RelativePace", "size"),
        MeanRelativePace=("RelativePace", "mean"),
        MedianRelativePace=("RelativePace", "median"),
    )
    .reset_index()
)


print(
    compound_stats.to_string(
        index=False,
        formatters={
            "MeanRelativePace": (
                lambda x: f"{x:+.3f}"
            ),
            "MedianRelativePace": (
                lambda x: f"{x:+.3f}"
            ),
        }
    )
)


# ============================================================
# SIMPLE MULTIVARIATE DIAGNOSTIC
# ============================================================
#
# This is NOT our final ML model.
#
# The purpose is to see whether LapNumber and TyreLife
# show separate effects once absolute circuit/driver pace
# has been removed.
# ============================================================

compound_dummies = pd.get_dummies(
    df["Compound"],
    prefix="Compound",
    drop_first=True,
    dtype=int
)


X = pd.concat(
    [
        df[
            [
                "LapNumber",
                "TyreLife",
            ]
        ],
        compound_dummies,
    ],
    axis=1
)

y = df["RelativePace"]


model = LinearRegression()

model.fit(
    X,
    y
)


predictions = model.predict(
    X
)


mae = mean_absolute_error(
    y,
    predictions
)

r2 = r2_score(
    y,
    predictions
)


print("\n========================================")
print("LINEAR RELATIVE-PACE DIAGNOSTIC")
print("========================================")

print("\nCoefficients:")

for feature, coefficient in zip(
    X.columns,
    model.coef_
):

    print(
        f"{feature}: "
        f"{coefficient:+.4f}"
    )


print(
    f"\nDiagnostic MAE: "
    f"{mae:.3f} s"
)

print(
    f"Diagnostic R2: "
    f"{r2:.3f}"
)