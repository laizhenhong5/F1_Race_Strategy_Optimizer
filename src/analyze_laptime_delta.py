import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/combined_feature_engineered.csv"


# ============================================================
# LOAD DATA
# ============================================================

print(f"Loading dataset: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

df["AbsoluteDelta"] = df["LapTimeDelta"].abs()


# ============================================================
# OVERALL DELTA STATISTICS
# ============================================================

print("\n========================================")
print("OVERALL LAP-TIME DELTA STATISTICS")
print("========================================")

print(f"Rows: {len(df)}")

print(
    f"Mean delta: "
    f"{df['LapTimeDelta'].mean():.3f} s"
)

print(
    f"Median delta: "
    f"{df['LapTimeDelta'].median():.3f} s"
)

print(
    f"Standard deviation: "
    f"{df['LapTimeDelta'].std():.3f} s"
)

print(
    f"Mean absolute delta: "
    f"{df['AbsoluteDelta'].mean():.3f} s"
)

print(
    f"Median absolute delta: "
    f"{df['AbsoluteDelta'].median():.3f} s"
)


# ============================================================
# DELTA DISTRIBUTION
# ============================================================

print("\n========================================")
print("HOW OFTEN ARE CONSECUTIVE LAPS SIMILAR?")
print("========================================")

thresholds = [
    0.10,
    0.20,
    0.30,
    0.50,
    1.00,
]

for threshold in thresholds:

    percentage = (
        (df["AbsoluteDelta"] <= threshold)
        .mean()
        * 100
    )

    print(
        f"|delta| <= {threshold:.2f} s: "
        f"{percentage:.1f}%"
    )


large_changes = (
    df["AbsoluteDelta"] > 1.0
).sum()

print(
    f"\nLaps with |delta| > 1.00 s: "
    f"{large_changes}"
)


# ============================================================
# STATISTICS BY RACE
# ============================================================

print("\n========================================")
print("DELTA STATISTICS BY RACE")
print("========================================")

race_stats = (
    df.groupby(
        ["Year", "GrandPrix"]
    )
    .agg(
        Rows=("LapTimeDelta", "size"),
        MeanDelta=("LapTimeDelta", "mean"),
        MedianDelta=("LapTimeDelta", "median"),
        MeanAbsoluteDelta=("AbsoluteDelta", "mean"),
        StdDelta=("LapTimeDelta", "std"),
    )
    .reset_index()
)

print(
    race_stats.to_string(
        index=False,
        formatters={
            "MeanDelta": lambda x: f"{x:.3f}",
            "MedianDelta": lambda x: f"{x:.3f}",
            "MeanAbsoluteDelta": lambda x: f"{x:.3f}",
            "StdDelta": lambda x: f"{x:.3f}",
        }
    )
)


# ============================================================
# STATISTICS BY COMPOUND
# ============================================================

print("\n========================================")
print("DELTA STATISTICS BY COMPOUND")
print("========================================")

compound_stats = (
    df.groupby("Compound")
    .agg(
        Rows=("LapTimeDelta", "size"),
        MeanDelta=("LapTimeDelta", "mean"),
        MedianDelta=("LapTimeDelta", "median"),
        MeanAbsoluteDelta=("AbsoluteDelta", "mean"),
    )
    .reset_index()
)

print(
    compound_stats.to_string(
        index=False,
        formatters={
            "MeanDelta": lambda x: f"{x:.3f}",
            "MedianDelta": lambda x: f"{x:.3f}",
            "MeanAbsoluteDelta": lambda x: f"{x:.3f}",
        }
    )
)


# ============================================================
# SIMPLE FEATURE RELATIONSHIPS
# ============================================================

print("\n========================================")
print("CORRELATION WITH LAP-TIME DELTA")
print("========================================")

for feature in [
    "LapNumber",
    "TyreLife",
]:

    pearson = df[
        [feature, "LapTimeDelta"]
    ].corr(
        method="pearson"
    ).iloc[0, 1]

    spearman = df[
        [feature, "LapTimeDelta"]
    ].corr(
        method="spearman"
    ).iloc[0, 1]

    print(
        f"{feature}: "
        f"Pearson = {pearson:.3f}, "
        f"Spearman = {spearman:.3f}"
    )


# ============================================================
# LARGEST LAP-TO-LAP CHANGES
# ============================================================

print("\n========================================")
print("10 LARGEST ABSOLUTE DELTAS")
print("========================================")

largest_changes = (
    df.sort_values(
        "AbsoluteDelta",
        ascending=False
    )
    [
        [
            "Year",
            "GrandPrix",
            "LapNumber",
            "Compound",
            "TyreLife",
            "PreviousLapTime",
            "LapTimeSeconds",
            "LapTimeDelta",
            "AbsoluteDelta",
        ]
    ]
    .head(10)
)

print(
    largest_changes.to_string(
        index=False
    )
)