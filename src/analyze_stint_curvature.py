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

OUTPUT_FILE = "data/quadratic_stint_dataset.csv"
ROLLING_WINDOW = 3

# Quadratic fits need a little more data than straight lines.
MIN_STINT_LAPS = 8


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
# HELPER: ADJUSTED R2
# ============================================================
#
# Ordinary R2 will almost always improve when another feature
# is added.
#
# Adjusted R2 penalises the quadratic model for using an
# additional term.
# ============================================================

def adjusted_r2(r2, n, predictors):

    denominator = (
        n
        - predictors
        - 1
    )

    if denominator <= 0:
        return float("nan")

    return (
        1
        - (
            (1 - r2)
            * (n - 1)
            / denominator
        )
    )


# ============================================================
# ANALYSE EACH STINT
# ============================================================

group_columns = [
    "Year",
    "GrandPrix",
    "Driver",
    "Stint",
    "Compound",
]


results = []


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
        stint_df[
            [
                "LapNumber",
                "TyreLife",
                "SmoothedLapTime",
            ]
        ]
        .dropna()
        .sort_values("TyreLife")
        .copy()
    )


    if len(stint_df) < MIN_STINT_LAPS:
        continue


    if stint_df["TyreLife"].nunique() < 3:
        continue


    # Use tyre age relative to the beginning of the stint.
    stint_df["AgeFromStart"] = (
        stint_df["TyreLife"]
        - stint_df["TyreLife"].min()
    )


    stint_df["AgeSquared"] = (
        stint_df["AgeFromStart"] ** 2
    )


    y = stint_df[
        "SmoothedLapTime"
    ]


    # ========================================================
    # LINEAR MODEL
    # ========================================================

    linear_model = LinearRegression()

    linear_model.fit(
        stint_df[
            ["AgeFromStart"]
        ],
        y,
    )


    linear_r2 = linear_model.score(
        stint_df[
            ["AgeFromStart"]
        ],
        y,
    )


    linear_adjusted_r2 = adjusted_r2(
        linear_r2,
        len(stint_df),
        predictors=1,
    )


    # ========================================================
    # QUADRATIC MODEL
    # ========================================================

    quadratic_model = LinearRegression()

    quadratic_model.fit(
        stint_df[
            [
                "AgeFromStart",
                "AgeSquared",
            ]
        ],
        y,
    )


    quadratic_r2 = quadratic_model.score(
        stint_df[
            [
                "AgeFromStart",
                "AgeSquared",
            ]
        ],
        y,
    )


    quadratic_adjusted_r2 = adjusted_r2(
        quadratic_r2,
        len(stint_df),
        predictors=2,
    )


    linear_term = (
        quadratic_model.coef_[0]
    )


    curvature = (
        quadratic_model.coef_[1]
    )


    results.append(
        {
            "Year": int(year),
            "GrandPrix": grand_prix,
            "Driver": driver,
            "Stint": int(stint),
            "Compound": compound,
            "Laps": len(stint_df),
            "StartTyreLife":(stint_df["TyreLife"].min()),

            "StartLap": (
                stint_df["LapNumber"].min()
            ),

            "MaxAgeFromStart":(
                stint_df["AgeFromStart"].max()
            ),

            "LinearR2": linear_r2,
            "LinearAdjustedR2": (
                linear_adjusted_r2
            ),

            "QuadraticR2": quadratic_r2,
            "QuadraticAdjustedR2": (
                quadratic_adjusted_r2
            ),

            "AdjustedR2Improvement": (
                quadratic_adjusted_r2
                - linear_adjusted_r2
            ),

            "QuadraticLinearTerm": (
                linear_term
            ),

            "Curvature": curvature,
        }
    )


results_df = pd.DataFrame(
    results
)

# ============================================================
# IDENTIFY WITHIN-STINT TURNING POINTS
# ============================================================

# For:
#
# pace = a + b*x + c*x^2
#
# turning point:
#
# x = -b / (2*c)
#
# We only care about positive curvature here because that can
# represent:
#
# faster -> optimum -> slower
#
# within the observed stint.

results_df["TurningPoint"] = float("nan")


positive_curve_mask = (
    results_df["Curvature"] > 0
)

results_df.loc[
    positive_curve_mask,
    "TurningPoint"
] = (
    -results_df.loc[
        positive_curve_mask,
        "QuadraticLinearTerm"
    ]
    /
    (
        2
        * results_df.loc[
            positive_curve_mask,
            "Curvature"
        ]
    )
)
results_df["FalloffWithinStint"] = (
    (results_df["Curvature"] > 0)
    & (results_df["TurningPoint"] > 0)
    & (
        results_df["TurningPoint"]
        <= results_df["MaxAgeFromStart"]
    )
)
# ============================================================
# OVERALL SUMMARY
# ============================================================

print("\n========================================")
print("STINT CURVATURE ANALYSIS")
print("========================================")

print(
    f"Stints analysed: "
    f"{len(results_df)}"
)


print(
    f"Mean linear adjusted R2: "
    f"{results_df['LinearAdjustedR2'].mean():.3f}"
)


print(
    f"Mean quadratic adjusted R2: "
    f"{results_df['QuadraticAdjustedR2'].mean():.3f}"
)


print(
    f"Median linear adjusted R2: "
    f"{results_df['LinearAdjustedR2'].median():.3f}"
)


print(
    f"Median quadratic adjusted R2: "
    f"{results_df['QuadraticAdjustedR2'].median():.3f}"
)


quadratic_better = (
    results_df[
        "AdjustedR2Improvement"
    ] > 0
).sum()


print(
    f"Stints where quadratic fit is better: "
    f"{quadratic_better} / "
    f"{len(results_df)}"
)


positive_curvature = (
    results_df[
        "Curvature"
    ] > 0
).sum()


print(
    f"Stints with positive curvature: "
    f"{positive_curvature} / "
    f"{len(results_df)}"
)


# ============================================================
# BY COMPOUND
# ============================================================

print("\n========================================")
print("CURVATURE BY COMPOUND")
print("========================================")


compound_summary = (
    results_df.groupby(
        "Compound"
    )
    .agg(
        Stints=(
            "Curvature",
            "size",
        ),
        MedianCurvature=(
            "Curvature",
            "median",
        ),
        MeanCurvature=(
            "Curvature",
            "mean",
        ),
        MedianR2Improvement=(
            "AdjustedR2Improvement",
            "median",
        ),
    )
    .reset_index()
)


positive_percentage = (
    results_df.assign(
        PositiveCurvature=(
            results_df["Curvature"] > 0
        )
    )
    .groupby("Compound")[
        "PositiveCurvature"
    ]
    .mean()
    * 100
)


compound_summary[
    "PositiveCurvaturePct"
] = (
    compound_summary[
        "Compound"
    ].map(
        positive_percentage
    )
)


print(
    compound_summary.to_string(
        index=False,
        formatters={
            "MedianCurvature": (
                lambda x: f"{x:+.5f}"
            ),
            "MeanCurvature": (
                lambda x: f"{x:+.5f}"
            ),
            "MedianR2Improvement": (
                lambda x: f"{x:+.3f}"
            ),
            "PositiveCurvaturePct": (
                lambda x: f"{x:.1f}%"
            ),
        }
    )
)


# ============================================================
# BIGGEST QUADRATIC IMPROVEMENTS
# ============================================================

print("\n========================================")
print("10 STINTS MOST HELPED BY CURVATURE")
print("========================================")


print(
    results_df.sort_values(
        "AdjustedR2Improvement",
        ascending=False,
    )
    .head(10)
    [
        [
            "Year",
            "GrandPrix",
            "Driver",
            "Stint",
            "Compound",
            "Laps",
            "LinearAdjustedR2",
            "QuadraticAdjustedR2",
            "AdjustedR2Improvement",
            "Curvature",
        ]
    ]
    .to_string(
        index=False,
        formatters={
            "LinearAdjustedR2": (
                lambda x: f"{x:.3f}"
            ),
            "QuadraticAdjustedR2": (
                lambda x: f"{x:.3f}"
            ),
            "AdjustedR2Improvement": (
                lambda x: f"{x:+.3f}"
            ),
            "Curvature": (
                lambda x: f"{x:+.5f}"
            ),
        }
    )
)

# ============================================================
# TURNING-POINT SUMMARY
# ============================================================
print("\n========================================")
print("WITHIN-STINT FALLOFF")
print("========================================")
falloff_count = (
    results_df[
        "FalloffWithinStint"
    ].sum()
)

print(
    f"Stints with observed optimum then falloff: "
    f"{falloff_count} / {len(results_df)}"
)

print("\nBy compound:")


falloff_summary = (
    results_df.groupby("Compound")
    .agg(
        Stints=(
            "Compound",
            "size",
        ),
        FalloffStints=(
            "FalloffWithinStint",
            "sum",
        ),
        MedianTurningPoint=(
            "TurningPoint",
            "median",
        ),
    )
    .reset_index()
)


falloff_summary[
    "FalloffPct"
] = (
    falloff_summary[
        "FalloffStints"
    ]
    / falloff_summary[
        "Stints"
    ]
    * 100
)

print(
    falloff_summary.to_string(
        index=False,
        formatters={
            "MedianTurningPoint": (
                lambda x: f"{x:.1f}"
            ),
            "FalloffPct": (
                lambda x: f"{x:.1f}%"
            ),
        }
    )
)


# ============================================================
# QUALITY-FILTERED FALLOFF
# ============================================================
#
# R2 >= 0.5 is used here as a diagnostic threshold, not a
# universal scientific rule.
#
# A "credible" falloff must:
#
# 1. Have positive curvature
# 2. Turn within the observed stint
# 3. Improve upon the linear fit
# 4. Have a reasonably strong quadratic fit
# ============================================================

results_df["CredibleFalloff"] = (
    results_df["FalloffWithinStint"]
    & (
        results_df[
            "AdjustedR2Improvement"
        ] > 0
    )
    & (
        results_df[
            "QuadraticAdjustedR2"
        ] >= 0.5
    )
)


# Convert the relative turning point into actual tyre age.
results_df["TurningTyreLife"] = (
    results_df["StartTyreLife"]
    + results_df["TurningPoint"]
)
print("\n========================================")
print("STINTS WITH OBSERVED FALLOFF")
print("========================================")

falloff_stints = (
    results_df[
        results_df[
            "FalloffWithinStint"
        ]
    ]
    .sort_values(
        [
            "Compound",
            "TurningPoint",
        ]
    )
)

print(
    falloff_stints[
        [
            "Year",
            "GrandPrix",
            "Driver",
            "Stint",
            "Compound",
            "Laps",
            "TurningPoint",
            "MaxAgeFromStart",
            "QuadraticAdjustedR2",
            "Curvature",
        ]
    ]
    .to_string(
        index=False,
        formatters={
            "TurningPoint": (
                lambda x: f"{x:.1f}"
            ),
            "MaxAgeFromStart": (
                lambda x: f"{x:.1f}"
            ),
            "QuadraticAdjustedR2": (
                lambda x: f"{x:.3f}"
            ),
            "Curvature": (
                lambda x: f"{x:+.5f}"
            ),
        }
    )
)

# ============================================================
# CREDIBLE FALLOFF SUMMARY
# ============================================================

print("\n========================================")
print("QUALITY-FILTERED FALLOFF")
print("========================================")


credible_count = (
    results_df[
        "CredibleFalloff"
    ].sum()
)


print(
    f"Credible falloff stints: "
    f"{credible_count} / {len(results_df)}"
)


credible_summary = (
    results_df.groupby("Compound")
    .agg(
        Stints=(
            "Compound",
            "size",
        ),
        CredibleFalloffStints=(
            "CredibleFalloff",
            "sum",
        ),
    )
    .reset_index()
)

credible_summary[
    "CredibleFalloffPct"
] = (
    credible_summary[
        "CredibleFalloffStints"
    ]
    / credible_summary[
        "Stints"
    ]
    * 100
)


print("\nBy compound:")

print(
    credible_summary.to_string(
        index=False,
        formatters={
            "CredibleFalloffPct": (
                lambda x: f"{x:.1f}%"
            ),
        }
    )
)

print("\n========================================")
print("CREDIBLE FALLOFF STINTS")
print("========================================")
credible_stints = (
    results_df[
        results_df[
            "CredibleFalloff"
        ]
    ]
    .sort_values(
        [
            "Compound",
            "TurningTyreLife",
        ]
    )
)
print(
    credible_stints[
        [
            "Year",
            "GrandPrix",
            "Driver",
            "Stint",
            "Compound",
            "Laps",
            "StartTyreLife",
            "TurningPoint",
            "TurningTyreLife",
            "QuadraticAdjustedR2",
            "AdjustedR2Improvement",
            "Curvature",
        ]
    ]
    .to_string(
        index=False,
        formatters={
            "TurningPoint": (
                lambda x: f"{x:.1f}"
            ),
            "TurningTyreLife": (
                lambda x: f"{x:.1f}"
            ),
            "QuadraticAdjustedR2": (
                lambda x: f"{x:.3f}"
            ),
            "AdjustedR2Improvement": (
                lambda x: f"{x:+.3f}"
            ),
            "Curvature": (
                lambda x: f"{x:+.5f}"
            ),
        }
    )
)
# ============================================================
# SAVE QUADRATIC STINT DATASET
# ============================================================

output_columns = [
    "Year",
    "GrandPrix",
    "Driver",
    "Stint",
    "Compound",
    "Laps",
    "StartLap",
    "StartTyreLife",
    "MaxAgeFromStart",
    "QuadraticLinearTerm",
    "Curvature",
    "LinearAdjustedR2",
    "QuadraticAdjustedR2",
    "AdjustedR2Improvement",
    "TurningPoint",
    "TurningTyreLife",
    "FalloffWithinStint",
    "CredibleFalloff",
]


quadratic_dataset = results_df[
    output_columns
].copy()


quadratic_dataset.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n========================================")
print("QUADRATIC STINT DATASET")
print("========================================")
print(
    f"Rows: "
    f"{len(quadratic_dataset)}"
)
print(
    f"Races: "
    f"{quadratic_dataset[['Year', 'GrandPrix']].drop_duplicates().shape[0]}"
)
print(
    f"Drivers: "
    f"{quadratic_dataset['Driver'].nunique()}"
)
print(
    f"Saved to: "
    f"{OUTPUT_FILE}"
)