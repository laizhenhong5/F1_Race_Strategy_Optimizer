# import pandas as pd
# from sklearn.linear_model import LinearRegression
# #================================
# #Settings
# #================================
# INPUT_FILES=[
#     #George Russell
#     "data/normal_racing_laps_las_vegas_2024.csv",
#     "data/normal_racing_laps_bahrain_2025.csv",
#     "data/normal_racing_laps_japan_2025.csv",
#     "data/normal_racing_laps_miami_2025_repaired.csv",
#     "data/normal_racing_laps_emilia_romagna_2025.csv",
#     "data/normal_racing_laps_barcelona_catalunya_2025.csv",
#     "data/normal_racing_laps_canada_2025.csv",
#     "data/normal_racing_laps_singapore_2025.csv",

#     #Kimi Antonelli
#     "data/normal_racing_laps_antonelli_bahrain_2025.csv",
#     "data/normal_racing_laps_antonelli_japan_2025.csv",
#     "data/normal_racing_laps_antonelli_canada_2025.csv",
#     "data/normal_racing_laps_antonelli_barcelona_catalunya_2025.csv",
#     "data/normal_racing_laps_antonelli_emilia_romagna_2025.csv",
#     "data/normal_racing_laps_antonelli_miami_2025_repaired.csv",
#     "data/normal_racing_laps_antonelli_singapore_2025.csv",
# ]

# OUTPUT_FILE= "data/stint_dataset.csv"
# ROLLING_WINDOW=3
# MIN_STINT_LAPS = 5
# #===============================
# #LOAD DATA
# #===============================
# datasets=[]
# for file in INPUT_FILES:
#     print(f"Loading: {file}")
#     df=pd.read_csv(file)
#     datasets.append(df)

# df=pd.concat(
#     datasets, ignore_index=True
# )

# numeric_columns = ["LapNumber", "LapTimeSeconds","TyreLife","Stint"]

# for column in numeric_columns:
#     df[column] = pd.to_numeric( df[column], errors="coerce")

# df=(df.sort_values(
#     ["Year", "GrandPrix","Driver","Stint", "LapNumber"]
# ).reset_index(drop=True))


# #====================================
# #IDENTIFY CONSECUTIVE LAP SEGMENTS
# #====================================
# stint_groups=["Year", "GrandPrix", "Driver","Stint"]

# df["LapDifference"]=(
#     df.groupby(stint_groups)[
#         "LapNumber"
#     ]
#     .diff()
# )

# df["NewSegment"]= (
#     (
#         df["LapDifference"].isna() | (df["lapDifference"]!=1)
#     )
#     .astype(int)
# )

# df["ConsecutiveSegment"] = (
#     df.groupby(stint_groups)[
#         "NewSegment"
#     ]
#     .cumsum()
# )


# # ============================================================
# # CREATE SMOOTHED LAP TIME
# # ============================================================

# smoothing_groups = (
#     stint_groups
#     + ["ConsecutiveSegment"]
# )


# df["SmoothedLapTime"] = (
#     df.groupby(smoothing_groups)[
#         "LapTimeSeconds"
#     ]
#     .transform(
#         lambda series:
#         series.rolling(
#             window=ROLLING_WINDOW,
#             center=True,
#             min_periods=1
#         ).median()
#     )
# )


# # ============================================================
# # BUILD ONE ROW PER STINT
# # ============================================================

# results = []


# group_columns = [
#     "Year",
#     "GrandPrix",
#     "Driver",
#     "Stint",
#     "Compound",
# ]


# for group_values, stint_df in df.groupby(
#     group_columns
# ):

#     (
#         year,
#         grand_prix,
#         driver,
#         stint,
#         compound,
#     ) = group_values


#     stint_df = (
#         stint_df[
#             [
#                 "LapNumber",
#                 "TyreLife",
#                 "LapTimeSeconds",
#                 "SmoothedLapTime",
#                 "ConsecutiveSegment",
#             ]
#         ]
#         .dropna()
#         .sort_values("TyreLife")
#     )


#     # Ignore short stints.
#     if len(stint_df) < MIN_STINT_LAPS:
#         continue


#     # Need tyre-age variation.
#     if stint_df["TyreLife"].nunique() < 2:
#         continue


#     X = stint_df[
#         ["TyreLife"]
#     ]


#     # ========================================================
#     # RAW TREND
#     # ========================================================

#     raw_model = LinearRegression()

#     raw_model.fit(
#         X,
#         stint_df["LapTimeSeconds"]
#     )


#     raw_slope = raw_model.coef_[0]

#     raw_r2 = raw_model.score(
#         X,
#         stint_df["LapTimeSeconds"]
#     )


#     # ========================================================
#     # SMOOTHED TREND
#     # ========================================================

#     smooth_model = LinearRegression()

#     smooth_model.fit(
#         X,
#         stint_df["SmoothedLapTime"]
#     )


#     smoothed_slope = (
#         smooth_model.coef_[0]
#     )

#     smoothed_r2 = (
#         smooth_model.score(
#             X,
#             stint_df["SmoothedLapTime"]
#         )
#     )


#     # ========================================================
#     # SAVE STINT
#     # ========================================================

#     results.append({

#         "Year": int(year),

#         "GrandPrix": grand_prix,

#         "Driver": driver,

#         "Stint": int(stint),

#         "Compound": compound,

#         "Laps": len(stint_df),

#         "StartLap": (
#             stint_df["LapNumber"].min()
#         ),

#         "EndLap": (
#             stint_df["LapNumber"].max()
#         ),

#         "StartTyreLife": (
#             stint_df["TyreLife"].min()
#         ),

#         "EndTyreLife": (
#             stint_df["TyreLife"].max()
#         ),

#         "TyreAgeRange": (
#             stint_df["TyreLife"].max()
#             - stint_df["TyreLife"].min()
#         ),

#         "NumSegments": (
#             stint_df[
#                 "ConsecutiveSegment"
#             ].nunique()
#         ),

#         "MedianLapTime": (
#             stint_df[
#                 "LapTimeSeconds"
#             ].median()
#         ),

#         "MedianSmoothedLapTime": (
#             stint_df[
#                 "SmoothedLapTime"
#             ].median()
#         ),

#         "RawSlope": raw_slope,

#         "RawR2": raw_r2,

#         "SmoothedSlope": smoothed_slope,

#         "SmoothedR2": smoothed_r2,
#     })


# # ============================================================
# # CREATE DATASET
# # ============================================================

# stint_dataset = pd.DataFrame(
#     results
# )


# stint_dataset = (
#     stint_dataset.sort_values(
#         [
#             "Year",
#             "GrandPrix",
#             "Driver",
#             "Stint",
#         ]
#     )
#     .reset_index(drop=True)
# )


# # ============================================================
# # SAVE
# # ============================================================

# stint_dataset.to_csv(
#     OUTPUT_FILE,
#     index=False
# )


# # ============================================================
# # SUMMARY
# # ============================================================

# print("\n========================================")
# print("STINT DATASET")
# print("========================================")

# print(
#     f"Rows: "
#     f"{len(stint_dataset)}"
# )

# print(
#     f"Drivers: "
#     f"{stint_dataset['Driver'].nunique()}"
# )

# print(
#     f"Races: "
#     f"{stint_dataset[['Year', 'GrandPrix']].drop_duplicates().shape[0]}"
# )


# print("\nStints by compound:")

# print(
#     stint_dataset[
#         "Compound"
#     ].value_counts()
# )


# print("\nFirst 10 rows:")

# print(
#     stint_dataset.head(10).to_string(
#         index=False
#     )
# )


# print(
#     f"\nSaved to: "
#     f"{OUTPUT_FILE}"
# )

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

OUTPUT_FILE = "data/stint_dataset.csv"

ROLLING_WINDOW = 3
MIN_STINT_LAPS = 5


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


numeric_columns = [
    "LapNumber",
    "LapTimeSeconds",
    "TyreLife",
    "Stint",
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
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
# IDENTIFY CONSECUTIVE LAP SEGMENTS
# ============================================================

stint_groups = [
    "Year",
    "GrandPrix",
    "Driver",
    "Stint",
]


df["LapDifference"] = (
    df.groupby(stint_groups)[
        "LapNumber"
    ]
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
    df.groupby(stint_groups)[
        "NewSegment"
    ]
    .cumsum()
)


# ============================================================
# CREATE SMOOTHED LAP TIME
# ============================================================

smoothing_groups = (
    stint_groups
    + ["ConsecutiveSegment"]
)


df["SmoothedLapTime"] = (
    df.groupby(smoothing_groups)[
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


# ============================================================
# BUILD ONE ROW PER STINT
# ============================================================

results = []


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
        stint_df[
            [
                "LapNumber",
                "TyreLife",
                "LapTimeSeconds",
                "SmoothedLapTime",
                "ConsecutiveSegment",
            ]
        ]
        .dropna()
        .sort_values("TyreLife")
    )


    # Ignore short stints.
    if len(stint_df) < MIN_STINT_LAPS:
        continue


    # Need tyre-age variation.
    if stint_df["TyreLife"].nunique() < 2:
        continue


    X = stint_df[
        ["TyreLife"]
    ]


    # ========================================================
    # RAW TREND
    # ========================================================

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


    # ========================================================
    # SMOOTHED TREND
    # ========================================================

    smooth_model = LinearRegression()

    smooth_model.fit(
        X,
        stint_df["SmoothedLapTime"]
    )


    smoothed_slope = (
        smooth_model.coef_[0]
    )

    smoothed_r2 = (
        smooth_model.score(
            X,
            stint_df["SmoothedLapTime"]
        )
    )


    # ========================================================
    # SAVE STINT
    # ========================================================

    results.append({

        "Year": int(year),

        "GrandPrix": grand_prix,

        "Driver": driver,

        "Stint": int(stint),

        "Compound": compound,

        "Laps": len(stint_df),

        "StartLap": (
            stint_df["LapNumber"].min()
        ),

        "EndLap": (
            stint_df["LapNumber"].max()
        ),

        "StartTyreLife": (
            stint_df["TyreLife"].min()
        ),

        "EndTyreLife": (
            stint_df["TyreLife"].max()
        ),

        "TyreAgeRange": (
            stint_df["TyreLife"].max()
            - stint_df["TyreLife"].min()
        ),

        "NumSegments": (
            stint_df[
                "ConsecutiveSegment"
            ].nunique()
        ),

        "MedianLapTime": (
            stint_df[
                "LapTimeSeconds"
            ].median()
        ),

        "MedianSmoothedLapTime": (
            stint_df[
                "SmoothedLapTime"
            ].median()
        ),

        "RawSlope": raw_slope,

        "RawR2": raw_r2,

        "SmoothedSlope": smoothed_slope,

        "SmoothedR2": smoothed_r2,
    })


# ============================================================
# CREATE DATASET
# ============================================================

stint_dataset = pd.DataFrame(
    results
)


stint_dataset = (
    stint_dataset.sort_values(
        [
            "Year",
            "GrandPrix",
            "Driver",
            "Stint",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# SAVE
# ============================================================

stint_dataset.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("STINT DATASET")
print("========================================")

print(
    f"Rows: "
    f"{len(stint_dataset)}"
)

print(
    f"Drivers: "
    f"{stint_dataset['Driver'].nunique()}"
)

print(
    f"Races: "
    f"{stint_dataset[['Year', 'GrandPrix']].drop_duplicates().shape[0]}"
)


print("\nStints by compound:")

print(
    stint_dataset[
        "Compound"
    ].value_counts()
)


print("\nFirst 10 rows:")

print(
    stint_dataset.head(10).to_string(
        index=False
    )
)


print(
    f"\nSaved to: "
    f"{OUTPUT_FILE}"
)