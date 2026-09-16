import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
# ============================================================
# SETTINGS
# ============================================================
INPUT_FILE = "data/combined_feature_engineered.csv"
FEATURES = [
    "LapNumber",
    "CompoundEncoded",
    "TyreLife",
]

TARGET = "LapTimeDelta"

# ============================================================
# LOAD DATASET
# ============================================================
print(f"Loading dataset: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

# ============================================================
# FIND AVAILABLE RACES
# ============================================================

races = (
    df[
        ["Year", "GrandPrix"]
    ]
    .drop_duplicates()
    .sort_values(
        ["Year", "GrandPrix"]
    )
    .reset_index(drop=True)
)


print("\n========================================")
print("AVAILABLE RACES")
print("========================================")
print(races.to_string(index=False))


# ============================================================
# LEAVE-ONE-RACE-OUT EVALUATION
# ============================================================

evaluation_results = []

total_absolute_error = 0.0
total_baseline_absolute_error = 0.0
total_test_rows = 0


for _, race in races.iterrows():

    test_year = int(race["Year"])
    test_grand_prix = race["GrandPrix"]

    print("\n========================================")
    print(
        f"TESTING: "
        f"{test_year} {test_grand_prix}"
    )
    print("========================================")


    # --------------------------------------------------------
    # TEST DATA
    # --------------------------------------------------------

    test_mask = (
        (df["Year"] == test_year)
        & (df["GrandPrix"] == test_grand_prix)
    )

    test_df = df[test_mask].copy()


    # --------------------------------------------------------
    # TRAINING DATA
    # --------------------------------------------------------

    train_df = df[~test_mask].copy()


    print(
        f"Training rows: {len(train_df)}"
    )

    print(
        f"Testing rows: {len(test_df)}"
    )


    # --------------------------------------------------------
    # FEATURES / TARGET
    # --------------------------------------------------------

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]


    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # ============================================================
    # RECONSTRUCT LAP TIMES
    # ============================================================

    actual_lap_times = test_df[
        "LapTimeSeconds"
    ].values

    predicted_lap_times = (
        test_df["PreviousLapTime"].values
        + y_pred
    )
    # ============================================================
    # NAIVE BASELINE
    # ============================================================

    # Simplest prediction:
    # assume the next lap takes exactly as long as the previous lap.
    baseline_predicted_lap_times = (
    test_df["PreviousLapTime"].values
    )

    # ============================================================
    # EVALUATE RECONSTRUCTED LAP TIMES
    # ============================================================

    mae = mean_absolute_error(
        actual_lap_times,
        predicted_lap_times
    )

    r2 = r2_score(
        actual_lap_times,
        predicted_lap_times
    )
    baseline_mae = mean_absolute_error(
        actual_lap_times,
        baseline_predicted_lap_times
    )

    baseline_r2 = r2_score(
        actual_lap_times,
        baseline_predicted_lap_times
    )

    print(
        f"MAE: {mae:.3f} seconds"
    )

    print(
        f"R2:  {r2:.3f}"
    )

    print(
        f"Baseline MAE: {baseline_mae:.3f} seconds"
    )

    print(
        f"Baseline R2:  {baseline_r2:.3f}"
   )



    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    evaluation_results.append({
        "TestYear": test_year,
        "TestRace": test_grand_prix,
        "TrainingRows": len(train_df),
        "TestingRows": len(test_df),
        "MAE": mae,
        "R2": r2,
        "BaselineMAE": baseline_mae,
        "BaselineR2": baseline_r2,
    })


    total_absolute_error += (
        abs(
            actual_lap_times
            - predicted_lap_times
        ).sum()
    )

    total_baseline_absolute_error += (
        abs(
            actual_lap_times
            - baseline_predicted_lap_times
        ).sum()
    )

    total_test_rows += len(test_df)



# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    evaluation_results
)


print("\n========================================")
print("LEAVE-ONE-RACE-OUT SUMMARY")
print("========================================")

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": lambda x: f"{x:.3f}",
            "R2": lambda x: f"{x:.3f}",
            "BaselineMAE": lambda x: f"{x:.3f}",
            "BaselineR2": lambda x: f"{x:.3f}",
        }
    )
)


# ============================================================
# OVERALL MAE
# ============================================================

overall_mae = (
    total_absolute_error
    / total_test_rows
)
overall_baseline_mae = (
    total_baseline_absolute_error
    / total_test_rows
)

print("\n========================================")
print("OVERALL RESULT")
print("========================================")

print(
    f"Total evaluated laps: "
    f"{total_test_rows}"
)

print(
    f"Overall MAE: "
    f"{overall_mae:.3f} seconds"
)

print(
    f"Overall ML MAE: "
    f"{overall_mae:.3f} seconds"
)

print(
    f"Overall baseline MAE: "
    f"{overall_baseline_mae:.3f} seconds"
)