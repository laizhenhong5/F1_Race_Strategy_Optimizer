import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/combined_feature_engineered.csv"

# ============================================================
# TRAIN / TEST RACE SETTINGS
# ============================================================

TRAIN_RACES = [
    (2025, "Bahrain"),
    (2025, "Singapore"),
]

TEST_RACE = (
    2024,
    "Las Vegas",
)
MODEL_FILE = "models/race_based_laptime_model.pkl"


# ============================================================
# LOAD COMBINED DATASET
# ============================================================

print(f"Loading dataset: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)


# ============================================================
# SORT DATA
# ============================================================

df = df.sort_values(
    [
        "Year",
        "GrandPrix",
        "Driver",
        "LapNumber",
    ]
).reset_index(drop=True)


# ============================================================
# DEFINE FEATURES AND TARGET
# ============================================================

features = [
    "LapNumber",
    "PreviousLapTime",
    "CompoundEncoded",
    "TyreLife",
    "StintLap",
]

target = "LapTimeSeconds"


# ============================================================
# CREATE TRAINING DATA
# ============================================================

train_mask = pd.Series(
    False,
    index=df.index
)

for year, grand_prix in TRAIN_RACES:

    train_mask |= (
        (df["Year"] == year)
        & (df["GrandPrix"] == grand_prix)
    )

train_df = df[train_mask].copy()


# ============================================================
# CREATE TEST DATA
# ============================================================

test_year, test_grand_prix = TEST_RACE

test_mask = (
    (df["Year"] == test_year)
    & (df["GrandPrix"] == test_grand_prix)
)

test_df = df[test_mask].copy()


# ============================================================
# CHECK DATA
# ============================================================

print("\n========================================")
print("RACE-BASED DATA SPLIT")
print("========================================")

print("\nTraining races:")

for year, grand_prix in TRAIN_RACES:
    print(
        f"- {year} {grand_prix}"
    )

print(
    f"\nTesting race: "
    f"{test_year} {test_grand_prix}"
)

print(
    f"\nTraining rows: {len(train_df)}"
)

print(
    f"Testing rows: {len(test_df)}"
)


# Stop early if either dataset is empty.
if train_df.empty:
    raise ValueError(
        "Training race was not found in the dataset."
    )

if test_df.empty:
    raise ValueError(
        "Testing race was not found in the dataset."
    )


# ============================================================
# BUILD X / y
# ============================================================

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICT
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# EVALUATE
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n========================================")
print("MODEL RESULTS")
print("========================================")

print(
    f"MAE: {mae:.3f} seconds"
)

print(
    f"R2: {r2:.3f}"
)


# ============================================================
# SHOW PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Year": test_df["Year"].values,
    "GrandPrix": test_df["GrandPrix"].values,
    "Driver": test_df["Driver"].values,
    "LapNumber": test_df["LapNumber"].values,
    "ActualLapTime": y_test.values,
    "PredictedLapTime": y_pred,
})

print("\n========================================")
print("PREDICTIONS")
print("========================================")

print(
    results.to_string(index=False)
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)

print(
    f"\nModel saved to: {MODEL_FILE}"
)