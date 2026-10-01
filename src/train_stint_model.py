import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/stint_dataset.csv"
OUTPUT_MODEL = "models/stint_slope_random_forest.pkl"

TARGET = "SmoothedSlope"

NUMERIC_FEATURES = [
    "StartLap",
    "StartTyreLife",
]

CATEGORICAL_FEATURES = [
    "Compound",
]

FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ============================================================
# LOAD DATA
# ============================================================

print(f"Loading dataset: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)


print("\n========================================")
print("TRAINING DATA")
print("========================================")

print(f"Stints: {len(df)}")
print(f"Drivers: {df['Driver'].nunique()}")

print(
    f"Races: "
    f"{df[['Year', 'GrandPrix']].drop_duplicates().shape[0]}"
)

print("\nCompound counts:")

print(
    df["Compound"].value_counts()
)


# ============================================================
# FEATURES / TARGET
# ============================================================

X = df[
    FEATURES
]

y = df[
    TARGET
]


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            NUMERIC_FEATURES,
        ),
        (
            "compound",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            CATEGORICAL_FEATURES,
        ),
    ]
)


# ============================================================
# RANDOM FOREST
# ============================================================
#
# Same conservative configuration that was evaluated using
# leave-one-race-out validation.
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=3,
                random_state=42,
            ),
        ),
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining Random Forest stint model...")

model.fit(
    X,
    y
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
    OUTPUT_MODEL
)


print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(
    f"Saved to: "
    f"{OUTPUT_MODEL}"
)


# ============================================================
# SIMPLE SANITY CHECK
# ============================================================
#
# These are NOT evaluation results.
# They simply verify that the saved model can generate
# predictions for example strategy inputs.
# ============================================================

examples = pd.DataFrame(
    [
        {
            "StartLap": 1,
            "StartTyreLife": 1,
            "Compound": "SOFT",
        },
        {
            "StartLap": 20,
            "StartTyreLife": 1,
            "Compound": "MEDIUM",
        },
        {
            "StartLap": 35,
            "StartTyreLife": 1,
            "Compound": "HARD",
        },
    ]
)


example_predictions = model.predict(
    examples
)


examples[
    "PredictedSlope"
] = example_predictions


print("\nExample predictions:")

print(
    examples.to_string(
        index=False
    )
)