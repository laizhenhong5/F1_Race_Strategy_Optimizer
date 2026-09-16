# import pandas as pd #loads to tool to clean and organize data
# from sklearn.model_selection import train_test_split  #splits the data into training and testing sets
# from sklearn.ensemble import RandomForestRegressor #imports an AI model that predicts numbers using decision trees
# from sklearn.metrics import mean_absolute_error, r2_score #tools to measure AI predicition accuracy
# import joblib #streamline
# import matplotlib.pyplot as plt

# # Load engineered data
# df = pd.read_csv("data/feature_engineered_russell_singapore_2025.csv")

# # Remove rows with missing values
# df = df.dropna().reset_index(drop=True)

# # Select features and target
# features = [
#     "LapNumber",
#     "CompoundEncoded",
#     "TyreLife",
#     "TyreLifeNormalized",
#     "Stint",
#     "StintLap",
#     "TrackStatus",
#     "IsPitLap"
# ]

# X = df[features]
# y = df["LapTimeSeconds"]

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Train model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Predict
# y_pred = model.predict(X_test)

# # Evaluate
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Model trained successfully.")
# print(f"MAE: {mae:.3f}")
# print(f"R2: {r2:.3f}")

# # Save model
# joblib.dump(model, "data/laptime_predictor_model.pkl")
# print("Model saved as data/laptime_predictor_model.pkl")

# # Plot actual vs predicted
# plt.figure(figsize=(8, 5))
# plt.scatter(y_test, y_pred)
# plt.xlabel("Actual Lap Time (seconds)")
# plt.ylabel("Predicted Lap Time (seconds)")
# plt.title("Actual vs Predicted Lap Time")
# plt.grid(True)
# plt.show()

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
# ============================================================
# SETTINGS
# ============================================================
INPUT_FILE = (
    "data/feature_engineered_normal_las_vegas_2024.csv"
)
MODEL_FILE = "models/laptime_random_forest.pkl"

# ============================================================
# LOAD DATA
# ============================================================
print(f"Loading dataset: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    "LapNumber"
).reset_index(drop=True)

# ============================================================
# DEFINE FEATURES AND TARGET
# ============================================================

features = [
    "LapNumber",
    "CompoundEncoded",
    "TyreLife",
    "StintLap",
    "TrackStatus",
    "Position",
]

target = "LapTimeSeconds"


X = df[features]

y = df[target]


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\n========================================")
print("TRAIN / TEST SPLIT")
print("========================================")

print(f"Total rows: {len(df)}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")


print(
    f"Training laps: "
    f"{df.iloc[:split_index]['LapNumber'].min():.0f}"
    f" → "
    f"{df.iloc[:split_index]['LapNumber'].max():.0f}"
)

print(
    f"Testing laps: "
    f"{df.iloc[split_index:]['LapNumber'].min():.0f}"
    f" → "
    f"{df.iloc[split_index:]['LapNumber'].max():.0f}"
)


# ============================================================
# TRAIN RANDOM FOREST
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
# MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# EVALUATE MODEL
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

print(f"MAE: {mae:.3f} seconds")
print(f"R2:  {r2:.3f}")


# ============================================================
# SHOW ACTUAL VS PREDICTED
# ============================================================

results = pd.DataFrame({
    "LapNumber": df.iloc[split_index:]["LapNumber"].values,
    "ActualLapTime": y_test.values,
    "PredictedLapTime": y_pred,
})

print("\n========================================")
print("PREDICTIONS")
print("========================================")

print(results.to_string(index=False))


# ============================================================
# SAVE MODEL
# ============================================================

import os

os.makedirs(
    os.path.dirname(MODEL_FILE),
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)

print(
    f"\nModel saved to: {MODEL_FILE}"
)