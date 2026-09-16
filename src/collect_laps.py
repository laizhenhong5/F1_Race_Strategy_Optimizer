#FIRST STEP
#All datasets used by the pipeline should be produced by the current version of collect_laps.py.
#previously inspect_laps.py
from pathlib import Path
import fastf1
import pandas as pd

# =========================
# SETTINGS (just change these values for different sessions)
# =========================

#These settings here are for the 2025 Spanish GP, Russell's race OR NEW ONES AFTER SPANISH/BARCELONA
YEAR = 2025
FASTF1_EVENT = "Canada"
GRAND_PRIX = "Canada"
SESSION = "R"
DRIVER = "RUS"
OUTPUT_FILE = "data/cleaned_russell_canada_2025.csv"

#DEFAULT COLLECT SETTINGS
# YEAR = 2025
# GRAND_PRIX = "Emilia Romagna"
# SESSION = "R"
# DRIVER = "RUS"
# OUTPUT_FILE = "data/cleaned_russell_emilia_romagna_2025.csv"

#set up fastf1 cache directory, ensure the cache directory exists before fasf1 uses it
cache_dir= Path ("cache")
cache_dir.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))  # Enable FastF1 cache

# =========================
# LOAD SESSION
# =========================

print(f"Loading {YEAR} {GRAND_PRIX}...")
#for 2025 spanish gp
session = fastf1.get_session(
    YEAR,
    FASTF1_EVENT,
    SESSION
)

#DEFAULT SETTINGS
# session = fastf1.get_session(
#     YEAR,
#     GRAND_PRIX,
#     SESSION
# )

session.load()
# =========================
# GET DRIVER LAP DATA
# =========================
# FastF1 currently warns that pick_driver() is deprecated.
# For now we keep it simple and use the replacement method.
laps = session.laps.pick_drivers(DRIVER)


# =========================
# SELECT USEFUL COLUMNS
# =========================

useful_columns = [
    "Driver",
    "LapNumber",
    "LapTime",
    "Compound",
    "TyreLife",
    "Stint",
    "TrackStatus",
    "PitInTime",
    "PitOutTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "Position"
]

available_columns = [
    column
    for column in useful_columns
    if column in laps.columns
]

df = laps[available_columns].copy()


# =========================
# CLEAN LAP DATA
# =========================

# Remove rows where we don't have a valid lap time
if "LapTime" in df.columns:
    df = df.dropna(subset=["LapTime"])

    df["LapTimeSeconds"] = (
        df["LapTime"].dt.total_seconds()
    )


# Convert sector times to seconds
sector_columns = [
    "Sector1Time",
    "Sector2Time",
    "Sector3Time"
]

for column in sector_columns:
    if column in df.columns:
        df[f"{column}Seconds"] = (
            df[column].dt.total_seconds()
        )

# ============================================================
# ADD RACE METADATA
# ============================================================

df["Year"] = YEAR
df["GrandPrix"] = GRAND_PRIX
df["Session"] = SESSION

# =========================
# DISPLAY RESULT
# =========================
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10))

print("\nNumber of laps:")
print(len(df))

# =========================
# SAVE CSV
# =========================

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to: {OUTPUT_FILE}")