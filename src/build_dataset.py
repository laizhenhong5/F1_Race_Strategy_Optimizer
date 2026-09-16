#load several races, collect lap data, combine everything into one csv
import fastf1
import pandas as pd

# Enable cache
fastf1.Cache.enable_cache("cache")

# Choose races to collect
races = [
    (2025, "Bahrain", "R"),
    (2025, "Miami", "R"),
    (2025, "Imola", "R"),
    (2025, "Spain", "R")
]

all_rows = []

for year, gp, session_type in races:
    print(f"Loading {year} {gp}...")
    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load()

        laps = session.laps

        # Keep only useful columns if they exist
        useful_cols = [
            "Time",
            "Driver",
            "LapNumber",
            "LapTime",
            "Compound",
            "TyreLife",
            "Stint",
            "TrackStatus",
            "PitInTime",
            "PitOutTime"
        ]

        available_cols = [col for col in useful_cols if col in laps.columns]
        laps = laps[available_cols].copy()

        # Convert LapTime to seconds
        if "LapTime" in laps.columns:
            laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

        # Add race info
        laps["Year"] = year
        laps["GrandPrix"] = gp

        all_rows.append(laps)

        print(f"Finished {year} {gp}, rows collected: {len(laps)}")

    except Exception as e:
        print(f"Failed to load {year} {gp}: {e}")

# Combine everything
if all_rows:
    df = pd.concat(all_rows, ignore_index=True)

    # Save raw combined dataset
    df.to_csv("data/f1_laps_raw.csv", index=False)

    print("\nCombined dataset saved as data/f1_laps_raw.csv")
    print(df.head())
    print("\nTotal rows:", len(df))
else:
    print("No data collected.")