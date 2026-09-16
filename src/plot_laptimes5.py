# import pandas as pd
# import matplotlib.pyplot as plt

# # 1. Load the Australia 2026 dataset
# df = pd.read_csv("cleaned_mercedes_australia_2026.csv")
# df["LapTimeSeconds"] = pd.to_numeric(df["LapTimeSeconds"], errors="coerce")

# # Define specific line styles for each driver to tell them apart
# driver_styles = {
#     "RUS": {"linestyle": "-", "marker": "o"},  # Solid line, circle dots
#     "ANT": {"linestyle": "--", "marker": "x"}  # Dashed line, X dots
# }

# # Define fixed colors for standard F1 tire compounds for clear visual parsing
# compound_colors = {
#     "SOFT": "red",
#     "MEDIUM": "darkorange",
#     "HARD": "black",
#     "INTERMEDIATE": "green",
#     "WET": "blue"
# }
# plt.figure(figsize=(12, 6)) # Make plot wider to read lap trends easier
# # 2. Group by Driver first, then by Compound
# for driver in df["Driver"].unique():
#     driver_df = df[df["Driver"] == driver]
#     style = driver_styles.get(driver, {"linestyle": "-", "marker": "o"})
    
#     for compound in driver_df["Compound"].dropna().unique():
#         subset = driver_df[driver_df["Compound"] == compound]
#         color = compound_colors.get(str(compound).upper(), None) # Auto-color tire type
        
#         # Plot each stint configuration
#         plt.plot(
#             subset["LapNumber"], 
#             subset["LapTimeSeconds"], 
#             linestyle=style["linestyle"],
#             marker=style["marker"], 
#             color=color,
#             label=f"{driver} ({compound})"
#         )

# # Graph styling and clean dark theme compatibility 
# plt.xlabel("Lap Number")
# plt.ylabel("Lap Time (seconds)")
# plt.title("Lap Time Comparison by Tire Compound - Mercedes Australia 2026")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Moves legend outside so it doesn't block lines
# plt.grid(True, linestyle=":", alpha=0.6)
# plt.tight_layout() # Prevents label cutoff
# plt.show()

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/cleaned_russell_singapore_2025.csv"
DRIVERS = ["RUS"]
TITLE = "Lap Time vs Lap Number"


# ============================================================
# LOAD DATA
# ============================================================

print(f"Loading dataset: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
df["LapTimeSeconds"] = pd.to_numeric(
    df["LapTimeSeconds"],
    errors="coerce"
)

df["LapNumber"] = pd.to_numeric(
    df["LapNumber"],
    errors="coerce"
)

# ============================================================
# REMOVE INVALID ROWS
# ============================================================
df = df.dropna(
    subset=["LapNumber", "LapTimeSeconds"]
)
# ============================================================
# FILTER DRIVERS
# ============================================================

df = df[df["Driver"].isin(DRIVERS)]


# ============================================================
# PLOT LAP TIMES
# ============================================================
plt.figure(figsize=(12, 6))
for driver in DRIVERS:

    driver_df = df[df["Driver"] == driver]

    for compound in driver_df["Compound"].dropna().unique():

        subset = driver_df[
            driver_df["Compound"] == compound
        ]

        plt.scatter(
            subset["TyreLife"],
            subset["LapTimeSeconds"],
            label=f"{driver} ({compound})"
        )
        # plt.plot(
        #     subset["LapNumber"],
        #     subset["LapTimeSeconds"],
        #     marker="o",
        #     label=f"{driver} ({compound})"
        # )

# ============================================================
# MARK PIT LAPS
# ============================================================
# for driver in DRIVERS:
#     driver_df = df[df["Driver"] == driver]
#     pit_laps = driver_df[
#         driver_df["PitInTime"].notna()
#     ]
#     for _, row in pit_laps.iterrows():
#         plt.axvline(
#             x=row["LapNumber"],
#             linestyle="--",
#             alpha=0.7
#         )
#         plt.text(
#             row["LapNumber"],
#             row["LapTimeSeconds"],
#             " PIT",
#             rotation=90,
#             verticalalignment="bottom"
#         )
# ============================================================
# GRAPH FORMATTING
# ============================================================
#plt.xlabel("Lap Number")
plt.xlabel("Tyre Life (laps)")  # Updated x-axis label for clarity
plt.ylabel("Lap Time (seconds)")
#plt.title(TITLE)
plt.title("Lap Time vs Tyre Age")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()