import pandas as pd
import matplotlib.pyplot as plt
#SHOWS THE PLOTTED LAP TIMES FOR RUSSELL IN LAS VEGAS 2024, GROUPED BY TIRE COMPOUND
df = pd.read_csv("cleaned_russell_las_vegas_2024.csv")
df["LapTimeSeconds"] = pd.to_numeric(df["LapTimeSeconds"], errors="coerce")

for compound in df["Compound"].dropna().unique():
    subset = df[df["Compound"] == compound]
    plt.plot(subset["LapNumber"], subset["LapTimeSeconds"], marker="o", label=compound)

plt.xlabel("Lap Number")
plt.ylabel("Lap Time (seconds)")
plt.title("Lap Time vs Lap Number by Tire Compound - Russell Las Vegas 2024")
plt.legend()
plt.grid(True)
plt.show()