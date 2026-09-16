import pandas as pd
import matplotlib.pyplot as plt
#SHOWS THE PLOTTED LAP TIMES FOR RUSSELL IN MIAMI 2026, GROUPED BY TIRE COMPOUND
df = pd.read_csv("cleaned_russell_miami_2026.csv")
df["LapTimeSeconds"] = pd.to_numeric(df["LapTimeSeconds"], errors="coerce")

for compound in df["Compound"].dropna().unique():
    subset = df[df["Compound"] == compound]
    plt.plot(subset["LapNumber"], subset["LapTimeSeconds"], marker="o", label=compound)

plt.xlabel("Lap Number")
plt.ylabel("Lap Time (seconds)")
plt.title("Lap Time vs Lap Number by Tire Compound - Russell Miami 2026")
plt.legend()
plt.grid(True)
plt.show()