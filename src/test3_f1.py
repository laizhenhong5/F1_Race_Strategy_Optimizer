import fastf1
import pandas as pd
import matplotlib.pyplot as plt

# Enable cache
fastf1.Cache.enable_cache("cache")

# Load one session
session = fastf1.get_session(2025, "Singapore", "R")
session.load()

#get lap data
laps = session.laps


laps = session.laps.pick_driver("RUS")

plt.plot(laps["LapNumber"], laps["LapTime"].dt.total_seconds())
plt.xlabel("Lap Number")
plt.ylabel("Lap Time (seconds)")
plt.title("Lap Time vs Lap Number - Russell")
plt.show()
print(laps.head())
print("\nColumns:")
print(laps.columns)