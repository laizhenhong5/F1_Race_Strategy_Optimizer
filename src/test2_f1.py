#this was sucessful
import fastf1
import pandas as pd
import matplotlib.pyplot as plt

# Enable cache
fastf1.Cache.enable_cache("cache")

# Load one session
session = fastf1.get_session(2026, "Miami", "R") #choose the gp, year
session.load()

#get lap data
laps = session.laps


laps = session.laps.pick_driver("RUS") #any driver

plt.plot(laps["LapNumber"], laps["LapTime"].dt.total_seconds())
plt.xlabel("Lap Number")
plt.ylabel("Lap Time (seconds)")
plt.title("Lap Time vs Lap Number - Russell")
plt.show()
print(laps.head())
print("\nColumns:")
print(laps.columns)