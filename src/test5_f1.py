import fastf1
import pandas as pd #load pandas data analysis library
import matplotlib.pyplot as plt #import plotting
# enable cache
fastf1.Cache.enable_cache("cache")
# Load session
session = fastf1.get_session(2026, "Montreal", "R")
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