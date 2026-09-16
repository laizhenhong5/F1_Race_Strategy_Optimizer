import fastf1
import pandas as pd
import matplotlib.pyplot as plt

# Enable cache
fastf1.Cache.enable_cache("cache")

# Load one session
session = fastf1.get_session(2026, "Australia", "R")
session.load()

# Get master lap data
all_laps = session.laps

# Filter data for Driver 1 (Russell) P1
rus_laps = all_laps.pick_driver("RUS")

# Filter data for Driver 2 (Antonelli) P2
ant_laps = all_laps.pick_driver("ANT")

# Plot Driver 1 (Solid Blue line)
plt.plot(rus_laps["LapNumber"], rus_laps["LapTime"].dt.total_seconds(), label="RUS", color="teal")

# Plot Driver 2 (Solid Red line)
plt.plot(ant_laps["LapNumber"], ant_laps["LapTime"].dt.total_seconds(), label="ANT", color="red")

# Graph styling
plt.xlabel("Lap Number")
plt.ylabel("Lap Time (seconds)")
plt.title("Lap Time Comparison - Australia 2026")
plt.legend() # Displays the driver labels
plt.grid(True, linestyle="--", alpha=0.6) # Makes the plot cleaner to read
plt.show()

# Print preview for both drivers
print("Russell Laps:")
print(rus_laps[["LapNumber", "LapTime"]].head())
print("\nAntonelli Laps:")
print(ant_laps[["LapNumber", "LapTime"]].head())