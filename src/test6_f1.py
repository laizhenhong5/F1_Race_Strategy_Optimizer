import fastf1
import pandas as pd
import matplotlib.pyplot as plt

# Enable cache
fastf1.Cache.enable_cache("cache")

# Load one session
session = fastf1.get_session(2025, "Singapore", "R")
session.load()

# Get master lap data
all_laps = session.laps

# Filter data for Driver 1 (Russell) P1
rus_laps = all_laps.pick_driver("RUS")

# Filter data for Driver 2 (Verstappen) P2
ver_laps = all_laps.pick_driver("VER")

# Plot Driver 1 (Solid Blue line)
plt.plot(rus_laps["LapNumber"], rus_laps["LapTime"].dt.total_seconds(), label="RUS", color="teal")

# Plot Driver 2 (Solid Red line)
plt.plot(ver_laps["LapNumber"], ver_laps["LapTime"].dt.total_seconds(), label="VER", color="red")

# Graph styling
plt.xlabel("Lap Number")
plt.ylabel("Lap Time (seconds)")
plt.title("Lap Time Comparison - Singapore 2025")
plt.legend() # Displays the driver labels
plt.grid(True, linestyle="--", alpha=0.6) # Makes the plot cleaner to read
plt.show()

# Print preview for both drivers
print("Russell Laps:")
print(rus_laps[["LapNumber", "LapTime"]].head())
print("\nVerstappen Laps:")
print(ver_laps[["LapNumber", "LapTime"]].head())