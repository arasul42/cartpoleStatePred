import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to the monitor log
log_file = os.path.join("logs", "monitor.csv")

# Skip first few rows of metadata
df = pd.read_csv(log_file, skiprows=1)

# Extract episode rewards and lengths
episode_rewards = df["r"]
episode_lengths = df["l"]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label='Episode Reward')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward Over Episodes")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("training_reward_plot.png", dpi=300)
plt.show()
