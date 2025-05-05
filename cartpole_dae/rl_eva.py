import cv2
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import os
import json

# --- Get the latest experiment folder ---
def get_experiment_path(exp_number=None, base_dir='experiments'):
    if exp_number is not None:
        exp_name = f"exp{exp_number}"
        exp_path = os.path.join(base_dir, exp_name)
        if not os.path.isdir(exp_path):
            raise FileNotFoundError(f"❌ Experiment folder '{exp_path}' not found.")
    else:
        exps = [d for d in os.listdir(base_dir) if d.startswith('exp') and d.replace('exp', '').isdigit()]
        if not exps:
            raise FileNotFoundError("❌ No experiment folders found.")
        latest = max(exps, key=lambda x: int(x.replace('exp', '')))
        exp_path = os.path.join(base_dir, latest)
    return exp_path

# --- SET THIS TO OVERRIDE ---
exp_number = None
exp_path = get_experiment_path(exp_number)
print(f"📂 Using experiment folder: {exp_path}")

full_state_rl_path = os.path.join(exp_path, "full_state_rl")
os.makedirs(full_state_rl_path, exist_ok=True)

# --- Load model ---
model = DQN.load("./logs/best_model/best_model")

# --- Evaluation settings ---
N_EPISODES = 10
timesteps = 500
labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Angular Velocity']
termination_bounds = {0: (-2.4, 2.4), 2: (-0.209, 0.209)}

# --- Normalization ranges ---
value_ranges = {
    'Cart Position': 4.8,
    'Cart Velocity': 4.0,
    'Pole Angle': 0.418,
    'Angular Velocity': 6.0
}

# --- Metric storage ---
all_mse = []
all_effort = []
all_switches = []
all_lengths = []
all_mae = []

# --- Evaluation loop ---
for run in range(N_EPISODES):
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    obs, _ = env.reset()
    states, actions = [], []



    video_path = os.path.join(full_state_rl_path, f"episode_{run + 1}.avi")
    frame = env.render()
    frame_shape = frame.shape  # (H, W, C)
    record_writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        30,
        (frame_shape[1], frame_shape[0])  # width, height
    )

    for t in range(timesteps):

        frame = env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        record_writer.write(frame_bgr)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        states.append(obs)
        actions.append(action)
        if done or truncated:
            print(f"🚫 Episode {run + 1} terminated at timestep {t + 1}")
            break

    env.close()

    states = np.array(states)
    actions = np.array(actions)

    # --- MSE ---
    reference = np.zeros(4)
    mse_per_state = np.mean((states - reference) ** 2, axis=0)
    mae_per_state = np.mean(np.abs(states - reference), axis=0)
    all_mse.append(mse_per_state)
    all_mae.append(mae_per_state)

    # --- Effort & Switching ---
    effort = 2 * actions - 1
    total_effort = np.sum(effort ** 2)
    switch_count = np.sum(effort[1:] != effort[:-1])
    all_effort.append(total_effort)
    all_switches.append(switch_count)
    all_lengths.append(len(states))

    # --- Plot ---
    plt.figure(figsize=(9.2, 6.7))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(states[:, i], label=labels[i])
        if i in termination_bounds:
            lower, upper = termination_bounds[i]
            plt.axhline(lower, linestyle='--', color='red', linewidth=1, label='Termination Boundary')
            plt.axhline(upper, linestyle='--', color='red', linewidth=1)
        plt.xlabel("Timestep")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(full_state_rl_path, f"episode_{run+1}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

print("📉 All plots saved to:", full_state_rl_path)

# --- Aggregated metrics ---
mean_rmse = np.sqrt(np.mean(all_mse, axis=0))
mean_mae = np.mean(all_mae, axis=0)
mean_effort = float(np.mean(all_effort))
mean_switches = float(np.mean(all_switches))
mean_length = float(np.mean(all_lengths))

# --- Normalized tracking MAE ---
percent_tracking_mae = {
    labels[i]: float((mean_mae[i] / value_ranges[labels[i]]) * 100)
    for i in range(4)
}

# --- Save JSON summary ---
summary = {
    "episodes": N_EPISODES,
    "mean_timesteps": mean_length,
    "mean_tracking_rmse": {labels[i]: float(mean_rmse[i]) for i in range(4)},
    "mean_control_effort": mean_effort,
    "mean_action_switches": mean_switches,
    "normalized_tracking_mae_percent": percent_tracking_mae
}

json_path = os.path.join(full_state_rl_path, "evaluation_summary.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=4)

print(f"✅ Summary with RMSE and normalized MAE saved to {json_path}")
