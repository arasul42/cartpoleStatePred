import os
import json
import torch
import numpy as np
import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
from model import CartPoleDynamicsModel
from pred_region import analyze_prediction_regions
from controllers import ControllerManager
import gc

# ==== Setup ====
gc.collect()
torch.cuda.empty_cache()

CONTROL_MODE = "rl"
RL_POLICY_PATH = "./logs/best_model_dqn/best_model"

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
EXP_DIR = get_experiment_path(exp_number)
print(f"📂 Using experiment folder: {EXP_DIR}")

value_ranges = {
    'Cart Position': 4.8,
    'Cart Velocity': 4.0,
    'Pole Angle': 0.418,
    'Angular Velocity': 6.0
}
labels = list(value_ranges.keys())

NUM_EPISODES = 10
summary_rmse = {key: [] for key in labels}
summary_tracking_mae = {key: [] for key in labels}
episode_summaries = []

PRED_RL_DIR = os.path.join(EXP_DIR, 'pred_state_rl')
os.makedirs(PRED_RL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(EXP_DIR, 'best_model.pth')
print(f"📦 Model loaded from '{MODEL_PATH}'")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CartPoleDynamicsModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

controller = ControllerManager(
    control_mode=CONTROL_MODE,
    model=model,
    device=device,
    rl_policy_path=RL_POLICY_PATH
)

termination_bounds = {0: (-2.4, 2.4), 2: (-0.209, 0.209)}

for episode in range(NUM_EPISODES):
    print(f"\n🎬 Starting evaluation episode {episode + 1}")
    episode_dir = os.path.join(PRED_RL_DIR, f'episode_{episode + 1}')
    os.makedirs(episode_dir, exist_ok=True)

    env = gym.make('CartPole-v1', render_mode='rgb_array')
    obs, _ = env.reset()

    video_path = os.path.join(episode_dir, "evaluation_video.avi")
    frame_shape = env.render().shape
    record_writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        30,
        (frame_shape[1], frame_shape[0])
    )

    frames_buffer, actions_buffer, log_true, log_pred, episode_resets, action_log = [], [], [], [], [], []

    for _ in range(4):
        pred_state = obs
        frame = env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        record_writer.write(frame_bgr)
        resized = cv2.resize(frame, (64, 64))
        preprocessed = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        frames_buffer.append(preprocessed)
        action = env.action_space.sample()
        actions_buffer.append([action])
        action_log.append(action)
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()

    for step in range(500):
        images_input = torch.from_numpy(np.array(frames_buffer, dtype=np.float32)).unsqueeze(0).to(device)
        actions_input = torch.tensor([actions_buffer], dtype=torch.float32).to(device)

        with torch.no_grad():
            pred_state = model(images_input, actions_input).cpu().numpy()[0]

        log_true.append(obs)
        log_pred.append(pred_state)

        frame = env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        record_writer.write(frame_bgr)

        resized = cv2.resize(frame, (64, 64))
        preprocessed = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        frames_buffer.pop(0)
        frames_buffer.append(preprocessed)

        action = controller.select_action(obs, frames_buffer, actions_buffer)
        action_log.append(action)

        actions_buffer.pop(0)
        actions_buffer.append([action])

        obs, _, done, _, _ = env.step(action)

        if done:
            obs, _ = env.reset()
            frames_buffer.clear()
            actions_buffer.clear()
            episode_resets.append(step)
            for _ in range(4):
                pred_state = obs
                frame = env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                record_writer.write(frame_bgr)
                resized = cv2.resize(frame, (64, 64))
                preprocessed = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
                frames_buffer.append(preprocessed)
                action = env.action_space.sample()
                actions_buffer.append([action])
                action_log.append(action)
                obs, _, done, _, _ = env.step(action)
                if done:
                    obs, _ = env.reset()
                    frames_buffer.clear()
                    actions_buffer.clear()

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    env.close()
    record_writer.release()

    log_true = np.array(log_true)
    log_pred = np.array(log_pred)
    abs_error = np.abs(log_true - log_pred)
    squared_error = (log_true - log_pred) ** 2

    mean_rmse = {labels[i]: float(np.sqrt(np.mean(squared_error[:, i]))) for i in range(4)}
    mean_abs_error = {labels[i]: float(np.mean(abs_error[:, i])) for i in range(4)}
    tracking_error = np.mean(np.abs(log_true), axis=0)

    for i in range(4):
        summary_rmse[labels[i]].append(mean_rmse[labels[i]])
        summary_tracking_mae[labels[i]].append(tracking_error[i])

    effort = 2 * np.array(action_log) - 1
    total_effort = np.sum(effort ** 2)
    switch_count = int(np.sum(effort[1:] != effort[:-1]))

    plt.figure(figsize=(10.2, 7.5))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(log_true[:, i], label='True')
        plt.plot(log_pred[:, i], label='Predicted', color='tab:orange')
        if i in termination_bounds:
            lower, upper = termination_bounds[i]
            plt.axhline(y=lower, linestyle=':', color='red', linewidth=1)
            plt.axhline(y=upper, linestyle=':', color='red', linewidth=1)
        plt.title(labels[i])
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(episode_dir, 'evaluation_results.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(abs_error[:, i], label='Prediction Error', color='tab:red', linewidth=0.5)
        for reset_step in episode_resets:
            plt.axvline(x=reset_step, color='blue', linestyle=':', linewidth=2)
        plt.xlabel("Timestep")
        plt.ylabel("Error")
        plt.title(f"{labels[i]} Error Over Time")
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(episode_dir, 'per_state_mae_rmse_over_time.png'))
    plt.close()

    episode_summary = {
        "episode": episode + 1,
        "mean_abs_error": mean_abs_error,
        "mean_rmse": mean_rmse,
        "tracking_error": {labels[i]: float(tracking_error[i]) for i in range(4)},
        "control_effort": float(total_effort),
        "action_switches": int(switch_count)
    }

    episode_summaries.append(episode_summary)
    with open(os.path.join(episode_dir, 'per_episode_summary.json'), 'w') as f:
        json.dump(episode_summary, f, indent=4)

avg_summary = {
    "episodes": NUM_EPISODES,
    "avg_mean_abs_error": {},
    "avg_mean_rmse": {},
    "avg_tracking_mae": {},
    "avg_control_effort": float(np.mean([e["control_effort"] for e in episode_summaries])),
    "avg_action_switches": float(np.mean([e["action_switches"] for e in episode_summaries])),
    "avg_percent_rmse": {},
    "avg_percent_tracking_mae": {}
}

for i, label in enumerate(labels):
    avg_summary["avg_mean_abs_error"][label] = float(np.mean([e["mean_abs_error"][label] for e in episode_summaries]))
    avg_summary["avg_mean_rmse"][label] = float(np.mean([e["mean_rmse"][label] for e in episode_summaries]))
    avg_summary["avg_tracking_mae"][label] = float(np.mean([e["tracking_error"][label] for e in episode_summaries]))
    avg_summary["avg_percent_rmse"][label] = 100 * np.mean(summary_rmse[label]) / value_ranges[label]
    avg_summary["avg_percent_tracking_mae"][label] = 100 * np.mean(summary_tracking_mae[label]) / value_ranges[label]

with open(os.path.join(PRED_RL_DIR, "summary.json"), 'w') as f:
    json.dump(avg_summary, f, indent=4)

with open(os.path.join(EXP_DIR, "normalized_summary.json"), 'w') as f:
    json.dump({
        "avg_percent_rmse": avg_summary["avg_percent_rmse"],
        "avg_percent_tracking_mae": avg_summary["avg_percent_tracking_mae"]
    }, f, indent=4)

print("\n✅ Evaluation complete.")
print("📄 Saved summary: summary.json")
print("📄 Saved normalized summary: normalized_summary.json")