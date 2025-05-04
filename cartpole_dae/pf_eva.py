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
from particle_filter import ModelDrivenParticleFilter
import gc

# ==== Setup ====
gc.collect()
torch.cuda.empty_cache()

CONTROL_MODE = "rl"
RL_POLICY_PATH = "dqn_learned_env_model"

def get_latest_experiment_folder(base_dir='experiments'):
    exps = [d for d in os.listdir(base_dir) if d.startswith('exp') and d.replace('exp', '').isdigit()]
    if not exps:
        raise FileNotFoundError("No experiment folder found.")
    latest = max(exps, key=lambda x: int(x.replace('exp', '')))
    return os.path.join(base_dir, latest)

def render_and_feed_model(env, obs, pred_state, step, show=True, record_writer=None, image_size=64):
    frame = env.render()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    true_text = f"True: pos={obs[0]:.2f}, vel={obs[1]:.2f}, angle={obs[2]:.2f}, ang_vel={obs[3]:.2f}"
    pred_text = f"Pred: pos={pred_state[0]:.2f}, vel={pred_state[1]:.2f}, angle={pred_state[2]:.2f}, ang_vel={pred_state[3]:.2f}"
    step_text = f"Step: {step}"

    cv2.putText(frame_bgr, step_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame_bgr, true_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(frame_bgr, pred_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    if show:
        cv2.imshow("CartPole Live", frame_bgr)
        cv2.waitKey(1)
    if record_writer:
        record_writer.write(frame_bgr)

    resized = cv2.resize(frame, (image_size, image_size))
    preprocessed = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return preprocessed

# ==== Experiment Setup ====
SEQ_LENGTH = 4
IMAGE_SIZE = 64
MAX_STEPS = 500
NUM_EPISODES = 10

EXP_DIR = get_latest_experiment_folder()
PRED_RL_DIR = os.path.join(EXP_DIR, 'pred_state_pf')
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

pf = ModelDrivenParticleFilter(model=model, device=device, num_particles=100, dt=1/100)

labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Angular Velocity']
termination_bounds = {0: (-2.4, 2.4), 2: (-0.209, 0.209)}
episode_summaries = []

# ==== Evaluation over Episodes ====
for episode in range(NUM_EPISODES):
    # pf.initialize()
    # pf.prev_pred = None
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

    frames_buffer, actions_buffer = [], []
    log_true, log_pred, episode_resets, action_log = [], [], [], []

    # Initialize buffers
    for _ in range(SEQ_LENGTH):
        frame = render_and_feed_model(env, obs, obs, step=0, show=True, record_writer=record_writer)
        frames_buffer.append(frame)
        action = env.action_space.sample()
        actions_buffer.append([action])
        action_log.append(action)
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()


    # ✅ Initialize particle filter using first NN prediction
    with torch.no_grad():
        images_input = torch.tensor([frames_buffer], dtype=torch.float32).to(device)
        actions_input = torch.tensor([actions_buffer], dtype=torch.float32).to(device)
        initial_pred = model(images_input, actions_input).cpu().numpy()[0]

    pf.initialize(initial_pred=initial_pred)



    for step in range(MAX_STEPS):
        pf.predict(frames_buffer, actions_buffer)

        images_input = torch.from_numpy(np.array(frames_buffer, dtype=np.float32)).unsqueeze(0).to(device)
        actions_input = torch.tensor([actions_buffer], dtype=torch.float32).to(device)
        with torch.no_grad():
            measurement = model(images_input, actions_input).cpu().numpy()[0]

        pf.update(measurement_pred=measurement)
        pf.resample()
        filtered_state, _ = pf.estimate(return_std=True)

        log_true.append(obs)
        log_pred.append(filtered_state)

        frame = render_and_feed_model(env, obs, filtered_state, step, show=True, record_writer=record_writer)

        frames_buffer.pop(0)
        frames_buffer.append(frame)

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
            for _ in range(SEQ_LENGTH):
                frame = render_and_feed_model(env, obs, obs, step, show=True, record_writer=record_writer)
                frames_buffer.append(frame)
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

    # ==== Logging & Metrics ====
    log_true = np.array(log_true)
    log_pred = np.array(log_pred)
    abs_error = np.abs(log_true - log_pred)
    rmse = np.sqrt((log_true - log_pred) ** 2)

    effort = 2 * np.array(action_log) - 1
    total_effort = np.sum(effort ** 2)
    switch_count = int(np.sum(effort[1:] != effort[:-1]))

    # ==== Plot State Trajectories ====
    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(log_true[:, i], label='True', color='black')
        plt.plot(log_pred[:, i], label='Filtered', linestyle='--', color='tab:red')
        if i in termination_bounds:
            plt.axhline(y=termination_bounds[i][0], linestyle=':', color='red')
            plt.axhline(y=termination_bounds[i][1], linestyle=':', color='red')
        plt.title(labels[i])
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(episode_dir, 'evaluation_results.png'))
    plt.close()

    # ==== Plot Error Over Time ====
    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(abs_error[:, i], label='Prediction Error', color='tab:red', linewidth=0.5)
        for reset_step in episode_resets:
            plt.axvline(x=reset_step, color='blue', linestyle=':', linewidth=1)
        plt.title(f"{labels[i]} Error Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Error")
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(episode_dir, 'per_state_mae_rmse_over_time.png'))
    plt.close()

    # ==== Save Episode Summary ====
    episode_summary = {
        "episode": episode + 1,
        "steps_taken": len(log_true),
        "mean_abs_error": {labels[i]: float(np.mean(abs_error[:, i])) for i in range(4)},
        "mean_rmse": {labels[i]: float(np.mean(rmse[:, i])) for i in range(4)},
        "control_effort": float(total_effort),
        "action_switches": switch_count
    }
    episode_summaries.append(episode_summary)
    with open(os.path.join(episode_dir, 'per_episode_summary.json'), 'w') as f:
        json.dump(episode_summary, f, indent=4)

# ==== Save Aggregate Summary ====
avg_summary = {
    "episodes": NUM_EPISODES,
    "avg_steps_taken": float(np.mean([e["steps_taken"] for e in episode_summaries])),
    "avg_mean_abs_error": {},
    "avg_mean_rmse": {},
    "avg_control_effort": float(np.mean([e["control_effort"] for e in episode_summaries])),
    "avg_action_switches": float(np.mean([e["action_switches"] for e in episode_summaries]))
}

for i, label in enumerate(labels):
    avg_summary["avg_mean_abs_error"][label] = float(np.mean([e["mean_abs_error"][label] for e in episode_summaries]))
    avg_summary["avg_mean_rmse"][label] = float(np.mean([e["mean_rmse"][label] for e in episode_summaries]))

with open(os.path.join(PRED_RL_DIR, "summary.json"), 'w') as f:
    json.dump(avg_summary, f, indent=4)

print(f"\n✅ Evaluation complete. Summary saved to: {os.path.join(PRED_RL_DIR, 'summary.json')}")
