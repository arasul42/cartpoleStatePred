# import gymnasium as gym
# from stable_baselines3 import DQN
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.callbacks import EvalCallback
# import torch

# # ✅ Function to create a properly wrapped environment
# def make_env():
#     return Monitor(gym.make("CartPole-v1"))

# # ✅ Vectorized environments
# env = DummyVecEnv([make_env])
# eval_env = DummyVecEnv([make_env])

# # ✅ Evaluation callback
# eval_callback = EvalCallback(
#     eval_env,
#     best_model_save_path="./logs/best_model/",
#     log_path="./logs/",
#     eval_freq=5000,
#     n_eval_episodes=5,
#     deterministic=True,
#     render=False,
# )

# # ✅ DQN model
# model = DQN(
#     policy="MlpPolicy",
#     env=env,
#     learning_rate=1e-4,
#     buffer_size=100000,
#     learning_starts=1000,
#     batch_size=64,
#     gamma=0.99,
#     train_freq=1,
#     target_update_interval=500,
#     exploration_fraction=0.1,
#     exploration_final_eps=0.02,
#     verbose=1,
#     tensorboard_log="./tensorboard_logs/",
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# )

# # ✅ Start training
# model.learn(total_timesteps=200_000, callback=eval_callback)
# # model.learn(total_timesteps=200_000)

# # ✅ Save model
# model.save("dqn_cartpole_model")
# print("✅ Model saved to dqn_cartpole_model.zip")

# # ✅ Close environments
# env.close()
# eval_env.close()








# env_eval = gym.make("CartPole-v1", render_mode='human')

# # Load the trained model
# loaded_model = DQN.load("dqn_cartpole_model")
# print("✅ Model loaded successfully")

# # Evaluate the loaded model for 500 timesteps with render mode set to 'human'
# obs, _ = env_eval.reset()
# total_reward = 0
# for _ in range(500):
#     env_eval.render()  # Render the environment in human mode
#     action, _states = loaded_model.predict(obs, deterministic=True)
#     obs, reward, done, _, _ = env_eval.step(action)
#     total_reward += reward
#     if done:
#         obs, _ = env_eval.reset()

# print(f"🎯 Total reward over 500 timesteps: {total_reward}")

# # Close the environment
# env_eval.close()


# import os
# import csv
# import gymnasium as gym
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from gymnasium import RewardWrapper
# from stable_baselines3 import DQN
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv

# # 🎯 Custom Reward Wrapper
# class ShapedCartPoleRewardWrapper(RewardWrapper):
#     def __init__(self, env, switch_penalty=0.05):
#         super().__init__(env)
#         self.prev_action = None
#         self.switch_penalty = switch_penalty

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         self.prev_action = None
#         return obs, info

#     def step(self, action):
#         obs, reward, done, truncated, info = self.env.step(action)
#         x, x_dot, theta, theta_dot = self.env.unwrapped.state

#         pos_penalty = 0.1 * abs(x)
#         angle_penalty = 1.0 * abs(theta)
#         switch_penalty = self.switch_penalty if self.prev_action is not None and action != self.prev_action else 0.0
#         self.prev_action = action

#         shaped_reward = 1.0 - pos_penalty - angle_penalty - switch_penalty
#         return obs, shaped_reward, done, truncated, info

# # ✅ Create log directory
# log_dir = "./logs/"
# os.makedirs(log_dir, exist_ok=True)

# # ✅ Environment creator
# def make_env(log_dir=None):
#     def _init():
#         base_env = gym.make("CartPole-v1")
#         shaped_env = ShapedCartPoleRewardWrapper(base_env)
#         return Monitor(shaped_env, filename=os.path.join(log_dir, "monitor.csv") if log_dir else None)
#     return _init

# # ✅ Create training environment with Monitor
# env = DummyVecEnv([make_env(log_dir)])

# # ✅ Define the DQN model
# model = DQN(
#     policy="MlpPolicy",
#     env=env,
#     learning_rate=1e-4,
#     buffer_size=100_000,
#     learning_starts=1000,
#     batch_size=64,
#     gamma=0.99,
#     train_freq=1,
#     target_update_interval=500,
#     exploration_fraction=0.1,
#     exploration_final_eps=0.02,
#     verbose=1,
#     tensorboard_log="./tensorboard_logs/",
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# )

# # ✅ Train the model
# model.learn(total_timesteps=120_000)
# model.save("dqn_cartpole_model")
# print("✅ Model saved to dqn_cartpole_model.zip")

# # ✅ Parse training episode rewards from monitor file
# monitor_file = os.path.join(log_dir, "monitor.csv")
# episode_rewards = []

# try:
#     with open(monitor_file, "r") as f:
#         lines = f.readlines()[2:]  # Skip header
#         for line in lines:
#             parts = line.strip().split(",")
#             reward = float(parts[0])  # First column: total reward
#             episode_rewards.append(reward)
# except Exception as e:
#     print(f"⚠️ Failed to read monitor.csv: {e}")

# # ✅ Save rewards to CSV
# with open("reward_log.csv", "w", newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(["episode", "total_reward"])
#     for i, r in enumerate(episode_rewards):
#         writer.writerow([i + 1, r])
# print("✅ reward_log.csv saved.")

# # ✅ Plot Reward vs Training Episode
# if episode_rewards:
#     def smooth(data, weight=10):
#         if len(data) < weight:
#             return np.array(data)
#         return np.convolve(data, np.ones(weight)/weight, mode='valid')

#     episodes = list(range(1, len(episode_rewards) + 1))
#     smoothed_rewards = smooth(episode_rewards)
#     aligned_episodes = episodes[-len(smoothed_rewards):]

#     plt.figure(figsize=(10, 7.5))
#     plt.plot(episodes, episode_rewards, label="Reward", alpha=0.4)
#     plt.plot(aligned_episodes, smoothed_rewards, linewidth=2)
#     plt.xlabel("Training Episode")
#     plt.ylabel("Total Reward")
#     plt.title("Reward vs Training Episode")
#     plt.legend()
#     plt.grid()
#     plt.savefig("reward_vs_training_episode.png", dpi=300)
#     plt.close()
#     print("✅ reward_vs_training_episode.png saved.")
# else:
#     print("⚠️ No training episodes found in monitor.csv.")

# # ✅ Cleanup
# env.close()



# # ✅ Evaluation with render (human mode)
# env_eval = gym.make("CartPole-v1", render_mode='human')
# obs, _ = env_eval.reset()
# loaded_model = DQN.load("dqn_cartpole_model")
# print("✅ Model loaded successfully")

# total_reward = 0
# for _ in range(500):
#     env_eval.render()
#     action, _ = loaded_model.predict(obs, deterministic=True)
#     obs, reward, done, _, _ = env_eval.step(action)
#     total_reward += reward
#     if done:
#         obs, _ = env_eval.reset()

# print(f"🎯 Total reward over 500 timesteps: {total_reward}")
# env_eval.close()


import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# 🎯 Shaped Reward Wrapper for Original CartPole
class ShapedCartPoleRewardWrapper(gym.Wrapper):
    def __init__(self, env, switch_penalty=0.05):
        super().__init__(env)
        self.prev_action = None
        self.switch_penalty = switch_penalty

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action = None
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Extract predicted state
        x, x_dot, theta, theta_dot = obs

        # Reward shaping
        pos_penalty = 0.1 * abs(x)
        angle_penalty = 1.0 * abs(theta)
        switch_penalty = 0.0
        if self.prev_action is not None and action != self.prev_action:
            switch_penalty = self.switch_penalty

        self.prev_action = action
        shaped_reward = 1.0 - pos_penalty - angle_penalty - switch_penalty

        return obs, shaped_reward, done, truncated, info

# ==== Environment ====
def make_env():
    base_env = gym.make("CartPole-v1")
    shaped_env = ShapedCartPoleRewardWrapper(base_env)
    return Monitor(shaped_env)

env = DummyVecEnv([make_env])
eval_env = DummyVecEnv([make_env])

# ==== Evaluation Callback ====
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/",
    eval_freq=5000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)

# ==== DQN Model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=1,
    target_update_interval=500,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    verbose=1,
    tensorboard_log="./tensorboard_logs/",
    device=device
)

# ==== Train ====
model.learn(total_timesteps=200_000, callback=eval_callback)
model.save("dqn_cartpole_shaped_model")
print("✅ Model saved to dqn_cartpole_shaped_model.zip")

# ==== Close ====
env.close()
eval_env.close()
