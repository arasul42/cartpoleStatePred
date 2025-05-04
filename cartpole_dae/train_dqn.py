import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from dqn_learned_env import DQNCartPoleFromPredictionEnv  # your predicted env
from model import CartPoleDynamicsModel


# 🎯 Custom Reward Wrapper for Predicted State Env
class ShapedPredictionRewardWrapper(gym.Wrapper):
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


# ==== Load Dynamics Model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dynamics_model = CartPoleDynamicsModel().to(device)
dynamics_model.load_state_dict(torch.load("experiments/exp16/best_model.pth"))  # adjust path as needed
dynamics_model.eval()

# ==== Environment ====
def make_env():
    base_env = DQNCartPoleFromPredictionEnv(model=dynamics_model, device=device)
    shaped_env = ShapedPredictionRewardWrapper(base_env, switch_penalty=0.05)
    return Monitor(shaped_env)

env = DummyVecEnv([make_env])
eval_env = DummyVecEnv([make_env])

# ==== Evaluation Callback ====
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model_dqn/",
    log_path="./logs/",
    eval_freq=5000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)

# ==== DQN Model ====
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
model.save("dqn_learned_env_model")
print("✅ Model saved to dqn_cartpole_model_predicted.zip")

# ==== Close ====
env.close()
eval_env.close()
