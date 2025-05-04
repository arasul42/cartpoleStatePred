import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import torch

# ✅ Function to create a properly wrapped environment
def make_env():
    return Monitor(gym.make("CartPole-v1"))

# ✅ Vectorized environments
env = DummyVecEnv([make_env])
eval_env = DummyVecEnv([make_env])

# ✅ Evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/",
    eval_freq=5000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)

# ✅ DQN model
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=1,
    target_update_interval=500,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    verbose=1,
    tensorboard_log="./tensorboard_logs/",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# ✅ Start training
model.learn(total_timesteps=200_000, callback=eval_callback)

# ✅ Save model
model.save("dqn_cartpole_model")
print("✅ Model saved to dqn_cartpole_model.zip")

# ✅ Close environments
env.close()
eval_env.close()
