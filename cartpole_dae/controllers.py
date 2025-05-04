# controllers.py

import torch
from stable_baselines3 import DQN, PPO
from mpc_controller import cem_control, mpc_control_batched, pid_control, PIDController

class ControllerManager:
    def __init__(self, control_mode="rl", model=None, device="cpu", rl_policy_path="./logs/best_model/best_model"):
        self.control_mode = control_mode.lower()
        self.model = model
        self.device = device
        self.rl_model = None

        if self.control_mode == "rl":
            try:
                self.rl_model = DQN.load(rl_policy_path)
                print(f"[ControllerManager] Loaded RL model from '{rl_policy_path}'")
            except Exception as e:
                raise RuntimeError(f"Failed to load RL model from '{rl_policy_path}': {e}")

        if self.control_mode == "pid":
            self.pid = PIDController(Kp=40.0, Ki=0.0, Kd=4.0)

        print(f"[ControllerManager] Initialized with mode: {self.control_mode.upper()}")

    def select_action(self, obs, frames_buffer=None, actions_buffer=None):
        if self.control_mode == "mpc":
            return cem_control(
                self.model,
                frames_buffer,
                actions_buffer,
                self.device,
                horizon=6,
                candidates=100
            )

        elif self.control_mode == "rl":
            action, _ = self.rl_model.predict(obs, deterministic=True)
            return int(action)

        elif self.control_mode == "pid":
            return pid_control(obs, self.pid)

        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")
