# dqn_learned_env.py

import gymnasium as gym
import numpy as np
import torch
import cv2
from gymnasium import spaces

class DQNCartPoleFromPredictionEnv(gym.Env):
    def __init__(self, model, device, image_size=64, seq_len=4):
        super().__init__()
        self.real_env = gym.make('CartPole-v1', render_mode='rgb_array')
        self.model = model.eval()
        self.device = device
        self.image_size = image_size
        self.seq_len = seq_len

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.frames_buffer = []
        self.actions_buffer = []

    def _preprocess_frame(self, frame):
        frame = cv2.resize(frame, (self.image_size, self.image_size))
        frame = frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        return frame

    def _get_predicted_state(self):
        with torch.no_grad():
            frames = torch.tensor(np.array([self.frames_buffer]), dtype=torch.float32).to(self.device)
            actions = torch.tensor(np.array([self.actions_buffer]), dtype=torch.float32).to(self.device)
            pred = self.model(frames, actions).cpu().numpy()[0]
        return pred.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, _ = self.real_env.reset()

        self.frames_buffer.clear()
        self.actions_buffer.clear()

        while len(self.frames_buffer) < self.seq_len:
            frame = self.real_env.render()
            self.frames_buffer.append(self._preprocess_frame(frame))

            action = self.real_env.action_space.sample()
            self.actions_buffer.append([action])

            obs, _, done, _, _ = self.real_env.step(action)

            # 🔒 Restart the whole process if episode ends too early
            if done:
                self.frames_buffer.clear()
                self.actions_buffer.clear()
                obs, _ = self.real_env.reset()

        return self._get_predicted_state(), {}


    def step(self, action):
        self.actions_buffer.pop(0)
        self.actions_buffer.append([action])

        obs, reward, done, truncated, info = self.real_env.step(action)
        frame = self.real_env.render()
        self.frames_buffer.pop(0)
        self.frames_buffer.append(self._preprocess_frame(frame))

        pred_state = self._get_predicted_state()
        return pred_state, reward, done, truncated, info

    def render(self):
        return self.real_env.render()

    def close(self):
        self.real_env.close()
