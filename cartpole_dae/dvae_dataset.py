
import os
import gymnasium as gym
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class DVAEDataset(Dataset):
    def __init__(self, env_name='CartPole-v1', seq_length=4, dataset_size=5000, image_size=64, cache_file='dvae_seq_cartpole.npz'):
        self.env_name = env_name
        self.seq_length = seq_length
        self.dataset_size = dataset_size
        self.image_size = image_size
        self.cache_file = cache_file

        if os.path.exists(cache_file):
            print(f"📂 Loading cached sequence dataset from {cache_file}")
            data = np.load(cache_file)
            self.x_seq = data['x_seq']
            self.a_seq = data['a_seq']
            self.states = data['states']
        else:
            print("📸 Collecting sequence-based dataset...")
            self.collect_data()
            np.savez_compressed(self.cache_file, x_seq=self.x_seq, a_seq=self.a_seq, states=self.states)
            print(f"✅ Dataset cached to {self.cache_file}")

    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (self.image_size, self.image_size))
        frame = frame.transpose(2, 0, 1)
        return frame.astype(np.float32) / 255.0

    def collect_data(self, bins_per_state=6, max_per_bin=2500, targeted_reset_ratio=0.5, show_preview=False, preview_interval=10):
        env = gym.make(self.env_name, render_mode='rgb_array')
        x_seq_list, a_seq_list, state_list = [], [], []

        state_limits = {
            0: (-2.4, 2.4),
            1: (-3.0, 3.0),
            2: (-0.209, 0.209),
            3: (-3.0, 3.0),
        }
        bin_edges = {i: np.linspace(*state_limits[i], bins_per_state + 1) for i in range(4)}
        bin_counts = {}

        def targeted_reset():
            cart_pos = np.random.uniform(-2.0, 2.0)
            cart_vel = np.random.uniform(-3.0, 3.0)
            pole_angle = np.random.uniform(-0.2, 0.2)
            ang_vel = np.random.uniform(-2.5, 2.5)
            env.reset()
            env.unwrapped.state = np.array([cart_pos, cart_vel, pole_angle, ang_vel])
            return env.unwrapped.state

        while len(x_seq_list) < self.dataset_size:
            use_targeted = np.random.rand() < targeted_reset_ratio
            obs = targeted_reset() if use_targeted else env.reset()[0]

            frame_buffer, action_buffer, state_buffer = [], [], []
            done = False
            while not done:
                frame = env.render()
                frame_buffer.append(self.preprocess_frame(frame))

                action = env.action_space.sample()
                action_buffer.append([float(action)])
                state_buffer.append(obs)

                obs, _, done, _, _ = env.step(action)

                if len(frame_buffer) >= self.seq_length + 1:
                    for i in range(len(frame_buffer) - self.seq_length):
                        s_next = np.array(state_buffer[i + self.seq_length])
                        bin_idx = tuple(np.digitize(s_next[j], bin_edges[j]) - 1 for j in range(4))
                        if any(b < 0 or b >= bins_per_state for b in bin_idx):
                            continue
                        if bin_counts.get(bin_idx, 0) >= max_per_bin:
                            continue

                        x_seq = np.array(frame_buffer[i:i + self.seq_length], dtype=np.float32)
                        a_seq = np.array(action_buffer[i:i + self.seq_length], dtype=np.float32)

                        x_seq_list.append(x_seq)
                        a_seq_list.append(a_seq)
                        state_list.append(s_next)
                        bin_counts[bin_idx] = bin_counts.get(bin_idx, 0) + 1

                        if len(x_seq_list) >= self.dataset_size:
                            break
                    if len(x_seq_list) >= self.dataset_size:
                        break

        self.x_seq = np.array(x_seq_list, dtype=np.float32)
        self.a_seq = np.array(a_seq_list, dtype=np.float32)
        self.states = np.array(state_list, dtype=np.float32)

    def __len__(self):
        return len(self.x_seq)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x_seq[idx]),   # (seq_len, 3, 64, 64)
            torch.from_numpy(self.a_seq[idx]),   # (seq_len, 1)
            torch.from_numpy(self.states[idx])   # (4,)
        )

def analyze_state_distribution(dataset):
    states = dataset.states
    labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Angular Velocity']

    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.hist(states[:, i], bins=50, color='skyblue', edgecolor='black')
        plt.title(labels[i])
        plt.grid(True)
        plt.xlabel("Value")
        plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig("state_distribution.png")
    plt.close()