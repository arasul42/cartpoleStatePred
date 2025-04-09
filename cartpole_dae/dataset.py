import gymnasium as gym
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class CleanCartPoleDataset(Dataset):
    def __init__(self, env_name='CartPole-v1', seq_length=4, dataset_size=1000, image_size=64):
        self.env = gym.make(env_name, render_mode='rgb_array')
        self.seq_length = seq_length
        self.dataset_size = dataset_size
        self.image_size = image_size
        
        self.sequences = []  # List of (frames, actions, next_state)
        self.collect_data()
    
    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (self.image_size, self.image_size))
        frame = frame.transpose(2, 0, 1)  # channels-first (C, H, W)
        frame = frame.astype(np.float32) / 255.0  # normalize to [0,1]
        return frame
    
    def collect_data(self):
        print("Collecting data...")
        frames_buffer, actions_buffer, states_buffer = [], [], []

        obs, _ = self.env.reset()
        collected_sequences = 0
        steps = 0

        while collected_sequences < self.dataset_size:
            # Render and preprocess frame
            frame = self.env.render()
            preprocessed_frame = self.preprocess_frame(frame)
            frames_buffer.append(preprocessed_frame)

            # Random action for data collection
            action = self.env.action_space.sample()
            actions_buffer.append([action])

            # Record state
            states_buffer.append(obs)

            # Step environment
            obs, _, done, _, _ = self.env.step(action)
            steps += 1

            if done:
                # If buffer is long enough, extract sequences
                if len(frames_buffer) >= self.seq_length + 1:
                    for i in range(len(frames_buffer) - self.seq_length):
                        seq_frames = np.array(frames_buffer[i:i + self.seq_length], dtype=np.float32)
                        seq_actions = np.array(actions_buffer[i:i + self.seq_length], dtype=np.float32)
                        next_state = np.array(states_buffer[i + self.seq_length], dtype=np.float32)
                        
                        self.sequences.append((seq_frames, seq_actions, next_state))
                        collected_sequences += 1

                        if collected_sequences >= self.dataset_size:
                            break


                # Reset buffers and environment
                frames_buffer.clear()
                actions_buffer.clear()
                states_buffer.clear()
                obs, _ = self.env.reset()

        print(f"Data collection complete. Collected {len(self.sequences)} sequences in {steps} steps.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_frames, seq_actions, next_state = self.sequences[idx]
        return (
            torch.from_numpy(seq_frames),
            torch.from_numpy(seq_actions),
            torch.from_numpy(next_state)
        )
