import os
import gymnasium as gym
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class CleanCartPoleDataset(Dataset):
    def __init__(self, env_name='CartPole-v1', seq_length=4, dataset_size=1000, image_size=64, cache_file='cartpole_sequences.npz'):
        self.env_name = env_name
        self.seq_length = seq_length
        self.dataset_size = dataset_size
        self.image_size = image_size
        self.cache_file = cache_file

        self.sequences = []

        if os.path.exists(cache_file):
            print(f"📂 Loading cached dataset from {cache_file}")
            data = np.load(cache_file)
            frames = data['frames']
            actions = data['actions']
            next_states = data['next_states']
            self.sequences = list(zip(frames, actions, next_states))

        else:
            print("📈 Generating dataset...")
            self.env = gym.make(env_name, render_mode='rgb_array')
            self.collect_data()
            self.save_sequences()
            print(f"✅ Dataset cached to {cache_file}")
    
    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (self.image_size, self.image_size))
        frame = frame.transpose(2, 0, 1)  # channels-first (C, H, W)
        return frame.astype(np.float32) / 255.0

    def save_sequences(self):
        frames = []
        actions = []
        next_states = []

        for seq_frames, seq_actions, next_state in self.sequences:
            frames.append(seq_frames)
            actions.append(seq_actions)
            next_states.append(next_state)

        np.savez_compressed(
            self.cache_file,
            frames=np.array(frames, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            next_states=np.array(next_states, dtype=np.float32)
        )
    
    #  def collect_data(self):
    #     print("Collecting data...")
    #     frames_buffer, actions_buffer, states_buffer = [], [], []

    #     obs, _ = self.env.reset()
    #     collected_sequences = 0
    #     steps = 0

    #     while collected_sequences < self.dataset_size:
    #         frame = self.env.render()
    #         frames_buffer.append(self.preprocess_frame(frame))

    #         action = self.env.action_space.sample()
    #         actions_buffer.append([action])

    #         states_buffer.append(obs)

    #         obs, _, done, _, _ = self.env.step(action)
    #         steps += 1

    #         if done:
    #             if len(frames_buffer) >= self.seq_length + 1:
    #                 for i in range(len(frames_buffer) - self.seq_length):
    #                     seq_frames = np.array(frames_buffer[i:i + self.seq_length], dtype=np.float32)
    #                     seq_actions = np.array(actions_buffer[i:i + self.seq_length], dtype=np.float32)
    #                     next_state = np.array(states_buffer[i + self.seq_length], dtype=np.float32)
                        
    #                     self.sequences.append((seq_frames, seq_actions, next_state))
    #                     collected_sequences += 1

    #                     if collected_sequences >= self.dataset_size:
    #                         break

    #             frames_buffer.clear()
    #             actions_buffer.clear()
    #             states_buffer.clear()
    #             obs, _ = self.env.reset()

    #     print(f"Data collection complete. Collected {len(self.sequences)} sequences in {steps} steps.")



    def collect_data(self, bins_per_state=5, max_per_bin=4000, targeted_reset_ratio=0.4,
                    show_preview=True, preview_interval=10):
        print("🔄 Starting uniform-bin-targeted data collection...")
        frames_buffer, actions_buffer, states_buffer = [], [], []

        collected_sequences = 0
        steps = 0

        # Define state limits and bin edges
        state_limits = {
            0: (-2.4, 2.4),     # cart position
            1: (-3.0, 3.0),     # cart velocity
            2: (-0.209, 0.209), # pole angle
            3: (-3.0, 3.0),     # angular velocity
        }
        bin_edges = {i: np.linspace(*state_limits[i], bins_per_state + 1) for i in range(4)}
        bin_counts = {}  # bin index tuple → count

        def targeted_reset():
            cart_pos = np.random.uniform(-1.5, 1.5)
            cart_vel = np.random.uniform(-3.0, 3.0)
            pole_angle = np.random.uniform(-0.18, 0.18)
            ang_vel = np.random.uniform(-2.5, 2.5)
            self.env.reset()
            self.env.unwrapped.state = np.array([cart_pos, cart_vel, pole_angle, ang_vel])
            return self.env.unwrapped.state

        while collected_sequences < self.dataset_size:
            use_targeted = np.random.rand() < targeted_reset_ratio
            obs = targeted_reset() if use_targeted else self.env.reset()[0]

            frames_buffer.clear()
            actions_buffer.clear()
            states_buffer.clear()
            done = False

            while not done and collected_sequences < self.dataset_size:
                frame = self.env.render()

                # === Live Preview ===
                if show_preview and steps % preview_interval == 0:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    info_lines = [
                        f"Step: {steps}",
                        f"Collected: {collected_sequences} / {self.dataset_size}",
                        f"State: pos={obs[0]:.2f}, vel={obs[1]:.2f}, ang={obs[2]:.3f}, ang_vel={obs[3]:.2f}"
                    ]
                    for i, text in enumerate(info_lines):
                        org = (10, 20 + 20 * i)
                        cv2.putText(frame_bgr, text, (org[0]+1, org[1]+1), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow("Collecting CartPole Dataset", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("🛑 Interrupted by user.")
                        cv2.destroyAllWindows()
                        return

                # Preprocess and store
                frames_buffer.append(self.preprocess_frame(frame))
                action = self.env.action_space.sample()
                actions_buffer.append([action])
                states_buffer.append(obs)

                obs, _, done, _, _ = self.env.step(action)
                steps += 1

                # Sample if enough sequence collected
                if len(frames_buffer) >= self.seq_length + 1:
                    for i in range(len(frames_buffer) - self.seq_length):
                        next_state = np.array(states_buffer[i + self.seq_length], dtype=np.float32)
                        bin_idx = tuple(np.digitize(next_state[j], bin_edges[j]) - 1 for j in range(4))

                        if any(b < 0 or b >= bins_per_state for b in bin_idx):
                            continue
                        if bin_counts.get(bin_idx, 0) >= max_per_bin:
                            continue

                        seq_frames = np.array(frames_buffer[i:i + self.seq_length], dtype=np.float32)
                        seq_actions = np.array(actions_buffer[i:i + self.seq_length], dtype=np.float32)

                        self.sequences.append((seq_frames, seq_actions, next_state))
                        bin_counts[bin_idx] = bin_counts.get(bin_idx, 0) + 1
                        collected_sequences += 1

                        if collected_sequences % 100 == 0:
                            print(f"📦 Collected {collected_sequences} / {self.dataset_size} samples...")

                        if collected_sequences >= self.dataset_size:
                            break

        print(f"✅ Finished collecting {collected_sequences} sequences in {steps} steps.")
        print(f"📊 Unique bins filled: {len(bin_counts)} / {bins_per_state**4}")
        cv2.destroyAllWindows()

















    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_frames, seq_actions, next_state = self.sequences[idx]
        return (
            torch.from_numpy(seq_frames),
            torch.from_numpy(seq_actions),
            torch.from_numpy(next_state)
        )
    
def analyze_state_distribution(dataset):
        all_states = []

        for _, _, next_state in dataset:
            all_states.append(next_state.numpy())

        all_states = np.stack(all_states)  # shape: [num_samples, 4]
        labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Angular Velocity']

        plt.figure(figsize=(8.5, 4.5))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.hist(all_states[:, i], bins=50, color='skyblue', edgecolor='black')
            plt.title(labels[i])
            plt.grid(True)
            plt.xlabel("Value")
            plt.ylabel("Count")

        plt.tight_layout()
        plt.savefig("state_distribution.png", dpi=300)
        plt.close()
