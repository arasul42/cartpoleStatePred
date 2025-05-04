import torch
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import json
import cv2
from model import CartPoleDynamicsModel

# Settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seq_length = 4
image_size = 64

# Load model
model = CartPoleDynamicsModel().to(device)
model.load_state_dict(torch.load('cartpole_dynamics_model.pth'))
model.eval()

# Setup environment
env = gym.make('CartPole-v1', render_mode='rgb_array')
obs, _ = env.reset()
done = False

# Containers
frames, actions, true_states = [], [], []

while not done:
    frame = env.render()
    frame_resized = cv2.resize(frame, (image_size, image_size))
    frame_rgb = cv2.resize(frame, (image_size, image_size))
    frame_tensor = torch.tensor(frame_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1)  # shape: [3, H, W]


    frames.append(frame_tensor)
    true_states.append(obs)

    action = env.action_space.sample()
    obs, _, done, _, _ = env.step(action)
    actions.append(action)

env.close()

# Convert to tensor sequences
frames = torch.stack(frames)  # [T, 1, H, W]
actions = torch.tensor(actions, dtype=torch.float32)  # [T]
true_states = np.array(true_states)  # [T, 4]

# Create sequences
X_images, X_actions, Y_states = [], [], []

for t in range(seq_length, len(frames)):
    seq_images = frames[t-seq_length:t]
    seq_actions = actions[t-seq_length:t]
    target_state = true_states[t]

    X_images.append(seq_images)
    X_actions.append(seq_actions)
    Y_states.append(target_state)

X_images = torch.stack(X_images).to(device)             # [N, 4, 1, H, W]
X_actions = torch.stack(X_actions).unsqueeze(-1).to(device)  # [N, 4, 1]
Y_states = np.array(Y_states)

# Run prediction
with torch.no_grad():
    pred_states = model(X_images, X_actions).cpu().numpy()

# Compute error
mse = np.mean((pred_states - Y_states) ** 2, axis=0)
rmse = np.sqrt(mse)

# Save error
error_dict = {
    'MSE': dict(zip(['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity At Tip'], mse.tolist())),
    'RMSE': dict(zip(['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity At Tip'], rmse.tolist()))
}

with open('real_episode_prediction_errors.json', 'w') as f:
    json.dump(error_dict, f, indent=4)

# Plotting
labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity At Tip']
time = range(len(pred_states))

plt.figure(figsize=(10, 7))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(time, Y_states[:, i], label='True')
    plt.plot(time, pred_states[:, i], '--', label='Predicted')
    plt.title(f"{labels[i]}\nRMSE = {rmse[i]:.4f}")
    plt.legend()

plt.tight_layout()
plt.savefig('real_episode_evaluation.png', dpi=300)

print("✅ Plot saved as 'real_episode_evaluation.png'")
print("✅ Prediction errors saved to 'real_episode_prediction_errors.json'")
