from matplotlib import pyplot as plt
import gymnasium as gym
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import DQN

# ===========================
# 1. DATA COLLECTION
# ===========================

# Initialize the environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Load pre-trained model
model = DQN.load("dqn_cartpole")

# Storage for states and observations
observations = []
actions = []

state, _ = env.reset()
done = False
true_states = []

while not done:
    action, _states = model.predict(state, deterministic=True)
    next_state, _, terminated, truncated, _ = env.step(action)

    # Capture frame and preprocess
    frame = env.render()
    frame = cv2.resize(frame, (64, 64))  # Resize for consistency
    frame = np.array(frame) / 255.0  # Normalize pixel values

    # Store data
    observations.append(frame)
    true_states.append(state)
    actions.append(action)  # Store actions

    state = next_state
    done = terminated or truncated

env.close()

# Convert to NumPy arrays
observations = np.array(observations)  # Shape: (num_frames, 64, 64, 3)
actions = np.array(actions).reshape(-1, 1)  # Shape: (num_frames, 1)
true_states = np.array(true_states)
# ===========================
# 2. DEFINE DVAE MODEL
# ===========================

class CartPoleDataset(Dataset):
    def __init__(self, observations, actions):
        self.observations = torch.FloatTensor(observations).permute(0, 3, 1, 2)  # (N, C, H, W)
        self.actions = torch.FloatTensor(actions)

    def __len__(self):
        return len(self.observations) - 1
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx], self.observations[idx + 1]

dataset = CartPoleDataset(observations, actions)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define Dynamic VAE (DVAE)
class DVAE(nn.Module):
    def __init__(self, latent_dim=4, action_dim=1):  # 4D latent space, 1D action
        super(DVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

        # Temporal transition model (GRU with action input)
        self.gru = nn.GRU(latent_dim + action_dim, latent_dim, batch_first=True)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def transition(self, z, action):
        z_action = torch.cat((z, action), dim=1).unsqueeze(1)  # Concatenate action with latent vector
        z, _ = self.gru(z_action)
        return z.squeeze(1)

    def decode(self, z):
        x = self.decoder_input(z).view(-1, 128, 8, 8)
        return self.decoder(x)

    def forward(self, x_t, action, x_tp1):
        mu_t, logvar_t = self.encode(x_t)
        mu_tp1, logvar_tp1 = self.encode(x_tp1)

        z_t = self.reparameterize(mu_t, logvar_t)
        z_tp1_pred = self.transition(z_t, action)

        recon_x_t = self.decode(z_t)
        recon_x_tp1 = self.decode(z_tp1_pred)

        return recon_x_t, recon_x_tp1, mu_tp1, logvar_tp1

# ===========================
# 3. TRAIN DVAE
# ===========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dvae = DVAE(latent_dim=4, action_dim=1).to(device)
optimizer = optim.Adam(dvae.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(10):  
    for x_t, action, x_tp1 in dataloader:
        x_t, action, x_tp1 = x_t.to(device), action.to(device), x_tp1.to(device)
        recon_x_t, recon_x_tp1, mu_tp1, logvar_tp1 = dvae(x_t, action, x_tp1)

        recon_loss = criterion(recon_x_t, x_t) + criterion(recon_x_tp1, x_tp1)
        kl_loss = -0.5 * torch.mean(1 + logvar_tp1 - mu_tp1.pow(2) - logvar_tp1.exp())
        loss = recon_loss + 0.001 * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


# ===========================
# 4. PARTICLE FILTER
# ===========================



class ParticleFilter:
    def __init__(self, num_particles, latent_dim):
        self.num_particles = num_particles
        self.latent_dim = latent_dim
        self.particles = np.random.randn(num_particles, latent_dim) * 0.1  # Initialize around 0
        self.weights = np.ones(num_particles) / num_particles  # Uniform weights

    def predict(self):
        """Propagate particles using a Gaussian noise model."""
        self.particles += np.random.randn(self.num_particles, self.latent_dim) * 0.05  

    def update(self, observation, encoder):
        """Update weights based on how close particles are to the encoded observation."""
        observation = torch.FloatTensor(observation).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        
        with torch.no_grad():
            encoded_obs, _ = encoder(observation)  # Only use mu
        encoded_obs = encoded_obs.cpu().numpy().flatten()

        distances = np.linalg.norm(self.particles - encoded_obs, axis=1)  # Euclidean distance
        
        # Normalize distances to avoid numerical issues
        distances = distances / (np.max(distances) + 1e-6)  

        self.weights = np.exp(-distances**2 / 0.1)  # Gaussian weighting
        self.weights /= np.sum(self.weights) + 1e-6  # Normalize

    def resample(self):
        """Resample particles using systematic resampling for better particle diversity."""
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # Ensure sum is exactly 1.0 due to floating-point precision
        step = 1.0 / self.num_particles
        positions = (np.arange(self.num_particles) + np.random.rand()) * step

        indices = np.searchsorted(cumulative_sum, positions)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)  # Reset weights

    def estimate(self):
        """Estimate latent state as a weighted mean of the particles."""
        return np.average(self.particles, weights=self.weights, axis=0)

# ===========================
# 5. STATE ESTIMATION
# ===========================

pf = ParticleFilter(num_particles=100, latent_dim=4)

estimated_states = []
for t in range(len(observations) - 1):
    pf.predict()
    pf.update(observations[t], dvae.encode)
    pf.resample()
    estimated_states.append(pf.estimate())

# Convert estimated states list to numpy array
estimated_states = np.array(estimated_states)

# Print an example estimated state
# print(f"Example estimated state: {estimated_states[0]}")


# Extract true and estimated states for plotting
time_steps = np.arange(len(true_states) - 1)  # Since we estimate one step less than total observations

true_positions = true_states[:-1, 0]  # Cart position
true_velocities = true_states[:-1, 1]  # Cart velocity
true_angles = true_states[:-1, 2]  # Pole angle
true_angular_velocities = true_states[:-1, 3]  # Pole angular velocity

estimated_positions = estimated_states[:, 0]
estimated_velocities = estimated_states[:, 1]
estimated_angles = estimated_states[:, 2]
estimated_angular_velocities = estimated_states[:, 3]

# Plot True vs Estimated States
fig, axs = plt.subplots(4, 1, figsize=(10, 12))

axs[0].plot(time_steps, true_positions, label="True Cart Position", color="blue")
axs[0].plot(time_steps, estimated_positions, label="Estimated Cart Position", linestyle="dashed", color="red")
axs[0].set_ylabel("Cart Position")
axs[0].legend()

axs[1].plot(time_steps, true_velocities, label="True Cart Velocity", color="blue")
axs[1].plot(time_steps, estimated_velocities, label="Estimated Cart Velocity", linestyle="dashed", color="red")
axs[1].set_ylabel("Cart Velocity")
axs[1].legend()

axs[2].plot(time_steps, true_angles, label="True Pole Angle", color="blue")
axs[2].plot(time_steps, estimated_angles, label="Estimated Pole Angle", linestyle="dashed", color="red")
axs[2].set_ylabel("Pole Angle")
axs[2].legend()

axs[3].plot(time_steps, true_angular_velocities, label="True Angular Velocity", color="blue")
axs[3].plot(time_steps, estimated_angular_velocities, label="Estimated Angular Velocity", linestyle="dashed", color="red")
axs[3].set_ylabel("Angular Velocity")
axs[3].set_xlabel("Time Steps")
axs[3].legend()

plt.suptitle("True vs Estimated CartPole States using Particle Filter over DVAE")
plt.savefig("test.png")

