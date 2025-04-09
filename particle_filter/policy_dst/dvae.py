import gymnasium as gym
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ===========================
# 1. DATA COLLECTION
# ===========================

# Initialize the environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Storage for observations and actions
observations = []
actions = []

state, _ = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Sample a random action
    next_state, _, terminated, truncated, _ = env.step(action)

    # Capture frame and preprocess
    frame = env.render()
    frame = cv2.resize(frame, (64, 64))  # Resize for consistency
    frame = np.array(frame) / 255.0  # Normalize pixel values

    # Store data
    observations.append(frame)
    actions.append(action)

    state = next_state
    done = terminated or truncated

env.close()

# Convert to NumPy arrays
observations = np.array(observations)  # Shape: (num_frames, 64, 64, 3)
actions = np.array(actions)  # Shape: (num_frames,)

# ===========================
# 2. DEFINE UNSUPERVISED DVAE WITH GRU TRANSITION MODEL
# ===========================

class CartPoleDataset(Dataset):
    def __init__(self, observations, actions):
        self.observations = torch.FloatTensor(observations).permute(0, 3, 1, 2)  # (C, H, W)
        self.actions = torch.LongTensor(actions)  # Actions as integers
    
    def __len__(self):
        return len(self.observations) - 1  
    
    def __getitem__(self, idx):
        return self.observations[idx], self.observations[idx + 1], self.actions[idx]

dataset = CartPoleDataset(observations, actions)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define DVAE with GRU-Based Transition Model
class DVAE(nn.Module):
    def __init__(self, latent_dim=8, action_dim=2):  # Latent space 8D, 2 actions (left, right)
        super(DVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128), nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # GRU Transition Model
        self.gru = nn.GRU(latent_dim + action_dim, latent_dim, batch_first=True)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * 8 * 8)  # Ensure correct transformation
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

    def transition(self, z, action, gru_hidden_state):
        """
        GRU-based transition function for predicting the next latent state.
        :param z: Current latent state (Tensor of shape [num_particles, latent_dim])
        :param action: Action taken (Tensor of shape [1] or [batch_size])
        :param gru_hidden_state: GRU hidden state from the previous step
        :return: Next latent state, updated GRU hidden state
        """

        # Ensure z has a batch dimension
        if len(z.shape) == 1:
            z = z.unsqueeze(0)  # Convert shape [latent_dim] -> [1, latent_dim]

        # Ensure action is 2D before one-hot encoding
        action = action.view(-1, 1)  # Shape: [batch_size, 1]
        action_one_hot = torch.nn.functional.one_hot(action, num_classes=2).float()
        action_one_hot = action_one_hot.squeeze(1)  # Ensure shape: [batch_size, action_dim]

        # Ensure action tensor matches z (num_particles)
        if action_one_hot.shape[0] == 1:
            action_one_hot = action_one_hot.repeat(z.shape[0], 1)  # Repeat action for each particle

        # Ensure z and action_one_hot have the same number of dimensions
        if len(z.shape) == 2 and len(action_one_hot.shape) == 1:
            action_one_hot = action_one_hot.unsqueeze(0)  # Convert shape [action_dim] -> [1, action_dim]

        # **Fix Here**: Check if action_one_hot batch size matches z batch size
        if action_one_hot.shape[0] != z.shape[0]:
            action_one_hot = action_one_hot.expand(z.shape[0], -1)  # Expand action tensor

        # Concatenate latent state and action for GRU input
        z_action = torch.cat([z, action_one_hot], dim=-1).unsqueeze(1)  # Add sequence dim

        # Forward pass through GRU
        z_tp1, gru_hidden_state = self.gru(z_action, gru_hidden_state)
        return z_tp1.squeeze(1), gru_hidden_state  # Remove sequence dim

























    def decode(self, z):
        x = self.decoder_input(z)  # Ensure correct transformation
        x = x.view(z.shape[0], 128, 8, 8)  # Fix: Correctly reshape
        return self.decoder(x)

    def forward(self, x_t, x_tp1, action, gru_hidden_state):
        mu_t, logvar_t = self.encode(x_t)
        z_t = self.reparameterize(mu_t, logvar_t)

        z_tp1_pred, gru_hidden_state = self.transition(z_t, action, gru_hidden_state)

        recon_x_t = self.decode(z_t)
        recon_x_tp1 = self.decode(z_tp1_pred)

        return recon_x_t, recon_x_tp1, mu_t, logvar_t, gru_hidden_state

# ===========================
# 3. TRAIN DVAE WITH GRU
# ===========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dvae = DVAE(latent_dim=8, action_dim=2).to(device)
optimizer = optim.Adam(dvae.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(50):  
    gru_hidden_state = None  # Initialize GRU hidden state
    for x_t, x_tp1, action in dataloader:
        x_t, x_tp1, action = x_t.to(device), x_tp1.to(device), action.to(device)

        recon_x_t, recon_x_tp1, mu_t, logvar_t, gru_hidden_state = dvae(x_t, x_tp1, action, gru_hidden_state)

        recon_loss = criterion(recon_x_t, x_t) + criterion(recon_x_tp1, x_tp1)
        kl_loss = -0.5 * torch.mean(1 + logvar_t - mu_t.pow(2) - logvar_t.exp())
        loss = recon_loss + 0.0005 * kl_loss  

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# ===========================
# 4. STATE ESTIMATION WITH PARTICLE FILTER
# ===========================

class ParticleFilter:
    def __init__(self, num_particles, latent_dim):
        self.num_particles = num_particles
        self.latent_dim = latent_dim
        self.particles = np.random.randn(num_particles, latent_dim) * 0.1
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, action, gru_hidden_state):
        self.particles, gru_hidden_state = dvae.transition(
            torch.FloatTensor(self.particles).to(device), 
            torch.LongTensor([action] * self.num_particles).to(device),
            gru_hidden_state
        )
        return gru_hidden_state

    def update(self, observation, encoder):
        observation = torch.FloatTensor(observation).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            encoded_obs, _ = encoder(observation)
        self.particles = encoded_obs.cpu().numpy().flatten()

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)

# ===========================
# 5. RUN PARTICLE FILTER
# ===========================

pf = ParticleFilter(num_particles=100, latent_dim=8)
gru_hidden_state = None

for t in range(len(observations) - 1):
    gru_hidden_state = pf.predict(actions[t], gru_hidden_state)
    pf.update(observations[t], dvae.encode)
