import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 🔹 Simulated Mass-Spring-Damper System (Ground Truth Model)
def mass_spring_damper(y_prev, u, dt=0.01):
    k, c, m = 5.0, 0.5, 1.0  # Spring constant, damping, mass
    y_dot = y_prev[1]  # Velocity
    y_ddot = (-k/m) * y_prev[0] - (c/m) * y_prev[1] + u / m  # Acceleration
    return np.array([y_prev[0] + y_dot * dt, y_prev[1] + y_ddot * dt])

# 🔹 Generate Training Data
num_steps = 1000
y_data, u_data = [], []
y = np.array([0.0, 0.0])  # Initial state

for _ in range(num_steps):
    u = np.random.uniform(-1, 1)  # Random force input
    y = mass_spring_damper(y, u)
    y_data.append(y)
    u_data.append(u)

y_data, u_data = np.array(y_data), np.array(u_data)

# Convert to PyTorch tensors
y_tensor = torch.tensor(y_data, dtype=torch.float32)
u_tensor = torch.tensor(u_data, dtype=torch.float32).unsqueeze(1)

# 🔹 Define the Dynamic Autoencoder
class DynamicAutoencoder(nn.Module):
    def __init__(self, latent_dim=4):
        super(DynamicAutoencoder, self).__init__()
        
        # Encoder: from [y, u] → latent z
        self.encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, latent_dim)
        )
        
        # Latent Dynamics: z_{t+1} = A z_t + B u_t
        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(1, latent_dim, bias=False)
        
        # Decoder: from z → y prediction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 2)
        )

    def forward(self, y, u, z_prev):
        z = self.encoder(torch.cat((y, u), dim=1))
        z_next = self.A(z_prev) + self.B(u)
        y_pred = self.decoder(z_next)
        return z_next, y_pred

# 🔹 Initialize Model and Optimizer
latent_dim = 4
model = DynamicAutoencoder(latent_dim)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
criterion = nn.MSELoss()

# 🔁 Training Loop (Recursive Latent State Propagation)
num_epochs = 500

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    z = torch.zeros(latent_dim)  # initial latent state
    y_preds = []

    for t in range(num_steps - 1):
        y_t = y_tensor[t].unsqueeze(0)
        u_t = u_tensor[t].unsqueeze(0)

        z_enc = model.encoder(torch.cat((y_t, u_t), dim=1)).squeeze(0)
        z = model.A(z_enc) + model.B(u_t).squeeze(0)  # propagate latent state
        y_pred = model.decoder(z.unsqueeze(0)).squeeze(0)
        y_preds.append(y_pred)

    y_preds = torch.stack(y_preds)
    loss = criterion(y_preds, y_tensor[1:])
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 🔹 Testing the Model
model.eval()
with torch.no_grad():
    z = torch.zeros(latent_dim)
    y_preds_test = []

    for t in range(num_steps - 1):
        y_t = y_tensor[t].unsqueeze(0)
        u_t = u_tensor[t].unsqueeze(0)

        z_enc = model.encoder(torch.cat((y_t, u_t), dim=1)).squeeze(0)
        z = model.A(z_enc) + model.B(u_t).squeeze(0)
        y_pred = model.decoder(z.unsqueeze(0)).squeeze(0)
        y_preds_test.append(y_pred)

    y_preds_test = torch.stack(y_preds_test)

# 🔹 Plot Results
plt.figure(figsize=(10,5))
plt.plot(y_data[1:, 0], label="True Position", color="blue")
plt.plot(y_preds_test[:, 0].numpy(), label="Predicted Position", linestyle="dashed", color="red")
plt.legend()
plt.xlabel("Time Step")
plt.ylabel("Position")
plt.title("Improved System Identification using Dynamic Autoencoder")
plt.grid()
plt.savefig("dae.png")
# plt.show()
