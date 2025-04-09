import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym

# Define the Q-network (ensure it matches the saved model's architecture)
class QNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)  # Change from 128 -> 64
        self.fc2 = torch.nn.Linear(64, 64)         # Change from 128 -> 64
        self.fc3 = torch.nn.Linear(64, output_dim) # Match last layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('CartPole-v1', render_mode="human")  # Render visualization

# Get state and action space dimensions
state_dim = env.observation_space.shape[0]  # 4 for CartPole
action_dim = env.action_space.n  # 2 (left or right)

# Load the model
model = QNetwork(state_dim, action_dim).to(device)
model.load_state_dict(torch.load('best_cartpole_model.pth', map_location=device, weights_only=True))

model.eval()  # Set to evaluation mode





# CartPole dynamics (simplified for particle filter)
def cartpole_dynamics(state, action, dt=0.02):
    x, x_dot, theta, theta_dot = state
    g = 9.8
    mc = 1.0  # Mass of the cart
    mp = 0.1  # Mass of the pole
    l = 0.5   # Half-length of the pole

    # Force from action
    force = action * 10.0  # Scale action to a reasonable force

    # Equations of motion
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    total_mass = mc + mp
    temp = (force + mp * l * theta_dot**2 * sin_theta) / total_mass
    theta_acc = (g * sin_theta - cos_theta * temp) / (l * (4/3 - mp * cos_theta**2 / total_mass))
    x_acc = temp - mp * l * theta_acc * cos_theta / total_mass

    # Update state
    x_dot += x_acc * dt
    x += x_dot * dt
    theta_dot += theta_acc * dt
    theta += theta_dot * dt

    return np.array([x, x_dot, theta, theta_dot])

# Particle filter class
class ParticleFilter:
    def __init__(self, num_particles, initial_state, process_noise, measurement_noise):
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.particles = np.tile(initial_state, (num_particles, 1)) + np.random.randn(num_particles, 4) * process_noise
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, action):
        for i in range(self.num_particles):
            self.particles[i] = cartpole_dynamics(self.particles[i], action)
            self.particles[i] += np.random.randn(4) * self.process_noise

    def update(self, measurement):
        for i in range(self.num_particles):
            # Likelihood of measurement given particle state
            diff = measurement - self.particles[i]
            likelihood = np.exp(-0.5 * np.dot(diff, diff) / self.measurement_noise**2)
            self.weights[i] = likelihood
        self.weights /= np.sum(self.weights)  # Normalize weights

    def resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        return np.average(self.particles, axis=0, weights=self.weights)

# Initialize environment and particle filter
env = gym.make('CartPole-v1',render_mode='human')
observation, _ = env.reset()
num_particles = 1000
process_noise = 0.1
measurement_noise = 0.1
pf = ParticleFilter(num_particles, observation, process_noise, measurement_noise)



def select_action(state):
    state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)  # Convert to tensor and add batch dimension
    with torch.no_grad():
        q_values = model(state_tensor)  # Get Q-values from the model
    return torch.argmax(q_values).item()  # Select action with highest Q-value

# Run simulation
states = []
estimates = []
for t in range(200):
    # True state (from environment)
    action = select_action(observation)
    observation, _, terminated, truncated, _ = env.step(action)
    states.append(observation)

    # Particle filter steps
    pf.predict(action)
    pf.update(observation)
    pf.resample()
    estimate = pf.estimate()
    estimates.append(estimate)

    if terminated or truncated:
        break

env.close()

# Convert to numpy arrays
states = np.array(states)
estimates = np.array(estimates)

# Plot results for all 4 states
# plt.figure(figsize=(14, 10))

# # Cart Position
# plt.subplot(2, 2, 1)
# plt.plot(states[:, 0], label='True Cart Position')
# plt.plot(estimates[:, 0], label='Estimated Cart Position', linestyle='--')
# plt.xlabel('Time Step')
# plt.ylabel('Cart Position')
# plt.legend()

# # Cart Velocity
# plt.subplot(2, 2, 2)
# plt.plot(states[:, 1], label='True Cart Velocity')
# plt.plot(estimates[:, 1], label='Estimated Cart Velocity', linestyle='--')
# plt.xlabel('Time Step')
# plt.ylabel('Cart Velocity')
# plt.legend()

# # Pole Angle
# plt.subplot(2, 2, 3)
# plt.plot(states[:, 2], label='True Pole Angle')
# plt.plot(estimates[:, 2], label='Estimated Pole Angle', linestyle='--')
# plt.xlabel('Time Step')
# plt.ylabel('Pole Angle')
# plt.legend()

# # Pole Angular Velocity
# plt.subplot(2, 2, 4)
# plt.plot(states[:, 3], label='True Pole Angular Velocity')
# plt.plot(estimates[:, 3], label='Estimated Pole Angular Velocity', linestyle='--')
# plt.xlabel('Time Step')
# plt.ylabel('Pole Angular Velocity')
# plt.legend()

# # Save the plot to a file
# plt.tight_layout()
# plt.savefig('particle_filter_all_states.png')  # Save as PNG file
# print("Plot saved as 'particle_filter_all_states.png'")


figsize_double_column = (7, 4)  # Double-column figure

# Generate 4 different plots suitable for an IEEE paper

# 1. Cart Position
plt.figure(figsize=figsize_double_column)
plt.plot(states[:, 0], label='True Cart Position')
plt.plot(estimates[:, 0], label='Estimated Cart Position', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Cart Position')
plt.legend()
plt.tight_layout()
plt.savefig('cart_position_known_dm.png')

# 2. Cart Velocity
plt.figure(figsize=figsize_double_column)
plt.plot(states[:, 1], label='True Cart Velocity')
plt.plot(estimates[:, 1], label='Estimated Cart Velocity', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Cart Velocity')
plt.legend()
plt.tight_layout()
plt.savefig('cart_velocity_known dm.png')

# 3. Pole Angle
plt.figure(figsize=figsize_double_column)
plt.plot(states[:, 2], label='True Pole Angle')
plt.plot(estimates[:, 2], label='Estimated Pole Angle', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Pole Angle')
plt.legend()
plt.tight_layout()
plt.savefig('pole_angle_known dm.png')

# 4. Pole Angular Velocity
plt.figure(figsize=figsize_double_column)
plt.plot(states[:, 3], label='True Pole Angular Velocity')
plt.plot(estimates[:, 3], label='Estimated Pole Angular Velocity', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Pole Angular Velocity')
plt.legend()
plt.tight_layout()
plt.savefig('pole_angular_velocity_known dm.png')

# Display confirmation
print("Plots saved as 'cart_position_ieee.png', 'cart_velocity_ieee.png', 'pole_angle_ieee.png', and 'pole_angular_velocity_ieee.png'.")
