import torch
import numpy as np
import matplotlib.pyplot as plt
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

# Particle Filter Parameters
num_particles = 1000
process_noise = 0.1
measurement_noise = 0.1
state_low = np.array([-4.8, -100, -0.418, -100])  # State lower bounds
state_high = np.array([4.8, 100, 0.418, 100])     # State upper bounds


import numpy as np

def cartpole_dynamics(state, action, dt=0.02):
    """
    Simulates the CartPole dynamics using an approximation of the system's equations of motion.

    Args:
        state (np.array): Current state [cart_position, cart_velocity, pole_angle, pole_angular_velocity].
        action (int): Action (0 = push left, 1 = push right).
        dt (float): Time step (default = 0.02s for CartPole-v1).

    Returns:
        np.array: Next state after applying action.
    """
    # CartPole physical constants (approximated from OpenAI Gym)
    gravity = 9.8       # Gravity (m/s^2)
    mass_cart = 1.0     # Mass of the cart (kg)
    mass_pole = 0.1     # Mass of the pole (kg)
    total_mass = mass_cart + mass_pole
    length = 0.5        # Half-length of the pole (m)
    force_mag = 10.0    # Force applied for action
    friction_cart = 0.1 # Cart friction coefficient
    friction_pole = 0.01 # Pole friction at the pivot

    x, x_dot, theta, theta_dot = state  # Unpack current state

    # Force applied based on action
    force = force_mag if action == 1 else -force_mag

    # Pole dynamics equations (approximated)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    temp = (force + mass_pole * length * theta_dot**2 * sin_theta) / total_mass
    theta_acc = (gravity * sin_theta - cos_theta * temp) / \
                (length * (4.0/3.0 - mass_pole * cos_theta**2 / total_mass))
    x_acc = temp - mass_pole * length * theta_acc * cos_theta / total_mass

    # Apply friction (simplified)
    x_acc -= friction_cart * x_dot
    theta_acc -= friction_pole * theta_dot

    # Integrate forward using Euler method
    x_new = x + dt * x_dot
    x_dot_new = x_dot + dt * x_acc
    theta_new = theta + dt * theta_dot
    theta_dot_new = theta_dot + dt * theta_acc

    return np.array([x_new, x_dot_new, theta_new, theta_dot_new])





# Particle Filter Class
class ParticleFilter:
    def __init__(self, num_particles, state_low, state_high, process_noise, measurement_noise):
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.particles = np.random.uniform(low=state_low, high=state_high, size=(num_particles, 4))
        self.weights = np.ones(num_particles) / num_particles

    # def predict(self):
    #     self.particles += np.random.randn(self.num_particles, 4) * self.process_noise

    def predict(self, action):
        for i in range(self.num_particles):
            self.particles[i] = cartpole_dynamics(self.particles[i], action) + np.random.randn(4) * self.process_noise


    def update(self, measurement):
        for i in range(self.num_particles):
            diff = measurement - self.particles[i]
            likelihood = np.exp(-0.5 * np.dot(diff, diff) / self.measurement_noise**2)
            self.weights[i] = likelihood
        weight_sum = np.sum(self.weights)
        if weight_sum == 0:  # Prevent division by zero
            self.weights = np.ones(self.num_particles) / self.num_particles  # Reset to uniform distribution
        else:
            self.weights /= weight_sum  # Normalize weights


    def resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        return np.average(self.particles, axis=0, weights=self.weights)

# Function to select the best action using the trained model
def select_action(state):
    state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)  # Convert to tensor and add batch dimension
    with torch.no_grad():
        q_values = model(state_tensor)  # Get Q-values from the model
    return torch.argmax(q_values).item()  # Select action with highest Q-value

# Run Simulation
pf = ParticleFilter(num_particles, state_low, state_high, process_noise, measurement_noise)
observation, _ = env.reset()

states = []
estimates = []
for t in range(1000):
    action = select_action(observation)  # Use trained model for action selection
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

# # Plot results
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

# # Save the plot
# plt.tight_layout()
# plt.savefig('q_learning_particle_filter.png')
# print("Plot saved as 'q_learning_particle_filter.png'")


figsize_single_column = (3.5, 2.5)  # Single-column figure
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
plt.savefig('cart_position_ieee.png')

# 2. Cart Velocity
plt.figure(figsize=figsize_double_column)
plt.plot(states[:, 1], label='True Cart Velocity')
plt.plot(estimates[:, 1], label='Estimated Cart Velocity', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Cart Velocity')
plt.legend()
plt.tight_layout()
plt.savefig('cart_velocity_ieee.png')

# 3. Pole Angle
plt.figure(figsize=figsize_double_column)
plt.plot(states[:, 2], label='True Pole Angle')
plt.plot(estimates[:, 2], label='Estimated Pole Angle', linestyle='--')
plt.xlabel('Time Step')

plt.ylabel('Pole Angle')
plt.legend()
plt.tight_layout()
plt.savefig('pole_angle_ieee.png')

# 4. Pole Angular Velocity
plt.figure(figsize=figsize_double_column)
plt.plot(states[:, 3], label='True Pole Angular Velocity')
plt.plot(estimates[:, 3], label='Estimated Pole Angular Velocity', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Pole Angular Velocity')
plt.legend()
plt.tight_layout()
plt.savefig('pole_angular_velocity_ieee.png')

# Display confirmation
print("Plots saved as 'cart_position_ieee.png', 'cart_velocity_ieee.png', 'pole_angle_ieee.png', and 'pole_angular_velocity_ieee.png'.")
