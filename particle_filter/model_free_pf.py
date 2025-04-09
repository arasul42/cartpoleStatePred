import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# Particle filter class (model-free)
class ParticleFilter:
    def __init__(self, num_particles, initial_state, process_noise, measurement_noise):
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.particles = np.tile(initial_state, (num_particles, 1)) + np.random.randn(num_particles, 4) * process_noise
        self.weights = np.ones(num_particles) / num_particles

    def predict(self):
        # Model-free prediction: add random noise
        self.particles += np.random.randn(self.num_particles, 4) * self.process_noise

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
env = gym.make('CartPole-v1')
observation, _ = env.reset()
num_particles = 1000
process_noise = 0.1
measurement_noise = 0.1
pf = ParticleFilter(num_particles, observation, process_noise, measurement_noise)

# Run simulation
states = []
estimates = []
for t in range(200):
    # True state (from environment)
    action = env.action_space.sample()  # Random action
    observation, _, terminated, truncated, _ = env.step(action)
    states.append(observation)

    # Particle filter steps
    pf.predict()  # Model-free prediction
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
plt.figure(figsize=(14, 10))

# Cart Position
plt.subplot(2, 2, 1)
plt.plot(states[:, 0], label='True Cart Position')
plt.plot(estimates[:, 0], label='Estimated Cart Position', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Cart Position')
plt.legend()

# Cart Velocity
plt.subplot(2, 2, 2)
plt.plot(states[:, 1], label='True Cart Velocity')
plt.plot(estimates[:, 1], label='Estimated Cart Velocity', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Cart Velocity')
plt.legend()

# Pole Angle
plt.subplot(2, 2, 3)
plt.plot(states[:, 2], label='True Pole Angle')
plt.plot(estimates[:, 2], label='Estimated Pole Angle', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Pole Angle')
plt.legend()

# Pole Angular Velocity
plt.subplot(2, 2, 4)
plt.plot(states[:, 3], label='True Pole Angular Velocity')
plt.plot(estimates[:, 3], label='Estimated Pole Angular Velocity', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Pole Angular Velocity')
plt.legend()

# Save the plot to a file
plt.tight_layout()
plt.savefig('model_free_particle_filter.png')  # Save as PNG file
print("Plot saved as 'model_free_particle_filter.png'")