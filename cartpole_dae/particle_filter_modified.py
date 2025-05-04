import numpy as np
import torch

class ModelDrivenParticleFilter_Modified:
    """
    Particle Filter where both transition and measurement are given by the same neural network model.
    Maintains uncertainty by adding process noise and resampling based on closeness to prediction.
    """

    def __init__(self, model, device, num_particles=100, process_noise_std=0.01):
        self.model = model.eval()
        self.device = device
        self.N = num_particles
        self.process_noise_std = process_noise_std

        self.particles = None  # shape: [N, 4]
        self.weights = np.ones(self.N) / self.N

    def initialize(self, bounds=None):
        # Initialize particles over the full state space
        # Default bounds: CartPole state [x, x_dot, theta, theta_dot]
        if bounds is None:
            bounds = {
                'x': (-2.4, 2.4),
                'x_dot': (-5.0, 5.0),
                'theta': (-0.209, 0.209),
                'theta_dot': (-5.0, 5.0)
            }

        x = np.random.uniform(*bounds['x'], size=self.N)
        x_dot = np.random.uniform(*bounds['x_dot'], size=self.N)
        theta = np.random.uniform(*bounds['theta'], size=self.N)
        theta_dot = np.random.uniform(*bounds['theta_dot'], size=self.N)

        self.particles = np.stack([x, x_dot, theta, theta_dot], axis=1)
        self.weights.fill(1.0 / self.N)

    def predict(self, frames_buffer, actions_buffer):
        # Predict using NN and inject process noise
        frames = torch.tensor([frames_buffer], dtype=torch.float32).to(self.device)
        actions = torch.tensor([actions_buffer], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred = self.model(frames, actions).cpu().numpy()[0]  # [x, x_dot, theta, theta_dot]

        noise = np.random.normal(0, self.process_noise_std, size=(self.N, 4))
        self.particles = pred + noise  # All particles centered around NN prediction

        return pred  # return the measurement to be reused

    def update(self, measurement, obs_noise_std=0.05):
        # Update weights based on how close particles are to measurement
        diffs = self.particles - measurement
        likelihoods = np.exp(-0.5 * np.sum((diffs / obs_noise_std) ** 2, axis=1))
        likelihoods += 1e-12  # prevent divide-by-zero
        self.weights = likelihoods / np.sum(likelihoods)

    def resample(self):
        indices = np.random.choice(self.N, size=self.N, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.N) / self.N

    def estimate(self, return_std=False):
        mean = np.average(self.particles, axis=0, weights=self.weights)
        if return_std:
            std = np.sqrt(np.average((self.particles - mean) ** 2, axis=0, weights=self.weights))
            return mean, std
        return mean

