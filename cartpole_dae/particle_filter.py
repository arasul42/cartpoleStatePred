import numpy as np
import torch

class ModelDrivenParticleFilter:
    def __init__(self, model, device, num_particles=200, dt=1/50):
        self.model = model.eval()
        self.device = device
        self.N = num_particles
        self.dt = dt

        self.particles = None  # [N, 4]
        self.weights = np.ones(self.N) / self.N
        self.prev_pred = None

    # def initialize(self):
    #     # Initialize full state: [x, x_dot, theta, theta_dot]
    #     x_range = [-2.4, 2.4]
    #     theta_range = [-0.209, 0.209]
    #     vel_range = [-5.0, 5.0]

    #     x = np.random.uniform(*x_range, self.N)
    #     x_dot = np.random.uniform(*vel_range, self.N)
    #     theta = np.random.uniform(*theta_range, self.N)
    #     theta_dot = np.random.uniform(*vel_range, self.N)

    #     self.particles = np.stack([x, x_dot, theta, theta_dot], axis=1)
    #     self.weights.fill(1.0 / self.N)


    def initialize(self, initial_pred):
        """
        Initialize particles around the first NN prediction.
        `initial_pred` should be a numpy array of shape (4,)
        """
        self.prev_pred = initial_pred.copy()
        self.particles = initial_pred + np.random.normal(0, 0.05, size=(self.N, 4))  # moderate noise
        self.weights = np.ones(self.N) / self.N




    def predict(self, frames_buffer, actions_buffer, process_noise_std=0.0001):
        # Predict next positions with model
        frames = torch.tensor(np.array([frames_buffer]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array([actions_buffer]), dtype=torch.float32).to(self.device)


        with torch.no_grad():
            pred = self.model(frames, actions).cpu().numpy()[0]  # shape: (4,)

        # Estimate velocities using position differences
        if self.prev_pred is not None:
            x_dot = (pred[0] - self.prev_pred[0]) / self.dt
            theta_dot = (pred[2] - self.prev_pred[2]) / self.dt
        else:
            x_dot = 0.0
            theta_dot = 0.0

        self.prev_pred = pred.copy()
        pred_state = np.array([pred[0], x_dot, pred[2], theta_dot])

        # Transition model = deterministic + noise
        noisy_particles = pred_state + np.random.normal(0, process_noise_std, size=(self.N, 4))
        self.particles = noisy_particles

    def update(self, measurement_pred, obs_noise_std=0.0005):
        # Weight particles by closeness to predicted measurement
        diffs = self.particles - measurement_pred
        likelihoods = np.exp(-0.5 * np.sum((diffs / obs_noise_std)**2, axis=1))
        likelihoods += 1e-12  # prevent division by zero
        self.weights = likelihoods / np.sum(likelihoods)

    def resample(self):
        indices = np.random.choice(self.N, size=self.N, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.N) / self.N

    def estimate(self, return_std=False):
        mean = np.average(self.particles, axis=0, weights=self.weights)
        if return_std:
            std = np.sqrt(np.average((self.particles - mean)**2, axis=0, weights=self.weights))
            return mean, std
        return mean
