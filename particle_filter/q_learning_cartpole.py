import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Hyperparameters
EPISODES = 100  # Number of episodes
BATCH_SIZE = 64  # Batch size for training
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_END = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Decay rate for exploration
LEARNING_RATE = 0.001  # Learning rate for the optimizer
MEMORY_SIZE = 10000  # Replay memory size

# Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Q-learning agent
class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Random action
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
            return torch.argmax(q_values).item()  # Greedy action

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # Sample a batch from memory
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Compute Q-values for current states
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # Compute Q-values for next states
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * GAMMA * next_q_values

        # Compute loss and update the model
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

# Initialize environment and agent
# Initialize environment and agent
env = gym.make("CartPole-v1", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = QLearningAgent(state_size, action_size)

best_reward = float('-inf')  # Track the best reward

# Training loop
for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Select action
        action = agent.act(state)

        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store experience in memory
        agent.memory.push(state, action, reward, next_state, done)

        # Train the agent
        agent.train()

        # Update state and total reward
        state = next_state
        total_reward += reward

    # Save the best model
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(agent.model.state_dict(), "best_cartpole_model.pth")
        print(f"New best model saved with reward: {best_reward}")

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

print("Training finished. Best model saved to best_cartpole_model.pth.")

# Close the environment
env.close()

# Load the best model for future use
agent.model.load_state_dict(torch.load("best_cartpole_model.pth"))
agent.model.eval()  # Set the model to evaluation mode
print("Best model loaded from best_cartpole_model.pth")
