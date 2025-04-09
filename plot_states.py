import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Initialize the environment
env = gym.make('CartPole-v1')
observation, _ = env.reset()

# Lists to store states, actions, and time steps
states = []
actions = []
timesteps = []

# Run a single episode
done = False
while not done:
    # Append the current state and time step
    states.append(observation)
    timesteps.append(len(timesteps))  # Time steps start from 0
    
    # Take a random action
    action = env.action_space.sample()
    actions.append(action)  # Store the action
    
    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Check if the episode is done
    done = terminated or truncated

# Close the environment
env.close()

# Convert states and actions to NumPy arrays for easier manipulation
states = np.array(states)
actions = np.array(actions)

# Plot each state variable and actions over time
plt.figure(figsize=(12, 10))

# Cart Position
plt.subplot(3, 2, 1)
plt.plot(timesteps, states[:, 0], label='Cart Position')
plt.xlabel('Time Step')
plt.ylabel('Cart Position')
plt.legend()

# Cart Velocity
plt.subplot(3, 2, 2)
plt.plot(timesteps, states[:, 1], label='Cart Velocity', color='orange')
plt.xlabel('Time Step')
plt.ylabel('Cart Velocity')
plt.legend()

# Pole Angle
plt.subplot(3, 2, 3)
plt.plot(timesteps, states[:, 2], label='Pole Angle', color='green')
plt.xlabel('Time Step')
plt.ylabel('Pole Angle')
plt.legend()

# Pole Angular Velocity
plt.subplot(3, 2, 4)
plt.plot(timesteps, states[:, 3], label='Pole Angular Velocity', color='red')
plt.xlabel('Time Step')
plt.ylabel('Pole Angular Velocity')
plt.legend()

# Actions
plt.subplot(3, 2, 5)
plt.step(timesteps, actions, label='Action', color='purple', where='post')
plt.xlabel('Time Step')
plt.ylabel('Action')
plt.yticks([0, 1], ['Left (0)', 'Right (1)'])  # Label actions
plt.legend()

# Save the plot to a file
plt.tight_layout()
plt.savefig('cartpole_states_and_actions.png')  # Save the plot as a PNG file
print("Plot saved as 'cartpole_states_and_actions.png'")