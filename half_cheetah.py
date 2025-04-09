import gymnasium as gym

env = gym.make("Humanoid-v4", render_mode="human")

for episode in range(10):
    print(f"Episode {episode + 1}")
    observation, info = env.reset()

    episode_over = False

    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated


env.close()

