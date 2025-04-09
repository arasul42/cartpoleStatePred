import gymnasium as gym
from frame_saver import FrameSaver

env = gym.make("CartPole-v1", render_mode="rgb_array")
saver = FrameSaver(save_dir="./frames", annotate=True)


for episode in range(10):
    print(f"Episode {episode + 1}")
    observation, info = env.reset()
    done = False
    frame_count = 0

    while not done:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(f"State: {observation}, Action: {action}, Reward: {reward}")

        if episode == 9:
            frame = env.render()
            saver.save_frame(frame, episode, frame_count, state=observation, action=action)
            frame_count += 1

env.close()
