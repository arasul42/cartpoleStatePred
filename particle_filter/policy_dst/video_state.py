import gymnasium as gym  # Use gymnasium instead of gym
import random
import numpy as np
import cv2

class DQN_CNN_Agent:
    def __init__(self, env_name):
        self.env_name = env_name       
        self.env = gym.make(env_name, render_mode="rgb_array")  # Specify render mode
        self.ROWS = 160
        self.COLS = 240
        self.REM_STEP = 4

        self.EPISODES = 10

        self.image_memory = np.zeros((self.REM_STEP, self.ROWS, self.COLS))

    def imshow(self, image, rem_step=0):
        """Display preprocessed image"""
        cv2.imshow(self.env_name + str(rem_step), image[rem_step,...])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()

    def GetImage(self):
        """Capture and preprocess the environment image"""
        img = self.env.render()
        if img is None:
            raise RuntimeError("Failed to render environment. Ensure render_mode='rgb_array'.")

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        img_resized = cv2.resize(img_gray, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        img_resized[img_resized < 255] = 0  # Binarize
        img_resized = img_resized / 255.0   # Normalize

        # Update image memory (shifting frames)
        self.image_memory = np.roll(self.image_memory, shift=1, axis=0)
        self.image_memory[0,:,:] = img_resized

        self.imshow(self.image_memory, 0)
        
        return np.expand_dims(self.image_memory, axis=0)  # Add batch dimension
    
    def reset(self):
        """Reset environment and return initial state"""
        obs, _ = self.env.reset()
        for _ in range(self.REM_STEP):
            state = self.GetImage()
        return state

    def step(self, action):
        """Execute action and return next state"""
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated  
        next_state = self.GetImage()
        return next_state, reward, done

    def run(self):
        """Run agent for EPISODES"""
        for episode in range(self.EPISODES):
            state = self.reset()
            for t in range(500):               
                action = self.env.action_space.sample()
                next_state, reward, done = self.step(action)
                if done:
                    break

if __name__ == "__main__":
    env_name = 'CartPole-v1'
    agent = DQN_CNN_Agent(env_name)
    agent.run()
