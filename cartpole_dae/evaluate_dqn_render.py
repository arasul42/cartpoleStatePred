# evaluate_dqn_render.py

import cv2
import torch
import numpy as np
from stable_baselines3 import DQN
from dqn_learned_env import DQNCartPoleFromPredictionEnv
from model import CartPoleDynamicsModel

def evaluate_and_render(model_path, dynamics_model_path, device, save_video=False, video_path="rendered_eval.avi"):
    # Load dynamics model
    model = CartPoleDynamicsModel().to(device)
    model.load_state_dict(torch.load(dynamics_model_path))
    model.eval()

    # Create environment
    env = DQNCartPoleFromPredictionEnv(model, device)
    dqn = DQN.load(model_path)

    obs, _ = env.reset()

    if save_video:
        frame = env.render()
        h, w, _ = frame.shape
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (w, h))
    else:
        writer = None

    for step in range(500):
        action, _ = dqn.predict(obs)
        obs, reward, done, _, _ = env.step(action)

        frame = env.render()
        cv2.imshow("CartPole Evaluation (Prediction)", frame)
        if writer:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) == ord('q') or done:
            break

    env.close()
    cv2.destroyAllWindows()
    if writer:
        writer.release()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate_and_render(
        model_path="dqn_pred_model.zip",
        dynamics_model_path="experiments/exp16/best_model.pth",
        device=device,
        save_video=True
    )
