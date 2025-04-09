import torch
import numpy as np
import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
from model import CartPoleDynamicsModel

# ==== Parameters ====
SEQ_LENGTH = 4
IMAGE_SIZE = 64
MODEL_PATH = 'cartpole_dynamics_model.pth'
WINDOW_NAME = "CartPole Evaluation"

# ==== Setup environment ====
env = gym.make('CartPole-v1', render_mode='rgb_array')
obs, _ = env.reset()

# ==== Load model ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CartPoleDynamicsModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

print(f"Model loaded from '{MODEL_PATH}'")

# ==== Buffers ====
frames_buffer = []
actions_buffer = []

log_true = []
log_pred = []

# ==== Frame preprocessing ====
def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    frame = frame.transpose(2, 0, 1)  # channels-first
    frame = frame.astype(np.float32) / 255.0
    return frame

# ==== Initialize buffers ====
for _ in range(SEQ_LENGTH):
    frame = env.render()
    frames_buffer.append(preprocess_frame(frame))
    action = env.action_space.sample()
    actions_buffer.append([action])
    obs, _, done, _, _ = env.step(action)
    if done:
        obs, _ = env.reset()

# ==== Evaluation loop ====
MAX_STEPS = 100
step = 0

cv2.namedWindow(WINDOW_NAME)

while step < MAX_STEPS:
    # Prepare inputs
    images_input = torch.tensor(np.array([frames_buffer]), dtype=torch.float32).to(device)
    actions_input = torch.tensor(np.array([actions_buffer]), dtype=torch.float32).to(device)

    # Predict next state
    with torch.no_grad():
        pred_state = model(images_input, actions_input).cpu().numpy()[0]

    # Log states
    log_true.append(obs)
    log_pred.append(pred_state)

    # Render frame
    frame = env.render()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Prepare text overlays
    true_state_text = f"True: pos={obs[0]:.2f}, vel={obs[1]:.2f}, angle={obs[2]:.2f}, ang_vel={obs[3]:.2f}"
    pred_state_text = f"Pred: pos={pred_state[0]:.2f}, vel={pred_state[1]:.2f}, angle={pred_state[2]:.2f}, ang_vel={pred_state[3]:.2f}"
    step_text = f"Step: {step}"

    # Overlay text on frame
    cv2.putText(frame_bgr, step_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, true_state_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, pred_state_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Show frame
    cv2.imshow(WINDOW_NAME, frame_bgr)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        print("Evaluation manually stopped.")
        break

    # Take random action for next step
    action = env.action_space.sample()
    actions_buffer.pop(0)
    actions_buffer.append([action])

    frames_buffer.pop(0)
    frames_buffer.append(preprocess_frame(frame))

    # Step environment
    obs, _, done, _, _ = env.step(action)
    step += 1

    if done:
        obs, _ = env.reset()

# ==== Cleanup ====
cv2.destroyAllWindows()
env.close()

print("Evaluation complete.")

# ==== Log to numpy arrays ====
log_true = np.array(log_true)
log_pred = np.array(log_pred)

# ==== Plot true vs predicted states ====
labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Angular Velocity']
plt.figure(figsize=(12, 8))

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(log_true[:, i], label='True')
    plt.plot(log_pred[:, i], label='Predicted', linestyle='--')
    plt.title(labels[i])
    plt.legend()

plt.tight_layout()
plt.savefig('evaluation_results.png')
plt.close()

print("Evaluation results saved to 'evaluation_results.png'.")
