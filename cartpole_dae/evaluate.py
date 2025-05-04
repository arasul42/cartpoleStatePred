import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import CleanCartPoleDataset
from model import CartPoleDynamicsModel
import json
import numpy as np

# Load dataset and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = CleanCartPoleDataset(seq_length=4, dataset_size=500, image_size=64)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

model = CartPoleDynamicsModel().to(device)
model.load_state_dict(torch.load('cartpole_dynamics_model.pth'))
model.eval()

# Evaluate on one batch
images, actions, next_state = next(iter(dataloader))
images, actions = images.to(device), actions.to(device)

with torch.no_grad():
    pred_state = model(images, actions).cpu().numpy()

true_state = next_state.numpy()
time = range(pred_state.shape[0])

# Calculate errors
mse = np.mean((pred_state - true_state) ** 2, axis=0)
rmse = np.sqrt(mse)

# Save errors to JSON
error_dict = {
    'MSE': {
        'Cart Position': float(mse[0]),
        'Cart Velocity': float(mse[1]),
        'Pole Angle': float(mse[2]),
        'Pole Velocity At Tip': float(mse[3])
    },
    'RMSE': {
        'Cart Position': float(rmse[0]),
        'Cart Velocity': float(rmse[1]),
        'Pole Angle': float(rmse[2]),
        'Pole Velocity At Tip': float(rmse[3])
    }
}

with open('prediction_errors.json', 'w') as f:
    json.dump(error_dict, f, indent=4)

# Plot true vs predicted states with RMSE
labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity At Tip']
plt.figure(figsize=(9.5, 6.8))

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(time, true_state[:, i], label='True')
    plt.plot(time, pred_state[:, i], label='Predicted', linestyle='--')
    plt.title(f"{labels[i]}\nRMSE = {rmse[i]:.4f}")
    plt.legend()

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=300)

print("✅ Plot saved as 'evaluation_results.png'")
print("✅ Prediction errors saved to 'prediction_errors.json'")
