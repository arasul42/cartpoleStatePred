import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import CleanCartPoleDataset
from model import CartPoleDynamicsModel

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

# Plot true vs predicted states
labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity At Tip']
plt.figure(figsize=(12, 8))

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(time, true_state[:, i], label='True')
    plt.plot(time, pred_state[:, i], label='Predicted', linestyle='--')
    plt.title(labels[i])
    plt.legend()

plt.tight_layout()
plt.savefig('evaluation_results.png')
