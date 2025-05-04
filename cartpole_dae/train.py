import os
import torch
from torch.utils.data import DataLoader, random_split
from dataset import CleanCartPoleDataset
from dataset import analyze_state_distribution
from model import CartPoleDynamicsModel
import matplotlib.pyplot as plt
import json

def create_exp_folder(base_dir='experiments'):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith('exp')]
    exp_nums = [int(d.replace('exp', '')) for d in existing if d.replace('exp', '').isdigit()]
    next_exp = max(exp_nums + [0]) + 1
    exp_path = os.path.join(base_dir, f'exp{next_exp}')
    os.makedirs(exp_path)
    return exp_path

# --- Configs ---
epochs = 100
batch_size = 32
patience = 10  # for early stopping
lr = 1e-3

exp_dir=create_exp_folder()

print(f"Experiment directory created: {exp_dir}")

config = {
    "epochs": epochs,
    "batch_size": batch_size,
    "patience": patience,
    "learning_rate": lr,
    "dataset_size": 200000,
    "seq_length": 4,
    "image_size": 64
}
with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)

# --- Dataset ---
dataset = CleanCartPoleDataset(
    seq_length=4,
    dataset_size=200000,
    image_size=64,
    cache_file='cartpole_sequences_200k.npz'
)

analyze_state_distribution(dataset) 




train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# --- Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CartPoleDynamicsModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,  # or whatever you set as learning rate
    steps_per_epoch=len(train_loader),
    epochs=epochs,
    pct_start=0.3,  # warm-up for 30% of total steps
    anneal_strategy='cos'  # cosine decay (default)
)

# --- Trackers ---
train_losses = []
val_losses = []
best_val_loss = float('inf')
epochs_no_improve = 0

# --- Training Loop ---
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, actions, next_state in train_loader:
        images, actions, next_state = images.to(device), actions.to(device), next_state.to(device)

        pred_state = model(images, actions)
        loss = loss_fn(pred_state, next_state)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # <-- this is where you call it

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, actions, next_state in val_loader:
            images, actions, next_state = images.to(device), actions.to(device), next_state.to(device)
            pred_state = model(images, actions)
            val_loss += loss_fn(pred_state, next_state).item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    # Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
        print("✅ Saved new best model.")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"⚠️  No improvement for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= patience:
        print(f"🛑 Early stopping after {epoch+1} epochs.")
        break

# --- Plot Losses ---
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(exp_dir, 'loss_plot.png'))
# plt.show()

print("📉 Training complete. Best model saved to 'best_model.pth'")



