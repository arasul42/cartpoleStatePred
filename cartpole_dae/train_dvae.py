import os
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from dvae_dataset import DVAEDataset, analyze_state_distribution
from dvae import SeqDVAE

# --- Configuration ---
seq_length = 8
epochs = 50
batch_size = 32
patience = 10
lr = 1e-4
latent_dim = 8
action_dim = 1
hidden_dim = 64

def create_exp_folder(base_dir='seq_dvae_experiments'):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith('exp')]
    exp_nums = [int(d.replace('exp', '')) for d in existing if d.replace('exp', '').isdigit()]
    next_exp = max(exp_nums + [0]) + 1
    exp_path = os.path.join(base_dir, f'exp{next_exp}')
    os.makedirs(exp_path)
    return exp_path

exp_dir = create_exp_folder()
print(f"📁 Logging to: {exp_dir}")

config = {
    "seq_length": seq_length,
    "epochs": epochs,
    "batch_size": batch_size,
    "patience": patience,
    "learning_rate": lr,
    "latent_dim": latent_dim,
    "hidden_dim": hidden_dim,
}
with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)

# --- Load Dataset ---
dataset = DVAEDataset(seq_length=seq_length, dataset_size=100000)
analyze_state_distribution(dataset)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# --- Model and Optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SeqDVAE(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# --- Training Loop ---
train_losses = []
val_losses = []
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for x_seq, a_seq, next_state in train_loader:
        x_input = x_seq[:, :-1].to(device)           # [B, 8, 3, 64, 64]
        a_input = a_seq[:, :-1].to(device).float()   # [B, 8, 1]
        x_target = x_seq[:, -1].to(device)           # [B, 3, 64, 64]

        x_pred, mu_0, logvar_0, mu_next, logvar_next = model(x_input, a_input)
        loss = model.compute_loss(x_target, x_pred, mu_0, logvar_0, mu_next, logvar_next)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- Validation ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x_seq, a_seq, next_state in val_loader:
            x_input = x_seq[:, :-1].to(device)
            a_input = a_seq[:, :-1].to(device).float()
            x_target = x_seq[:, -1].to(device)

            x_pred, mu_0, logvar_0, mu_next, logvar_next = model(x_input, a_input)
            val_loss = model.compute_loss(x_target, x_pred, mu_0, logvar_0, mu_next, logvar_next)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
        print("✅ New best model saved.")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= patience:
        print("🛑 Early stopping.")
        break

# --- Plot Loss ---
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("SeqDVAE Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(exp_dir, "loss_plot.png"))
print("📉 Training complete.")
