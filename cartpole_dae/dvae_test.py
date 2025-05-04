# import torch
# import matplotlib.pyplot as plt
# from dvae import SeqDVAE
# from dvae_dataset import DVAEDataset

# # Config
# latent_dim = 8
# action_dim = 1
# hidden_dim = 64
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load trained model
# model = SeqDVAE(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
# model.load_state_dict(torch.load("seq_dvae_experiments/exp4/best_model.pth"))
# model.eval()
# print("✅ Loaded trained SeqDVAE model.")

# # Load one sample
# dataset = DVAEDataset()
# x_seq, a_seq, true_states = dataset[45]  # x_seq: [9, 3, 64, 64], a_seq: [9, 1]

# # Format inputs
# x_input = x_seq[:-1].unsqueeze(0).to(device)        # [1, 8, 3, 64, 64]
# a_input = a_seq[:-1].unsqueeze(0).float().to(device) # [1, 8, 1]
# x_target = x_seq[-1].unsqueeze(0).to(device)        # [1, 3, 64, 64]

# # Inference
# with torch.no_grad():
#     x_pred, mu_0, logvar_0, mu_next, logvar_next = model(x_input, a_input)

# # Visualization helper
# def show_frame(tensor_img, title, subplot_idx, edge_color="black"):
#     img = tensor_img.squeeze().cpu().permute(1, 2, 0).numpy()
#     plt.subplot(1, 3, subplot_idx)
#     plt.imshow(img)
#     plt.title(title, color=edge_color)
#     plt.axis('off')
#     ax = plt.gca()
#     for spine in ax.spines.values():
#         spine.set_edgecolor(edge_color)
#         spine.set_linewidth(3)

# # Plot frame 7 (last input), true frame 8, predicted frame 8
# plt.figure(figsize=(12, 4))
# show_frame(x_seq[-2], "Last Input Frame (x₇)", 1, "blue")
# show_frame(x_target[0], "True Frame (x₈)", 2, "green")
# show_frame(x_pred[0], "Predicted Frame (x̂₈)", 3, "red")
# plt.tight_layout()
# plt.savefig("dvae_predict_frame8.png")
# plt.close()


import torch
import matplotlib.pyplot as plt
from dvae import SeqDVAE
from dvae_dataset import DVAEDataset

# ---------- Configuration ----------
latent_dim = 8
action_dim = 1
hidden_dim = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "dvae_experiments/exp1/best_model.pth"

# ---------- Load Model ----------
model = SeqDVAE(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print("✅ Loaded trained SeqDVAE model.")

# ---------- Load Dataset ----------
dataset = DVAEDataset()
x_seq_full, a_seq_full, _ = dataset[45]  # x_seq_full: [8, 3, 64, 64], a_seq_full: [8, 1]

# ---------- Split into Input and Target ----------
x_input = x_seq_full[:-1].unsqueeze(0).to(device)        # [1, 7, 3, 64, 64]
a_input = a_seq_full[:-1].unsqueeze(0).float().to(device) # [1, 7, 1]
x_target = x_seq_full[-1].unsqueeze(0).to(device)         # [1, 3, 64, 64]

# ---------- Predict Final Frame ----------
with torch.no_grad():
    x_pred, mu_0, logvar_0, mu_next, logvar_next = model(x_input, a_input)

# ---------- Visualization ----------
def show_sequence_and_prediction(x_input, x_pred, x_target, save_path="dvae_sequence_pred.png"):
    """
    Show all input frames + true 8th frame + predicted 8th frame
    """
    x_input_np = x_input.squeeze(0).cpu().numpy()  # [T, 3, 64, 64]
    num_frames = x_input_np.shape[0]
    x_pred_np = x_pred.squeeze(0).cpu().permute(1, 2, 0).numpy()
    x_target_np = x_target.squeeze(0).cpu().permute(1, 2, 0).numpy()

    plt.figure(figsize=(14, 6))

    # Input frames
    for i in range(num_frames):
        frame = x_input_np[i].transpose(1, 2, 0)
        plt.subplot(2, num_frames + 1, i + 1)
        plt.imshow(frame)
        plt.title(f"Input Frame {i}")
        plt.axis("off")

    # Predicted frame
    plt.subplot(2, num_frames + 1, num_frames + 1)
    plt.imshow(x_pred_np)
    plt.title("Predicted Frame")
    plt.axis("off")

    # True frame
    plt.subplot(2, num_frames + 1, num_frames + 2)
    plt.imshow(x_target_np)
    plt.title("True Frame")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📷 Saved visualization to {save_path}")

# ---------- Run Visualization ----------
show_sequence_and_prediction(x_input, x_pred, x_target, save_path="dvae_sequence_pred.png")
