import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqDVAE(nn.Module):
    def __init__(self, latent_dim=8, action_dim=1, hidden_dim=64):
        super(SeqDVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Encoder CNN
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # (64, 16, 16)
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(64 * 16 * 16, latent_dim)

        # GRU for sequence modeling
        self.gru = nn.GRU(latent_dim + action_dim, hidden_dim, batch_first=True)

        # Latent prediction
        self.fc_gru_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_gru_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder from latent to image
        self.fc_decoder = nn.Linear(latent_dim, 64 * 16 * 16)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode_frame(self, x):
        h = self.encoder_cnn(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decoder(z).view(-1, 64, 16, 16)
        return self.decoder_deconv(h)

    def forward(self, x_seq, a_seq):
        """
        x_seq: [B, T, 3, 64, 64]
        a_seq: [B, T, 1]
        Returns:
            x̂_{t+1}, mu_z_0, logvar_z_0
        """
        B, T, C, H, W = x_seq.shape

        # Flatten sequence dimension
        x_seq_flat = x_seq.view(B * T, C, H, W)
        mu_flat, logvar_flat = self.encode_frame(x_seq_flat)
        z_flat = self.reparameterize(mu_flat, logvar_flat)

        # Reshape back to sequence
        z_seq = z_flat.view(B, T, -1)
        z_mu_seq = mu_flat.view(B, T, -1)
        z_logvar_seq = logvar_flat.view(B, T, -1)

        # Concatenate z_t and a_t
        gru_input = torch.cat([z_seq, a_seq], dim=-1)  # [B, T, L+A]

        # GRU sequence modeling
        _, h_T = self.gru(gru_input)  # h_T: [1, B, H]
        h_T = h_T.squeeze(0)          # [B, H]

        # Predict next latent z_{t+1}
        mu_next = self.fc_gru_mu(h_T)
        logvar_next = self.fc_gru_logvar(h_T)
        z_next = self.reparameterize(mu_next, logvar_next)

        # Decode to image
        x_pred = self.decode(z_next)

        # Return prediction and first-step encoding for KL loss
        return x_pred, z_mu_seq[:, 0], z_logvar_seq[:, 0], mu_next, logvar_next

    def compute_loss(self, x_target, x_pred, mu_0, logvar_0, mu_next, logvar_next, beta=1.0):
        recon_loss = F.mse_loss(x_pred, x_target, reduction='sum')
        kl_z0 = -0.5 * torch.sum(1 + logvar_0 - mu_0.pow(2) - logvar_0.exp())
        kl_zT = -0.5 * torch.sum(1 + logvar_next - mu_next.pow(2) - logvar_next.exp())
        return recon_loss + beta * (kl_z0 + kl_zT)
