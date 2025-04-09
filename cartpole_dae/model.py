import torch
import torch.nn as nn

class FrameEncoder(nn.Module):
    def __init__(self):
        super(FrameEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
        )
    
    def forward(self, x):
        b, seq_len, c, h, w = x.shape
        x = x.view(b * seq_len, c, h, w)
        features = self.cnn(x)
        return features.view(b, seq_len, -1)

class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SequenceEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
    
    def forward(self, x):
        _, h_n = self.gru(x)
        return h_n.squeeze(0)

class StatePredictor(nn.Module):
    def __init__(self, hidden_dim, output_dim=4):
        super(StatePredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, h):
        return self.fc(h)

class CartPoleDynamicsModel(nn.Module):
    def __init__(self):
        super(CartPoleDynamicsModel, self).__init__()
        self.frame_encoder = FrameEncoder()
        self.hidden_dim = 128
        self.sequence_encoder = SequenceEncoder(input_dim=128 + 1, hidden_dim=self.hidden_dim)
        self.state_predictor = StatePredictor(hidden_dim=self.hidden_dim)
    
    def forward(self, images, actions):
        img_features = self.frame_encoder(images)
        x = torch.cat([img_features, actions], dim=-1)
        hidden = self.sequence_encoder(x)
        state_pred = self.state_predictor(hidden)
        return state_pred
