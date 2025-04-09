import torch
from torch.utils.data import DataLoader
from dataset import CleanCartPoleDataset
from model import CartPoleDynamicsModel

# Prepare dataset and dataloader
dataset = CleanCartPoleDataset(seq_length=4, dataset_size=200000, image_size=64)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Prepare model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CartPoleDynamicsModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, actions, next_state in dataloader:
        images, actions, next_state = images.to(device), actions.to(device), next_state.to(device)
        
        pred_state = model(images, actions)
        loss = loss_fn(pred_state, next_state)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}')

# Save the trained model
torch.save(model.state_dict(), 'cartpole_dynamics_model.pth')
print("Model saved to cartpole_dynamics_model.pth")
