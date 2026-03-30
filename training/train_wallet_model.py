import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn

from backend.models.wallet_analyze_model import WalletClassifier

static_dim = 10 
seq_dim = 5      
hidden_dim = 128
gru_hidden = 64

model = WalletClassifier(
    static_dim=static_dim,
    seq_dim=seq_dim,
    hidden_dim=hidden_dim,
    gru_hidden=gru_hidden
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    batch_size = 10
    seq_length = 20 
    
    static_features = torch.randn(batch_size, static_dim)  #
    seq_features = torch.randn(batch_size, seq_length, seq_dim)  
    labels = torch.randint(0, 2, (batch_size,))  

    optimizer.zero_grad()
    outputs = model(static_features, seq_features)  
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0: 
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save model
save_path = os.path.join(project_root, "backend/saved_models/wallet_classifier.pth")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
