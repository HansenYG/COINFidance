import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn

from backend.models.coin_analyze_model import ScamCoinClassifier

numeric_dim = 10 
binary_dim = 5      
text_dim = 128
hidden_dim = 64

model = ScamCoinClassifier(
    numeric_dim=numeric_dim,
    binary_dim=binary_dim,
    text_dim=text_dim,
    hidden_dim=hidden_dim
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    batch_size = 10
    
    numeric_features = torch.randn(batch_size, numeric_dim)      
    binary_features = torch.randint(0, 2, (batch_size, binary_dim)).float()  
    text_embeddings = torch.randn(batch_size, text_dim)          
    labels = torch.randint(0, 2, (batch_size,))                  

    optimizer.zero_grad()
    outputs = model(numeric_features, binary_features, text_embeddings)  # Pass all 3 inputs
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0: 
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save model
save_path = os.path.join(project_root, "backend/saved_models/coin_classifier.pth")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")