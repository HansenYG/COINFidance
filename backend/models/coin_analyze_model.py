import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return x + self.block(x)


class ScamCoinClassifier(nn.Module):
    def __init__(
        self,
        numeric_dim,      
        binary_dim,       
        text_dim,         
        hidden_dim=128
    ):
        super().__init__()

        # Branch 1: numeric features
        self.numeric_branch = nn.Sequential(
            nn.Linear(numeric_dim, hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim)
        )

        # Branch 2: binary flags
        self.binary_branch = nn.Sequential(
            nn.Linear(binary_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Branch 3: textual embeddings
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim)
        )

        # Combined classifier head
        combined_dim = hidden_dim * 2 + hidden_dim // 2
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # scam / not scam
        )

    def forward(self, numeric, binary, text_embed):
        n = self.numeric_branch(numeric)
        b = self.binary_branch(binary)
        t = self.text_branch(text_embed)

        combined = torch.cat([n, b, t], dim=-1)
        return self.classifier(combined)