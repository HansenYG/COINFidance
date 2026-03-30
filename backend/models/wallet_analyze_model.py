import torch
import torch.nn as nn

class ResidualMLP(nn.Module):
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


class WalletClassifier(nn.Module):
    def __init__(
        self,
        static_dim,       # static wallet features
        seq_dim,          # per‑transaction feature dimension
        hidden_dim=128,
        gru_hidden=64
    ):
        super().__init__()

        # Branch 1: static features (e.g., wallet age, cluster risk, flags)
        self.static_branch = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim)
        )

        # Branch 2: sequence features (transaction history)
        self.gru = nn.GRU(
            input_size=seq_dim,
            hidden_size=gru_hidden,
            batch_first=True
        )

        self.seq_projection = nn.Sequential(
            nn.Linear(gru_hidden, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim)
        )

        # Combined classifier
        combined_dim = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # suspicious / normal
        )

    def forward(self, static_features, seq_features):
        # static branch
        s = self.static_branch(static_features)

        # sequence branch
        _, h = self.gru(seq_features)   # h: (1, batch, gru_hidden)
        h = h.squeeze(0)
        t = self.seq_projection(h)

        # combine
        combined = torch.cat([s, t], dim=-1)
        return self.classifier(combined)