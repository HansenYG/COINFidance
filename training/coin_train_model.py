"""
Train the ScamCoinClassifier on realistic synthetic data.

The synthetic generator builds two distinguishable populations:
  - Legitimate tokens: higher liquidity, broader holder distribution,
    older creation, audited contracts, neutral sentiment text embeddings.
  - Scam tokens: low / locked liquidity, concentrated top-holder %, brand
    new, anonymous deployer, hype-y / suspicious text embeddings.

This gives the network real signal to learn from, instead of pure noise.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from backend.models.coin_analyze_model import ScamCoinClassifier

# ── Hyperparameters ──────────────────────────────────────────────────────────
NUMERIC_DIM = 10
BINARY_DIM = 5
TEXT_DIM = 128
HIDDEN_DIM = 64

NUM_SAMPLES = 8000
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
VAL_SPLIT = 0.2
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Synthetic data generator ─────────────────────────────────────────────────
def generate_coin_dataset(n_samples: int):
    """
    Numeric features (10): [liquidity_usd, market_cap, holder_count,
        top10_holder_pct, age_days, tx_count_24h, volume_24h,
        price_volatility, dev_wallet_pct, contract_size_kb]
    Binary features (5): [contract_verified, has_audit, ownership_renounced,
        liquidity_locked, has_website]
    Text embeddings (128): pretend BERT-like vector summarising name +
        description sentiment.
    """
    n_legit = n_samples // 2
    n_scam = n_samples - n_legit

    # ---- Legitimate distribution ----
    legit_numeric = np.column_stack([
        np.random.lognormal(mean=14, sigma=1.0, size=n_legit),     # liquidity
        np.random.lognormal(mean=16, sigma=1.2, size=n_legit),     # market cap
        np.random.lognormal(mean=8, sigma=0.7, size=n_legit),      # holders
        np.random.beta(2, 8, size=n_legit) * 100,                  # top10 %
        np.random.uniform(180, 2000, size=n_legit),                # age days
        np.random.lognormal(mean=6, sigma=1.0, size=n_legit),      # tx count
        np.random.lognormal(mean=12, sigma=1.0, size=n_legit),     # volume
        np.random.uniform(0.01, 0.15, size=n_legit),               # volatility
        np.random.beta(2, 20, size=n_legit) * 100,                 # dev wallet %
        np.random.uniform(5, 40, size=n_legit),                    # contract size
    ])
    legit_binary = np.column_stack([
        (np.random.rand(n_legit) < 0.95).astype(np.float32),  # verified
        (np.random.rand(n_legit) < 0.65).astype(np.float32),  # audit
        (np.random.rand(n_legit) < 0.55).astype(np.float32),  # renounced
        (np.random.rand(n_legit) < 0.85).astype(np.float32),  # locked liq
        (np.random.rand(n_legit) < 0.90).astype(np.float32),  # website
    ])
    legit_text = np.random.normal(loc=0.2, scale=0.8, size=(n_legit, TEXT_DIM))

    # ---- Scam distribution ----
    scam_numeric = np.column_stack([
        np.random.lognormal(mean=8, sigma=1.5, size=n_scam),       # low liquidity
        np.random.lognormal(mean=10, sigma=1.5, size=n_scam),      # low cap
        np.random.lognormal(mean=4, sigma=1.0, size=n_scam),       # few holders
        np.random.beta(8, 2, size=n_scam) * 100,                   # high top10 %
        np.random.uniform(0, 30, size=n_scam),                     # very new
        np.random.lognormal(mean=3, sigma=1.5, size=n_scam),       # tx count
        np.random.lognormal(mean=7, sigma=1.5, size=n_scam),       # low volume
        np.random.uniform(0.20, 1.50, size=n_scam),                # high volatility
        np.random.beta(8, 3, size=n_scam) * 100,                   # high dev %
        np.random.uniform(2, 15, size=n_scam),                     # contract size
    ])
    scam_binary = np.column_stack([
        (np.random.rand(n_scam) < 0.40).astype(np.float32),
        (np.random.rand(n_scam) < 0.05).astype(np.float32),
        (np.random.rand(n_scam) < 0.20).astype(np.float32),
        (np.random.rand(n_scam) < 0.25).astype(np.float32),
        (np.random.rand(n_scam) < 0.35).astype(np.float32),
    ])
    scam_text = np.random.normal(loc=-0.3, scale=1.1, size=(n_scam, TEXT_DIM))

    # Combine
    numeric = np.vstack([legit_numeric, scam_numeric]).astype(np.float32)
    binary = np.vstack([legit_binary, scam_binary]).astype(np.float32)
    text = np.vstack([legit_text, scam_text]).astype(np.float32)
    labels = np.concatenate([
        np.zeros(n_legit, dtype=np.int64),
        np.ones(n_scam, dtype=np.int64),
    ])

    # Flip 10% of binary flags to simulate noisy / partial info.
    binary_flip = (np.random.rand(*binary.shape) < 0.10).astype(np.float32)
    binary = np.abs(binary - binary_flip).astype(np.float32)

    # Normalize numeric features (log-scale large columns first)
    log_cols = [0, 1, 2, 5, 6]
    numeric[:, log_cols] = np.log1p(np.maximum(numeric[:, log_cols], 0))
    mean = numeric.mean(axis=0, keepdims=True)
    std = numeric.std(axis=0, keepdims=True) + 1e-6
    numeric = (numeric - mean) / std

    # Add Gaussian overlap noise on normalized space (numeric & text) so the
    # two populations aren't trivially separable -- forces real learning.
    numeric += np.random.normal(0, 0.5, numeric.shape).astype(np.float32)
    text += np.random.normal(0, 0.6, text.shape).astype(np.float32)

    # Flip 5% of labels (mislabelled / undiscovered scams in real data).
    flip_mask = np.random.rand(n_samples) < 0.05
    labels[flip_mask] = 1 - labels[flip_mask]

    # Shuffle
    idx = np.random.permutation(n_samples)
    return (
        torch.from_numpy(numeric[idx]),
        torch.from_numpy(binary[idx]),
        torch.from_numpy(text[idx]),
        torch.from_numpy(labels[idx]),
    )


# ── Training loop ────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for num, bn, txt, lbl in loader:
            num, bn, txt, lbl = num.to(device), bn.to(device), txt.to(device), lbl.to(device)
            out = model(num, bn, txt)
            loss = criterion(out, lbl)
            loss_total += loss.item() * lbl.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == lbl).sum().item()
            total += lbl.size(0)
    return loss_total / total, correct / total


def main():
    print(f"Device: {device}")
    print("Generating synthetic coin dataset...")
    numeric, binary, text, labels = generate_coin_dataset(NUM_SAMPLES)

    dataset = TensorDataset(numeric, binary, text, labels)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = ScamCoinClassifier(
        numeric_dim=NUMERIC_DIM,
        binary_dim=BINARY_DIM,
        text_dim=TEXT_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    save_path = os.path.join(project_root, "backend/saved_models/coin_classifier.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for num, bn, txt, lbl in train_loader:
            num, bn, txt, lbl = num.to(device), bn.to(device), txt.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(num, bn, txt)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * lbl.size(0)
            running_correct += (out.argmax(dim=1) == lbl).sum().item()
            running_total += lbl.size(0)

        scheduler.step()
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        val_loss, val_acc = evaluate(model, val_loader)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to {save_path}")


if __name__ == "__main__":
    main()
