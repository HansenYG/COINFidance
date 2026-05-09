"""
Train the WalletClassifier on realistic synthetic data.

Static features (10):
  [num_txs, total_value, avg_value, failed_txs, fail_ratio,
   unique_receivers, avg_gas_price, std_gas_price,
   avg_time_delta, min_time_delta]
Sequence features per tx (5):
  [value, gas_price, gas_used, is_error, time_stamp]

Two distinguishable populations are generated:
  - Normal wallets: mature usage, many unique receivers, low fail ratio,
    long & varied inter-tx time deltas, moderate gas prices.
  - Suspicious wallets: bot-like burst patterns, very small inter-tx
    delays, repetitive receivers, higher fail ratio, abnormal gas spikes.

These match the feature extraction logic in backend/app.py:extract_wallet_features.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from backend.models.wallet_analyze_model import WalletClassifier

# ── Hyperparameters ──────────────────────────────────────────────────────────
STATIC_DIM = 10
SEQ_DIM = 5
HIDDEN_DIM = 128
GRU_HIDDEN = 64
SEQ_LENGTH = 20

NUM_SAMPLES = 6000
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
VAL_SPLIT = 0.2
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Synthetic data generator ─────────────────────────────────────────────────
def _build_normal_sequence():
    """Build a SEQ_LENGTH x SEQ_DIM tensor for a normal wallet."""
    n = SEQ_LENGTH
    values = np.random.lognormal(mean=16, sigma=1.5, size=n)            # wei
    gas_prices = np.random.normal(loc=30e9, scale=8e9, size=n).clip(1e9)
    gas_used = np.random.normal(loc=80_000, scale=20_000, size=n).clip(21_000)
    is_error = (np.random.rand(n) < 0.03).astype(np.float32)
    base_ts = np.random.uniform(1_600_000_000, 1_700_000_000)
    deltas = np.random.exponential(scale=3600 * 12, size=n)             # hours
    timestamps = base_ts + np.cumsum(deltas)
    return np.column_stack([values, gas_prices, gas_used, is_error, timestamps])


def _build_suspicious_sequence():
    """Build a SEQ_LENGTH x SEQ_DIM tensor for a suspicious / bot wallet."""
    n = SEQ_LENGTH
    values = np.random.lognormal(mean=12, sigma=2.5, size=n)
    gas_prices = np.random.normal(loc=120e9, scale=40e9, size=n).clip(5e9)
    gas_used = np.random.normal(loc=60_000, scale=15_000, size=n).clip(21_000)
    is_error = (np.random.rand(n) < 0.20).astype(np.float32)
    base_ts = np.random.uniform(1_650_000_000, 1_700_000_000)
    deltas = np.random.exponential(scale=15, size=n)                    # seconds → bursty
    timestamps = base_ts + np.cumsum(deltas)
    return np.column_stack([values, gas_prices, gas_used, is_error, timestamps])


def _static_from_seq(seq, label):
    """Compute static features mirroring backend/app.py extraction."""
    num_txs = SEQ_LENGTH
    values = seq[:, 0]
    gas_prices = seq[:, 1]
    is_error = seq[:, 3]
    timestamps = seq[:, 4]

    total_value = float(values.sum())
    avg_value = float(values.mean())
    failed_txs = int(is_error.sum())
    fail_ratio = failed_txs / num_txs

    # Unique receivers: normal wallets ~ many; suspicious ~ few
    if label == 0:
        unique_receivers = int(np.random.randint(int(num_txs * 0.6), num_txs + 1))
    else:
        unique_receivers = int(np.random.randint(1, max(2, int(num_txs * 0.3))))

    avg_gas = float(gas_prices.mean())
    std_gas = float(gas_prices.std())

    deltas = np.diff(timestamps)
    avg_dt = float(deltas.mean()) if len(deltas) else 0.0
    min_dt = float(deltas.min()) if len(deltas) else 0.0

    return np.array([
        num_txs, total_value, avg_value, failed_txs, fail_ratio,
        unique_receivers, avg_gas, std_gas, avg_dt, min_dt,
    ], dtype=np.float32)


def generate_wallet_dataset(n_samples: int):
    n_normal = n_samples // 2
    n_suspicious = n_samples - n_normal

    seqs = []
    statics = []
    labels = []

    for _ in range(n_normal):
        s = _build_normal_sequence()
        seqs.append(s)
        statics.append(_static_from_seq(s, label=0))
        labels.append(0)
    for _ in range(n_suspicious):
        s = _build_suspicious_sequence()
        seqs.append(s)
        statics.append(_static_from_seq(s, label=1))
        labels.append(1)

    seqs = np.stack(seqs).astype(np.float32)
    statics = np.stack(statics).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)

    # Per-feature normalization (log-scale heavy-tailed columns)
    log_static_cols = [1, 2, 6, 7, 8, 9]
    statics[:, log_static_cols] = np.log1p(np.maximum(statics[:, log_static_cols], 0))
    s_mean = statics.mean(axis=0, keepdims=True)
    s_std = statics.std(axis=0, keepdims=True) + 1e-6
    statics = (statics - s_mean) / s_std

    log_seq_cols = [0, 1, 2, 4]
    seqs[:, :, log_seq_cols] = np.log1p(np.maximum(seqs[:, :, log_seq_cols], 0))
    seq_flat = seqs.reshape(-1, SEQ_DIM)
    q_mean = seq_flat.mean(axis=0, keepdims=True)
    q_std = seq_flat.std(axis=0, keepdims=True) + 1e-6
    seqs = (seqs - q_mean) / q_std

    # Gaussian overlap noise after normalization, so populations aren't
    # trivially separable and the model has to learn real signal.
    statics += np.random.normal(0, 0.5, statics.shape).astype(np.float32)
    seqs += np.random.normal(0, 0.5, seqs.shape).astype(np.float32)

    # Flip 5% of labels to simulate real-world label noise.
    flip_mask = np.random.rand(len(labels)) < 0.05
    labels[flip_mask] = 1 - labels[flip_mask]

    idx = np.random.permutation(n_samples)
    return (
        torch.from_numpy(statics[idx]),
        torch.from_numpy(seqs[idx]),
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
        for st, sq, lbl in loader:
            st, sq, lbl = st.to(device), sq.to(device), lbl.to(device)
            out = model(st, sq)
            loss = criterion(out, lbl)
            loss_total += loss.item() * lbl.size(0)
            correct += (out.argmax(dim=1) == lbl).sum().item()
            total += lbl.size(0)
    return loss_total / total, correct / total


def main():
    print(f"Device: {device}")
    print("Generating synthetic wallet dataset...")
    static, seq, labels = generate_wallet_dataset(NUM_SAMPLES)

    dataset = TensorDataset(static, seq, labels)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = WalletClassifier(
        static_dim=STATIC_DIM,
        seq_dim=SEQ_DIM,
        hidden_dim=HIDDEN_DIM,
        gru_hidden=GRU_HIDDEN,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    save_path = os.path.join(project_root, "backend/saved_models/wallet_classifier.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for st, sq, lbl in train_loader:
            st, sq, lbl = st.to(device), sq.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(st, sq)
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
