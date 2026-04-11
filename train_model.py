#!/usr/bin/env python3
"""
LiveGrid Phase 2 — LSTM Model Training

Trains a PyTorch LSTM model to predict power grid node failures
10 ticks ahead using engineered time-series features.

Architecture:
    Input (10 timesteps × N features)
    → LSTM (hidden=64, 2 layers, dropout=0.2)
    → Linear(64 → 1)
    → Sigmoid → failure probability

Usage:
    python train_model.py

Input:
    output/features.csv

Output:
    output/model.pt — trained model weights + scaler parameters
"""

from __future__ import annotations

import os
import sys
import json
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ── Constants ────────────────────────────────────────────────────────

FEATURES_CSV = os.path.join("output", "features.csv")
MODEL_PATH = os.path.join("output", "model.pt")

SEQUENCE_LENGTH = 10  # Sliding window size (last 10 ticks)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
TRAIN_SPLIT = 0.8  # 80% train, 20% test (split by run_id)

# Features used for model input
FEATURE_COLUMNS = [
    "load_ratio",
    "voltage",
    "frequency",
    "current_load",
    "capacity",
    "load_ratio_rolling_mean_5",
    "load_ratio_rolling_std_5",
    "load_ratio_delta",
    "neighbor_avg_load",
    "ticks_since_warning",
]

LABEL_COLUMN = "will_fail_within_10_ticks"


# ── Dataset ──────────────────────────────────────────────────────────

class GridSequenceDataset(Dataset):
    """
    PyTorch Dataset for grid node time-series sequences.

    Creates sliding windows of SEQUENCE_LENGTH ticks for each
    (run_id, node_id) combination. Each sample is a tuple of:
    - features: tensor of shape (SEQUENCE_LENGTH, num_features)
    - label: scalar tensor (0 or 1)

    The label is taken from the LAST tick in the window.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            sequences: Array of shape (N, SEQUENCE_LENGTH, num_features).
            labels: Array of shape (N,) with binary labels.
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (sequence tensor, label tensor).
        """
        return self.sequences[idx], self.labels[idx]


# ── Model ────────────────────────────────────────────────────────────

class GridLSTM(nn.Module):
    """
    LSTM model for grid failure prediction.

    Takes a sequence of node feature vectors and predicts the
    probability of failure within the next 10 ticks.

    Architecture:
        LSTM(input_size, hidden=64, layers=2, dropout=0.2)
        → Linear(64, 1) on the last hidden state
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        """
        Initialize the LSTM model.

        Args:
            input_size: Number of features per timestep.
            hidden_size: LSTM hidden state dimension.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout rate between LSTM layers.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Logits tensor of shape (batch, 1).
        """
        # LSTM output: (batch, seq_len, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state from the final LSTM layer
        last_hidden = h_n[-1]  # (batch, hidden_size)

        # Classify
        logits = self.classifier(last_hidden)  # (batch, 1)
        return logits


# ── Data Preparation ─────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    """
    Load the engineered features CSV.

    Returns:
        DataFrame with features and labels.

    Raises:
        FileNotFoundError: If features.csv doesn't exist.
    """
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(
            f"Features file not found at '{FEATURES_CSV}'. "
            f"Run feature_engineering.py first."
        )
    print(f"  Loading {FEATURES_CSV}...")
    df = pd.read_csv(FEATURES_CSV)
    print(f"  Loaded {len(df):,} rows")
    return df


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding window sequences from the feature DataFrame.

    For each (run_id, node_id) group, creates overlapping windows
    of seq_len consecutive ticks. The label is taken from the last
    tick in each window.

    Args:
        df: DataFrame sorted by (run_id, node_id, tick).
        feature_cols: Column names to use as features.
        label_col: Column name for the binary label.
        seq_len: Window size.

    Returns:
        Tuple of (sequences array, labels array) where:
        - sequences: shape (N, seq_len, num_features)
        - labels: shape (N,)
    """
    print(f"  Building sequences (window={seq_len})...")

    sequences = []
    labels = []

    df = df.sort_values(["run_id", "node_id", "tick"]).reset_index(drop=True)

    for (run_id, node_id), group in df.groupby(["run_id", "node_id"]):
        features = group[feature_cols].values.astype(np.float32)
        target = group[label_col].values.astype(np.float32)

        # Create sliding windows
        for i in range(len(features) - seq_len + 1):
            seq = features[i:i + seq_len]
            label = target[i + seq_len - 1]  # Label from last tick in window
            sequences.append(seq)
            labels.append(label)

    sequences_arr = np.array(sequences)
    labels_arr = np.array(labels)

    print(f"  Built {len(sequences_arr):,} sequences")
    return sequences_arr, labels_arr


def split_by_run_id(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_SPLIT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test sets by run_id.

    Ensures all ticks from the same run stay together to prevent
    temporal data leakage.

    Args:
        df: Full feature DataFrame.
        train_ratio: Fraction of runs for training.

    Returns:
        Tuple of (train DataFrame, test DataFrame).
    """
    run_ids = df["run_id"].unique()
    np.random.shuffle(run_ids)

    split_idx = int(len(run_ids) * train_ratio)
    train_runs = set(run_ids[:split_idx])
    test_runs = set(run_ids[split_idx:])

    train_df = df[df["run_id"].isin(train_runs)].reset_index(drop=True)
    test_df = df[df["run_id"].isin(test_runs)].reset_index(drop=True)

    print(f"  Train: {len(train_df):,} rows ({len(train_runs)} runs)")
    print(f"  Test: {len(test_df):,} rows ({len(test_runs)} runs)")

    return train_df, test_df


# ── Training ─────────────────────────────────────────────────────────

def train_model(
    model: GridLSTM,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    test_loader: Optional[DataLoader] = None,
) -> GridLSTM:
    """
    Train the LSTM model.

    Prints training loss per epoch, and optionally test metrics.

    Args:
        model: The GridLSTM model.
        train_loader: Training data DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        num_epochs: Number of training epochs.
        test_loader: Optional test DataLoader for epoch-end evaluation.

    Returns:
        The trained model.
    """
    print(f"\n{'=' * 60}")
    print(f"  Training LSTM ({num_epochs} epochs)")
    print(f"{'=' * 60}")
    print(f"  {'Epoch':<8} {'Loss':<12} {'Train Acc':<12} {'Test Acc':<12} {'Test F1':<10}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    model.train()

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        train_preds = []
        train_labels = []

        for sequences, labels in train_loader:
            optimizer.zero_grad()
            logits = model(sequences).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_preds.extend(preds.detach().numpy())
            train_labels.extend(labels.numpy())

        avg_loss = total_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)

        # Test metrics
        test_acc_str = ""
        test_f1_str = ""
        if test_loader and (epoch % 5 == 0 or epoch == num_epochs):
            test_preds, test_labels = evaluate_predictions(model, test_loader)
            test_acc = accuracy_score(test_labels, test_preds)
            test_f1 = f1_score(test_labels, test_preds, zero_division=0)
            test_acc_str = f"{test_acc:.4f}"
            test_f1_str = f"{test_f1:.4f}"

        print(
            f"  {epoch:<8} {avg_loss:<12.6f} {train_acc:<12.4f} "
            f"{test_acc_str:<12} {test_f1_str:<10}"
        )

    return model


def evaluate_predictions(
    model: GridLSTM,
    data_loader: DataLoader,
) -> tuple[list, list]:
    """
    Generate predictions from the model.

    Args:
        model: Trained GridLSTM model.
        data_loader: DataLoader to evaluate.

    Returns:
        Tuple of (predicted labels list, true labels list).
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in data_loader:
            logits = model(sequences).squeeze(1)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    model.train()
    return all_preds, all_labels


def save_model(
    model: GridLSTM,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    feature_cols: list[str],
    path: str,
) -> None:
    """
    Save the trained model and scaler parameters.

    Saves a checkpoint dict containing:
    - model_state_dict: model weights
    - scaler_mean: StandardScaler means
    - scaler_scale: StandardScaler scales
    - feature_columns: list of feature column names
    - model_config: architecture hyperparameters

    Args:
        model: Trained model.
        scaler_mean: Scaler mean values.
        scaler_scale: Scaler scale values.
        feature_cols: Feature column names in order.
        path: Output file path.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "feature_columns": feature_cols,
        "model_config": {
            "input_size": len(feature_cols),
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "sequence_length": SEQUENCE_LENGTH,
        },
    }
    torch.save(checkpoint, path)
    print(f"\n💾 Model saved to: {os.path.abspath(path)}")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    """
    Run the full training pipeline.

    Loads features, builds sequences, trains LSTM, evaluates,
    and saves the model.
    """
    print("=" * 60)
    print("  ⚡ LiveGrid — LSTM Model Training")
    print("=" * 60 + "\n")

    np.random.seed(42)
    torch.manual_seed(42)

    # Step 1: Load data
    df = load_features()

    # Step 2: Split by run_id
    print("\n  Splitting data by run_id...")
    train_df, test_df = split_by_run_id(df)

    # Step 3: Fit scaler on training data only
    print("\n  Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    train_df[FEATURE_COLUMNS] = scaler.fit_transform(
        train_df[FEATURE_COLUMNS].values
    )
    test_df[FEATURE_COLUMNS] = scaler.transform(
        test_df[FEATURE_COLUMNS].values
    )
    print(f"  Scaled {len(FEATURE_COLUMNS)} features")

    # Step 4: Build sequences
    print("\n  Building training sequences...")
    train_seqs, train_labels = build_sequences(
        train_df, FEATURE_COLUMNS, LABEL_COLUMN, SEQUENCE_LENGTH
    )
    print("  Building test sequences...")
    test_seqs, test_labels = build_sequences(
        test_df, FEATURE_COLUMNS, LABEL_COLUMN, SEQUENCE_LENGTH
    )

    # Step 5: Class balance
    pos_count = train_labels.sum()
    neg_count = len(train_labels) - pos_count
    pos_weight = neg_count / max(pos_count, 1)
    print(f"\n  Class balance (train):")
    print(f"    Positive: {int(pos_count):,} ({pos_count / len(train_labels) * 100:.1f}%)")
    print(f"    Negative: {int(neg_count):,} ({neg_count / len(train_labels) * 100:.1f}%)")
    print(f"    pos_weight: {pos_weight:.2f}")

    # Step 6: Create DataLoaders
    train_dataset = GridSequenceDataset(train_seqs, train_labels)
    test_dataset = GridSequenceDataset(test_seqs, test_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    )

    # Step 7: Initialize model
    input_size = len(FEATURE_COLUMNS)
    model = GridLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    print(f"\n  Model: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Step 8: Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight])
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Step 9: Train
    model = train_model(
        model, train_loader, criterion, optimizer,
        num_epochs=NUM_EPOCHS, test_loader=test_loader,
    )

    # Step 10: Final evaluation
    print(f"\n{'=' * 60}")
    print("  Final Test Evaluation")
    print(f"{'=' * 60}")

    test_preds, test_true = evaluate_predictions(model, test_loader)

    acc = accuracy_score(test_true, test_preds)
    prec = precision_score(test_true, test_preds, zero_division=0)
    rec = recall_score(test_true, test_preds, zero_division=0)
    f1 = f1_score(test_true, test_preds, zero_division=0)

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    if f1 >= 0.75:
        print(f"\n  ✅ F1 target achieved! ({f1:.4f} >= 0.75)")
    else:
        print(f"\n  ⚠️  F1 below target ({f1:.4f} < 0.75)")

    print(f"\n  Detailed classification report:")
    print(classification_report(
        test_true, test_preds,
        target_names=["No Failure", "Will Fail"],
        zero_division=0,
    ))

    # Step 11: Save model
    os.makedirs("output", exist_ok=True)
    save_model(
        model,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        feature_cols=FEATURE_COLUMNS,
        path=MODEL_PATH,
    )

    print(f"\n{'=' * 60}")
    print("  ✅ Training complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
