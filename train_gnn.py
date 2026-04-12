"""
LiveGrid Phase 5 — Train Graph Attention Network

Reads output/features.csv, builds per-tick PyG Data objects,
trains a 2-layer GAT, and saves to output/gnn_model.pt.

Usage:
    cd LiveGrid
    python3 train_gnn.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score

# ── Project imports ──────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnn_model import (
    GridGAT,
    NODE_ORDER,
    NODE_TO_IDX,
    NUM_NODES,
    NUM_FEATURES,
    EDGE_INDEX,
    GNN_FEATURE_COLUMNS,
)

# ── Config ───────────────────────────────────────────────────────────

FEATURES_PATH = "output/features.csv"
MODEL_SAVE_PATH = "output/gnn_model.pt"
SCALER_SAVE_PATH = "output/gnn_scaler.pkl"

EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_SPLIT = 0.8


# ── Data Loading ─────────────────────────────────────────────────────

def load_and_prepare_data() -> tuple[list[Data], list[Data]]:
    """
    Load features.csv, engineer extra columns for the GNN,
    build one PyG Data object per tick per run, and split by run_id.
    """
    print("📂 Loading features.csv ...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"   {len(df):,} rows, {df['run_id'].nunique()} runs, {df['node_id'].nunique()} nodes")

    # ── Engineer additional columns ──────────────────────────────────

    # Sort so lags work correctly
    df = df.sort_values(["run_id", "node_id", "tick"]).reset_index(drop=True)

    # load_ratio_lag1
    df["load_ratio_lag1"] = df.groupby(["run_id", "node_id"])["load_ratio"].shift(1).fillna(df["load_ratio"])

    # load_ratio_lag3
    df["load_ratio_lag3"] = df.groupby(["run_id", "node_id"])["load_ratio"].shift(3).fillna(df["load_ratio"])

    # is_warning (binary)
    df["is_warning"] = (df["status"] == "WARNING").astype(float)

    # ── Normalise features ───────────────────────────────────────────

    feature_cols = GNN_FEATURE_COLUMNS
    # Verify all columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"⚠️  Missing columns: {missing}")
        raise ValueError(f"Missing feature columns: {missing}")

    # Compute scaler stats
    means = df[feature_cols].mean().values.astype(np.float32)
    stds = df[feature_cols].std().values.astype(np.float32)
    stds[stds < 1e-6] = 1.0  # avoid div-by-0

    # Save scaler for inference
    import pickle
    with open(SCALER_SAVE_PATH, "wb") as f:
        pickle.dump({"mean": means, "std": stds, "columns": feature_cols}, f)
    print(f"💾 Scaler saved to {SCALER_SAVE_PATH}")

    # Normalise in-place
    df[feature_cols] = (df[feature_cols] - means) / stds

    # ── Label ────────────────────────────────────────────────────────

    label_col = "will_fail_within_10_ticks"
    pos_count = df[label_col].sum()
    neg_count = len(df) - pos_count
    print(f"   Labels: {int(pos_count):,} positive, {int(neg_count):,} negative ({pos_count/len(df)*100:.1f}% pos)")

    # ── Split by run_id ──────────────────────────────────────────────

    run_ids = sorted(df["run_id"].unique())
    np.random.seed(42)
    np.random.shuffle(run_ids)
    split_idx = int(len(run_ids) * TRAIN_SPLIT)
    train_runs = set(run_ids[:split_idx])
    val_runs = set(run_ids[split_idx:])
    print(f"   Train runs: {len(train_runs)}, Val runs: {len(val_runs)}")

    # ── Build PyG Data objects ───────────────────────────────────────

    edge_index = EDGE_INDEX

    def build_graphs(run_set: set) -> list[Data]:
        graphs: list[Data] = []
        subset = df[df["run_id"].isin(run_set)]

        for (run_id, tick), group in subset.groupby(["run_id", "tick"]):
            if len(group) != NUM_NODES:
                continue  # skip incomplete ticks

            # Sort nodes into canonical order
            node_map = {row["node_id"]: row for _, row in group.iterrows()}
            if not all(n in node_map for n in NODE_ORDER):
                continue

            x = np.zeros((NUM_NODES, NUM_FEATURES), dtype=np.float32)
            y = np.zeros((NUM_NODES, 1), dtype=np.float32)

            for idx, node_id in enumerate(NODE_ORDER):
                row = node_map[node_id]
                x[idx] = [row[c] for c in feature_cols]
                y[idx, 0] = row[label_col]

            data = Data(
                x=torch.from_numpy(x),
                edge_index=edge_index,
                y=torch.from_numpy(y),
            )
            graphs.append(data)

        return graphs

    print("🔨 Building graph snapshots ...")
    train_graphs = build_graphs(train_runs)
    val_graphs = build_graphs(val_runs)
    print(f"   Train: {len(train_graphs):,} graphs, Val: {len(val_graphs):,} graphs")

    return train_graphs, val_graphs


# ── Training Loop ────────────────────────────────────────────────────

def train() -> None:
    """Train the GNN and save the best model."""
    train_graphs, val_graphs = load_and_prepare_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    model = GridGAT().to(device)
    print(f"🧠 Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Class imbalance: compute pos_weight from training data
    all_labels = torch.cat([g.y for g in train_graphs])
    pos = all_labels.sum().item()
    neg = len(all_labels) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    print(f"⚖️  pos_weight = {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_f1 = 0.0
    best_state = None

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val F1':>8}")
    print("-" * 42)

    for epoch in range(1, EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        np.random.shuffle(train_graphs)

        for data in train_graphs:
            data = data.to(device)
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = criterion(logits, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_graphs)

        # ── Validate ─────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels_list = []

        with torch.no_grad():
            for data in val_graphs:
                data = data.to(device)
                logits = model(data.x, data.edge_index)
                loss = criterion(logits, data.y)
                val_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                labels = data.y.cpu().numpy().flatten().astype(int)

                all_preds.extend(preds)
                all_labels_list.extend(labels)

        val_loss /= len(val_graphs)
        f1 = f1_score(all_labels_list, all_preds, zero_division=0)

        print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>10.4f} | {f1:>8.4f}")

        scheduler.step(val_loss)

        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict().copy()

    # ── Save best model ──────────────────────────────────────────────

    print(f"\n✅ Best validation F1: {best_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics on val set
    model.eval()
    all_preds = []
    all_labels_list = []
    with torch.no_grad():
        for data in val_graphs:
            data = data.to(device)
            logits = model(data.x, data.edge_index)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            labels = data.y.cpu().numpy().flatten().astype(int)
            all_preds.extend(preds)
            all_labels_list.extend(labels)

    precision = precision_score(all_labels_list, all_preds, zero_division=0)
    recall = recall_score(all_labels_list, all_preds, zero_division=0)
    f1_final = f1_score(all_labels_list, all_preds, zero_division=0)

    print(f"\n📊 Final Metrics:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1_final:.4f}")

    target_met = "✅" if f1_final >= 0.78 else "⚠️  Below target"
    print(f"   Target F1 ≥ 0.78: {target_met}")

    # Save
    torch.save({
        "model_state_dict": best_state or model.state_dict(),
        "model_config": {
            "in_channels": NUM_FEATURES,
            "hidden_channels": 64,
            "heads": 8,
            "dropout": 0.2,
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1_final,
        },
    }, MODEL_SAVE_PATH)

    print(f"💾 Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    t0 = time.time()
    train()
    elapsed = time.time() - t0
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
