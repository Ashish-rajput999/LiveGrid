#!/usr/bin/env python3
"""
LiveGrid Phase 2+5 — Prediction Interface

Loads the trained LSTM or GNN model and provides failure prediction
for live grid node readings.

LSTM (Phase 2): predict() — uses 10-tick sequences per node
GNN  (Phase 5): predict_gnn() — uses one full graph snapshot per tick

Usage:
    python predict.py

Requires:
    output/model.pt     — trained LSTM model from train_model.py
    output/gnn_model.pt — trained GNN model from train_gnn.py  (optional)
"""

from __future__ import annotations

import json
import os
import sys
import io

import numpy as np
import pandas as pd
import torch

from train_model import GridLSTM, FEATURE_COLUMNS, SEQUENCE_LENGTH


# ── Constants ────────────────────────────────────────────────────────

MODEL_PATH = os.path.join("output", "model.pt")
NEIGHBOR_MAP_JSON = os.path.join("output", "neighbor_map.json")


# ── Model Loading ────────────────────────────────────────────────────

def load_model(path: str = MODEL_PATH) -> tuple[GridLSTM, np.ndarray, np.ndarray, dict]:
    """
    Load the trained model and scaler parameters from a checkpoint.

    Args:
        path: Path to the model checkpoint file.

    Returns:
        Tuple of (model, scaler_mean, scaler_scale, config_dict).

    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. Run train_model.py first."
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    config = checkpoint["model_config"]
    model = GridLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return (
        model,
        checkpoint["scaler_mean"],
        checkpoint["scaler_scale"],
        config,
    )


# ── Prediction ───────────────────────────────────────────────────────

def predict(
    model: GridLSTM,
    sequences: dict[str, np.ndarray],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> dict[str, float]:
    """
    Predict failure probability for each node.

    Args:
        model: Trained GridLSTM model (in eval mode).
        sequences: Dict mapping node_id to feature array of shape
                   (SEQUENCE_LENGTH, num_features). Features must be
                   in the same order as FEATURE_COLUMNS.
        scaler_mean: StandardScaler mean values for normalization.
        scaler_scale: StandardScaler scale values for normalization.

    Returns:
        Dict mapping node_id to failure probability (0.0 to 1.0).
    """
    predictions: dict[str, float] = {}

    for node_id, features in sequences.items():
        # Normalize features using the saved scaler parameters
        normalized = (features - scaler_mean) / scaler_scale

        # Convert to tensor: (1, seq_len, num_features)
        tensor = torch.FloatTensor(normalized).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor).squeeze()
            prob = torch.sigmoid(logits).item()

        predictions[node_id] = round(prob, 4)

    return predictions


# ── GNN Model (Phase 5) ────────────────────────────────────────────────

GNN_MODEL_PATH = os.path.join("output", "gnn_model.pt")
GNN_SCALER_PATH = os.path.join("output", "gnn_scaler.pkl")


def load_gnn_model(
    path: str = GNN_MODEL_PATH,
    scaler_path: str = GNN_SCALER_PATH,
):
    """
    Load the trained GNN model and its scaler from disk.

    Returns:
        (model, scaler_dict) where scaler_dict has 'mean', 'std', 'columns'

    Raises:
        FileNotFoundError: If either file is missing.
    """
    import pickle
    from gnn_model import GridGAT

    if not os.path.exists(path):
        raise FileNotFoundError(f"GNN model not found at '{path}'. Run train_gnn.py.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"GNN scaler not found at '{scaler_path}'. Run train_gnn.py.")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["model_config"]

    model = GridGAT(
        in_channels=cfg["in_channels"],
        hidden_channels=cfg["hidden_channels"],
        heads=cfg["heads"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def predict_gnn(
    model,
    node_readings: dict[str, dict],
    scaler: dict,
) -> dict[str, float]:
    """
    Predict failure probability using the GNN for one full graph snapshot.

    Args:
        model:         Trained GridGAT model (eval mode).
        node_readings: Dict of {node_id: {feature: value}} for ONE tick.
                       Must include all 10 nodes.
        scaler:        Dict with 'mean', 'std', 'columns'.

    Returns:
        Dict mapping node_id to failure probability (0.0 – 1.0).
    """
    from gnn_model import NODE_ORDER, EDGE_INDEX, GNN_FEATURE_COLUMNS

    means = scaler["mean"]
    stds = scaler["std"]
    columns = scaler["columns"]

    x = np.zeros((len(NODE_ORDER), len(columns)), dtype=np.float32)
    for idx, node_id in enumerate(NODE_ORDER):
        readings = node_readings.get(node_id, {})
        for j, col in enumerate(columns):
            x[idx, j] = float(readings.get(col, 0.0))

    # Normalise
    x = (x - means) / stds

    x_tensor = torch.from_numpy(x)
    with torch.no_grad():
        logits = model(x_tensor, EDGE_INDEX)
        probs = torch.sigmoid(logits).squeeze(-1).numpy()

    return {
        node_id: round(float(probs[idx]), 4)
        for idx, node_id in enumerate(NODE_ORDER)
    }


# ── Feature Extraction from Simulation ───────────────────────────────

def extract_features_from_simulation(
    num_ticks: int = 20,
    seed: int = 999,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """
    Run a short simulation and extract features for prediction.

    Runs the LiveGrid simulator for num_ticks, then engineers the
    required features from the last SEQUENCE_LENGTH ticks.

    Args:
        num_ticks: Number of ticks to simulate.
        seed: Random seed for the simulation.

    Returns:
        Tuple of:
        - Dict mapping node_id to feature array (SEQUENCE_LENGTH, num_features)
        - Raw DataFrame for reference
    """
    from livegrid.config import SimulationConfig
    from livegrid.engine.engine import SimulationEngine
    from livegrid.grid.grid import Grid
    from livegrid.logging.logger import DataLogger
    from livegrid.scenarios.heatwave import HeatWaveScenario

    # Load neighbor map
    with open(NEIGHBOR_MAP_JSON) as f:
        neighbor_map = json.load(f)

    # Run simulation silently
    grid = Grid.build_sample_grid()
    config = SimulationConfig(
        total_ticks=num_ticks,
        random_seed=seed,
        load_fluctuation_pct=0.03,
    )
    logger = DataLogger()

    # Add a mild heat wave to make things interesting (not too aggressive
    # so we get a realistic range of failure probabilities in the demo)
    scenarios = [HeatWaveScenario(start_tick=10, duration=8, load_increase_pct=0.10)]

    engine = SimulationEngine(
        grid=grid, logger=logger, scenarios=scenarios, config=config,
    )

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        engine.run()
    finally:
        sys.stdout = old_stdout

    # Convert to DataFrame
    rows = logger.get_node_data()
    df = pd.DataFrame(rows)

    # Sort by node and tick
    df = df.sort_values(["node_id", "tick"]).reset_index(drop=True)

    # ── Engineer features on-the-fly ──

    # Rolling features per node
    grouped = df.groupby("node_id")["load_ratio"]

    df["load_ratio_rolling_mean_5"] = grouped.transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    ).round(4)

    df["load_ratio_rolling_std_5"] = grouped.transform(
        lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
    ).round(4)

    df["load_ratio_delta"] = grouped.transform(
        lambda x: x.diff().fillna(0)
    ).round(4)

    # Neighbor average load
    load_pivot = df.pivot_table(
        index="tick", columns="node_id", values="load_ratio", aggfunc="first",
    )
    neighbor_avg = {}
    for node_id, neighbors in neighbor_map.items():
        valid = [n for n in neighbors if n in load_pivot.columns]
        if valid:
            neighbor_avg[node_id] = load_pivot[valid].mean(axis=1)
        else:
            neighbor_avg[node_id] = pd.Series(0.0, index=load_pivot.index)

    df["neighbor_avg_load"] = df.apply(
        lambda row: round(
            neighbor_avg.get(row["node_id"], pd.Series(0.0)).get(row["tick"], 0.0), 4
        ),
        axis=1,
    )

    # Ticks since warning
    tsw_values = []
    for node_id, group in df.groupby("node_id"):
        last_warn = -999
        for _, row in group.iterrows():
            if row["status"] == "WARNING":
                last_warn = row["tick"]
                tsw_values.append(0)
            elif last_warn >= 0:
                tsw_values.append(row["tick"] - last_warn)
            else:
                tsw_values.append(999)
    df["ticks_since_warning"] = tsw_values

    # Extract last SEQUENCE_LENGTH ticks per node
    sequences: dict[str, np.ndarray] = {}
    max_tick = df["tick"].max()

    for node_id in df["node_id"].unique():
        node_df = df[df["node_id"] == node_id].sort_values("tick")
        # Get last SEQUENCE_LENGTH rows
        tail = node_df.tail(SEQUENCE_LENGTH)
        if len(tail) >= SEQUENCE_LENGTH:
            features = tail[FEATURE_COLUMNS].values.astype(np.float32)
            sequences[node_id] = features

    return sequences, df


# ── Demo ─────────────────────────────────────────────────────────────

def main() -> None:
    """
    Run a prediction demo.

    Loads the trained model, runs a 20-tick simulation, extracts
    features from the last 10 ticks, and predicts failure probability
    for each node.
    """
    print("=" * 60)
    print("  ⚡ LiveGrid — Failure Prediction Demo")
    print("=" * 60 + "\n")

    # Step 1: Load model
    print("  Loading trained model...")
    model, scaler_mean, scaler_scale, config = load_model()
    print(f"  Model loaded: LSTM(input={config['input_size']}, "
          f"hidden={config['hidden_size']}, layers={config['num_layers']})")

    # Step 2: Run simulation and extract features
    print("\n  Running 20-tick simulation...")
    sequences, raw_df = extract_features_from_simulation(num_ticks=20, seed=999)
    print(f"  Extracted sequences for {len(sequences)} nodes")

    # Step 3: Predict
    print("\n  Running predictions...")
    predictions = predict(model, sequences, scaler_mean, scaler_scale)

    # Step 4: Display results
    print(f"\n{'=' * 60}")
    print("  Failure Predictions (next 10 ticks)")
    print(f"{'=' * 60}")
    print(f"  {'Node ID':<12} {'Probability':<14} {'Risk Level':<12}")
    print(f"  {'-'*12} {'-'*14} {'-'*12}")

    for node_id in sorted(predictions.keys()):
        prob = predictions[node_id]

        # Risk level based on probability
        if prob >= 0.7:
            risk = "🔴 HIGH"
        elif prob >= 0.4:
            risk = "🟡 MEDIUM"
        else:
            risk = "🟢 LOW"

        print(f"  {node_id:<12} {prob:<14.4f} {risk}")

    # Show current grid state for context
    max_tick = raw_df["tick"].max()
    current_state = raw_df[raw_df["tick"] == max_tick][
        ["node_id", "current_load", "capacity", "load_ratio", "status"]
    ].sort_values("node_id")

    print(f"\n{'=' * 60}")
    print(f"  Current Grid State (tick {max_tick})")
    print(f"{'=' * 60}")
    print(f"  {'Node':<12} {'Load':<10} {'Capacity':<10} {'Ratio':<8} {'Status':<8}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    for _, row in current_state.iterrows():
        print(
            f"  {row['node_id']:<12} {row['current_load']:<10.1f} "
            f"{row['capacity']:<10.0f} {row['load_ratio']:<8.2%} "
            f"{row['status']:<8}"
        )

    print(f"\n{'=' * 60}")
    print("  ✅ Prediction complete.")
    print(f"{'=' * 60}\n")

    return predictions


if __name__ == "__main__":
    main()
