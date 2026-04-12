"""
LiveGrid Phase 5 — Graph Attention Network (GNN) Model

2-layer GAT architecture:
  - Input: 11 per-node features (same as LSTM feature set)
  - Layer 1: GATConv, 8 heads, hidden_dim=64
  - Layer 2: GATConv, 1 head, output_dim=1
  - Output: per-node failure probability (sigmoid)

The graph structure (edges) is fixed from build_sample_grid().
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


# ── Fixed graph topology from build_sample_grid() ────────────────────

# Node ordering (index → node_id)
NODE_ORDER = [
    "GEN-1", "GEN-2",
    "SUB-1", "SUB-2", "SUB-3",
    "DIST-1", "DIST-2", "DIST-3", "DIST-4", "DIST-5",
]
NODE_TO_IDX = {n: i for i, n in enumerate(NODE_ORDER)}
NUM_NODES = len(NODE_ORDER)

# Edges from build_sample_grid() — bidirectional
_EDGES_RAW = [
    ("GEN-1", "SUB-1"),
    ("GEN-1", "GEN-2"),
    ("GEN-2", "SUB-2"),
    ("SUB-1", "SUB-2"),
    ("SUB-1", "SUB-3"),
    ("SUB-2", "SUB-3"),
    ("SUB-1", "DIST-1"),
    ("DIST-1", "DIST-2"),
    ("SUB-2", "DIST-3"),
    ("SUB-3", "DIST-4"),
    ("SUB-3", "DIST-5"),
]


def get_edge_index() -> torch.Tensor:
    """Return edge_index tensor of shape [2, 2*E] (bidirectional)."""
    src, dst = [], []
    for a, b in _EDGES_RAW:
        src.extend([NODE_TO_IDX[a], NODE_TO_IDX[b]])
        dst.extend([NODE_TO_IDX[b], NODE_TO_IDX[a]])
    return torch.tensor([src, dst], dtype=torch.long)


# Cached for import-time use
EDGE_INDEX = get_edge_index()

# ── Feature columns (11) ──────────────────────────────────────────────

GNN_FEATURE_COLUMNS = [
    "load_ratio",
    "voltage",
    "frequency",
    "load_ratio_rolling_mean_5",
    "load_ratio_rolling_std_5",
    "load_ratio_delta",
    "neighbor_avg_load",
    "ticks_since_warning",
    # Lag features (computed in train_gnn.py)
    "load_ratio_lag1",
    "load_ratio_lag3",
    "is_warning",
]

NUM_FEATURES = len(GNN_FEATURE_COLUMNS)  # 11


# ── Model ────────────────────────────────────────────────────────────

class GridGAT(nn.Module):
    """
    2-layer Graph Attention Network for per-node failure prediction.

    Input:  x of shape [N, NUM_FEATURES]
    Output: logits of shape [N, 1]  (apply sigmoid for probabilities)
    """

    def __init__(
        self,
        in_channels: int = NUM_FEATURES,
        hidden_channels: int = 64,
        heads: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.dropout = dropout

        # Layer 1: multi-head attention
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,  # output: hidden_channels * heads
        )

        # Layer 2: single-head output
        self.conv2 = GATConv(
            hidden_channels * heads,
            1,
            heads=1,
            dropout=dropout,
            concat=False,  # output: 1
        )

        self.relu = nn.ELU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          [N, in_channels]  — node features
            edge_index: [2, E]            — graph connectivity
        Returns:
            logits:     [N, 1]            — raw pre-sigmoid scores
        """
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv2(x, edge_index)
        return x  # raw logits
