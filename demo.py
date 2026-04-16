#!/usr/bin/env python3
"""
LiveGrid Demo Script — Interview-Ready Self-Demonstrating Simulation

Runs a scripted sequence that shows off the full LiveGrid pipeline:
  - Normal operation (ticks 1–20)
  - Heat wave injection (tick 21)
  - Forced node failure + cascade (tick 35)
  - Live risk table every 5 ticks
  - Grid reset (tick 60)

Usage:
    cd LiveGrid
    python3 demo.py
"""

from __future__ import annotations

import io
import os
import sys
import time

# ── Imports ──────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from livegrid.config import SimulationConfig
from livegrid.engine.engine import SimulationEngine
from livegrid.grid.grid import Grid
from livegrid.logging.logger import DataLogger
from livegrid.models.node import NodeStatus
from livegrid.scenarios.heatwave import HeatWaveScenario
from livegrid.scenarios.sudden_failure import SuddenFailureScenario

# ── Try loading GNN, fall back to LSTM ───────────────────────────────

from predict import load_gnn_model, predict_gnn, load_model, predict
from collections import deque

MODEL_TYPE = "LSTM"
gnn_model = None
gnn_scaler = None
lstm_model = None
lstm_mean = None
lstm_scale = None

try:
    gnn_model, gnn_scaler = load_gnn_model()
    MODEL_TYPE = "GNN"
    print(f"✅ GNN model loaded")
except FileNotFoundError:
    try:
        lstm_model, lstm_mean, lstm_scale, _ = load_model()
        print(f"✅ LSTM model loaded (GNN not found)")
    except FileNotFoundError:
        print("⚠️  No model found — risk scores will be 0.0")

# ── Helpers ───────────────────────────────────────────────────────────

def suppress(fn, *args, **kwargs):
    """Run fn silently (suppress stdout)."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return fn(*args, **kwargs)
    finally:
        sys.stdout = old


def get_risk_scores(grid: Grid, node_buffers: dict, neighbor_map: dict,
                    last_warning: dict, tick: int) -> dict[str, float]:
    """Compute risk scores using GNN or LSTM."""
    scores: dict[str, float] = {n.id: 0.0 for n in grid.all_nodes()}

    if MODEL_TYPE == "GNN" and gnn_model is not None:
        from gnn_model import NODE_ORDER
        readings: dict[str, dict] = {}
        for node in grid.all_nodes():
            buf = node_buffers.get(node.id)
            if buf and len(buf) >= 2:
                cur, prev = buf[-1], buf[-2]
                neighbors = neighbor_map.get(node.id, [])
                n_loads = [node_buffers[n][-1]["load_ratio"]
                           for n in neighbors if n in node_buffers and node_buffers[n]]
                readings[node.id] = {
                    "load_ratio": cur["load_ratio"],
                    "voltage": cur["voltage"],
                    "frequency": cur["frequency"],
                    "load_ratio_rolling_mean_5": cur["load_ratio"],
                    "load_ratio_rolling_std_5": 0.0,
                    "load_ratio_delta": cur["load_ratio"] - prev["load_ratio"],
                    "neighbor_avg_load": sum(n_loads) / len(n_loads) if n_loads else 0.0,
                    "ticks_since_warning": 0.0 if cur["status"] == "WARNING" else 999.0,
                    "load_ratio_lag1": prev["load_ratio"],
                    "load_ratio_lag3": list(buf)[-4]["load_ratio"] if len(buf) >= 4 else prev["load_ratio"],
                    "is_warning": 1.0 if cur["status"] == "WARNING" else 0.0,
                }
        if len(readings) == 10:
            scores = predict_gnn(gnn_model, readings, gnn_scaler)

    elif lstm_model is not None and tick >= 10:
        from train_model import SEQUENCE_LENGTH
        from backend.main import build_sequences_from_buffers
        seqs = build_sequences_from_buffers(node_buffers, neighbor_map, last_warning, tick)
        if seqs:
            scores = predict(lstm_model, seqs, lstm_mean, lstm_scale)

    return scores


def status_icon(node, risk_scores: dict) -> str:
    s = str(node.status)
    if s == "FAILED": return "💀"
    risk = risk_scores.get(node.id, 0.0)
    if risk > 0.7: return "🔴"
    if risk > 0.4: return "🟡"
    return "🟢"


def print_risk_table(grid: Grid, risk_scores: dict, tick: int, scenario: str | None) -> None:
    """Print a formatted risk table."""
    scenario_str = scenario.replace("_", " ") if scenario else "none"
    print(f"\n{'─'*62}")
    print(f"  Tick {tick:>3}  │  Model: {MODEL_TYPE:<4}  │  Scenario: {scenario_str}")
    print(f"{'─'*62}")
    print(f"  {'NODE':<10} {'LOAD':>6}  {'RISK':>6}  {'STATUS':<20}")
    print(f"  {'─'*8} {'─'*6}  {'─'*6}  {'─'*18}")

    for node in sorted(grid.all_nodes(), key=lambda n: n.id):
        if str(node.status) == "FAILED":
            load_str = "FAILED"
            risk_str = "   —  "
        else:
            load_str = f"{node.load_ratio*100:.0f}%"
            risk_str = f"{risk_scores.get(node.id, 0.0):.2f}"
        icon = status_icon(node, risk_scores)
        print(f"  {node.id:<10} {load_str:>6}  {risk_str:>6}  {icon} {str(node.status)}")

    print(f"{'─'*62}")


# ── Main Demo ─────────────────────────────────────────────────────────

def run_demo() -> None:
    print("\n" + "═"*62)
    print("  ⚡ LiveGrid — Real-Time Power Grid Failure Prediction Demo")
    print("═"*62 + "\n")
    time.sleep(0.5)

    # Setup
    grid = Grid.build_sample_grid()
    config = SimulationConfig(total_ticks=999999, random_seed=42)
    logger = DataLogger()
    scenarios: list = []
    engine = SimulationEngine(grid=grid, logger=logger, scenarios=scenarios, config=config)
    suppress(logger.log_tick, tick=0, nodes=grid.all_nodes())

    # Buffers
    node_buffers: dict[str, deque] = {n.id: deque(maxlen=15) for n in grid.all_nodes()}
    neighbor_map: dict[str, list[str]] = {n.id: n.neighbors.copy() for n in grid.all_nodes()}
    last_warning: dict[str, int] = {}
    risk_scores: dict[str, float] = {n.id: 0.0 for n in grid.all_nodes()}

    active_scenario: str | None = None

    def record_buffers(tick: int) -> None:
        for node in grid.all_nodes():
            s = str(node.status)
            node_buffers[node.id].append({
                "tick": tick,
                "load_ratio": round(node.load_ratio, 4),
                "voltage": round(node.voltage, 2),
                "frequency": round(node.frequency, 2),
                "current_load": round(node.current_load, 2),
                "capacity": node.capacity,
                "status": s,
            })
            if s == "WARNING":
                last_warning[node.id] = tick

    # Seed buffers
    record_buffers(0)

    print("  Phase 1: Normal operation (ticks 1–20)")
    print("  Watch the grid stabilise...\n")
    time.sleep(0.5)

    for tick in range(1, 61):
        suppress(engine._run_tick, tick)
        record_buffers(tick)
        risk_scores = get_risk_scores(grid, node_buffers, neighbor_map, last_warning, tick)

        # Set risk_score on node objects for status_icon
        for node in grid.all_nodes():
            node.risk_score = risk_scores.get(node.id, 0.0)

        # ── Scenario events ─────────────────────────────────────────

        if tick == 21:
            hw = HeatWaveScenario(start_tick=21, duration=30, load_increase_pct=0.20)
            scenarios.append(hw)
            engine.scenarios = scenarios
            active_scenario = "heat_wave"
            print("\n  🌡️  HEAT WAVE INJECTED (+20% load) — watch the risk scores rise")
            time.sleep(0.3)

        if tick == 35:
            sf = SuddenFailureScenario(target_node_id="DIST-3", trigger_tick=35)
            scenarios.append(sf)
            engine.scenarios = scenarios
            suppress(engine._run_tick, tick)
            record_buffers(tick)
            print("\n  💥 FAILURE INJECTED at DIST-3 — cascade propagating to neighbours")
            time.sleep(0.3)

        if tick == 60:
            active_scenario = None
            # Rebuild the grid to show recovery
            print("\n  🔄 Resetting grid for recovery demo...")
            grid = Grid.build_sample_grid()
            scenarios = []
            engine = SimulationEngine(grid=grid, logger=DataLogger(), scenarios=scenarios, config=config)
            node_buffers = {n.id: deque(maxlen=15) for n in grid.all_nodes()}
            neighbor_map = {n.id: n.neighbors.copy() for n in grid.all_nodes()}
            last_warning = {}
            risk_scores = {n.id: 0.0 for n in grid.all_nodes()}

        # Print table every 5 ticks, or at key events
        if tick % 5 == 0 or tick in (21, 35, 36):
            print_risk_table(grid, risk_scores, tick, active_scenario)
            time.sleep(0.1)

    # Reset
    print("\n\n  ✅ Grid restored — demo complete")
    print("═"*62)
    print(f"\n  Model used: {MODEL_TYPE}")
    high_risk = [n.id for n in grid.all_nodes()
                 if risk_scores.get(n.id, 0.0) > 0.7 and str(n.status) != "FAILED"]
    if high_risk:
        print(f"  ⚠️  Still elevated risk: {', '.join(high_risk)}")
    print("\n  To see the live dashboard:  http://localhost:3000")
    print("  Backend API docs:           http://localhost:8000/docs\n")


if __name__ == "__main__":
    run_demo()
