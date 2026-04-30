"""
LiveGrid Phase 3+4 — FastAPI Real-Time Backend

Runs the power grid simulation tick-by-tick in a background task,
computes ML predictions on-the-fly, and streams results to connected
WebSocket clients every second.

Endpoints:
    GET  /api/grid               — current grid snapshot with risk scores
    GET  /api/history            — last 50 tick snapshots
    POST /api/simulate-failure   — what-if cascade simulation (Phase 4)
    GET  /api/explain/{node_id}  — explainable risk analysis (Phase 4)
    WS   /ws/live                — real-time grid state stream

Usage:
    cd LiveGrid
    PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import random
import sys
import os
import time
from collections import deque
from typing import Any

from pydantic import BaseModel

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Ensure project root is importable ────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from livegrid.config import SimulationConfig
from livegrid.engine.engine import SimulationEngine
from livegrid.grid.grid import Grid
from livegrid.logging.logger import DataLogger
from livegrid.models.node import Node, NodeStatus, NodeType
from livegrid.scenarios.heatwave import HeatWaveScenario
from livegrid.scenarios.sudden_failure import SuddenFailureScenario
from predict import load_model, predict, load_gnn_model, predict_gnn
from train_model import FEATURE_COLUMNS, SEQUENCE_LENGTH


# ── App Setup ────────────────────────────────────────────────────────

app = FastAPI(title="LiveGrid Real-Time API", version="3.0.0")


def _build_cors_origins() -> list[str]:
    """Build allowed origins list from env vars + sensible defaults."""
    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    # Allow any Vercel preview/production URL
    origins.append("https://livegrid.vercel.app")
    # Allow exact URL(s) provided via FRONTEND_ORIGINS (comma-separated)
    raw = os.getenv("FRONTEND_ORIGINS", "").strip()
    if raw:
        extra = [o.strip() for o in raw.split(",") if o.strip()]
        origins.extend(extra)
    return list(dict.fromkeys(origins))  # deduplicate, preserve order

app.add_middleware(
    CORSMiddleware,
    allow_origins=_build_cors_origins(),
    allow_origin_regex=r"https://.*\.vercel\.app",  # all *.vercel.app URLs
    allow_credentials=False,  # no cookies/auth — False lets the regex work correctly
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Shared State ─────────────────────────────────────────────────────

class SimulationState:
    """Mutable shared state for the simulation loop."""

    def __init__(self) -> None:
        self.tick: int = 0
        self.grid: Grid | None = None
        self.engine: SimulationEngine | None = None
        self.logger: DataLogger | None = None

        # LSTM model (Phase 2)
        self.model = None
        self.scaler_mean = None
        self.scaler_scale = None

        # GNN model (Phase 5)
        self.gnn_model = None
        self.gnn_scaler = None
        self.model_type: str = "LSTM"  # "GNN" or "LSTM"

        # Rolling buffer: {node_id: deque of dicts}  — last SEQUENCE_LENGTH ticks
        self.node_buffers: dict[str, deque] = {}

        # Per-node warning tracker for ticks_since_warning feature
        self.last_warning_tick: dict[str, int] = {}

        # Risk scores from the latest prediction
        self.risk_scores: dict[str, float] = {}

        # Snapshot history for /api/history (last 50 ticks)
        self.history: deque[dict[str, Any]] = deque(maxlen=50)

        # Currently active scenario info
        self.active_scenario: str | None = None
        self.scenario_end_tick: int = 0

        # Next scenario injection tick
        self.next_scenario_tick: int = 0

        # Active scenarios list for the engine
        self.scenarios: list = []

        # WebSocket connections
        self.ws_clients: set[WebSocket] = set()

        # Neighbor map (built from grid topology)
        self.neighbor_map: dict[str, list[str]] = {}


state = SimulationState()


# ── Feature Engineering (on-the-fly) ─────────────────────────────────

def compute_features_for_node(
    node_id: str,
    buffer: deque[dict],
    neighbor_map: dict[str, list[str]],
    all_current_readings: dict[str, dict],
    last_warning_tick: dict[str, int],
    current_tick: int,
) -> list[float]:
    """
    Compute the 10 LSTM features for a single node from its rolling buffer.

    Returns feature vector in FEATURE_COLUMNS order:
        load_ratio, voltage, frequency, current_load, capacity,
        load_ratio_rolling_mean_5, load_ratio_rolling_std_5,
        load_ratio_delta, neighbor_avg_load, ticks_since_warning
    """
    reading = buffer[-1]  # latest tick

    load_ratio = reading["load_ratio"]
    voltage = reading["voltage"]
    frequency = reading["frequency"]
    current_load = reading["current_load"]
    capacity = reading["capacity"]

    # Rolling stats on load_ratio (last 5 ticks)
    recent_ratios = [b["load_ratio"] for b in list(buffer)[-5:]]
    rolling_mean = sum(recent_ratios) / len(recent_ratios)
    if len(recent_ratios) > 1:
        variance = sum((r - rolling_mean) ** 2 for r in recent_ratios) / (len(recent_ratios) - 1)
        rolling_std = variance ** 0.5
    else:
        rolling_std = 0.0

    # Delta
    if len(buffer) >= 2:
        load_ratio_delta = load_ratio - list(buffer)[-2]["load_ratio"]
    else:
        load_ratio_delta = 0.0

    # Neighbor average load
    neighbors = neighbor_map.get(node_id, [])
    if neighbors:
        neighbor_loads = [
            all_current_readings[n]["load_ratio"]
            for n in neighbors
            if n in all_current_readings
        ]
        neighbor_avg = sum(neighbor_loads) / len(neighbor_loads) if neighbor_loads else 0.0
    else:
        neighbor_avg = 0.0

    # Ticks since warning
    lwt = last_warning_tick.get(node_id, -999)
    if reading["status"] == "WARNING":
        ticks_since = 0
    elif lwt >= 0:
        ticks_since = current_tick - lwt
    else:
        ticks_since = 999

    return [
        load_ratio, voltage, frequency, current_load, capacity,
        rolling_mean, rolling_std, load_ratio_delta,
        neighbor_avg, ticks_since,
    ]


def build_sequences_from_buffers(
    node_buffers: dict[str, deque],
    neighbor_map: dict[str, list[str]],
    last_warning_tick: dict[str, int],
    current_tick: int,
) -> dict[str, Any]:
    """
    Build LSTM-ready sequences from rolling buffers for all nodes.

    Returns dict of {node_id: numpy array of shape (SEQUENCE_LENGTH, 10)}.
    Only returns nodes with full SEQUENCE_LENGTH history.
    """
    import numpy as np

    # Build current readings lookup
    all_current: dict[str, dict] = {}
    for nid, buf in node_buffers.items():
        if buf:
            all_current[nid] = buf[-1]

    sequences: dict[str, Any] = {}

    for node_id, buffer in node_buffers.items():
        if len(buffer) < SEQUENCE_LENGTH:
            continue

        # Build feature sequence for each of the last SEQUENCE_LENGTH ticks
        feat_seq = []
        buffer_list = list(buffer)

        for i in range(len(buffer_list) - SEQUENCE_LENGTH, len(buffer_list)):
            # Create a temporary sub-buffer up to tick i
            sub_buffer = deque(buffer_list[: i + 1], maxlen=SEQUENCE_LENGTH)
            tick_at_i = buffer_list[i].get("tick", current_tick)
            features = compute_features_for_node(
                node_id, sub_buffer, neighbor_map,
                all_current, last_warning_tick, tick_at_i,
            )
            feat_seq.append(features)

        sequences[node_id] = np.array(feat_seq, dtype=np.float32)

    return sequences


# ── Grid Snapshot Builder ────────────────────────────────────────────

def build_grid_snapshot() -> dict[str, Any]:
    """Build the JSON-serializable grid snapshot."""
    grid = state.grid
    if grid is None:
        return {"tick": 0, "nodes": [], "edges": [], "active_scenario": None, "failed_count": 0}

    nodes = []
    for node in grid.all_nodes():
        nodes.append({
            "id": node.id,
            "type": str(node.node_type),
            "capacity": node.capacity,
            "current_load": round(node.current_load, 2),
            "load_ratio": round(node.load_ratio, 4),
            "voltage_kv": round(node.voltage, 2),
            "frequency_hz": round(node.frequency, 2),
            "status": str(node.status),
            "risk_score": state.risk_scores.get(node.id, 0.0),
        })

    edges = [[e.source, e.target] for e in grid._edges]

    failed_count = len(grid.get_failed_nodes())

    return {
        "tick": state.tick,
        "nodes": nodes,
        "edges": edges,
        "active_scenario": state.active_scenario,
        "failed_count": failed_count,
        "model_type": state.model_type,
    }


# ── Simulation Loop ─────────────────────────────────────────────────

def init_simulation() -> None:
    """Initialize or reset the simulation."""
    import io

    grid = Grid.build_sample_grid()
    config = SimulationConfig(
        total_ticks=999999,  # We control ticks manually
        random_seed=None,     # Non-deterministic for variety
        load_fluctuation_pct=0.03,
    )
    logger = DataLogger()

    state.grid = grid
    state.logger = logger
    state.scenarios = []
    state.tick = 0
    state.node_buffers = {n.id: deque(maxlen=SEQUENCE_LENGTH + 5) for n in grid.all_nodes()}
    state.last_warning_tick = {}
    state.risk_scores = {}
    state.active_scenario = None
    state.scenario_end_tick = 0
    state.next_scenario_tick = random.randint(30, 60)

    # Build neighbor map from grid topology
    state.neighbor_map = {}
    for node in grid.all_nodes():
        state.neighbor_map[node.id] = node.neighbors.copy()

    # Create engine (we'll call _run_tick manually)
    state.engine = SimulationEngine(
        grid=grid, logger=logger, scenarios=state.scenarios, config=config,
    )

    # Log initial state
    logger.log_tick(tick=0, nodes=grid.all_nodes())

    # Buffer the initial readings
    _record_tick_to_buffers(0)


def _record_tick_to_buffers(tick: int) -> None:
    """Record current node states into the rolling buffers."""
    grid = state.grid
    if grid is None:
        return

    for node in grid.all_nodes():
        reading = {
            "tick": tick,
            "load_ratio": round(node.load_ratio, 4),
            "voltage": round(node.voltage, 2),
            "frequency": round(node.frequency, 2),
            "current_load": round(node.current_load, 2),
            "capacity": node.capacity,
            "status": str(node.status),
        }
        state.node_buffers[node.id].append(reading)

        # Track warning ticks
        if node.status == NodeStatus.WARNING:
            state.last_warning_tick[node.id] = tick


def inject_scenario(tick: int) -> None:
    """Inject a random scenario at the current tick."""
    grid = state.grid
    if grid is None:
        return

    choice = random.choice(["heat_wave", "sudden_failure"])

    if choice == "heat_wave":
        duration = random.randint(10, 25)
        intensity = round(random.uniform(0.15, 0.35), 2)
        scenario = HeatWaveScenario(
            start_tick=tick,
            duration=duration,
            load_increase_pct=intensity,
        )
        state.active_scenario = "heat_wave"
        state.scenario_end_tick = tick + duration
        state.scenarios.append(scenario)
    else:
        # Pick a random non-failed node
        operational = grid.get_operational_nodes()
        if operational:
            target = random.choice(operational)
            scenario = SuddenFailureScenario(
                target_node_id=target.id,
                trigger_tick=tick,
            )
            state.active_scenario = "sudden_failure"
            state.scenario_end_tick = tick + 1
            state.scenarios.append(scenario)

    # Schedule next scenario
    state.next_scenario_tick = tick + random.randint(30, 60)


def _get_neighbor_avg(node_id: str) -> float:
    """Return average load_ratio of active neighbours (for GNN feature)."""
    neighbors = state.neighbor_map.get(node_id, [])
    loads = []
    for nid in neighbors:
        buf = state.node_buffers.get(nid)
        if buf and len(buf) >= 1:
            loads.append(buf[-1]["load_ratio"])
    return sum(loads) / len(loads) if loads else 0.0


def _get_ticks_since_warning(node_id: str, current_tick: int) -> float:
    """Return ticks elapsed since last WARNING state for a node."""
    buf = state.node_buffers.get(node_id)
    if buf and buf[-1]["status"] == "WARNING":
        return 0.0
    lwt = state.last_warning_tick.get(node_id, -999)
    return float(current_tick - lwt) if lwt >= 0 else 999.0


async def simulation_loop() -> None:
    """Main simulation background task — runs forever."""
    import io

    # Try GNN first (Phase 5), fall back to LSTM (Phase 2)
    try:
        gnn_model, gnn_scaler = load_gnn_model()
        state.gnn_model = gnn_model
        state.gnn_scaler = gnn_scaler
        state.model_type = "GNN"
        print("✅ GNN model loaded — using Graph Attention Network")
    except Exception as e:
        print(f"ℹ️  GNN unavailable ({e}), trying LSTM...")
        state.model_type = "LSTM"

    # Load LSTM as fallback (or always, if GNN unavailable)
    try:
        model, scaler_mean, scaler_scale, config = load_model()
        state.model = model
        state.scaler_mean = scaler_mean
        state.scaler_scale = scaler_scale
        if state.model_type == "LSTM":
            print("✅ LSTM model loaded successfully")
    except FileNotFoundError as e:
        if state.gnn_model is None:
            print(f"⚠️  No model found: {e}")
            print("   Risk scores will be 0.0 until a model is available")

    init_simulation()
    print("⚡ Simulation started")

    while True:
        await asyncio.sleep(1.0)

        state.tick += 1
        tick = state.tick

        # Reset after 120 ticks
        if tick > 120:
            print(f"\n🔄 Resetting simulation at tick {tick}")
            init_simulation()
            state.tick = 0
            continue

        # Inject scenario if due
        if tick >= state.next_scenario_tick:
            inject_scenario(tick)

        # Clear active scenario label if it ended
        if state.active_scenario and tick > state.scenario_end_tick:
            state.active_scenario = None

        # Suppress engine print output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            state.engine._run_tick(tick)
        finally:
            sys.stdout = old_stdout

        # Record to buffers
        _record_tick_to_buffers(tick)

        # Run ML prediction each tick
        if state.model_type == "GNN" and state.gnn_model is not None:
            # GNN: build one snapshot (current tick readings) for all nodes
            try:
                node_readings: dict[str, dict] = {}
                for node in state.grid.all_nodes():
                    buf = state.node_buffers.get(node.id)
                    if buf and len(buf) >= 2:
                        cur = buf[-1]
                        prev = buf[-2]
                        node_readings[node.id] = {
                            "load_ratio": cur["load_ratio"],
                            "voltage": cur["voltage"],
                            "frequency": cur["frequency"],
                            "load_ratio_rolling_mean_5": cur.get("rolling_mean", cur["load_ratio"]),
                            "load_ratio_rolling_std_5": cur.get("rolling_std", 0.0),
                            "load_ratio_delta": cur["load_ratio"] - prev["load_ratio"],
                            "neighbor_avg_load": _get_neighbor_avg(node.id),
                            "ticks_since_warning": _get_ticks_since_warning(node.id, tick),
                            "load_ratio_lag1": prev["load_ratio"],
                            "load_ratio_lag3": list(buf)[-4]["load_ratio"] if len(buf) >= 4 else prev["load_ratio"],
                            "is_warning": 1.0 if cur["status"] == "WARNING" else 0.0,
                        }
                if len(node_readings) == 10:  # all nodes have data
                    state.risk_scores = predict_gnn(state.gnn_model, node_readings, state.gnn_scaler)
            except Exception as e:
                print(f"⚠️  GNN prediction error at tick {tick}: {e}")
        elif state.model is not None and tick >= SEQUENCE_LENGTH:
            # LSTM fallback
            try:
                sequences = build_sequences_from_buffers(
                    state.node_buffers,
                    state.neighbor_map,
                    state.last_warning_tick,
                    tick,
                )
                if sequences:
                    state.risk_scores = predict(
                        state.model, sequences,
                        state.scaler_mean, state.scaler_scale,
                    )
            except Exception as e:
                print(f"⚠️  LSTM prediction error at tick {tick}: {e}")

        # Build snapshot
        snapshot = build_grid_snapshot()
        state.history.append(snapshot)

        # Broadcast to WebSocket clients
        if state.ws_clients:
            payload = json.dumps(snapshot)
            disconnected = set()
            for ws in state.ws_clients:
                try:
                    await ws.send_text(payload)
                except Exception:
                    disconnected.add(ws)
            state.ws_clients -= disconnected


# ── Lifecycle Events ─────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """Start the simulation background task."""
    asyncio.create_task(simulation_loop())


# ── REST Endpoints ───────────────────────────────────────────────────

@app.get("/api/grid")
async def get_grid() -> dict[str, Any]:
    """Return current grid state as JSON."""
    return build_grid_snapshot()


@app.get("/api/history")
async def get_history() -> list[dict[str, Any]]:
    """Return last 50 tick snapshots."""
    return list(state.history)


# ── Phase 4 — What-If Cascade Simulator ──────────────────────────────

CUSTOMER_MAP = {
    "GENERATOR": 500_000,
    "SUBSTATION": 150_000,
    "DISTRIBUTION": 50_000,
}


class FailureRequest(BaseModel):
    node_id: str


@app.post("/api/simulate-failure")
async def simulate_failure(req: FailureRequest) -> dict[str, Any]:
    """
    Simulate a what-if node failure on a deep copy of the live grid.

    Forces the specified node to FAILED, runs cascade propagation,
    and returns the full cascade sequence without affecting the live sim.
    """
    grid = state.grid
    if grid is None:
        raise HTTPException(status_code=503, detail="Simulation not running")

    # Validate node exists
    target = grid.get_node(req.node_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Node '{req.node_id}' not found")
    if str(target.status) == "FAILED":
        raise HTTPException(status_code=400, detail=f"Node '{req.node_id}' is already FAILED")

    # Deep copy the grid
    grid_copy = _deep_copy_grid(grid)
    config = state.engine.config if state.engine else None

    # Run cascade on the copy
    cascade_seq = _run_cascade_simulation(grid_copy, req.node_id, config)

    # Compute results
    all_failed_ids = {step["node_id"] for step in cascade_seq}
    all_node_ids = {n.id for n in grid_copy.all_nodes()}
    survived = sorted(all_node_ids - all_failed_ids)

    # Estimate customers affected
    customers = 0
    for step in cascade_seq:
        node = grid_copy.get_node(step["node_id"])
        if node:
            customers += CUSTOMER_MAP.get(str(node.node_type), 50_000)

    return {
        "triggered_node": req.node_id,
        "cascade_sequence": cascade_seq,
        "total_failed": len(cascade_seq),
        "survived": survived,
        "estimated_customers_affected": customers,
    }


def _deep_copy_grid(original: Grid) -> Grid:
    """Create a deep copy of a Grid for what-if simulation."""
    new_grid = Grid()

    for node in original.all_nodes():
        new_node = Node(
            id=node.id,
            node_type=node.node_type,
            capacity=node.capacity,
            current_load=node.current_load,
            voltage=node.voltage,
            frequency=node.frequency,
            status=node.status,
            neighbors=node.neighbors.copy(),
        )
        new_grid._nodes[node.id] = new_node

    new_grid._edges = copy.deepcopy(original._edges)
    return new_grid


def _run_cascade_simulation(
    grid: Grid,
    trigger_node_id: str,
    config: SimulationConfig | None,
) -> list[dict[str, Any]]:
    """
    Force-fail a node and propagate cascading failures.

    Returns a list of cascade steps in order.
    """
    cascade_seq: list[dict[str, Any]] = []
    step_counter = [0]

    nominal_voltage = config.nominal_voltage if config else 230.0
    nominal_frequency = config.nominal_frequency if config else 50.0
    warning_threshold = config.warning_threshold if config else 0.9
    failure_threshold = config.failure_threshold if config else 1.0
    max_depth = config.max_cascade_depth if config else 10

    # Step 1: Force the trigger node to fail
    trigger = grid.get_node(trigger_node_id)
    if trigger is None:
        return cascade_seq

    trigger.fail()
    step_counter[0] += 1
    cascade_seq.append({
        "step": step_counter[0],
        "node_id": trigger_node_id,
        "reason": "manually triggered",
    })

    # Recursive cascade
    def _cascade(failed_id: str, depth: int) -> None:
        if depth >= max_depth:
            return

        failed_node = grid.get_node(failed_id)
        if failed_node is None:
            return

        stranded_load = failed_node.current_load
        if stranded_load <= 0:
            return

        failed_node.current_load = 0.0

        active_neighbors = grid.get_active_neighbors(failed_id)
        if not active_neighbors:
            return

        load_per_neighbor = stranded_load / len(active_neighbors)
        secondary: list[str] = []

        for neighbor in active_neighbors:
            neighbor.current_load += load_per_neighbor
            neighbor.update_electrical_state(
                nominal_voltage=nominal_voltage,
                nominal_frequency=nominal_frequency,
                warning_threshold=warning_threshold,
                failure_threshold=failure_threshold,
            )

            if neighbor.status == NodeStatus.FAILED:
                secondary.append(neighbor.id)
                step_counter[0] += 1
                cascade_seq.append({
                    "step": step_counter[0],
                    "node_id": neighbor.id,
                    "reason": f"overloaded by cascade from {failed_id}",
                })

        for nid in secondary:
            _cascade(nid, depth + 1)

    _cascade(trigger_node_id, 0)
    return cascade_seq


# ── Phase 4 — Explainable Alerts ─────────────────────────────────────

@app.get("/api/explain/{node_id}")
async def explain_node(node_id: str) -> dict[str, Any]:
    """
    Generate a human-readable explanation for a node's current risk score.

    Uses rule-based analysis of the last 10 ticks of node history.
    """
    grid = state.grid
    if grid is None:
        raise HTTPException(status_code=503, detail="Simulation not running")

    node = grid.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    buffer = state.node_buffers.get(node_id)
    if not buffer or len(buffer) < 2:
        raise HTTPException(status_code=425, detail="Not enough history yet")

    buf_list = list(buffer)
    current = buf_list[-1]
    risk = state.risk_scores.get(node_id, 0.0)

    # ── Primary driver: which metric changed most ──
    primary_driver = _compute_primary_driver(buf_list)

    # ── Contributing factors ──
    factors = _compute_contributing_factors(
        node_id, buf_list, state.neighbor_map, grid,
    )

    # ── Counterfactual ──
    counterfactual = _compute_counterfactual(current, risk)

    # ── Time to critical ──
    ttc = _compute_time_to_critical(buf_list)

    return {
        "node_id": node_id,
        "risk_score": round(risk, 4),
        "status": str(node.status),
        "explanation": {
            "primary_driver": primary_driver,
            "contributing_factors": factors,
            "counterfactual": counterfactual,
            "time_to_critical": ttc,
        },
    }


def _compute_primary_driver(buf_list: list[dict]) -> str:
    """Determine which single metric changed most over the buffer window."""
    window = buf_list[-min(8, len(buf_list)):]
    if len(window) < 2:
        return "Insufficient history for trend analysis"

    first, last = window[0], window[-1]
    ticks = len(window)

    load_delta = last["load_ratio"] - first["load_ratio"]
    voltage_delta = last["voltage"] - first["voltage"]
    # Normalize: load_ratio is 0-1 scale, voltage is ~200-230 scale
    load_change_pct = abs(load_delta) * 100  # as percentage points
    voltage_change_abs = abs(voltage_delta)

    if load_change_pct >= voltage_change_abs and load_change_pct > 1:
        direction = "increased" if load_delta > 0 else "decreased"
        return f"Load has {direction} {load_change_pct:.0f}% over the last {ticks} ticks"
    elif voltage_change_abs > 1:
        direction = "dropped" if voltage_delta < 0 else "risen"
        return f"Voltage has {direction} {voltage_change_abs:.1f}kV over the last {ticks} ticks"
    else:
        # Check if neighbours have been failing
        return "Node metrics are relatively stable — risk driven by model pattern recognition"


def _compute_contributing_factors(
    node_id: str,
    buf_list: list[dict],
    neighbor_map: dict[str, list[str]],
    grid: Grid,
) -> list[str]:
    """Build a list of contributing factors for the risk score."""
    factors: list[str] = []

    # Factor 1: Active scenario
    if state.active_scenario:
        scenario_name = state.active_scenario.replace("_", " ")
        factors.append(f"Active {scenario_name} scenario is pushing all nodes higher")

    # Factor 2: High connectivity
    neighbors = neighbor_map.get(node_id, [])
    if len(neighbors) >= 3:
        factors.append(
            f"{node_id} has {len(neighbors)} neighbours — "
            f"{'highest' if len(neighbors) >= 4 else 'high'} connectivity in grid"
        )

    # Factor 3: Voltage dropping
    window = buf_list[-min(5, len(buf_list)):]
    if len(window) >= 2:
        v_delta = window[-1]["voltage"] - window[0]["voltage"]
        if v_delta < -1.0:
            factors.append(
                f"Voltage has dropped {abs(v_delta):.1f}kV in last {len(window)} ticks"
            )

    # Factor 4: Neighbour failures
    failed_neighbors = [
        n for n in neighbors
        if grid.get_node(n) and str(grid.get_node(n).status) == "FAILED"
    ]
    if failed_neighbors:
        factors.append(
            f"{len(failed_neighbors)} neighbouring node(s) have failed: "
            f"{', '.join(failed_neighbors)}"
        )

    # Factor 5: Load ratio is in warning zone
    current_ratio = buf_list[-1]["load_ratio"]
    if current_ratio > 0.85:
        factors.append(
            f"Load ratio at {current_ratio:.0%} — approaching failure threshold"
        )

    if not factors:
        factors.append("No significant contributing factors detected")

    return factors


def _compute_counterfactual(current: dict, risk: float) -> str:
    """Calculate what load reduction would bring risk below 0.4."""
    load_ratio = current["load_ratio"]

    if risk < 0.4:
        return "Risk is already below the warning threshold (0.4)"

    # We need load_ratio to drop enough to get risk < 0.4
    # Approximate: if current load_ratio maps to current risk,
    # a proportional reduction should help
    # Target: reduce load_ratio to ~0.6 (a safe zone)
    target_ratio = 0.6
    if load_ratio <= target_ratio:
        target_ratio = load_ratio * 0.7  # need a bigger reduction

    reduction_needed = load_ratio - target_ratio
    reduction_pct = (reduction_needed / load_ratio * 100) if load_ratio > 0 else 0

    estimated_new_risk = max(0.05, risk * (target_ratio / max(load_ratio, 0.01)))

    return (
        f"If load decreased by {reduction_pct:.0f}%, "
        f"risk would drop from {risk:.2f} to approximately {estimated_new_risk:.2f}"
    )


def _compute_time_to_critical(buf_list: list[dict]) -> str:
    """Extrapolate current load trend to estimate ticks until load_ratio hits 1.0."""
    window = buf_list[-min(5, len(buf_list)):]
    if len(window) < 2:
        return "Insufficient data for prediction"

    current_ratio = window[-1]["load_ratio"]

    if current_ratio >= 1.0:
        return "Node is already at or beyond failure threshold"

    # Linear trend: average delta per tick
    deltas = [
        window[i]["load_ratio"] - window[i - 1]["load_ratio"]
        for i in range(1, len(window))
    ]
    avg_delta = sum(deltas) / len(deltas)

    if avg_delta <= 0:
        return "Load is stable or decreasing — no imminent failure projected"

    ticks_to_failure = (1.0 - current_ratio) / avg_delta

    if ticks_to_failure > 100:
        return "Estimated 100+ ticks before potential failure at current trend"

    low = max(1, int(ticks_to_failure * 0.7))
    high = max(low + 1, int(ticks_to_failure * 1.3))

    return f"Estimated {low}–{high} ticks before potential failure at current trend"


# ── WebSocket ────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket) -> None:
    """
    WebSocket endpoint for real-time grid state streaming.

    Clients connect here and receive the grid snapshot JSON
    automatically every 1 second (pushed by the simulation loop).
    """
    await ws.accept()
    state.ws_clients.add(ws)
    print(f"🔌 WebSocket client connected ({len(state.ws_clients)} total)")

    try:
        # Keep connection alive — listen for client messages (ping/pong)
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        state.ws_clients.discard(ws)
        print(f"🔌 WebSocket client disconnected ({len(state.ws_clients)} total)")
