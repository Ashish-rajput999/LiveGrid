#!/usr/bin/env python3
"""
LiveGrid — Test Suite

Run with:  python test_livegrid.py

Tests all core components: Node physics, Grid graph operations,
Scenario behavior, Cascade logic, and Data logging.
"""

import os
import sys


# ── Helpers ──────────────────────────────────────────────────────────

PASS = 0
FAIL = 0


def test(name: str, condition: bool, detail: str = "") -> None:
    """Run a single assertion test."""
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        msg = f"  ❌ {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# ── Test 1: Node Model ──────────────────────────────────────────────

def test_node_model():
    section("Test 1: Node Model")

    from livegrid.models.node import Node, NodeStatus, NodeType

    # Create a node
    node = Node(
        id="TEST-1",
        node_type=NodeType.GENERATOR,
        capacity=500.0,
        current_load=250.0,
    )

    test("Node created with correct ID", node.id == "TEST-1")
    test("Load ratio is 0.5", node.load_ratio == 0.5)
    test("Node is not overloaded at 50%", not node.is_overloaded())
    test("Status is OK", node.status == NodeStatus.OK)

    # Test overload detection
    node.current_load = 600.0
    test("Load ratio > 1 when overloaded", node.load_ratio > 1.0)
    test("Node IS overloaded at 120%", node.is_overloaded())

    # Test electrical state update
    node.current_load = 250.0
    node.status = NodeStatus.OK
    node.update_electrical_state()
    test("Voltage drops with load", node.voltage < 230.0)
    test("Voltage is reasonable", 200.0 < node.voltage < 230.0)
    test("Frequency is at nominal (load < 50%)", node.frequency == 50.0)

    # Test warning threshold
    node.current_load = 460.0  # 92% load
    node.update_electrical_state()
    test("Status is WARNING at 92% load", node.status == NodeStatus.WARNING)

    # Test failure threshold
    node.current_load = 510.0  # 102% load
    node.status = NodeStatus.OK  # Reset so update can trigger
    node.update_electrical_state()
    test("Status is FAILED at 102% load", node.status == NodeStatus.FAILED)
    test("Voltage is 0 after failure", node.voltage == 0.0)
    test("Frequency is 0 after failure", node.frequency == 0.0)

    # Test fail() stays failed
    node.update_electrical_state()
    test("Failed nodes stay failed on update", node.status == NodeStatus.FAILED)

    # Test restore
    node.restore()
    test("Restored node is OK", node.status == NodeStatus.OK)
    test("Restored node has 0 load", node.current_load == 0.0)

    # Test snapshot
    snap = node.snapshot(tick=5)
    test("Snapshot has tick field", snap["tick"] == 5)
    test("Snapshot has node_id", snap["node_id"] == "TEST-1")
    test("Snapshot has all 9 fields", len(snap) == 9)


# ── Test 2: Edge Model ──────────────────────────────────────────────

def test_edge_model():
    section("Test 2: Edge Model")

    from livegrid.models.edge import Edge

    edge = Edge(source="A", target="B", capacity=400.0)

    test("Edge source is A", edge.source == "A")
    test("Edge target is B", edge.target == "B")
    test("Edge is active by default", edge.active)
    test("Edge to_dict works", edge.to_dict()["capacity"] == 400.0)


# ── Test 3: Grid Graph ──────────────────────────────────────────────

def test_grid_graph():
    section("Test 3: Grid Graph")

    from livegrid.grid.grid import Grid
    from livegrid.models.node import Node, NodeStatus, NodeType
    from livegrid.models.edge import Edge

    grid = Grid()

    # Add nodes
    grid.add_node(Node(id="A", node_type=NodeType.GENERATOR, capacity=100, current_load=50))
    grid.add_node(Node(id="B", node_type=NodeType.SUBSTATION, capacity=80, current_load=40))
    grid.add_node(Node(id="C", node_type=NodeType.DISTRIBUTION, capacity=60, current_load=30))

    test("Grid has 3 nodes", grid.node_count == 3)

    # Add edges
    grid.add_edge(Edge(source="A", target="B"))
    grid.add_edge(Edge(source="B", target="C"))

    test("Grid has 2 edges", grid.edge_count == 2)

    # Neighbor queries
    neighbors_of_b = grid.get_neighbors("B")
    test("B has 2 neighbors (A and C)", len(neighbors_of_b) == 2)

    neighbor_ids = {n.id for n in neighbors_of_b}
    test("B's neighbors are A and C", neighbor_ids == {"A", "C"})

    # Bidirectional
    neighbors_of_a = grid.get_neighbors("A")
    test("A has 1 neighbor (B)", len(neighbors_of_a) == 1)

    # Active neighbor filtering
    node_c = grid.get_node("C")
    node_c.fail()
    active = grid.get_active_neighbors("B")
    test("B has 1 active neighbor after C fails", len(active) == 1)
    test("Active neighbor is A", active[0].id == "A")

    # Duplicate node check
    try:
        grid.add_node(Node(id="A", node_type=NodeType.GENERATOR, capacity=100, current_load=50))
        test("Duplicate node raises error", False)
    except ValueError:
        test("Duplicate node raises error", True)

    # Sample grid
    sample = Grid.build_sample_grid()
    test("Sample grid has 10 nodes", sample.node_count == 10)
    test("Sample grid has 11 edges", sample.edge_count == 11)
    test("GEN-1 exists", sample.get_node("GEN-1") is not None)
    test("DIST-5 exists", sample.get_node("DIST-5") is not None)


# ── Test 4: Scenarios ───────────────────────────────────────────────

def test_scenarios():
    section("Test 4: Scenarios")

    from livegrid.grid.grid import Grid
    from livegrid.scenarios.heatwave import HeatWaveScenario
    from livegrid.scenarios.sudden_failure import SuddenFailureScenario

    # --- Heat Wave ---
    hw = HeatWaveScenario(start_tick=10, duration=5, load_increase_pct=0.5)

    test("HeatWave name is 'heat_wave'", hw.name == "heat_wave")
    test("HeatWave inactive before start", not hw.is_active(9))
    test("HeatWave active at start", hw.is_active(10))
    test("HeatWave active during duration", hw.is_active(14))
    test("HeatWave inactive after end", not hw.is_active(15))

    grid = Grid.build_sample_grid()
    gen1_load_before = grid.get_node("GEN-1").current_load

    events = hw.apply(grid, tick=10)
    gen1_load_after = grid.get_node("GEN-1").current_load

    test("HeatWave increases load", gen1_load_after > gen1_load_before)
    test("HeatWave increase is ~50%",
         abs(gen1_load_after - gen1_load_before * 1.5) < 0.01,
         f"expected {gen1_load_before * 1.5}, got {gen1_load_after}")
    test("HeatWave returns events", len(events) > 0)

    # Second apply should not re-apply surge
    gen1_load_second = grid.get_node("GEN-1").current_load
    hw.apply(grid, tick=11)
    test("HeatWave doesn't re-apply surge", grid.get_node("GEN-1").current_load == gen1_load_second)

    # --- Sudden Failure ---
    grid2 = Grid.build_sample_grid()
    sf = SuddenFailureScenario(target_node_id="SUB-1", trigger_tick=20)

    test("SuddenFailure inactive before trigger", not sf.is_active(19))
    test("SuddenFailure active at trigger", sf.is_active(20))

    from livegrid.models.node import NodeStatus
    test("SUB-1 is OK before failure", grid2.get_node("SUB-1").status == NodeStatus.OK)

    events = sf.apply(grid2, tick=20)
    test("SUB-1 is FAILED after scenario", grid2.get_node("SUB-1").status == NodeStatus.FAILED)
    test("SuddenFailure returns events", len(events) > 0)
    test("SuddenFailure doesn't fire again", not sf.is_active(20))


# ── Test 5: Data Logger ─────────────────────────────────────────────

def test_data_logger():
    section("Test 5: Data Logger")

    from livegrid.logging.logger import DataLogger
    from livegrid.models.node import Node, NodeType

    logger = DataLogger()

    test("Logger starts empty (nodes)", logger.node_row_count == 0)
    test("Logger starts empty (events)", logger.event_count == 0)

    # Log some tick data
    nodes = [
        Node(id="N1", node_type=NodeType.GENERATOR, capacity=100, current_load=50),
        Node(id="N2", node_type=NodeType.SUBSTATION, capacity=80, current_load=60),
    ]
    logger.log_tick(tick=0, nodes=nodes)
    logger.log_tick(tick=1, nodes=nodes)

    test("Logger has 4 node rows (2 nodes × 2 ticks)", logger.node_row_count == 4)

    # Log events
    logger.log_event(tick=1, event_type="TEST", description="test event")
    test("Logger has 1 event", logger.event_count == 1)

    # CSV export
    test_dir = "output/test"
    os.makedirs(test_dir, exist_ok=True)

    node_path = logger.to_csv(f"{test_dir}/test_nodes.csv")
    test("Node CSV created", os.path.exists(node_path))

    events_path = logger.events_to_csv(f"{test_dir}/test_events.csv")
    test("Events CSV created", os.path.exists(events_path))

    # Verify CSV content
    with open(node_path) as f:
        lines = f.readlines()
        test("Node CSV has header + 4 rows", len(lines) == 5)
        test("Node CSV header is correct",
             "tick,node_id,node_type,capacity,current_load" in lines[0])

    # Cleanup
    os.remove(f"{test_dir}/test_nodes.csv")
    os.remove(f"{test_dir}/test_events.csv")
    os.rmdir(test_dir)


# ── Test 6: Cascade Logic ───────────────────────────────────────────

def test_cascade_logic():
    section("Test 6: Cascade Logic")

    from livegrid.config import SimulationConfig
    from livegrid.engine.engine import SimulationEngine
    from livegrid.grid.grid import Grid
    from livegrid.logging.logger import DataLogger
    from livegrid.models.node import Node, NodeStatus, NodeType
    from livegrid.models.edge import Edge

    # Build a small test grid:  A(cap=100) -- B(cap=50) -- C(cap=50)
    # If A fails with 100MW load, B and C each get 50MW — but B can barely
    # hold it and C might cascade.
    grid = Grid()
    grid.add_node(Node(id="A", node_type=NodeType.GENERATOR, capacity=100, current_load=90))
    grid.add_node(Node(id="B", node_type=NodeType.SUBSTATION, capacity=80, current_load=40))
    grid.add_node(Node(id="C", node_type=NodeType.DISTRIBUTION, capacity=50, current_load=45))

    grid.add_edge(Edge(source="A", target="B"))
    grid.add_edge(Edge(source="B", target="C"))

    logger = DataLogger()
    config = SimulationConfig(
        total_ticks=1,
        load_fluctuation_pct=0.0,  # No noise — deterministic
        random_seed=42,
    )

    engine = SimulationEngine(grid=grid, logger=logger, config=config)

    # Manually fail node A and trigger cascade
    node_a = grid.get_node("A")
    node_a.fail()
    engine._cascade("A", tick=1, depth=0)

    # A had 90MW load. After fail, its load is set to 0 by cascade.
    # Only neighbor of A is B. B gets all 90MW.
    # B was at 40MW, now at 130MW with cap=80 → B should fail too.
    node_b = grid.get_node("B")
    test("Node A load zeroed after cascade", node_a.current_load == 0.0)
    test("Node B failed from cascade", node_b.status == NodeStatus.FAILED,
         f"B status={node_b.status}, load={node_b.current_load}")

    # B's load (130MW) redistributes to its active neighbors.
    # B's neighbors are A (failed) and C (active).
    # So C gets all of B's load → C also exceeds capacity and cascade-fails.
    node_c = grid.get_node("C")
    test("Node C also cascade-failed", node_c.status == NodeStatus.FAILED,
         f"C status={node_c.status}, load={node_c.current_load}")

    # Check events were logged
    cascade_events = [e for e in logger.get_events() if e["event_type"] == "CASCADE"]
    test("Cascade events were logged", len(cascade_events) >= 2,
         f"got {len(cascade_events)} cascade events")


# ── Test 7: Full Simulation Run ─────────────────────────────────────

def test_full_simulation():
    section("Test 7: Full Simulation (end-to-end)")

    from livegrid.config import SimulationConfig
    from livegrid.engine.engine import SimulationEngine
    from livegrid.grid.grid import Grid
    from livegrid.logging.logger import DataLogger
    from livegrid.scenarios.heatwave import HeatWaveScenario
    from livegrid.scenarios.sudden_failure import SuddenFailureScenario

    grid = Grid.build_sample_grid()
    logger = DataLogger()
    config = SimulationConfig(total_ticks=50, random_seed=42, output_dir="output/test_run")
    scenarios = [
        HeatWaveScenario(start_tick=10, duration=10, load_increase_pct=0.2),
        SuddenFailureScenario(target_node_id="DIST-5", trigger_tick=30),
    ]

    engine = SimulationEngine(grid=grid, logger=logger, scenarios=scenarios, config=config)

    # This should complete without exceptions
    try:
        engine.run(ticks=50)
        test("Full simulation completed without errors", True)
    except Exception as e:
        test("Full simulation completed without errors", False, str(e))

    # Verify data was logged
    test("Node data logged (should be 510 rows = 10 nodes × 51 ticks)",
         logger.node_row_count == 510,
         f"got {logger.node_row_count}")
    test("Events were logged", logger.event_count > 0,
         f"got {logger.event_count} events")

    # Export and verify
    os.makedirs(config.output_dir, exist_ok=True)
    path = logger.to_csv(f"{config.output_dir}/test_nodes.csv")
    test("CSV file was written", os.path.exists(path))

    # Cleanup
    import shutil
    shutil.rmtree(config.output_dir, ignore_errors=True)


# ── Test 8: Config Validation ───────────────────────────────────────

def test_config_validation():
    section("Test 8: Config Validation")

    from livegrid.config import SimulationConfig

    # Valid config
    try:
        c = SimulationConfig(total_ticks=50)
        test("Valid config accepted", True)
    except ValueError:
        test("Valid config accepted", False)

    # Invalid: negative ticks
    try:
        SimulationConfig(total_ticks=-1)
        test("Negative ticks rejected", False)
    except ValueError:
        test("Negative ticks rejected", True)

    # Invalid: warning >= failure threshold
    try:
        SimulationConfig(warning_threshold=1.0, failure_threshold=0.9)
        test("Warning >= failure rejected", False)
    except ValueError:
        test("Warning >= failure rejected", True)

    # Invalid: fluctuation > 1
    try:
        SimulationConfig(load_fluctuation_pct=1.5)
        test("Fluctuation > 1 rejected", False)
    except ValueError:
        test("Fluctuation > 1 rejected", True)


# ── Runner ───────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 50)
    print("  ⚡ LiveGrid Test Suite")
    print("=" * 50)

    test_node_model()
    test_edge_model()
    test_grid_graph()
    test_scenarios()
    test_data_logger()
    test_cascade_logic()
    test_full_simulation()
    test_config_validation()

    print(f"\n{'=' * 50}")
    print(f"  Results: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print(f"{'=' * 50}\n")

    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
