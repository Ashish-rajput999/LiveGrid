"""
Power grid graph model.

Manages the collection of nodes and edges that form the grid topology.
Provides graph operations (neighbor lookup, active filtering) and a
factory method for building sample test grids.
"""

from __future__ import annotations

from typing import Optional

from livegrid.models.node import Node, NodeStatus, NodeType
from livegrid.models.edge import Edge


class Grid:
    """
    Graph representation of a power grid.

    Nodes are stored in a dict keyed by ID for O(1) lookup.
    Edges are stored in a list and also indexed by node for fast
    neighbor traversal. The neighbor relationship is bidirectional —
    adding an edge A→B also makes B a neighbor of A.

    Usage:
        grid = Grid.build_sample_grid()
        node = grid.get_node("GEN-1")
        neighbors = grid.get_active_neighbors("GEN-1")
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []

    # --- Node operations ---

    def add_node(self, node: Node) -> None:
        """
        Add a node to the grid.

        Raises:
            ValueError: If a node with the same ID already exists.
        """
        if node.id in self._nodes:
            raise ValueError(f"Node '{node.id}' already exists in the grid")
        self._nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID, or None if not found."""
        return self._nodes.get(node_id)

    def all_nodes(self) -> list[Node]:
        """Return all nodes in the grid."""
        return list(self._nodes.values())

    @property
    def node_count(self) -> int:
        """Number of nodes in the grid."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in the grid."""
        return len(self._edges)

    # --- Edge operations ---

    def add_edge(self, edge: Edge) -> None:
        """
        Add an edge (transmission line) to the grid.

        Automatically updates the neighbor lists of both endpoints
        to maintain bidirectional adjacency.

        Raises:
            ValueError: If either endpoint node doesn't exist.
        """
        if edge.source not in self._nodes:
            raise ValueError(f"Source node '{edge.source}' not found in grid")
        if edge.target not in self._nodes:
            raise ValueError(f"Target node '{edge.target}' not found in grid")

        self._edges.append(edge)

        # Maintain bidirectional adjacency
        source_node = self._nodes[edge.source]
        target_node = self._nodes[edge.target]

        if edge.target not in source_node.neighbors:
            source_node.neighbors.append(edge.target)
        if edge.source not in target_node.neighbors:
            target_node.neighbors.append(edge.source)

    # --- Graph queries ---

    def get_neighbors(self, node_id: str) -> list[Node]:
        """
        Get all neighbor nodes of a given node.

        Args:
            node_id: The ID of the node to find neighbors for.

        Returns:
            List of neighbor Node objects (regardless of status).
        """
        node = self._nodes.get(node_id)
        if node is None:
            return []
        return [
            self._nodes[nid]
            for nid in node.neighbors
            if nid in self._nodes
        ]

    def get_active_neighbors(self, node_id: str) -> list[Node]:
        """
        Get only non-FAILED neighbors of a given node.

        Used during cascade propagation to find nodes that can
        absorb redistributed load.

        Args:
            node_id: The ID of the node to find active neighbors for.

        Returns:
            List of neighbor Node objects with status != FAILED.
        """
        return [
            n for n in self.get_neighbors(node_id)
            if n.status != NodeStatus.FAILED
        ]

    def get_failed_nodes(self) -> list[Node]:
        """Return all nodes currently in FAILED status."""
        return [n for n in self._nodes.values() if n.status == NodeStatus.FAILED]

    def get_operational_nodes(self) -> list[Node]:
        """Return all nodes NOT in FAILED status."""
        return [n for n in self._nodes.values() if n.status != NodeStatus.FAILED]

    # --- Display ---

    def summary(self) -> str:
        """Generate a human-readable summary of the grid state."""
        lines = [
            "=" * 70,
            f"  GRID STATUS — {self.node_count} nodes, {self.edge_count} edges",
            "=" * 70,
        ]

        ok_count = sum(1 for n in self._nodes.values() if n.status == NodeStatus.OK)
        warn_count = sum(1 for n in self._nodes.values() if n.status == NodeStatus.WARNING)
        fail_count = sum(1 for n in self._nodes.values() if n.status == NodeStatus.FAILED)

        lines.append(f"  Status: {ok_count} OK | {warn_count} WARNING | {fail_count} FAILED")
        lines.append("-" * 70)

        for node in sorted(self._nodes.values(), key=lambda n: n.id):
            status_icon = {"OK": "🟢", "WARNING": "🟡", "FAILED": "🔴"}.get(
                node.status.value, "⚪"
            )
            lines.append(
                f"  {status_icon} {node.id:<12} [{node.node_type}] "
                f"Load: {node.current_load:>7.1f}/{node.capacity:.0f} MW "
                f"({node.load_ratio:.0%})  "
                f"V={node.voltage:.1f}kV  f={node.frequency:.2f}Hz"
            )

        lines.append("=" * 70)
        return "\n".join(lines)

    # --- Factory ---

    @classmethod
    def build_sample_grid(cls) -> "Grid":
        """
        Build a realistic 10-node sample grid.

        Topology:
            2 Generators (GEN-1, GEN-2) — high capacity, backbone
            3 Substations (SUB-1, SUB-2, SUB-3) — medium capacity, hub nodes
            5 Distribution (DIST-1..5) — lower capacity, leaf-ish nodes

        Network structure (partially meshed):

            GEN-1 ──── SUB-1 ──── DIST-1
              │          │  ╲        │
              │          │   ╲     DIST-2
              │          │    ╲
            GEN-2 ──── SUB-2 ── SUB-3
                         │        │  ╲
                       DIST-3   DIST-4  DIST-5

        This topology allows for cascading failures: if SUB-1 fails, load
        redistributes to GEN-1, GEN-2, SUB-2, DIST-1, and SUB-3.
        """
        grid = cls()

        # --- Generators (high capacity, moderate initial load) ---
        grid.add_node(Node(
            id="GEN-1",
            node_type=NodeType.GENERATOR,
            capacity=500.0,
            current_load=280.0,
            voltage=230.0,
            frequency=50.0,
        ))
        grid.add_node(Node(
            id="GEN-2",
            node_type=NodeType.GENERATOR,
            capacity=500.0,
            current_load=310.0,
            voltage=230.0,
            frequency=50.0,
        ))

        # --- Substations (medium capacity, moderate load) ---
        grid.add_node(Node(
            id="SUB-1",
            node_type=NodeType.SUBSTATION,
            capacity=300.0,
            current_load=195.0,
            voltage=230.0,
            frequency=50.0,
        ))
        grid.add_node(Node(
            id="SUB-2",
            node_type=NodeType.SUBSTATION,
            capacity=300.0,
            current_load=210.0,
            voltage=230.0,
            frequency=50.0,
        ))
        grid.add_node(Node(
            id="SUB-3",
            node_type=NodeType.SUBSTATION,
            capacity=300.0,
            current_load=180.0,
            voltage=230.0,
            frequency=50.0,
        ))

        # --- Distribution nodes (lower capacity, closer to full) ---
        grid.add_node(Node(
            id="DIST-1",
            node_type=NodeType.DISTRIBUTION,
            capacity=150.0,
            current_load=105.0,
            voltage=230.0,
            frequency=50.0,
        ))
        grid.add_node(Node(
            id="DIST-2",
            node_type=NodeType.DISTRIBUTION,
            capacity=150.0,
            current_load=120.0,
            voltage=230.0,
            frequency=50.0,
        ))
        grid.add_node(Node(
            id="DIST-3",
            node_type=NodeType.DISTRIBUTION,
            capacity=150.0,
            current_load=95.0,
            voltage=230.0,
            frequency=50.0,
        ))
        grid.add_node(Node(
            id="DIST-4",
            node_type=NodeType.DISTRIBUTION,
            capacity=150.0,
            current_load=110.0,
            voltage=230.0,
            frequency=50.0,
        ))
        grid.add_node(Node(
            id="DIST-5",
            node_type=NodeType.DISTRIBUTION,
            capacity=150.0,
            current_load=100.0,
            voltage=230.0,
            frequency=50.0,
        ))

        # --- Edges (transmission lines) ---
        # Generator connections to substations
        grid.add_edge(Edge(source="GEN-1", target="SUB-1", capacity=400.0))
        grid.add_edge(Edge(source="GEN-1", target="GEN-2", capacity=600.0))
        grid.add_edge(Edge(source="GEN-2", target="SUB-2", capacity=400.0))

        # Substation interconnections (mesh backbone)
        grid.add_edge(Edge(source="SUB-1", target="SUB-2", capacity=350.0))
        grid.add_edge(Edge(source="SUB-1", target="SUB-3", capacity=350.0))
        grid.add_edge(Edge(source="SUB-2", target="SUB-3", capacity=350.0))

        # Distribution connections
        grid.add_edge(Edge(source="SUB-1", target="DIST-1", capacity=200.0))
        grid.add_edge(Edge(source="DIST-1", target="DIST-2", capacity=150.0))
        grid.add_edge(Edge(source="SUB-2", target="DIST-3", capacity=200.0))
        grid.add_edge(Edge(source="SUB-3", target="DIST-4", capacity=200.0))
        grid.add_edge(Edge(source="SUB-3", target="DIST-5", capacity=200.0))

        return grid
