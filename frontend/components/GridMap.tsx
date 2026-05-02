"use client";

import React, { useRef, useEffect, useCallback, useState } from "react";
import * as d3 from "d3";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

// ── Types ────────────────────────────────────────────────────────────

export interface GridNode {
  id: string;
  type: string;
  capacity: number;
  current_load: number;
  load_ratio: number;
  voltage_kv: number;
  frequency_hz: number;
  status: string;
  risk_score: number;
}

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  type: string;
  capacity: number;
  current_load: number;
  load_ratio: number;
  voltage_kv: number;
  frequency_hz: number;
  status: string;
  risk_score: number;
  radius: number;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  source: string | SimNode;
  target: string | SimNode;
}

interface CascadeStep {
  step: number;
  node_id: string;
  reason: string;
}

interface CascadeResult {
  triggered_node: string;
  cascade_sequence: CascadeStep[];
  total_failed: number;
  survived: string[];
  estimated_customers_affected: number;
}

interface GridMapProps {
  nodes: GridNode[];
  edges: [string, string][];
  onNodeClick?: (node: GridNode | null) => void;
  activeScenario?: string | null;
}

// ── Helpers ──────────────────────────────────────────────────────────

function getNodeRadius(type: string): number {
  switch (type) {
    case "GENERATOR":
      return 28;
    case "SUBSTATION":
      return 22;
    case "DISTRIBUTION":
      return 16;
    default:
      return 16;
  }
}

function getNodeColor(status: string, risk: number): string {
  if (status === "FAILED") return "#4b5563"; // grey
  if (risk > 0.7) return "#ef4444"; // red
  if (risk > 0.4) return "#eab308"; // yellow
  return "#22c55e"; // green
}

function getNodeGlow(status: string, risk: number): string {
  if (status === "FAILED") return "none";
  if (risk > 0.7) return "0 0 20px rgba(239,68,68,0.6), 0 0 40px rgba(239,68,68,0.3)";
  if (risk > 0.4) return "0 0 14px rgba(234,179,8,0.4)";
  return "0 0 10px rgba(34,197,94,0.3)";
}

function getTypeIcon(type: string): string {
  switch (type) {
    case "GENERATOR":
      return "⚡";
    case "SUBSTATION":
      return "🔌";
    case "DISTRIBUTION":
      return "🏠";
    default:
      return "●";
  }
}

// ── Component ────────────────────────────────────────────────────────

export default function GridMap({ nodes, edges, onNodeClick, activeScenario }: GridMapProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<d3.Simulation<SimNode, SimLink> | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const initializedRef = useRef(false);

  // ── Phase 4: Cascade Simulator State ──────────────────────────────
  const [confirmPopup, setConfirmPopup] = useState<{ nodeId: string; x: number; y: number } | null>(null);
  const [simLoading, setSimLoading] = useState(false);
  const [simResult, setSimResult] = useState<CascadeResult | null>(null);
  // Which node IDs are currently highlighted by the cascade animation
  const [simHighlighted, setSimHighlighted] = useState<Set<string>>(new Set());
  const simAnimTimers = useRef<ReturnType<typeof setTimeout>[]>([]);

  // ── Phase 4: Explain API Cache ─────────────────────────────────────
  // {nodeId: primaryDriver string}
  const explainCache = useRef<Map<string, string>>(new Map());

  // Resize observer
  useEffect(() => {
    const container = svgRef.current?.parentElement;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setDimensions({ width, height });
        }
      }
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Initialize D3 force simulation once
  useEffect(() => {
    if (!svgRef.current || nodes.length === 0 || initializedRef.current) return;

    const svg = d3.select(svgRef.current);
    const { width, height } = dimensions;

    // Clear any defaults
    svg.selectAll("*").remove();

    // Defs for glow filter
    const defs = svg.append("defs");

    const filter = defs
      .append("filter")
      .attr("id", "glow")
      .attr("x", "-50%")
      .attr("y", "-50%")
      .attr("width", "200%")
      .attr("height", "200%");
    filter
      .append("feGaussianBlur")
      .attr("stdDeviation", "4")
      .attr("result", "coloredBlur");
    const feMerge = filter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "coloredBlur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // Build simulation data
    const simNodes: SimNode[] = nodes.map((n) => ({
      ...n,
      radius: getNodeRadius(n.type),
      x: width / 2 + (Math.random() - 0.5) * 200,
      y: height / 2 + (Math.random() - 0.5) * 200,
    }));

    const simLinks: SimLink[] = edges.map(([s, t]) => ({
      source: s,
      target: t,
    }));

    // Create groups
    svg.append("g").attr("class", "edges-group");
    svg.append("g").attr("class", "nodes-group");
    svg.append("g").attr("class", "labels-group");

    // Create force simulation
    const simulation = d3
      .forceSimulation<SimNode>(simNodes)
      .force(
        "link",
        d3
          .forceLink<SimNode, SimLink>(simLinks)
          .id((d) => d.id)
          .distance(120)
      )
      .force("charge", d3.forceManyBody().strength(-500))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force(
        "collision",
        d3.forceCollide<SimNode>().radius((d) => d.radius + 15)
      )
      .force("x", d3.forceX(width / 2).strength(0.05))
      .force("y", d3.forceY(height / 2).strength(0.05));

    // Draw edges
    const edgeSelection = svg
      .select(".edges-group")
      .selectAll<SVGLineElement, SimLink>("line")
      .data(simLinks)
      .join("line")
      .attr("stroke", "#2a2e3a")
      .attr("stroke-width", 2)
      .attr("stroke-opacity", 0.6);

    // Draw pulse rings (behind nodes)
    const pulseSelection = svg
      .select(".nodes-group")
      .selectAll<SVGCircleElement, SimNode>("circle.pulse-ring")
      .data(simNodes, (d) => d.id)
      .join("circle")
      .attr("class", "pulse-ring")
      .attr("r", (d) => d.radius + 8)
      .attr("fill", "none")
      .attr("stroke", "transparent")
      .attr("stroke-width", 2);

    // Draw nodes
    const nodeSelection = svg
      .select(".nodes-group")
      .selectAll<SVGCircleElement, SimNode>("circle.node-circle")
      .data(simNodes, (d) => d.id)
      .join("circle")
      .attr("class", "node-circle")
      .attr("r", (d) => d.radius)
      .attr("fill", (d) => getNodeColor(d.status, d.risk_score))
      .attr("stroke", "#1a1d27")
      .attr("stroke-width", 2.5)
      .attr("cursor", "pointer")
      .attr("filter", "url(#glow)");

    // Draw labels
    const labelSelection = svg
      .select(".labels-group")
      .selectAll<SVGTextElement, SimNode>("text")
      .data(simNodes, (d) => d.id)
      .join("text")
      .attr("text-anchor", "middle")
      .attr("dy", (d) => d.radius + 16)
      .attr("fill", "#8b8fa3")
      .attr("font-size", "11px")
      .attr("font-weight", "500")
      .attr("font-family", "'Inter', sans-serif")
      .text((d) => d.id);

    // Hover handlers
    nodeSelection
      .on("mouseover", function (event: MouseEvent, d: SimNode) {
        const tooltip = tooltipRef.current;
        if (!tooltip) return;

        const cached = explainCache.current.get(d.id);
        const driverLine = cached
          ? `<div style="color:#8b8fa3;margin-top:4px;font-style:italic;font-size:11px;">⚡ ${cached}</div>`
          : "";

        tooltip.innerHTML = `
          <div style="font-weight:600;font-size:13px;margin-bottom:4px;">${getTypeIcon(d.type)} ${d.id}</div>
          <div style="color:#8b8fa3;">Type: <span style="color:#e4e6ed;">${d.type}</span></div>
          <div style="color:#8b8fa3;">Load: <span style="color:#e4e6ed;">${(d.load_ratio * 100).toFixed(1)}%</span> (${d.current_load.toFixed(0)}/${d.capacity}MW)</div>
          <div style="color:#8b8fa3;">Voltage: <span style="color:#e4e6ed;">${d.voltage_kv.toFixed(1)} kV</span></div>
          <div style="color:#8b8fa3;">Frequency: <span style="color:#e4e6ed;">${d.frequency_hz.toFixed(2)} Hz</span></div>
          <div style="color:#8b8fa3;">Risk: <span style="color:${getNodeColor(d.status, d.risk_score)};font-weight:600;">${d.risk_score.toFixed(3)}</span></div>
          <div style="color:#8b8fa3;">Status: <span style="color:${d.status === 'FAILED' ? '#ef4444' : d.status === 'WARNING' ? '#eab308' : '#22c55e'};font-weight:600;">${d.status}</span></div>
          ${driverLine}
        `;
        tooltip.classList.add("visible");

        // Fetch explain if not cached (fire-and-forget, updates tooltip on next hover)
        if (!cached && d.status !== "FAILED") {
          fetch(`${API_BASE}/api/explain/${d.id}`)
            .then((r) => r.ok ? r.json() : null)
            .then((data) => {
              if (data?.explanation?.primary_driver) {
                explainCache.current.set(d.id, data.explanation.primary_driver);
              }
            })
            .catch(() => { });
        }
      })
      .on("mousemove", function (event: MouseEvent) {
        const tooltip = tooltipRef.current;
        if (!tooltip) return;
        const svgRect = svgRef.current?.getBoundingClientRect();
        if (!svgRect) return;
        tooltip.style.left = `${event.clientX - svgRect.left + 15}px`;
        tooltip.style.top = `${event.clientY - svgRect.top - 10}px`;
      })
      .on("mouseout", function () {
        const tooltip = tooltipRef.current;
        if (tooltip) tooltip.classList.remove("visible");
      })
      .on("click", function (event: MouseEvent, d: SimNode) {
        // If FAILED, just select; otherwise show cascade confirm popup
        if (d.status === "FAILED") {
          setSelectedNodeId((prev) => (prev === d.id ? null : d.id));
          const gridNode = nodes.find((n) => n.id === d.id) || null;
          onNodeClick?.(gridNode);
          return;
        }
        // Clear any existing simulation
        setSimResult(null);
        setSimHighlighted(new Set());
        simAnimTimers.current.forEach(clearTimeout);
        simAnimTimers.current = [];

        const svgRect = svgRef.current?.getBoundingClientRect();
        const px = svgRect ? event.clientX - svgRect.left : event.clientX;
        const py = svgRect ? event.clientY - svgRect.top : event.clientY;
        setConfirmPopup({ nodeId: d.id, x: px, y: py });

        setSelectedNodeId((prev) => (prev === d.id ? null : d.id));
        const gridNode = nodes.find((n) => n.id === d.id) || null;
        onNodeClick?.(gridNode);
      });

    // Drag behavior
    const drag = d3
      .drag<SVGCircleElement, SimNode>()
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });

    nodeSelection.call(drag);

    // Tick handler
    simulation.on("tick", () => {
      edgeSelection
        .attr("x1", (d) => (d.source as SimNode).x!)
        .attr("y1", (d) => (d.source as SimNode).y!)
        .attr("x2", (d) => (d.target as SimNode).x!)
        .attr("y2", (d) => (d.target as SimNode).y!);

      nodeSelection.attr("cx", (d) => d.x!).attr("cy", (d) => d.y!);
      pulseSelection.attr("cx", (d) => d.x!).attr("cy", (d) => d.y!);
      labelSelection.attr("x", (d) => d.x!).attr("y", (d) => d.y!);
    });

    simulationRef.current = simulation;
    initializedRef.current = true;

    return () => {
      simulation.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes.length > 0, dimensions.width, dimensions.height]);

  // Update node visuals on data change (D3 update pattern — no re-create)
  useEffect(() => {
    if (!svgRef.current || !initializedRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    const simulation = simulationRef.current;
    if (!simulation) return;

    // Update node data on the simulation nodes
    const simNodes = simulation.nodes();
    const nodeMap = new Map(nodes.map((n) => [n.id, n]));

    simNodes.forEach((sn) => {
      const fresh = nodeMap.get(sn.id);
      if (fresh) {
        sn.status = fresh.status;
        sn.risk_score = fresh.risk_score;
        sn.load_ratio = fresh.load_ratio;
        sn.current_load = fresh.current_load;
        sn.voltage_kv = fresh.voltage_kv;
        sn.frequency_hz = fresh.frequency_hz;
      }
    });

    // Update visual properties with transitions
    // If simulation is active, overlay sim colours on top of live colours
    svg
      .select(".nodes-group")
      .selectAll<SVGCircleElement, SimNode>("circle.node-circle")
      .data(simNodes, (d) => d.id)
      .transition()
      .duration(400)
      .attr("fill", (d) => {
        if (simHighlighted.has(d.id)) return "#7f1d1d"; // dark red for sim cascade
        return getNodeColor(d.status, d.risk_score);
      })
      .attr("r", (d) => (d.status === "FAILED" ? d.radius * 0.7 : d.radius))
      .attr("opacity", (d) => (d.status === "FAILED" ? 0.5 : 1));

    // Update pulse rings
    svg
      .select(".nodes-group")
      .selectAll<SVGCircleElement, SimNode>("circle.pulse-ring")
      .data(simNodes, (d) => d.id)
      .attr("stroke", (d) =>
        d.risk_score > 0.7 && d.status !== "FAILED"
          ? getNodeColor(d.status, d.risk_score)
          : "transparent"
      )
      .attr("class", (d) =>
        d.risk_score > 0.7 && d.status !== "FAILED"
          ? "pulse-ring node-pulse"
          : "pulse-ring"
      );

    // Update edge visuals based on connected node loads
    const edgesGroup = svg.select(".edges-group");
    edgesGroup
      .selectAll<SVGLineElement, SimLink>("line")
      .transition()
      .duration(400)
      .attr("stroke", (d) => {
        const srcNode = d.source as SimNode;
        const tgtNode = d.target as SimNode;
        if (srcNode.status === "FAILED" || tgtNode.status === "FAILED") {
          return "#1f2129";
        }
        const avgRisk = (srcNode.risk_score + tgtNode.risk_score) / 2;
        if (avgRisk > 0.7) return "rgba(239,68,68,0.5)";
        if (avgRisk > 0.4) return "rgba(234,179,8,0.4)";
        return "rgba(34,197,94,0.25)";
      })
      .attr("stroke-width", (d) => {
        const srcNode = d.source as SimNode;
        const tgtNode = d.target as SimNode;
        if (srcNode.status === "FAILED" || tgtNode.status === "FAILED") return 1;
        const avgLoad = (srcNode.load_ratio + tgtNode.load_ratio) / 2;
        return Math.max(1.5, Math.min(6, avgLoad * 6));
      })
      .attr("stroke-opacity", (d) => {
        const srcNode = d.source as SimNode;
        const tgtNode = d.target as SimNode;
        return srcNode.status === "FAILED" || tgtNode.status === "FAILED"
          ? 0.2
          : 0.7;
      });

    // Highlight selected node
    svg
      .select(".nodes-group")
      .selectAll<SVGCircleElement, SimNode>("circle.node-circle")
      .attr("stroke", (d) => {
        if (simHighlighted.has(d.id)) return "#ef4444";
        return d.id === selectedNodeId ? "#6366f1" : "#1a1d27";
      })
      .attr("stroke-width", (d) => {
        if (simHighlighted.has(d.id)) return 4;
        return d.id === selectedNodeId ? 3.5 : 2.5;
      });
  }, [nodes, selectedNodeId, simHighlighted]);

  // ── Cascade simulation handler ─────────────────────────────────────
  const runCascadeSimulation = useCallback(async (nodeId: string) => {
    setSimLoading(true);
    setConfirmPopup(null);
    try {
      const res = await fetch(`${API_BASE}/api/simulate-failure`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_id: nodeId }),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(`Simulation error: ${err.detail}`);
        return;
      }
      const result: CascadeResult = await res.json();
      setSimResult(result);

      // Animate cascade steps 600ms apart
      simAnimTimers.current.forEach(clearTimeout);
      simAnimTimers.current = [];
      const highlighted = new Set<string>();

      result.cascade_sequence.forEach((step, i) => {
        const t = setTimeout(() => {
          highlighted.add(step.node_id);
          setSimHighlighted(new Set(highlighted));
        }, i * 600);
        simAnimTimers.current.push(t);
      });
    } catch (e) {
      alert("Failed to connect to simulation API");
    } finally {
      setSimLoading(false);
    }
  }, []);

  const clearSimulation = useCallback(() => {
    simAnimTimers.current.forEach(clearTimeout);
    simAnimTimers.current = [];
    setSimResult(null);
    setSimHighlighted(new Set());
    setConfirmPopup(null);
  }, []);

  // Re-center on resize
  useEffect(() => {
    const simulation = simulationRef.current;
    if (!simulation) return;

    simulation
      .force("center", d3.forceCenter(dimensions.width / 2, dimensions.height / 2))
      .force("x", d3.forceX(dimensions.width / 2).strength(0.05))
      .force("y", d3.forceY(dimensions.height / 2).strength(0.05));

    simulation.alpha(0.1).restart();
  }, [dimensions]);

  // Selected node detail panel
  const selectedNode = nodes.find((n) => n.id === selectedNodeId);

  return (
    <div id="grid-map" className="relative w-full h-full overflow-hidden">
      {/* SVG Canvas */}
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        className="w-full h-full"
        onClick={(e) => {
          // Close popup if clicking on SVG background
          if ((e.target as SVGElement).tagName === "svg" || (e.target as SVGElement).tagName === "g") {
            setConfirmPopup(null);
          }
        }}
      />

      {/* Tooltip */}
      <div ref={tooltipRef} className="node-tooltip" />

      {/* ── Cascade Confirm Popup ── */}
      {confirmPopup && !simResult && (
        <div
          className="absolute z-50 bg-[#0f1117]/95 border border-red-500/40 rounded-xl px-4 py-3 shadow-2xl backdrop-blur-md"
          style={{ left: Math.min(confirmPopup.x, dimensions.width - 220), top: Math.max(10, confirmPopup.y - 100) }}
        >
          <p className="text-xs text-[var(--color-text-secondary)] mb-1">What-If Simulator</p>
          <p className="text-sm font-semibold text-white mb-3">
            Simulate failure of <span className="text-red-400">{confirmPopup.nodeId}</span>?
          </p>
          <div className="flex gap-2">
            <button
              id={`sim-confirm-${confirmPopup.nodeId}`}
              disabled={simLoading}
              onClick={() => runCascadeSimulation(confirmPopup.nodeId)}
              className="flex-1 bg-red-600 hover:bg-red-500 disabled:opacity-50 text-white text-xs font-semibold px-3 py-1.5 rounded-lg transition-colors"
            >
              {simLoading ? "Running..." : "⚡ Simulate"}
            </button>
            <button
              onClick={() => setConfirmPopup(null)}
              className="flex-1 bg-[var(--color-surface)] hover:bg-[var(--color-surface-hover)] text-[var(--color-text-secondary)] text-xs px-3 py-1.5 rounded-lg transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* ── Cascade Result Overlay ── */}
      {simResult && (
        <div className="absolute bottom-4 right-4 left-4 md:left-auto md:w-96 bg-[#0f1117]/96 border border-red-500/50 rounded-xl p-4 shadow-2xl backdrop-blur-md z-40">
          <div className="flex items-center justify-between mb-3">
            <div>
              <p className="text-xs text-red-400 font-semibold uppercase tracking-widest">⚠ Cascade Simulation</p>
              <p className="text-sm font-bold text-white mt-0.5">
                {simResult.total_failed} node{simResult.total_failed !== 1 ? "s" : ""} would fail
              </p>
            </div>
            <button
              id="clear-simulation-btn"
              onClick={clearSimulation}
              className="text-xs text-[var(--color-text-secondary)] hover:text-white bg-[var(--color-surface)] px-3 py-1.5 rounded-lg transition-colors"
            >
              Clear
            </button>
          </div>

          {/* Cascade sequence */}
          <div className="space-y-1 mb-3 max-h-40 overflow-y-auto">
            {simResult.cascade_sequence.map((step) => (
              <div
                key={step.step}
                className={`flex items-start gap-2 text-xs transition-opacity duration-300 ${simHighlighted.has(step.node_id) ? "opacity-100" : "opacity-30"
                  }`}
              >
                <span className="text-red-400 font-mono shrink-0">{step.step}.</span>
                <span className="font-semibold text-white shrink-0">{step.node_id}</span>
                <span className="text-[var(--color-text-secondary)]">{step.reason}</span>
              </div>
            ))}
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 gap-2 pt-3 border-t border-[var(--color-border)]">
            <div className="bg-red-950/40 rounded-lg p-2 text-center">
              <div className="text-xs text-red-400 font-semibold">{simResult.total_failed} Failed</div>
              <div className="text-[10px] text-[var(--color-text-secondary)] mt-0.5">
                {simResult.survived.length} survived
              </div>
            </div>
            <div className="bg-orange-950/40 rounded-lg p-2 text-center">
              <div className="text-xs text-orange-400 font-semibold">
                ~{(simResult.estimated_customers_affected / 1000).toFixed(0)}K
              </div>
              <div className="text-[10px] text-[var(--color-text-secondary)] mt-0.5">customers affected</div>
            </div>
          </div>
        </div>
      )}

      {/* Selected Node Detail Panel */}
      {selectedNode && (
        <div className="absolute bottom-4 left-4 bg-[var(--color-surface)]/95 backdrop-blur-md border border-[var(--color-border)] rounded-xl p-4 w-72 shadow-2xl">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <span className="text-lg">{getTypeIcon(selectedNode.type)}</span>
              <h3 className="text-sm font-bold text-white">{selectedNode.id}</h3>
            </div>
            <button
              onClick={() => {
                setSelectedNodeId(null);
                onNodeClick?.(null);
              }}
              className="text-[var(--color-text-secondary)] hover:text-white text-xs"
            >
              ✕
            </button>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-[var(--color-background)]/60 rounded-lg p-2">
              <div className="text-[var(--color-text-secondary)]">Status</div>
              <div
                className="font-semibold"
                style={{
                  color: getNodeColor(selectedNode.status, selectedNode.risk_score),
                }}
              >
                {selectedNode.status}
              </div>
            </div>
            <div className="bg-[var(--color-background)]/60 rounded-lg p-2">
              <div className="text-[var(--color-text-secondary)]">Risk</div>
              <div
                className="font-semibold"
                style={{
                  color: getNodeColor(selectedNode.status, selectedNode.risk_score),
                }}
              >
                {selectedNode.risk_score.toFixed(3)}
              </div>
            </div>
            <div className="bg-[var(--color-background)]/60 rounded-lg p-2">
              <div className="text-[var(--color-text-secondary)]">Load</div>
              <div className="font-semibold text-white">
                {(selectedNode.load_ratio * 100).toFixed(1)}%
              </div>
            </div>
            <div className="bg-[var(--color-background)]/60 rounded-lg p-2">
              <div className="text-[var(--color-text-secondary)]">Power</div>
              <div className="font-semibold text-white">
                {selectedNode.current_load.toFixed(0)}/{selectedNode.capacity}MW
              </div>
            </div>
            <div className="bg-[var(--color-background)]/60 rounded-lg p-2">
              <div className="text-[var(--color-text-secondary)]">Voltage</div>
              <div className="font-semibold text-white">
                {selectedNode.voltage_kv.toFixed(1)} kV
              </div>
            </div>
            <div className="bg-[var(--color-background)]/60 rounded-lg p-2">
              <div className="text-[var(--color-text-secondary)]">Freq</div>
              <div className="font-semibold text-white">
                {selectedNode.frequency_hz.toFixed(2)} Hz
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="absolute top-4 left-4 bg-[var(--color-surface)]/80 backdrop-blur-sm border border-[var(--color-border)] rounded-lg px-3 py-2 text-xs text-[var(--color-text-secondary)]">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-green-500" />
            Low
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-yellow-500" />
            Medium
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
            High
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-gray-500" />
            Failed
          </span>
        </div>
      </div>
    </div>
  );
}
