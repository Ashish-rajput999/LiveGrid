"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import StatsBar from "@/components/StatsBar";
import AlertFeed, { AlertItem } from "@/components/AlertFeed";
import GridMap, { GridNode } from "@/components/GridMap";

// ── Types ────────────────────────────────────────────────────────────

interface GridSnapshot {
  tick: number;
  nodes: GridNode[];
  edges: [string, string][];
  active_scenario: string | null;
  failed_count: number;
  model_type: string;
}

// ── Constants ────────────────────────────────────────────────────────

const WS_URL = "ws://localhost:8000/ws/live";
const RECONNECT_DELAY = 3000;
const MAX_ALERTS = 20;

// ── Page Component ───────────────────────────────────────────────────

export default function Dashboard() {
  const [gridData, setGridData] = useState<GridSnapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [selectedNode, setSelectedNode] = useState<GridNode | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const prevNodesRef = useRef<Map<string, GridNode>>(new Map());
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const alertIdCounter = useRef(0);

  // ── Alert Generation ───────────────────────────────────────────────

  const generateAlerts = useCallback(
    (newNodes: GridNode[], scenario: string | null) => {
      const prevMap = prevNodesRef.current;
      const newAlerts: AlertItem[] = [];
      const now = new Date().toLocaleTimeString("en-US", {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });

      for (const node of newNodes) {
        const prev = prevMap.get(node.id);

        // Node just failed
        if (prev && prev.status !== "FAILED" && node.status === "FAILED") {
          newAlerts.push({
            id: `alert-${++alertIdCounter.current}`,
            time: now,
            nodeId: node.id,
            level: "FAILED",
            risk: node.risk_score,
            message: `Node FAILED — load was ${(node.load_ratio * 100).toFixed(0)}%`,
          });
        }

        // Risk crossed HIGH threshold
        if (
          node.status !== "FAILED" &&
          node.risk_score > 0.7 &&
          (!prev || prev.risk_score <= 0.7)
        ) {
          newAlerts.push({
            id: `alert-${++alertIdCounter.current}`,
            time: now,
            nodeId: node.id,
            level: "CRITICAL",
            risk: node.risk_score,
            message: `load at ${(node.load_ratio * 100).toFixed(0)}%${
              scenario ? ` — ${scenario.replace(/_/g, " ")} active` : ""
            }`,
          });
        }

        // Risk crossed MEDIUM threshold
        if (
          node.status !== "FAILED" &&
          node.risk_score > 0.4 &&
          node.risk_score <= 0.7 &&
          (!prev || prev.risk_score <= 0.4)
        ) {
          newAlerts.push({
            id: `alert-${++alertIdCounter.current}`,
            time: now,
            nodeId: node.id,
            level: "WARNING",
            risk: node.risk_score,
            message: `load at ${(node.load_ratio * 100).toFixed(0)}%${
              scenario ? ` — ${scenario.replace(/_/g, " ")} active` : ""
            }`,
          });
        }

        // Node recovered from failure
        if (prev && prev.status === "FAILED" && node.status !== "FAILED") {
          newAlerts.push({
            id: `alert-${++alertIdCounter.current}`,
            time: now,
            nodeId: node.id,
            level: "RECOVERED",
            risk: node.risk_score,
            message: `Node recovered — back online`,
          });
        }
      }

      if (newAlerts.length > 0) {
        setAlerts((prev) => [...newAlerts, ...prev].slice(0, MAX_ALERTS));
      }

      // Update prev map
      const newMap = new Map<string, GridNode>();
      for (const n of newNodes) {
        newMap.set(n.id, n);
      }
      prevNodesRef.current = newMap;
    },
    []
  );

  // ── WebSocket Connection ───────────────────────────────────────────

  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("⚡ WebSocket connected");
      setConnected(true);
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };

    ws.onmessage = (event) => {
      try {
        const data: GridSnapshot = JSON.parse(event.data);
        setGridData(data);
        generateAlerts(data.nodes, data.active_scenario);
      } catch (e) {
        console.error("Failed to parse WebSocket message:", e);
      }
    };

    ws.onclose = () => {
      console.log("🔌 WebSocket disconnected");
      setConnected(false);
      wsRef.current = null;
      // Auto-reconnect
      reconnectTimerRef.current = setTimeout(connectWebSocket, RECONNECT_DELAY);
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      ws.close();
    };
  }, [generateAlerts]);

  useEffect(() => {
    connectWebSocket();

    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);

  // ── Send keepalive ping ────────────────────────────────────────────

  useEffect(() => {
    const interval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send("ping");
      }
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  // ── Computed Stats ─────────────────────────────────────────────────

  const tick = gridData?.tick ?? 0;
  const nodesData = gridData?.nodes ?? [];
  const edges = gridData?.edges ?? [];
  const activeScenario = gridData?.active_scenario ?? null;
  const modelType = gridData?.model_type ?? "LSTM";

  const okCount = nodesData.filter(
    (n) => n.status === "OK"
  ).length;
  const warningCount = nodesData.filter(
    (n) => n.status === "WARNING"
  ).length;
  const failedCount = nodesData.filter(
    (n) => n.status === "FAILED"
  ).length;
  const avgRisk =
    nodesData.length > 0
      ? nodesData.reduce((sum, n) => sum + n.risk_score, 0) / nodesData.length
      : 0;

  // ── Render ─────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* Top Stats Bar */}
      <StatsBar
        tick={tick}
        activeScenario={activeScenario}
        okCount={okCount}
        warningCount={warningCount}
        failedCount={failedCount}
        avgRisk={avgRisk}
        connected={connected}
        modelType={modelType}
      />

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Center — Grid Map */}
        <main className="flex-1 relative bg-[var(--color-background)]">
          {!connected && nodesData.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="text-4xl mb-4 animate-pulse">⚡</div>
                <h2 className="text-lg font-semibold text-white mb-2">
                  Connecting to LiveGrid...
                </h2>
                <p className="text-sm text-[var(--color-text-secondary)]">
                  Make sure the backend is running on port 8000
                </p>
                <div className="mt-4 flex items-center justify-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-accent)] animate-bounce [animation-delay:0ms]" />
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-accent)] animate-bounce [animation-delay:150ms]" />
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-accent)] animate-bounce [animation-delay:300ms]" />
                </div>
              </div>
            </div>
          ) : (
            <GridMap
              nodes={nodesData}
              edges={edges}
              onNodeClick={setSelectedNode}
            />
          )}
        </main>

        {/* Right Sidebar — Alert Feed */}
        <div className="w-80 shrink-0">
          <AlertFeed alerts={alerts} />
        </div>
      </div>
    </div>
  );
}
