"use client";

import React from "react";

interface StatsBarProps {
  tick: number;
  activeScenario: string | null;
  okCount: number;
  warningCount: number;
  failedCount: number;
  avgRisk: number;
  connected: boolean;
  modelType?: string;
}

export default function StatsBar({
  tick,
  activeScenario,
  okCount,
  warningCount,
  failedCount,
  avgRisk,
  connected,
  modelType = "LSTM",
}: StatsBarProps) {
  const scenarioLabel = activeScenario
    ? activeScenario
        .replace(/_/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase())
    : "None";

  const riskColor =
    avgRisk > 0.7
      ? "text-red-400"
      : avgRisk > 0.4
        ? "text-yellow-400"
        : "text-green-400";

  return (
    <header
      id="stats-bar"
      className="w-full border-b border-[var(--color-border)] bg-[var(--color-surface)]/80 backdrop-blur-sm"
    >
      <div className="max-w-[1800px] mx-auto px-6 py-3 flex items-center justify-between gap-4 flex-wrap">
        {/* Logo + Title */}
        <div className="flex items-center gap-3">
          <span className="text-2xl">⚡</span>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white">
              LiveGrid
            </h1>
            <p className="text-xs text-[var(--color-text-secondary)]">
              Real-Time Power Grid Monitor
            </p>
          </div>
        </div>

        {/* Stat Cards */}
        <div className="flex items-center gap-3 flex-wrap">
          {/* Tick */}
          <div className="stat-card flex items-center gap-2">
            <span className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider">
              Tick
            </span>
            <span className="text-lg font-semibold tabular-nums text-white">
              {tick}
            </span>
          </div>

          {/* Scenario */}
          <div className="stat-card flex items-center gap-2">
            <span className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider">
              Scenario
            </span>
            <span
              className={`text-sm font-medium ${
                activeScenario ? "text-orange-400" : "text-[var(--color-text-secondary)]"
              }`}
            >
              {activeScenario ? (
                <>
                  {activeScenario === "heat_wave" ? "🌡️" : "💥"}{" "}
                  {scenarioLabel}
                </>
              ) : (
                "—"
              )}
            </span>
          </div>

          {/* Node Counts */}
          <div className="stat-card flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-green-500" />
              <span className="text-sm font-medium text-green-400">
                {okCount}
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-yellow-500" />
              <span className="text-sm font-medium text-yellow-400">
                {warningCount}
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-red-500" />
              <span className="text-sm font-medium text-red-400">
                {failedCount}
              </span>
            </div>
          </div>

          {/* Model Badge */}
          <div className="stat-card flex items-center gap-2">
            <span className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider">
              Model
            </span>
            <span
              className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                modelType === "GNN"
                  ? "bg-purple-600/30 text-purple-300 border border-purple-500/40"
                  : "bg-blue-600/30 text-blue-300 border border-blue-500/40"
              }`}
            >
              {modelType}
            </span>
          </div>

          {/* Avg Risk */}
          <div className="stat-card flex items-center gap-2">
            <span className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider">
              Avg Risk
            </span>
            <span className={`text-lg font-semibold tabular-nums ${riskColor}`}>
              {avgRisk.toFixed(2)}
            </span>
          </div>

          {/* Connection Status */}
          <div className="stat-card flex items-center gap-2">
            <span
              id="connection-indicator"
              className={`connection-dot ${connected ? "connected" : "disconnected"}`}
            />
            <span className="text-xs text-[var(--color-text-secondary)]">
              {connected ? "Live" : "Offline"}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
