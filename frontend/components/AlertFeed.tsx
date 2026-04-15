"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";

const API_BASE = "http://localhost:8000";

export interface AlertItem {
  id: string;
  time: string;
  nodeId: string;
  level: "CRITICAL" | "WARNING" | "FAILED" | "RECOVERED";
  risk: number;
  message: string;
}

interface ExplanationData {
  node_id: string;
  risk_score: number;
  status: string;
  explanation: {
    primary_driver: string;
    contributing_factors: string[];
    counterfactual: string;
    time_to_critical: string;
  };
}

interface AlertFeedProps {
  alerts: AlertItem[];
}

function getLevelEmoji(level: AlertItem["level"]): string {
  switch (level) {
    case "CRITICAL": return "🔴";
    case "WARNING":  return "🟡";
    case "FAILED":   return "⚫";
    case "RECOVERED":return "🟢";
  }
}

function getLevelColor(level: AlertItem["level"]): string {
  switch (level) {
    case "CRITICAL":  return "border-red-500/30 bg-red-500/5";
    case "WARNING":   return "border-yellow-500/30 bg-yellow-500/5";
    case "FAILED":    return "border-gray-500/30 bg-gray-500/5";
    case "RECOVERED": return "border-green-500/30 bg-green-500/5";
  }
}

function getTtcBadgeColor(ttc: string): string {
  if (ttc.includes("1–") || ttc.includes("2–") || ttc.includes("3–")) return "bg-red-600/80 text-white";
  if (ttc.includes("stable") || ttc.includes("decreasing") || ttc.includes("below")) return "bg-green-600/80 text-white";
  return "bg-yellow-600/80 text-white";
}

// ── Single Alert Card ─────────────────────────────────────────────────

function AlertCard({ alert }: { alert: AlertItem }) {
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [explanation, setExplanation] = useState<ExplanationData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchExplain = useCallback(async () => {
    if (explanation || loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/explain/${alert.nodeId}`);
      if (!res.ok) {
        const err = await res.json();
        setError(err.detail ?? "Failed to load explanation");
        return;
      }
      const data: ExplanationData = await res.json();
      setExplanation(data);
    } catch {
      setError("Could not connect to backend");
    } finally {
      setLoading(false);
    }
  }, [alert.nodeId, explanation, loading]);

  const handleClick = () => {
    const next = !expanded;
    setExpanded(next);
    if (next) fetchExplain();
  };

  const canExpand = alert.level === "CRITICAL" || alert.level === "WARNING";

  return (
    <div className={`alert-enter rounded-lg border ${getLevelColor(alert.level)} overflow-hidden`}>
      {/* Header row */}
      <div
        className={`px-3 py-2.5 ${canExpand ? "cursor-pointer hover:bg-white/5 transition-colors" : ""}`}
        onClick={canExpand ? handleClick : undefined}
        id={`alert-${alert.id}`}
      >
        <div className="flex items-start gap-2">
          <span className="text-sm mt-0.5 shrink-0">{getLevelEmoji(alert.level)}</span>
          <div className="min-w-0 flex-1">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-semibold text-white">{alert.nodeId}</span>
              <div className="flex items-center gap-1.5 shrink-0">
                <span className="text-[10px] text-[var(--color-text-secondary)] tabular-nums">
                  {alert.time}
                </span>
                {canExpand && (
                  <span className="text-[10px] text-[var(--color-text-secondary)]">
                    {expanded ? "▲" : "▼"}
                  </span>
                )}
              </div>
            </div>
            <p className="text-xs text-[var(--color-text-secondary)] mt-0.5 leading-relaxed">
              {alert.level === "CRITICAL" || alert.level === "WARNING"
                ? `${alert.level} (risk: ${alert.risk.toFixed(2)}) — ${alert.message}`
                : alert.message}
            </p>
          </div>
        </div>
      </div>

      {/* Expanded explanation */}
      {expanded && canExpand && (
        <div className="border-t border-white/10 px-3 py-3 space-y-3">
          {loading && (
            <div className="text-xs text-[var(--color-text-secondary)] animate-pulse">
              Loading explanation...
            </div>
          )}

          {error && (
            <div className="text-xs text-red-400">{error}</div>
          )}

          {explanation && (
            <>
              {/* Primary driver */}
              <div>
                <p className="text-[10px] text-[var(--color-text-secondary)] uppercase tracking-wider mb-1">
                  Primary Driver
                </p>
                <p className="text-xs font-semibold text-white leading-relaxed">
                  {explanation.explanation.primary_driver}
                </p>
              </div>

              {/* Contributing factors */}
              {explanation.explanation.contributing_factors.length > 0 && (
                <div>
                  <p className="text-[10px] text-[var(--color-text-secondary)] uppercase tracking-wider mb-1">
                    Contributing Factors
                  </p>
                  <ul className="space-y-1">
                    {explanation.explanation.contributing_factors.map((f, i) => (
                      <li key={i} className="text-xs text-[var(--color-text-secondary)] flex gap-1.5">
                        <span className="text-yellow-500 shrink-0">•</span>
                        <span>{f}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Counterfactual */}
              <div className="bg-[var(--color-background)]/60 rounded-lg px-2.5 py-2">
                <p className="text-[10px] text-[var(--color-text-secondary)] uppercase tracking-wider mb-1">
                  What-If
                </p>
                <p className="text-xs text-[var(--color-text-secondary)] italic leading-relaxed">
                  {explanation.explanation.counterfactual}
                </p>
              </div>

              {/* Time to critical */}
              <div className="flex items-center gap-2">
                <p className="text-[10px] text-[var(--color-text-secondary)] uppercase tracking-wider">
                  Time to Critical
                </p>
                <span
                  className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${getTtcBadgeColor(
                    explanation.explanation.time_to_critical
                  )}`}
                >
                  {explanation.explanation.time_to_critical}
                </span>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ── AlertFeed Container ───────────────────────────────────────────────

export default function AlertFeed({ alerts }: AlertFeedProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = 0;
    }
  }, [alerts.length]);

  return (
    <aside
      id="alert-feed"
      className="flex flex-col h-full bg-[var(--color-surface)]/50 border-l border-[var(--color-border)] backdrop-blur-sm"
    >
      {/* Header */}
      <div className="px-4 py-3 border-b border-[var(--color-border)] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm">🚨</span>
          <h2 className="text-sm font-semibold text-white tracking-tight">Alert Feed</h2>
        </div>
        <span className="text-xs text-[var(--color-text-secondary)] bg-[var(--color-surface)] px-2 py-0.5 rounded-full">
          {alerts.length}
        </span>
      </div>

      {/* Hint */}
      {alerts.some((a) => a.level === "CRITICAL" || a.level === "WARNING") && (
        <div className="px-3 pt-2">
          <p className="text-[10px] text-[var(--color-text-secondary)] italic">
            💡 Click WARNING / CRITICAL alerts to see AI explanation
          </p>
        </div>
      )}

      {/* Alerts List */}
      <div ref={containerRef} className="flex-1 overflow-y-auto px-3 py-2 space-y-2">
        {alerts.length === 0 && (
          <div className="text-center text-[var(--color-text-secondary)] text-xs py-8">
            No alerts yet. Monitoring grid...
          </div>
        )}
        {alerts.map((alert) => (
          <AlertCard key={alert.id} alert={alert} />
        ))}
      </div>
    </aside>
  );
}
