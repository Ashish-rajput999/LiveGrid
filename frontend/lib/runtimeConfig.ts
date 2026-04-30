const trimTrailingSlash = (value: string) => value.replace(/\/+$/, "");

// NEXT_PUBLIC_API_BASE_URL  — REST base URL   (e.g. https://livegrid-backend.onrender.com)
// NEXT_PUBLIC_WS_URL        — WebSocket URL   (e.g. wss://livegrid-backend.onrender.com/ws/live)
// Both fall back to localhost for local development.

const rawApiBase = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
const rawWsUrl =
  process.env.NEXT_PUBLIC_WS_URL?.trim() ??
  process.env.NEXT_PUBLIC_BACKEND_WS_URL?.trim(); // legacy alias

export const API_BASE_URL = rawApiBase
  ? trimTrailingSlash(rawApiBase)
  : "http://localhost:8000";

export const WS_URL = rawWsUrl
  ? rawWsUrl
  : API_BASE_URL.replace(/^http/i, "ws") + "/ws/live";
