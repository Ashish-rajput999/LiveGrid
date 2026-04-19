#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  LiveGrid — start.sh
#  Starts the backend (FastAPI) and frontend (Next.js) together.
#  Usage:  ./start.sh
#  Stop:   Ctrl+C  (kills both background processes cleanly)
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Load nvm if node is not already in PATH ────────────────────
# nvm installs node outside the default $PATH in non-interactive shells.
if ! command -v node &>/dev/null; then
    export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
    # shellcheck source=/dev/null
    [ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"
    # Also try Homebrew / system locations as fallback
    for _p in /opt/homebrew/bin /usr/local/bin; do
        [ -x "$_p/node" ] && export PATH="$_p:$PATH" && break
    done
fi

BACKEND_LOG="logs/backend.log"
FRONTEND_LOG="logs/frontend.log"
BACKEND_PID=""
FRONTEND_PID=""

# ── Colours ────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}[LiveGrid]${NC} $1"; }
success() { echo -e "${GREEN}[LiveGrid]${NC} $1"; }
warn()    { echo -e "${YELLOW}[LiveGrid]${NC} $1"; }
error()   { echo -e "${RED}[LiveGrid] ERROR:${NC} $1"; }

# ── Cleanup on Ctrl+C ──────────────────────────────────────────
cleanup() {
    echo ""
    info "Shutting down LiveGrid..."
    if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" && info "Backend stopped (PID $BACKEND_PID)"
    fi
    if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill "$FRONTEND_PID" && info "Frontend stopped (PID $FRONTEND_PID)"
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Banner ─────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}⚡ LiveGrid — Real-Time Power Grid Failure Prediction${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────${NC}"
echo ""

# ── 1. Check prerequisites ─────────────────────────────────────
info "Checking prerequisites..."

if ! command -v python3 &>/dev/null; then
    error "python3 is not installed or not in PATH."
    error "Install it from https://www.python.org/downloads/ and re-run."
    exit 1
fi
success "python3 found: $(python3 --version)"

if ! command -v node &>/dev/null; then
    error "node is not installed or not in PATH."
    error "Install it from https://nodejs.org/ (or via nvm) and re-run."
    exit 1
fi
success "node found: $(node --version)"

# ── 2. Python virtual environment ─────────────────────────────
if [[ ! -d "venv" ]]; then
    info "Creating Python virtual environment at venv/ ..."
    python3 -m venv venv
    success "Virtual environment created."
else
    info "Virtual environment already exists — skipping creation."
fi

# Activate venv
# shellcheck source=/dev/null
source venv/bin/activate
success "Virtual environment activated."

# ── 3. Install Python dependencies ────────────────────────────
SENTINEL="venv/.deps_installed"
if [[ ! -f "$SENTINEL" ]]; then
    info "Installing Python dependencies from backend/requirements.txt ..."
    pip install --quiet --upgrade pip
    pip install --quiet -r backend/requirements.txt
    # torch-geometric needs special handling (no PyPI wheel for all platforms)
    if ! python3 -c "import torch_geometric" 2>/dev/null; then
        warn "torch-geometric not found — attempting install..."
        pip install --quiet torch-geometric || \
            warn "Could not auto-install torch-geometric. GNN will fall back to LSTM."
    fi
    touch "$SENTINEL"
    success "Python dependencies installed."
else
    info "Python dependencies already installed — skipping."
fi

# ── 4. Frontend npm deps ───────────────────────────────────────
if [[ ! -d "frontend/node_modules" ]]; then
    info "Installing frontend npm dependencies..."
    (cd frontend && npm install --silent)
    success "Frontend dependencies installed."
else
    info "Frontend node_modules already present — skipping."
fi

# ── 5. Create logs directory ──────────────────────────────────
mkdir -p logs

# ── 6. Start backend ──────────────────────────────────────────
# Use the venv's binaries explicitly — 'source activate' does not
# propagate into background subshells reliably on all platforms.
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
VENV_UVICORN="$SCRIPT_DIR/venv/bin/uvicorn"
info "Starting backend on http://localhost:8000 ..."
PYTHONPATH="$SCRIPT_DIR" "$VENV_UVICORN" backend.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level warning \
    > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
info "Backend PID: $BACKEND_PID — logs → $BACKEND_LOG"

# ── 7. Wait for backend to be ready ───────────────────────────
# PyTorch takes 15-25s to import on first run — give it up to 60s.
info "Waiting for backend to be ready (up to 60s — PyTorch loads slowly)..."
READY=false
for i in $(seq 1 120); do
    if curl -sf http://localhost:8000/api/grid > /dev/null 2>&1; then
        READY=true
        break
    fi
    # Show a dot every 5s so the user knows it's still working
    if (( i % 10 == 0 )); then
        echo -ne "  [${i}s elapsed]\r"
    fi
    sleep 0.5
done

if [[ "$READY" == false ]]; then
    error "Backend did not become ready within 60 seconds."
    error "Check logs at $BACKEND_LOG for details."
    echo ""
    tail -20 "$BACKEND_LOG"
    kill "$BACKEND_PID" 2>/dev/null || true
    exit 1
fi
success "Backend is ready."

# ── 8. Start frontend ─────────────────────────────────────────
info "Starting frontend on http://localhost:3000 ..."
(cd frontend && npm run dev) \
    > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
info "Frontend PID: $FRONTEND_PID — logs → $FRONTEND_LOG"

# ── 9. Done ───────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}✅ LiveGrid is running → open http://localhost:3000${NC}"
echo ""
echo -e "  Backend:   http://localhost:8000/api/grid"
echo -e "  Logs:      $BACKEND_LOG  |  $FRONTEND_LOG"
echo -e "  Stop:      Ctrl+C"
echo ""

# Wait for background jobs (keeps script alive until Ctrl+C)
wait
