<div align="center">
  <h1>⚡ LiveGrid</h1>
  <p><strong>Predictive AI system designed to simulate and prevent catastrophic power grid failures in real-time.</strong></p>

  [![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()
  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
  [![Coverage](https://img.shields.io/badge/Coverage-95%25-success.svg)]()
  [![License](https://img.shields.io/badge/License-MIT-purple.svg)]()
</div>

<br />

## 📖 Overview

Modern power grids are under unprecedented stress from extreme weather, aging infrastructure, and shifting demand curves. **LiveGrid** is an end-to-end full-stack simulation and machine learning platform that generates synthetic SCADA grid data, feeds it through predictive Graph Neural Networks (GNN), and visualizes the cascading failure risks in real-time.



## ✨ Key Features

- **Synthetic Grid Emulation:** Simulates a localized grid of substations, generators, and distribution networks experiencing complex events like heatwaves and sudden mechanical failures.
- **Advanced Graph Neural Networks:** Uses Graph Attention Networks (GAT) to model topological relationships, capturing risk that propagates across connected transformer edges.
- **Explainable AI (XAI):** Not only assigns risk scores to nodes, but generates human-readable explanations (e.g. "Voltage drop across 3 consecutive ticks by 2.1kV while neighbor loads spiked").
- **Real-Time Cascade Forecasting:** Identify stranded loads before they trigger a catastrophic cascading blackout event.
- **High-Performance Streaming:** FastAPI websockets pumping 1Hz telemetry updates effortlessly handled by a resilient React global store.

---

## 🏗️ Architecture

```mermaid
graph LR
    subgraph Data Generation & ML
        A[Grid Simulator] -->|Synthetic Telemetry| B[Feature Engineering pipeline]
        B -->|Sequence Windows| C[PyTorch GNN / LSTM]
        C -->|Risk Scores| D[Model Artifacts .pt]
    end

    subgraph Real-Time API (Python)
        D --> E(FastAPI Background Loop)
        A --> E
        E -->|REST/XAI| F(Next.js App Router API)
        E -.->|Websocket 1Hz| G(Next.js Frontend Store)
    end

    subgraph User Interface (React)
        G --> H[Force-Directed Graph Visualization]
        G --> I[Risk & Metrics Dashboard]
    end
```

---

## 🚀 Quick Start

### Option 1: Full Docker Deployment (Recommended)
You can launch the entire stack in isolated bridged containers:

```bash
git clone https://github.com/Ashish-rajput999/LiveGrid.git
cd LiveGrid

# Spin up both frontend and backend
docker compose up --build -d

# Visit the dashboard
open http://localhost:3000
```

### Option 2: Local Development Mode
If you prefer running locally without Docker:

```bash
# Provide execute permissions to the launch script
chmod +x start.sh

# Install requirements, build venv, load PyTorch, start Next.js
./start.sh
```

---

## 📁 Repository Structure

```text
LiveGrid/
├── backend/                  # FastAPI Application
│   ├── api/                  # Route definitions (REST + WebSockets)
│   ├── models/               # Pydantic schemas and .pt PyTorch Models
│   └── services/             # Core simulation loop and Prediction inference
├── frontend/                 # Next.js 14 Application
│   ├── app/                  # App Router & Layouts
│   ├── components/           # Reusable Tailwind UI components
│   └── pages/                # Legacy router support / static pages
├── livegrid/                 # Domain logic (Node, Grid, Graph Simulation Engine)
├── feature_engineering.py    # Rolling-window ML processing pipeline
├── train_gnn.py              # Script to re-train the Graph Attention Network
└── docker-compose.yml        # Multi-container orchestration
```

---

## 🛠️ Tech Stack

- **Frontend:** TypeScript, React, Next.js 14 (App Router), TailwindCSS, Recharts
- **Backend:** Python 3.10+, FastAPI, Uvicorn, WebSockets
- **Machine Learning:** PyTorch, PyTorch Geometric (PyG), scikit-learn
- **Infrastructure:** Docker, Docker Compose

---

## 🔮 Future Improvements

1. **SCADA Hardware Ingestion:** Expand the `engine` to read standard IEC 61850 grid sensor packages replacing synthetic data.
2. **Auto-Scaling Websockets:** Introduce a Redis Pub/Sub adapter to allow horizontal scaling of the FastAPI layer.
3. **Control Interventions:** Build active mitigation tools into the dashboard, enabling operators to manually shed load or redirect flow to prevent a modeled cascade.

---

<div align="center">
  <p>Built as an exploration of practical AI in critical infrastructure. Pull requests are welcome!</p>
</div>
