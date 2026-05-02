# LiveGrid Deployment Guide (Vercel + Render)

This project should be deployed as two services:
- Frontend (`frontend/`) on **Vercel**
- Backend (`backend/main.py`) on **Render** using `Dockerfile.backend`

## 1. Deploy Backend on Render

### Create service
1. Push this repo to GitHub.
2. In Render, choose **New +** -> **Blueprint**.
3. Select this repository. Render will detect `render.yaml`.
4. Create the service `livegrid-backend`.

### Required environment variables (Render)
Set these in Render for backend service:
- `PYTHONPATH=/app`
- `FRONTEND_ORIGINS=https://<your-vercel-domain>`

If you use a custom frontend domain later, append it as comma-separated:
- `FRONTEND_ORIGINS=https://<your-vercel-domain>,https://<your-custom-domain>`

### Verify backend
After deploy, open:
- `https://<render-backend-domain>/api/grid`

You should get JSON with `tick`, `nodes`, `edges`.

## 2. Deploy Frontend on Vercel

### Create project
1. In Vercel, click **Add New Project**.
2. Import this repository.
3. Set **Root Directory** to `frontend`.
4. Framework preset: Next.js.

### Required environment variables (Vercel)
Add these in Vercel Project Settings -> Environment Variables:
- `NEXT_PUBLIC_API_BASE_URL=https://<render-backend-domain>`
- `NEXT_PUBLIC_WS_URL=wss://<render-backend-domain>/ws/live`

Important:
- Use `https://` for API URL.
- Use `wss://` for WebSocket URL.
- No trailing slash on base API URL.

### Verify frontend
Open deployed Vercel URL and confirm:
- Dashboard loads.
- Connection indicator shows connected.
- Live ticks update every second.

## 3. Order of deployment
Deploy in this order to avoid broken URLs:
1. Render backend first.
2. Copy Render backend domain.
3. Configure Vercel env vars.
4. Deploy/redeploy Vercel frontend.
5. Set `FRONTEND_ORIGINS` in Render to Vercel URL and redeploy Render.

## 4. Common issues and fixes

### CORS error in browser
- Cause: `FRONTEND_ORIGINS` missing/wrong on Render.
- Fix: set exact Vercel domain in `FRONTEND_ORIGINS` and redeploy backend.

### WebSocket does not connect
- Cause: using `ws://` on production HTTPS site.
- Fix: `NEXT_PUBLIC_WS_URL` must be `wss://.../ws/live`.

### Backend starts but no predictions
- App now auto-loads model artifacts from `backend/models/` by default.
- If you move files, set:
  - `LIVEGRID_MODEL_PATH`
  - `LIVEGRID_GNN_MODEL_PATH`
  - `LIVEGRID_GNN_SCALER_PATH`
  - `LIVEGRID_NEIGHBOR_MAP_PATH`

### GNN dependency issues on host
- App now falls back to LSTM automatically if GNN cannot load.

