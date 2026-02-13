# Multi-Agent Dashboard UI (Next.js)

This folder contains the custom UI for the multi-agent system.

It connects to the FastAPI backend (`app/api_server.py`) via a streaming SSE proxy route:

- `POST /api/adk/run-sse`

## Prerequisites

- Node.js (LTS recommended)
- Backend running (default: `http://localhost:8001`)

## Setup

```bash
npm install
```

## Environment Variables

Create `frontend/.env.local` if needed.

- `ADK_API_URL` (server-side)
  - Default: `http://localhost:8001`
  - Used by `src/app/api/adk/run-sse/route.ts` to proxy the event-stream from the backend.

- `NEXT_PUBLIC_PERSIST_SESSIONS` (optional)
  - `true` enables local persistence of session history in `localStorage`.

## Run

```bash
npm run dev
```

Open `http://localhost:3000`.

## Features

### Simple View

- Runs tasks and shows agent activity and logs
- Saves a shared monitoring state to `localStorage` for live monitoring

### Advanced View (Monitoring)

- Opens with a `session` query parameter
- Polls the shared monitoring state from `localStorage` to show live agent statuses/logs

### Data Agent Artifacts

The UI can render structured artifacts embedded in the agent stream:

- Table previews
- Inline Plotly charts (loaded from Plotly CDN at runtime)
