# Communication Outline: Workers <-> Coordinator (Control + P2P Data Plane)

## Scope
This document focuses **only** on how workers, clients, and dashboards connect to and communicate with the coordinator and the P2P network. Sharding/ML details are intentionally ignored.

## Components and Roles (Communication Surface)
- **Coordinator API server**: FastAPI app providing REST endpoints and an in-process DHT bootstrap. `coordinator/main.py`
- **DHT bootstrap (simulated)**: Simple in-memory peer registry used to provide initial peers. `coordinator/dht_bootstrap.py`
- **Worker GUI app**: Registers workers, polls heartbeat, launches Petals server. `worker_app/main.py`
- **Worker headless script**: Registers, heartbeats, and runs real Petals server. `worker_app/petals_mac_worker.py`
- **Worker API client**: HTTP client wrapper for coordinator. `worker_app/coordinator_client.py`
- **Client GUI app**: Queries network, registers users, connects to distributed model. `client_app/main.py`
- **Client CLI**: Minimal Petals client using coordinator bootstrap peers. `client_app/petals_client.py`
- **Client API client**: HTTP client wrapper for coordinator. `client_app/coordinator_client.py`
- **Dashboards**:
  - Inline dashboard served from coordinator at `/dashboard`. `coordinator/main.py`
  - Standalone static dashboard. `admin_dashboard/index.html`

## Protocols, Libraries, and Ports
### Control Plane (Coordinator REST API)
- **Protocol**: HTTP + JSON
- **Server libraries**: FastAPI, Uvicorn, Pydantic (schemas), CORS middleware
  - `coordinator/requirements.txt`, `coordinator/main.py`, `coordinator/models.py`
- **Client libraries**: `requests` (Python) and `fetch` (browser)
  - `worker_app/coordinator_client.py`, `client_app/coordinator_client.py`, `admin_dashboard/index.html`, `coordinator/main.py` (inline dashboard JS)
- **Port**: `API_PORT` (default **8000**)
  - `coordinator/config.py`, `docker-compose.yml`

### Data Plane (Petals / Hivemind P2P)
- **Protocol**: Petals/Hivemind P2P network using multiaddr-like peer strings
- **Worker/server libraries**: `petals`, `hivemind`, `torch`, `transformers`
  - `worker_app/requirements.txt`, `worker_app/petals_worker.py`, `worker_app/petals_mac_worker.py`
- **Client libraries**: `petals`, `hivemind`, `transformers`, `torch`
  - `client_app/requirements.txt`, `client_app/petals_client.py`, `client_app/chat_engine.py`
- **Ports**:
  - **DHT bootstrap**: `DHT_PORT` (default **31337**) in coordinator config
    - `coordinator/config.py`
  - **Worker P2P server port**: `--port` CLI arg (default **31337**) in `petals_mac_worker.py`
  - **Coordinator DHT registration for workers**: `DHT_PORT + 1` (default **31338**)
    - `coordinator/main.py` (`approve_worker` registers peer with `port=DHT_PORT + 1`)

## Coordinator REST API (Control Plane)
### Key Models (Request/Response Shapes)
Defined in `coordinator/models.py` and used across clients.

- **WorkerRegistration** (request body for `/workers/register`):
  - `worker_id`, `nickname`, `hardware_type`, `device_name`, `vram_gb`, `requested_layers?`
- **LayerAssignment** (response from `/workers/register`):
  - `layers_start`, `layers_end`, `model_name`, `initial_peers`, `is_idle`, `status`
- **WorkerHeartbeat** (request body for `/workers/heartbeat`):
  - `worker_id`, `tokens_served_since_last`, `status`
- **NetworkInfo** (response from `/network/info`):
  - `model_name`, `total_layers`, `bootstrap_peers`, `online_workers`, `coverage_percent`, `ready_for_inference`, etc.
- **UsageLog** (request body for `/usage/log`):
  - `user_id`, `prompt_tokens`, `completion_tokens`, `worker_ids`

### Endpoints and Who Uses Them
- `GET /`
  - **Used by**: Worker GUI, Client GUI, Admin dashboard
  - **Purpose**: health check and basic status
  - **Files**: `worker_app/coordinator_client.py`, `client_app/coordinator_client.py`, `admin_dashboard/index.html`
- `GET /network/info`
  - **Used by**: Worker GUI/CLI, Client GUI/CLI
  - **Purpose**: returns `bootstrap_peers` / model info for P2P connection
  - **Files**: `worker_app/coordinator_client.py`, `client_app/coordinator_client.py`, `client_app/petals_client.py`
- `GET /network/health`
  - **Used by**: Admin dashboard (standalone)
  - **Purpose**: detailed layer coverage
  - **File**: `admin_dashboard/index.html`
- `GET /network/stats`
  - **Used by**: Inline dashboard and standalone admin dashboard
  - **Purpose**: overall stats and coverage
  - **Files**: `coordinator/main.py` (inline dashboard JS), `admin_dashboard/index.html`
- `PUT /network/quantization?mode=...`
  - **Used by**: Inline dashboard
  - **Purpose**: change quantization (affects new worker assignments)
  - **File**: `coordinator/main.py` (inline dashboard JS)
- `PUT /network/target-workers?count=...` + `GET /network/target-workers`
  - **Purpose**: store recommended split target; not used directly by clients
  - **File**: `coordinator/main.py`
- `GET /workers`
  - **Used by**: Dashboards
  - **Purpose**: list workers and statuses
- `POST /workers/register`
  - **Used by**: Worker GUI, headless worker
  - **Purpose**: register a worker (initially `pending`)
  - **Files**: `worker_app/coordinator_client.py`, `worker_app/petals_mac_worker.py`
- `POST /workers/heartbeat`
  - **Used by**: Worker GUI, headless worker
  - **Purpose**: liveness + tokens served + approval status polling
  - **Files**: `worker_app/coordinator_client.py`, `worker_app/main.py`, `worker_app/petals_mac_worker.py`
- `POST /workers/{id}/approve` / `POST /workers/{id}/reject`
  - **Used by**: Inline dashboard
  - **Purpose**: transition worker out of `pending`
  - **File**: `coordinator/main.py` (inline dashboard JS)
- `DELETE /workers/{id}`
  - **Used by**: Worker GUI, headless worker
  - **Purpose**: explicit unregister on shutdown
  - **Files**: `worker_app/coordinator_client.py`, `worker_app/petals_mac_worker.py`
- `POST /users/register` + `GET /users/{id}`
  - **Used by**: Client GUI
  - **Purpose**: track user stats
  - **File**: `client_app/coordinator_client.py`
- `POST /usage/log` + `GET /usage/recent`
  - **Used by**: Client GUI (log usage), dashboards (recent logs)
  - **Files**: `client_app/main.py`, `admin_dashboard/index.html`, `coordinator/main.py`
- `GET /dashboard`
  - **Used by**: Browser
  - **Purpose**: serves inline HTML dashboard that calls the endpoints above
  - **File**: `coordinator/main.py`

## Worker <-> Coordinator Flow (Control Plane)
### GUI Worker (`worker_app/main.py`)
1. **Health check**: `GET /` via `CoordinatorClient.health_check()`
2. **Register**: `POST /workers/register` with hardware info
   - Response provides `initial_peers`, `layers_start/end`, and `status`
3. **Pending approval**: if `status == pending`, GUI starts heartbeat polling
   - Heartbeat interval: **30s** (`HEARTBEAT_INTERVAL` in `worker_app/config.py`)
4. **Approval detection**: heartbeat response returns `worker_status` + layers
   - If approved with layers, worker starts Petals server
   - If approved with no layers, worker becomes **standby**
5. **Ongoing heartbeats**: continues to send token counts to `/workers/heartbeat`
6. **Shutdown**: `DELETE /workers/{id}` on stop/exit

### Headless Worker (`worker_app/petals_mac_worker.py`)
1. **Register**: `POST /workers/register`
2. **Heartbeat thread**: `POST /workers/heartbeat` every **30s**
3. **Start Petals server**: CLI `python -m petals.cli.run_server` using `initial_peers` and assigned layers
4. **Shutdown**: `DELETE /workers/{id}` on Ctrl+C

## Coordinator Behavior (Control Plane)
- **Startup**: initializes DB + starts DHT bootstrap (`start_dht()`) `coordinator/main.py`
- **Worker registration**: stores worker as `pending` and returns `initial_peers` `coordinator/main.py`
- **Approval**:
  - Assigns layers via `health_monitor.assign_layers_to_worker(...)`
  - Registers approved workers into the DHT using `PUBLIC_HOST` and `DHT_PORT + 1`
  - `coordinator/main.py`
- **Heartbeat handling**:
  - Updates last_seen and tokens in DB
  - If worker is online and has assigned layers, updates DHT heartbeat
  - `coordinator/main.py` and `coordinator/database.py`
- **Stale cleanup**:
  - Workers marked offline after **60s** without heartbeat (`HEARTBEAT_TIMEOUT_SECONDS`)
  - Background cleanup loop runs every **30s**
  - `coordinator/main.py`, `coordinator/config.py`

## DHT / Peer Discovery (Data Plane Bootstrap)
- **SimpleDHT** (MVP): in-process registry that tracks peer IDs, addresses, ports, and last_seen
  - `coordinator/dht_bootstrap.py`
- **Peer address format**: `"/ip4/{address}/tcp/{port}/p2p/{peer_id}"`
  - Returned in `initial_peers` / `bootstrap_peers`
- **Bootstrap peer** is always included:
  - `"/ip4/{PUBLIC_HOST}/tcp/{DHT_PORT}/p2p/QmBootstrap"`
- **Stale DHT peers** cleaned every **30s**, removed after **120s** idle
  - `coordinator/dht_bootstrap.py`
- **Production note**: Code includes a commented `hivemind.DHT` implementation placeholder

## P2P Data Plane: Petals Worker and Client
### Worker side
- **Simulation**: `worker_app/petals_worker.py` simulates network connection and token serving
- **Real server**: `worker_app/petals_mac_worker.py` runs:
  - `python -m petals.cli.run_server <model> --initial_peers ... --block_indices <start:end> --port <port> [--public_ip ...]`
- **Initial peers** come from coordinator `initial_peers` (DHT bootstrap + registered peers)

### Client side
- **GUI app**: uses `ChatEngine.connect()` with `initial_peers` from `/network/info`
  - `client_app/main.py`, `client_app/chat_engine.py`
- **CLI app**: uses `AutoDistributedModelForCausalLM.from_pretrained(..., initial_peers=bootstrap_peers)`
  - `client_app/petals_client.py`

## Dashboard Communication
### Inline dashboard (`/dashboard`)
- Served by coordinator; JS calls:
  - `/network/stats`, `/workers`, `/usage/recent`
  - Control actions: `/workers/{id}/approve`, `/workers/{id}/reject`, `/network/quantization`
  - `coordinator/main.py`

### Standalone dashboard (`admin_dashboard/index.html`)
- Uses browser `fetch` against a configurable coordinator URL:
  - `/`, `/network/stats`, `/network/health`, `/workers`, `/usage/recent?limit=15`

## File Map (Communication-Related)
- `coordinator/main.py`: REST API, dashboard, DHT updates
- `coordinator/models.py`: request/response schemas
- `coordinator/dht_bootstrap.py`: DHT bootstrap + peer list
- `coordinator/config.py`: ports, timeouts, host
- `worker_app/coordinator_client.py`: HTTP client used by worker apps
- `worker_app/main.py`: worker GUI flow (register, heartbeat, start server)
- `worker_app/petals_mac_worker.py`: headless worker, real Petals server
- `worker_app/petals_worker.py`: simulated Petals worker / placeholder
- `client_app/coordinator_client.py`: HTTP client used by client apps
- `client_app/main.py`: client GUI flow (network info + usage logging)
- `client_app/chat_engine.py`: simulated Petals client (comments show real integration)
- `client_app/petals_client.py`: real Petals CLI client
- `admin_dashboard/index.html`: standalone dashboard fetch calls
- `docker-compose.yml`: exposes ports 8000 and 31337
