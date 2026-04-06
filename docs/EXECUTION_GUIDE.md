# Execution Guide — Critical Node Detection Framework

**Project:** Multi-Attribute Critical Node Detection using CRITIC-TOPSIS  
**Student:** Akshat Agrawal (23MIC0079)  

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Method 1: Docker (1-Click Run — Recommended)](#2-method-1-docker-1-click-run--recommended)
3. [Method 2: Manual Setup (Without Docker)](#3-method-2-manual-setup-without-docker)
4. [Running the Test Suite](#4-running-the-test-suite)
5. [API Documentation](#5-api-documentation)
6. [Accessing the Application](#6-accessing-the-application)
7. [Project Structure](#7-project-structure)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

### For Docker Method (Recommended)
- **Docker Desktop** (v4.0+) — Download from [docker.com](https://www.docker.com/products/docker-desktop/)
- **Docker Compose** (included with Docker Desktop)

### For Manual Method
- **Python 3.11+** — Download from [python.org](https://www.python.org/downloads/)
- **Node.js 20+** — Download from [nodejs.org](https://nodejs.org/)
- **pip** (Python package manager, included with Python)
- **npm** (Node package manager, included with Node.js)

---

## 2. Method 1: Docker (1-Click Run — Recommended)

This is the simplest way to run the entire application. One command builds and starts both the backend and frontend.

### Step 1: Ensure Docker Desktop Is Running
Open Docker Desktop and wait until the engine is fully started.

### Step 2: Navigate to the Project Directory
```bash
cd /path/to/critical_node_detection
```

### Step 3: Build and Run
```bash
docker-compose up --build
```

This will:
- Build the **Python 3.11 backend** container (FastAPI + all algorithm modules)
- Build the **Node.js 20 frontend** container (React 19 + Vite)
- Start both services with health-check monitoring

### Step 4: Access the Application
- **Frontend Dashboard:** http://localhost:5173
- **Backend API Docs:** http://localhost:8000/docs

### Step 5: Stop the Application
Press `Ctrl+C` in the terminal, then:
```bash
docker-compose down
```

### Troubleshooting Port Conflicts
If ports 8000 or 5173 are already in use:
```bash
docker-compose down
lsof -t -i:8000,5173 | xargs kill -9
docker-compose up --build
```

---

## 3. Method 2: Manual Setup (Without Docker)

This method runs the backend and frontend in two separate terminal windows.

### Step 1: Install Python Dependencies

```bash
cd /path/to/critical_node_detection
pip install -r requirements.txt
```

The `requirements.txt` includes:
- `fastapi` — REST API framework
- `uvicorn` — ASGI server
- `networkx` — Graph algorithms
- `numpy` — Numerical computing
- `pandas` — Data manipulation
- `scipy` — Scientific computing (bootstrap statistics)

### Step 2: Start the Backend Server (Terminal 1)

```bash
cd /path/to/critical_node_detection
uvicorn api.main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

### Step 3: Install Frontend Dependencies (Terminal 2)

```bash
cd /path/to/critical_node_detection/frontend
npm install
```

### Step 4: Start the Frontend Dev Server (Terminal 2)

```bash
npm run dev
```

You should see:
```
  VITE v6.x.x  ready in XXXms

  ➜  Local:   http://localhost:5173/
```

### Step 5: Access the Application
- **Frontend Dashboard:** http://localhost:5173
- **Backend API Docs:** http://localhost:8000/docs

---

## 4. Running the Test Suite

The project includes a `pytest` mathematical verification suite.

### Run All Tests
```bash
cd /path/to/critical_node_detection
pytest tests/ -v
```

### Expected Output
```
tests/test_critic_topsis.py::test_extreme_density_complete_graph PASSED
tests/test_critic_topsis.py::test_disconnected_network PASSED

======================== 2 passed in X.XXs ========================
```

### What the Tests Verify
- **`test_extreme_density_complete_graph`**: CRITIC correctly falls back to equal weights when all nodes are identical (complete graph K₁₀).
- **`test_disconnected_network`**: TOPSIS does not crash on division-by-zero when all distances are 0 (empty graph with no edges).

---

## 5. API Documentation

Once the backend is running, visit **http://localhost:8000/docs** for the interactive Swagger UI.

### Key API Endpoints

| Endpoint | Method | Description |
|:---|:---|:---|
| `POST /discovery` | POST | Run full CRITIC-TOPSIS pipeline. Returns rankings, weights, and network statistics. |
| `POST /impact` | POST | Run targeted attack simulation. Returns collapse curves and effectiveness scores. |
| `POST /cascade` | POST | Run cascading failure simulation with configurable tolerance factor. |
| `POST /temporal` | POST | Run temporal evolution analysis. Returns rising stars and rank drift. |
| `POST /domain` | POST | Run domain-aware analysis (Social, Infrastructure, etc.). |
| `POST /robustness` | POST | Run bootstrap stability + adversarial robustness + sensitivity analysis. |

### Example API Request (using curl)

```bash
curl -X POST http://localhost:8000/discovery \
  -H "Content-Type: application/json" \
  -d '{"network": "karate", "top_k": 10}'
```

### Custom Dataset Upload (via API)

```bash
curl -X POST http://localhost:8000/discovery \
  -H "Content-Type: application/json" \
  -d '{"network": "custom", "edges": [[0,1],[1,2],[2,3],[3,0],[0,2]], "top_k": 5}'
```

---

## 6. Accessing the Application

### Using the Web Dashboard

1. Open **http://localhost:5173** in your browser.
2. Select a benchmark network from the dropdown (e.g., "Karate Club").
3. Follow the 8-step guided pipeline:
   - **① Introduction** — Overview and custom dataset upload
   - **② Discovery** — Run CRITIC-TOPSIS analysis
   - **③ Impact** — Targeted attack simulation
   - **④ Temporal** — Temporal evolution & rising stars
   - **⑤ Domain** — Domain-aware weighted analysis
   - **⑥ Explainability** — Node-level explanations
   - **⑦ Robustness** — Bootstrap + adversarial testing
   - **⑧ Conclusion** — Summary and data export

### Custom Dataset Upload (via UI)

1. Navigate to the **① Introduction** section.
2. Drag and drop a `.csv` or `.txt` file containing an edge list.
3. File format: one edge per line, two integers separated by a comma or space:
   ```
   0,1
   1,2
   2,3
   3,0
   0,2
   ```
4. The system will parse the file and switch to "Custom Network" mode.

---

## 7. Project Structure

```
critical_node_detection/
│
├── api/                        # FastAPI REST backend
│   ├── __init__.py
│   └── main.py                 # All API endpoint definitions
│
├── frontend/                   # React 19 + Vite frontend
│   ├── src/
│   │   ├── sections/           # 8 UI sections (Hero, Discovery, Impact, etc.)
│   │   ├── components/         # Reusable React components (Navbar, Hero)
│   │   ├── App.jsx             # Main application component
│   │   ├── api.js              # API client functions
│   │   └── index.css           # Global styles
│   ├── package.json
│   ├── vite.config.js
│   └── Dockerfile              # Frontend container definition
│
├── src/                        # Core algorithmic logic (Python)
│   ├── centralities.py         # 7 centrality measure computations
│   ├── critic.py               # CRITIC objective weighting
│   ├── topsis.py               # TOPSIS ranking
│   ├── evaluation.py           # Targeted attack simulation
│   ├── cascading_failure.py    # Load-based cascade model
│   ├── uncertainty.py          # Bootstrap confidence intervals
│   ├── adversarial.py          # Sybil & edge-manipulation attacks
│   ├── temporal_analysis.py    # Temporal ranking evolution
│   ├── sensitivity_analysis.py # Centrality removal sensitivity
│   ├── domain_weights.py       # Domain-specific weight adjustment
│   ├── explainable_ai.py       # Node-level explanations
│   ├── scalability.py          # Performance benchmarking
│   └── data_loading.py         # Dataset loading utilities
│
├── tests/                      # pytest test suite
│   └── test_critic_topsis.py   # Mathematical edge-case tests
│
├── results/                    # Pre-computed analysis results (PNG plots)
│   ├── karate_club/
│   ├── les_miserables/
│   ├── power_grid/
│   └── ...
│
├── docs/                       # Project documentation
│   ├── MASTER_REPORT.md        # Final project report
│   ├── CODES.md                # Source code documentation
│   └── EXECUTION_GUIDE.md      # This file
│
├── legacy/                     # Archived monolithic prototype
│   └── app.py                  # Original Streamlit script
│
├── docker-compose.yml          # 1-click Docker orchestration
├── Dockerfile.backend          # Backend container definition
├── requirements.txt            # Python dependencies
├── .dockerignore               # Docker build exclusions
└── README.md                   # Project overview
```

---

## 8. Troubleshooting

### "Module not found" errors when starting the backend
Ensure you are in the project root directory (not inside `api/` or `src/`):
```bash
cd /path/to/critical_node_detection
uvicorn api.main:app --reload --port 8000
```

### Frontend shows "Network Error" or blank page
- Ensure the backend is running on port 8000.
- Check that the Vite proxy is configured (see `frontend/vite.config.js`).

### Docker "unhealthy" container errors
```bash
docker-compose down
lsof -t -i:8000,5173 | xargs kill -9
docker-compose up --build
```

### pytest "import error"
Run from the project root:
```bash
cd /path/to/critical_node_detection
pytest tests/ -v
```

### Port already in use
```bash
# Find and kill processes on ports 8000 and 5173
lsof -t -i:8000 | xargs kill -9
lsof -t -i:5173 | xargs kill -9
```

---

*End of Execution Guide*
