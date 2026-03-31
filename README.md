# 🌐 Critical Node Detection (CRITIC-TOPSIS Framework)

A **modern, full-stack, AI-powered execution system** for identifying critical nodes in complex networks using a multi-attribute decision-making approach. This project implements a sophisticated hybrid model that fuses **7 centrality measures** via **CRITIC** weighting and **TOPSIS** ranking.

---

## ⚡ 1-Click Launch (Docker)

The fastest way to run the entire suite (Backend API + Frontend UI) on any machine:

```bash
docker-compose up --build
```

- **Frontend UI:** [http://localhost:5173/](http://localhost:5173/)
- **Backend API:** [http://localhost:8000/docs](http://localhost:8000/docs) (Interactive Swagger UI)

---

## 🎯 Core Methodology

Traditional methods use a **single centrality metric** (e.g. only degree), which ignores crucial network properties. Our framework fuses information from **7 complementary dimensions**:

1.  **Degree Centrality** (Hub Detection)
2.  **Betweenness** (Bridge Detection)
3.  **Closeness** (Global Reach)
4.  **Eigenvector** (Influence Ranking)
5.  **PageRank** (Stability/Importance)
6.  **K-Shell Decomposition** (Core-Periphery)
7.  **H-Index** (Local Spread)

### 🔬 The Algorithm
1.  **Metric Computation**: Parallelized calculation of all centralities.
2.  **CRITIC Weighting**: Objectively assigns weights based on **Standard Deviation** (contrast intensity) and **Intercriteria Correlation** (redundancy filtering).
3.  **TOPSIS Ranking**: Identifies the "ideal" node by minimizing Euclidean distance to the theoretical high-impact best and maximizing distance from the worst.

---

## 🏗️ New Modern Architecture

The project has transitioned from a monolithic prototype to a professional **Decoupled REST Architecture**:

-   **Backend:** FastAPI (Python 3.11) with automated serialization and JSON response schemas.
-   **Frontend:** React 19 + Vite + Recharts + Force-Graph-2D for interactive visualizations.
-   **Simulation:** Real-time **Cascading Failure** and **Targeted Attack** engines.
-   **Feature Support:** **Custom Dataset Upload** (CSV/TXT edge lists), **Temporal Rankings**, and **Domain-Aware Analysis**.

---

## 📁 Project Structure

```bash
critical_node_detection/
├── api/                  # ⚙️ FastAPI REST Endpoints
├── frontend/             # 🎨 React 19 Frontend Dashboard
│   ├── src/sections/     # UI Sections (Impact, Discovery, Robustness)
│   └── Dockerfile        # Frontend Containerization
├── src/                  # 🧬 Core Algorithmic Logic (Shared)
│   ├── critic.py         # Advanced Weighting
│   ├── topsis.py         # Closeness Ranking
│   └── evaluation.py     # Attack Simulation Engine
├── legacy/               # 📜 Original Streamlit Prototype (Archived)
├── tests/                 # 🧪 Pytest Suite (Mathematical Rigor)
├── docker-compose.yml    # 📦 1-Click Orchestration
├── Dockerfile.backend    # 🛡️ Backend Containerization
└── README.md
```

---

## 🧪 Testing & Verification

We use `pytest` to mathematically verify that the CRITIC-TOPSIS logic is robust against extreme network topologies (e.g., zero-variance or disconnected graphs).

```bash
# Run the test suite
pytest tests/
```

---

## 📚 Academic References

1.  **CRITIC:** Diakoulaki, D. et al. (1995). "Determining objective weights in multiple criteria problems."
2.  **TOPSIS:** Hwang, C.L. & Yoon, K. (1981). "Multiple Attribute Decision Making."
3.  **Complex Networks:** Newman, M.E.J. (2010). "Networks: An Introduction."

---

## 🛠️ Requirements (Manual Run)

-   **Python 3.11+**: `pip install -r requirements.txt`
-   **Node.js 20+**: `cd frontend && npm install`
-   **Backend**: `uvicorn api.main:app --port 8000`
-   **Frontend**: `cd frontend && npm run dev`

---
Copyright © 2026 Critical Node Detection Framework.
