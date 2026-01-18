# Critical Node Detection using CRITIC-TOPSIS Framework

A Python implementation for detecting critical nodes in complex networks using a multi-attribute decision-making approach that combines multiple centrality measures.

## 🎯 Project Overview

This project implements a framework to identify the most important nodes in a network by:
1. Computing 7 centrality measures (degree, betweenness, closeness, eigenvector, PageRank, k-shell, H-index)
2. Using **CRITIC** (CRiteria Importance Through Intercriteria Correlation) for objective weight determination
3. Using **TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution) for final ranking
4. Validating results through targeted attack simulations

## 📁 Project Structure

```
critical_node_detection/
├── src/
│   ├── __init__.py
│   ├── data_loading.py      # Load benchmark networks
│   ├── centralities.py      # Compute 7 centrality measures
│   ├── critic.py            # CRITIC weighting method
│   ├── topsis.py            # TOPSIS ranking method
│   ├── evaluation.py        # Attack simulation experiments
│   ├── visualization.py     # Plotting functions
│   └── main_pipeline.py     # Complete experiment pipeline
├── data/
│   ├── synthetic/           # Generated networks
│   └── real_networks/       # Downloaded benchmark networks
├── results/                  # Experiment outputs
├── docs/
│   ├── THEORY.md            # Theoretical background
│   └── REPORT_SKELETON.md   # Academic report outline
├── test_installation.py     # Verify setup
└── README.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd critical_node_detection
python3 -m venv venv
source venv/bin/activate
pip install networkx numpy pandas scipy matplotlib seaborn scikit-learn
```

### 2. Verify Installation

```bash
python test_installation.py
```

### 3. Run Experiments

```bash
cd src
python main_pipeline.py
```

This runs the complete pipeline on 4 benchmark networks and saves results to `results/`.

## 📊 Output Files

For each network, the pipeline generates:
- `centralities.csv` - Raw centrality values for all nodes
- `topsis_ranking.csv` - Final CRITIC-TOPSIS rankings
- `critic_weights.csv` - Computed CRITIC weights
- `effectiveness.csv` - Attack effectiveness scores
- `attack_curves.png` - Node removal curves
- `centrality_heatmap.png` - Centrality visualization
- `network.png` - Network diagram with critical nodes highlighted
- `summary.png` - Combined results figure

## 🔬 Using Individual Modules

### Compute Centralities
```python
import networkx as nx
from centralities import compute_all_centralities

G = nx.karate_club_graph()
df = compute_all_centralities(G)
print(df.head())
```

### Compute CRITIC Weights
```python
from critic import compute_critic_weights

weights, details = compute_critic_weights(df)
print(f"Weights: {weights}")
```

### Perform TOPSIS Ranking
```python
from topsis import topsis_rank, get_critical_nodes

results, details = topsis_rank(df, weights)
top_10 = get_critical_nodes(results, k=10)
print(f"Top 10 critical nodes: {top_10}")
```

### Run Attack Simulation
```python
from evaluation import compare_attack_methods, get_ranking_from_topsis

rankings = {
    'CRITIC-TOPSIS': get_ranking_from_topsis(results),
    'degree': df['degree'].sort_values(ascending=False).index.tolist()
}
attack_results = compare_attack_methods(G, rankings)
```

## 📈 Benchmark Networks

| Network | Nodes | Edges | Type | Description |
|---------|-------|-------|------|-------------|
| Karate Club | 34 | 78 | Social | Zachary's karate club friendships |
| Les Miserables | 77 | 254 | Literature | Character co-appearances |
| Florentine Families | 15 | 20 | Historical | Renaissance marriage ties |
| Barabasi-Albert | 100 | 291 | Synthetic | Scale-free model |

## 📚 Documentation

- See `docs/THEORY.md` for detailed explanations of all methods
- See `docs/REPORT_SKELETON.md` for academic report outline

## 🛠️ Requirements

- Python 3.8+
- networkx
- numpy
- pandas
- scipy
- matplotlib
- seaborn
- scikit-learn

## 📖 References

1. Diakoulaki, D., et al. (1995). "Determining objective weights in multiple criteria problems: The CRITIC method."
2. Hwang, C.L., & Yoon, K. (1981). "Multiple Attribute Decision Making: Methods and Applications."
3. Newman, M.E.J. (2010). "Networks: An Introduction."
