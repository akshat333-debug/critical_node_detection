# Source Code Documentation — Critical Node Detection Framework

**Project:** Multi-Attribute Critical Node Detection using CRITIC-TOPSIS  
**Student:** Akshat Agrawal (23MIC0079)  
**Course:** CSI3020 — Advanced Graph Algorithms  

This document contains all the important source code, algorithms, and logic used throughout the project, organized by module.

---

## Table of Contents

1. [Centrality Computation (centralities.py)](#1-centrality-computation)
2. [CRITIC Objective Weighting (critic.py)](#2-critic-objective-weighting)
3. [TOPSIS Ranking (topsis.py)](#3-topsis-ranking)
4. [Targeted Attack Simulation (evaluation.py)](#4-targeted-attack-simulation)
5. [Cascading Failure Engine (cascading_failure.py)](#5-cascading-failure-engine)
6. [Bootstrap Uncertainty Quantification (uncertainty.py)](#6-bootstrap-uncertainty-quantification)
7. [Adversarial Robustness Testing (adversarial.py)](#7-adversarial-robustness-testing)
8. [Temporal Evolution Analysis (temporal_analysis.py)](#8-temporal-evolution-analysis)
9. [Sensitivity Analysis (sensitivity_analysis.py)](#9-sensitivity-analysis)
10. [FastAPI REST Backend (api/main.py)](#10-fastapi-rest-backend)
11. [Mathematical Unit Tests (tests/test_critic_topsis.py)](#11-mathematical-unit-tests)

---

## 1. Centrality Computation

**File:** `src/centralities.py`  
**Purpose:** Compute all 7 centrality measures for every node in the graph, producing the Decision Matrix.

### 1.1 H-Index Centrality (Custom Implementation)

Unlike the other 6 metrics (which use NetworkX built-ins), H-Index is implemented from scratch:

```python
def compute_hindex(G: nx.Graph) -> Dict[int, int]:
    """
    H-index of a node = largest h such that the node has at least h
    neighbors with degree >= h.
    """
    h_index = {}
    degrees = dict(G.degree())

    for node in G.nodes():
        neighbor_degrees = sorted(
            [degrees[n] for n in G.neighbors(node)], reverse=True
        )

        if not neighbor_degrees:
            h_index[node] = 0
            continue

        h = 0
        for i, deg in enumerate(neighbor_degrees):
            if deg >= i + 1:
                h = i + 1
            else:
                break
        h_index[node] = h

    return h_index
```

### 1.2 Building the Decision Matrix

```python
def compute_all_centralities(G: nx.Graph, verbose: bool = True) -> pd.DataFrame:
    """
    Compute all centrality measures and assemble the Decision Matrix X.
    Returns DataFrame with nodes as rows, 7 centralities as columns.
    """
    nodes = list(G.nodes())

    degree      = compute_degree_centrality(G)          # nx.degree_centrality
    betweenness = compute_betweenness_centrality(G)     # nx.betweenness_centrality
    closeness   = compute_closeness_centrality(G)       # nx.closeness_centrality
    eigenvector = compute_eigenvector_centrality(G)     # nx.eigenvector_centrality
    pagerank    = compute_pagerank(G)                   # nx.pagerank
    kshell      = compute_kshell(G)                     # nx.core_number
    hindex      = compute_hindex(G)                     # Custom implementation

    data = {
        'degree':      [degree[n] for n in nodes],
        'betweenness': [betweenness[n] for n in nodes],
        'closeness':   [closeness[n] for n in nodes],
        'eigenvector': [eigenvector[n] for n in nodes],
        'pagerank':    [pagerank[n] for n in nodes],
        'kshell':      [kshell[n] for n in nodes],
        'hindex':      [hindex[n] for n in nodes],
    }

    df = pd.DataFrame(data, index=nodes)
    df.index.name = 'node'
    return df
```

---

## 2. CRITIC Objective Weighting

**File:** `src/critic.py`  
**Purpose:** Derive objective criterion weights based on contrast intensity and inter-criteria conflict.

### 2.1 Min-Max Normalization (with Zero-Variance Fallback)

```python
def normalize_minmax(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale each column to [0, 1].
    Handles zero-variance columns (e.g. complete graphs) by setting to 0.5.
    """
    df_norm = df.copy()
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val - min_val > 0:
            df_norm[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0.5  # Fallback for zero variance
    return df_norm
```

### 2.2 CRITIC Weight Computation (Core Algorithm)

```python
def compute_critic_weights(df: pd.DataFrame,
                           normalization: str = 'minmax',
                           verbose: bool = True) -> Tuple[pd.Series, Dict]:
    """
    CRITIC weight derivation:
      Step 1: Normalize data (min-max or z-score)
      Step 2: Compute standard deviation σ_j for each criterion
      Step 3: Compute Pearson correlation matrix ρ_jk
      Step 4: Information content  C_j = σ_j × Σ(1 - ρ_jk)
      Step 5: Normalized weights   w_j = C_j / Σ C_j
    """
    # Step 1: Normalize
    if normalization == 'minmax':
        df_norm = normalize_minmax(df)
    else:
        df_norm = normalize_zscore(df)

    # Step 2: Standard deviations (contrast intensity)
    std_devs = df_norm.std()

    # Step 3: Correlation matrix (redundancy detection)
    corr_matrix = df_norm.corr()

    # Step 4: Information content
    info_content = {}
    for col in df.columns:
        conflict_sum = 0
        for other_col in df.columns:
            corr_val = corr_matrix.loc[col, other_col]
            if pd.isna(corr_val):
                corr_val = 0  # Treat NaN as no correlation
            conflict_sum += (1 - corr_val)
        info_content[col] = std_devs[col] * conflict_sum

    info_content = pd.Series(info_content)

    # Handle edge case: all info_content is 0 (e.g. complete graph)
    if info_content.sum() == 0 or pd.isna(info_content.sum()):
        weights = pd.Series({col: 1.0 / len(df.columns) for col in df.columns})
    else:
        # Step 5: Normalize to get weights
        weights = info_content / info_content.sum()

    details = {
        'normalized_df': df_norm,
        'std_devs': std_devs,
        'correlation_matrix': corr_matrix,
        'info_content': info_content
    }

    return weights, details
```

---

## 3. TOPSIS Ranking

**File:** `src/topsis.py`  
**Purpose:** Rank all nodes by their closeness to the theoretical ideal-best solution.

### 3.1 Vector Normalization

```python
def normalize_vector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vector normalization: r_ij = x_ij / sqrt(Σ x_ij²)
    Ensures the sum of squares of each column equals 1.
    """
    df_norm = df.copy()
    for col in df.columns:
        norm_factor = np.sqrt((df[col] ** 2).sum())
        if norm_factor > 0:
            df_norm[col] = df[col] / norm_factor
        else:
            df_norm[col] = 0
    return df_norm
```

### 3.2 Ideal Solutions (PIS and NIS)

```python
def compute_ideal_solutions(df_weighted, criteria_types=None):
    """
    PIS (A+) = column-wise maximum (best possible node)
    NIS (A-) = column-wise minimum (worst possible node)
    All centralities are 'benefit' criteria (higher = better).
    """
    if criteria_types is None:
        criteria_types = {col: 'benefit' for col in df_weighted.columns}

    ideal_best  = pd.Series(index=df_weighted.columns, dtype=float)
    ideal_worst = pd.Series(index=df_weighted.columns, dtype=float)

    for col in df_weighted.columns:
        if criteria_types.get(col, 'benefit') == 'benefit':
            ideal_best[col]  = df_weighted[col].max()
            ideal_worst[col] = df_weighted[col].min()
        else:
            ideal_best[col]  = df_weighted[col].min()
            ideal_worst[col] = df_weighted[col].max()

    return ideal_best, ideal_worst
```

### 3.3 Euclidean Distance & Closeness Coefficient

```python
def compute_distances(df_weighted, ideal_best, ideal_worst):
    """
    D+ = sqrt(Σ (v_ij - v_j+)²)   — distance to ideal best
    D- = sqrt(Σ (v_ij - v_j-)²)   — distance to ideal worst
    """
    dist_best  = np.sqrt(((df_weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((df_weighted - ideal_worst) ** 2).sum(axis=1))
    return dist_best, dist_worst


def compute_closeness_coefficient(dist_best, dist_worst):
    """
    C_i* = D- / (D+ + D-)
    Range: [0, 1]. Higher = more critical node.
    """
    total_dist = dist_best + dist_worst
    closeness = np.where(total_dist > 0, dist_worst / total_dist, 0)
    return pd.Series(closeness, index=dist_best.index, name='closeness')
```

### 3.4 Complete TOPSIS Pipeline

```python
def topsis_rank(df, weights, verbose=True):
    """
    Full TOPSIS ranking:
      1. Vector normalization
      2. Apply CRITIC weights → weighted normalized matrix
      3. Compute PIS (A+) and NIS (A-)
      4. Compute Euclidean distances S+ and S-
      5. Compute closeness coefficient C*
      6. Rank nodes by descending C*
    """
    df_normalized = normalize_vector(df)
    df_weighted   = apply_weights(df_normalized, weights)
    ideal_best, ideal_worst = compute_ideal_solutions(df_weighted)
    dist_best, dist_worst   = compute_distances(df_weighted, ideal_best, ideal_worst)
    closeness = compute_closeness_coefficient(dist_best, dist_worst)

    results = pd.DataFrame({
        'closeness': closeness,
        'dist_to_best': dist_best,
        'dist_to_worst': dist_worst
    })
    results['rank'] = results['closeness'].rank(ascending=False).astype(int)
    results = results.sort_values('rank')

    return results, {
        'normalized': df_normalized,
        'weighted': df_weighted,
        'ideal_best': ideal_best,
        'ideal_worst': ideal_worst
    }
```

---

## 4. Targeted Attack Simulation

**File:** `src/evaluation.py`  
**Purpose:** Validate critical node rankings by simulating node removal and measuring connectivity collapse.

### 4.1 Attack Simulation

```python
def simulate_targeted_attack(G, node_ranking, fractions=None, verbose=True):
    """
    Iteratively remove top-ranked nodes and measure LCC collapse.
    Default fractions: [1%, 2%, 5%, 10%, 15%, 20%, 25%, 30%]
    """
    if fractions is None:
        fractions = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    original_size = G.number_of_nodes()
    G_copy = G.copy()
    nodes_removed = 0

    results = [{'fraction_removed': 0.0, 'lcc_fraction': 1.0, ...}]

    for frac in fractions:
        target_removed = int(frac * original_size)
        while nodes_removed < target_removed and nodes_removed < len(node_ranking):
            node = node_ranking[nodes_removed]
            if node in G_copy:
                G_copy.remove_node(node)
            nodes_removed += 1

        results.append({
            'fraction_removed': frac,
            'lcc_size': get_largest_component_size(G_copy),
            'lcc_fraction': get_largest_component_fraction(G_copy, original_size),
            'efficiency': compute_global_efficiency(G_copy),
        })

    return pd.DataFrame(results)
```

### 4.2 Attack Effectiveness Metric

```python
def compute_attack_effectiveness(attack_results, metric='lcc_fraction'):
    """
    Effectiveness = 1 - (AUC under LCC curve / max_area)
    Higher = the method's ranking causes faster network collapse.
    Uses numpy.trapezoid for numerical integration.
    """
    for method, df in attack_results.items():
        x, y = df['fraction_removed'].values, df[metric].values
        auc = np.trapezoid(y, x)
        max_area = x.max()
        eff = 1 - (auc / max_area if max_area > 0 else 0)
```

---

## 5. Cascading Failure Engine

**File:** `src/cascading_failure.py`  
**Purpose:** Model real-world load redistribution and avalanche propagation.

### 5.1 Cascading Failure Simulation

```python
def simulate_cascading_failure(G, initial_failures, capacity_factor=1.2,
                                load_method='betweenness', max_iterations=100):
    """
    Model: Each node has Capacity = InitialLoad × capacity_factor.
    When load > capacity after redistribution, the node fails.

    Steps:
      1. Compute initial loads (betweenness-based)
      2. Set capacities = load × (1 + α)
      3. Remove initial top-ranked failures
      4. Recompute loads on surviving network
      5. Any node where new_load > capacity → fails
      6. Repeat until no new failures (avalanche stops)
    """
    H = G.copy()
    initial_loads = compute_all_loads(H, load_method)
    capacities = {node: load * capacity_factor
                  for node, load in initial_loads.items()}

    all_failed = set(initial_failures)
    H.remove_nodes_from(initial_failures)

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        if H.number_of_nodes() == 0:
            break

        current_loads = compute_all_loads(H, load_method)
        new_failures = [node for node in H.nodes()
                        if current_loads.get(node, 0) > capacities.get(node, float('inf'))]

        if not new_failures:
            break  # Cascade stopped

        all_failed.update(new_failures)
        H.remove_nodes_from(new_failures)

    return {
        'total_failures': len(all_failed),
        'cascade_iterations': iteration,
        'survival_rate': (G.number_of_nodes() - len(all_failed)) / G.number_of_nodes(),
    }
```

---

## 6. Bootstrap Uncertainty Quantification

**File:** `src/uncertainty.py`  
**Purpose:** Measure ranking stability using bootstrap resampling.

### 6.1 Bootstrap Ranking

```python
def bootstrap_rankings(G, n_bootstrap=100, sample_fraction=0.8):
    """
    For each iteration:
      1. Sample 80% of edges with replacement
      2. Reconstruct graph, recompute centralities + CRITIC + TOPSIS
      3. Record each node's rank

    Returns distribution of ranks per node.
    """
    all_ranks = {node: [] for node in G.nodes()}
    edges = list(G.edges())

    for i in range(n_bootstrap):
        n_sample = int(len(edges) * sample_fraction)
        sample_indices = np.random.choice(len(edges), size=n_sample, replace=True)

        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        for idx in sample_indices:
            H.add_edge(*edges[idx])

        try:
            df = compute_all_centralities(H, verbose=False)
            weights, _ = compute_critic_weights(df, verbose=False)
            results, _ = topsis_rank(df, weights, verbose=False)

            for node in G.nodes():
                if node in results.index:
                    all_ranks[node].append(int(results.loc[node, 'rank']))
        except:
            continue

    return all_ranks
```

### 6.2 Ranking Stability Metrics

```python
def compute_ranking_stability(rank_distributions):
    """
    Classify nodes:
      Stable:   σ < 2    → rank barely changes under perturbation
      Unstable: σ >= 5   → rank fluctuates heavily
    """
    stabilities = [np.std(ranks) for ranks in rank_distributions.values()
                   if len(ranks) > 1]
    return {
        'mean_rank_std': np.mean(stabilities),
        'stable_nodes': sum(1 for s in stabilities if s < 2),
        'unstable_nodes': sum(1 for s in stabilities if s >= 5),
    }
```

---

## 7. Adversarial Robustness Testing

**File:** `src/adversarial.py`  
**Purpose:** Test if strategic edge manipulation can fool the CRITIC-TOPSIS ranking.

### 7.1 Sybil Attack

```python
def sybil_attack(G, n_sybils=5, target_node=None):
    """
    Add fake 'sybil' nodes connected to target to inflate its importance.
    Tests if CRITIC-TOPSIS is fooled by artificial hub creation.
    """
    H = G.copy()
    sybil_nodes = []
    max_node_val = max(H.nodes()) if all(isinstance(n, int) for n in H.nodes()) else -1

    for i in range(n_sybils):
        sybil = max_node_val + i + 1
        sybil_nodes.append(sybil)
        H.add_node(sybil)

        if target_node is not None:
            H.add_edge(sybil, target_node)

        existing = [n for n in G.nodes() if n != target_node]
        if existing:
            H.add_edge(sybil, np.random.choice(existing))

    return H, sybil_nodes
```

### 7.2 Robustness Grading

```python
def test_robustness(G, target_node=None, n_trials=5):
    """
    Run 8 attack variants (3 edge-addition, 3 edge-removal, 2 sybil).
    A successful attack = rank change ≥ 3 positions.

    Robustness Grade:
      A: < 20% of attacks succeed
      B: < 40% of attacks succeed
      C: < 60% of attacks succeed
      D: ≥ 60% of attacks succeed
    """
    vulnerability_score = len(successful_attacks) / len(all_results) * 100
    grade = 'A' if vulnerability_score < 20 else \
            'B' if vulnerability_score < 40 else \
            'C' if vulnerability_score < 60 else 'D'
```

---

## 8. Temporal Evolution Analysis

**File:** `src/temporal_analysis.py`  
**Purpose:** Track how node criticality changes over time and predict future critical nodes.

### 8.1 Network Snapshot Generation

```python
def create_network_snapshot(G, remove_fraction=0.1, add_fraction=0.1):
    """
    Simulate network evolution by randomly adding/removing edges.
    Models real-world topology drift over time.
    """
    H = G.copy()
    edges = list(H.edges())

    # Remove some edges
    n_remove = int(len(edges) * remove_fraction)
    remove_indices = np.random.choice(len(edges), size=n_remove, replace=False)
    for idx in remove_indices:
        H.remove_edge(*edges[idx])

    # Add new edges
    nodes = list(H.nodes())
    n_add = int(len(edges) * add_fraction)
    added = 0
    while added < n_add:
        u, v = np.random.choice(nodes, size=2, replace=False)
        if not H.has_edge(u, v):
            H.add_edge(u, v)
            added += 1

    return H
```

### 8.2 Rising Star Detection

```python
def predict_future_critical(snapshots, top_k=10):
    """
    Identify nodes trending toward criticality ('Rising Stars').
    Trend = last_rank - first_rank. Negative trend = becoming more critical.
    """
    analysis = analyze_temporal_rankings(snapshots, top_k)

    current_top = set(analysis.nsmallest(top_k, f't{len(snapshots)-1}_rank')['node'])

    # Rising stars: not yet in top-k but improving rapidly
    not_current = analysis[~analysis['node'].isin(current_top)]
    rising = not_current[not_current['trend'] < 0].nsmallest(5, 'trend')

    # Stable critical: consistently in top-k with low variance
    stable = analysis[analysis['node'].isin(current_top)].nsmallest(5, 'stability')

    return {
        'current_critical': list(current_top),
        'rising_stars': rising[['node', 'trend', 'avg_rank']].to_dict('records'),
        'stable_critical': stable[['node', 'stability', 'avg_rank']].to_dict('records'),
    }
```

### 8.3 Adaptive CRITIC Weights (Exponential Decay)

```python
def compute_adaptive_weights(snapshots, decay=0.3):
    """
    Exponentially-weighted average of CRITIC weights across time.
    Recent snapshots matter more: time_weight = exp(-decay × (T - t))
    """
    n = len(snapshots)
    all_weights = []
    for G in snapshots:
        df = compute_all_centralities(G, verbose=False)
        weights, _ = compute_critic_weights(df, verbose=False)
        all_weights.append(weights)

    # Exponential time weights
    time_weights = np.array([np.exp(-decay * (n - 1 - t)) for t in range(n)])
    time_weights = time_weights / time_weights.sum()

    # Weighted average
    adaptive = {}
    for m in all_weights[0].index:
        vals = np.array([w[m] for w in all_weights])
        adaptive[m] = np.dot(vals, time_weights)

    result = pd.Series(adaptive)
    return result / result.sum()
```

---

## 9. Sensitivity Analysis

**File:** `src/sensitivity_analysis.py`  
**Purpose:** Measure how much the ranking depends on individual centrality metrics.

### 9.1 Centrality Removal Impact

```python
def sensitivity_to_centrality_removal(G):
    """
    Remove each centrality one at a time and measure top-10 overlap
    with the full-metric ranking.
    High impact = the metric provides unique, irreplaceable information.
    """
    df = compute_all_centralities(G, verbose=False)
    weights_base, _ = compute_critic_weights(df, verbose=False)
    results_base, _ = topsis_rank(df, weights_base, verbose=False)
    ranking_base = results_base.sort_values('rank').index.tolist()

    results = []
    for col in df.columns:
        df_reduced = df.drop(columns=[col])
        weights, _ = compute_critic_weights(df_reduced, verbose=False)
        results_reduced, _ = topsis_rank(df_reduced, weights, verbose=False)
        ranking_reduced = results_reduced.sort_values('rank').index.tolist()

        similarity = compute_ranking_similarity(ranking_base, ranking_reduced)
        results.append({
            'removed_centrality': col,
            'top_10_overlap': similarity['top_k_overlap'],
            'impact': 100 - similarity['top_k_overlap']
        })

    return pd.DataFrame(results).sort_values('impact', ascending=False)
```

---

## 10. FastAPI REST Backend

**File:** `api/main.py`  
**Purpose:** Serve the CRITIC-TOPSIS pipeline as a RESTful API for the React frontend.

### 10.1 API Initialization & Middleware

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Critical Node Detection API",
    description="CRITIC-TOPSIS multi-attribute framework.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 10.2 Pydantic Request Schema

```python
from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):
    network: str = Field("karate", description="Name of the benchmark network")
    edges: list = Field(default=None, description="Custom edge list [[u,v], ...]")
    top_k: int = Field(10, ge=1, le=100)
```

### 10.3 Key API Endpoints

| Endpoint | Method | Purpose |
|:---|:---|:---|
| `/discovery` | POST | Run full CRITIC-TOPSIS pipeline, return rankings + weights |
| `/impact` | POST | Run targeted attack simulation, return collapse curves |
| `/cascade` | POST | Run cascading failure simulation |
| `/temporal` | POST | Run temporal evolution + rising star analysis |
| `/domain` | POST | Run domain-aware weighted analysis |
| `/robustness` | POST | Run bootstrap + adversarial + sensitivity analysis |

---

## 11. Mathematical Unit Tests

**File:** `tests/test_critic_topsis.py`  
**Purpose:** Prove algorithm stability under extreme graph conditions.

### 11.1 Complete Graph Edge Case (Zero Variance)

```python
def test_extreme_density_complete_graph():
    """
    Complete Graph K_n: every node is identical.
    All centralities are the same → σ_j = 0 → info_content = 0.
    CRITIC must fall back to equal weights without crashing.
    """
    G = nx.complete_graph(10)
    df_centrality = compute_all_centralities(G, verbose=False)
    weights, details = compute_critic_weights(df_centrality, verbose=False)

    expected_weight = 1.0 / len(df_centrality.columns)
    for col in df_centrality.columns:
        assert np.isclose(weights[col], expected_weight), \
            f"Expected uniform weight {expected_weight}, got {weights[col]}"
```

### 11.2 Disconnected Graph Edge Case (Zero Distance)

```python
def test_disconnected_network():
    """
    Empty Graph (no edges): all nodes are isolated.
    D+ and D- are both 0 → TOPSIS must not divide by zero.
    All nodes should have identical rank (tied).
    """
    G = nx.empty_graph(5)
    df_centrality = compute_all_centralities(G, verbose=False)
    weights, _ = compute_critic_weights(df_centrality, verbose=False)
    results, _ = topsis_rank(df_centrality, weights, verbose=False)

    assert len(results['rank'].unique()) == 1, \
        "All nodes in a disconnected graph should tie with the exact same rank"
```

---

*End of Source Code Documentation*
