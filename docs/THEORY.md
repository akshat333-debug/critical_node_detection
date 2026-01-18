# Critical Node Detection: CRITIC-TOPSIS Framework

## Theory and Concepts

This document explains the core concepts behind the multi-attribute critical node detection framework.

---

## 1. Critical Node Detection: What and Why

**What is it?**
Critical node detection identifies the most important nodes in a network—those whose removal would maximally disrupt network functionality. Unlike simple degree-based approaches, sophisticated methods consider multiple aspects of node importance.

**Why does it matter?**
- **Internet/Infrastructure**: Identifying routers or power substations whose failure would cascade into widespread outages
- **Social Networks**: Finding influential spreaders for viral marketing or disease containment
- **Biological Networks**: Discovering drug targets (proteins) that would disrupt disease pathways
- **Security**: Protecting key nodes against targeted attacks

**Example**: In a social network, a node with moderate connections but positioned as a "bridge" between communities may be more critical than a highly connected node within a single community.

---

## 2. Centrality Measures Explained

### 2.1 Degree Centrality
**Intuition**: How many direct connections does a node have?
- Formula: `C_D(v) = degree(v) / (n-1)`
- High degree = local hub, many direct contacts
- **Limitation**: Ignores global network structure

### 2.2 Betweenness Centrality
**Intuition**: How often does a node lie on shortest paths between other pairs?
- Formula: `C_B(v) = Σ σ_st(v) / σ_st` (fraction of shortest paths through v)
- High betweenness = bridge/broker, controls information flow
- **Limitation**: Computationally expensive O(n·m); ignores non-geodesic paths

### 2.3 Closeness Centrality
**Intuition**: How quickly can a node reach all others?
- Formula: `C_C(v) = (n-1) / Σ d(v,u)`
- High closeness = central position, can disseminate information quickly
- **Limitation**: Sensitive to disconnected components

### 2.4 Eigenvector Centrality
**Intuition**: A node is important if connected to other important nodes (recursive definition)
- Computed as the principal eigenvector of the adjacency matrix
- High eigenvector = well-connected to influential nodes
- **Limitation**: Concentrated in dense cores; less useful for periphery

### 2.5 PageRank
**Intuition**: Similar to eigenvector but with damping (random surfer model)
- Models a random walker who follows edges with probability α and jumps randomly with 1-α
- More robust than eigenvector centrality
- Originally used by Google to rank web pages

### 2.6 K-Shell (Core Number)
**Intuition**: How deep is a node in the network's core structure?
- Computed by iteratively removing nodes with degree < k
- High k-shell = part of the dense network backbone
- **Limitation**: Many nodes share the same k-shell value (coarse ranking)

### 2.7 H-Index
**Intuition**: Balance between having many neighbors AND those neighbors being well-connected
- H-index = largest h such that node has ≥ h neighbors with degree ≥ h
- Captures local influence quality, not just quantity

---

## 3. When Centralities Disagree

A node can rank high in one centrality but low in another:

| Scenario | High In | Low In |
|----------|---------|--------|
| Bridge between communities | Betweenness | Degree, K-shell |
| Hub in single community | Degree, Eigenvector | Betweenness |
| Peripheral connector | Closeness | K-shell |
| Core node with redundant neighbors | K-shell | Betweenness |

**This is why we need multi-attribute methods**: No single centrality captures all aspects of node importance.

---

## 4. CRITIC Method (Objective Weighting)

**Problem**: How do we combine 7 centrality measures? What weight should each get?

**CRITIC Idea**: Let the data determine weights objectively:
1. **High variance** = criterion distinguishes well between alternatives → more weight
2. **Low correlation with others** = criterion provides unique information → more weight

**Formula**:
```
Information Content: C_j = σ_j × Σ(1 - r_jk)
Weight: w_j = C_j / Σ C_j
```

Where:
- σ_j = standard deviation of criterion j (after normalization)
- r_jk = Pearson correlation between criteria j and k

**Example**: If betweenness has high variance (distinguishes nodes well) and low correlation with degree (provides different information), it gets high weight.

---

## 5. TOPSIS Method (Ranking)

**Problem**: Given weighted criteria, how do we rank nodes?

**TOPSIS Idea**: The best alternative should be:
- Closest to the ideal best solution (A⁺)
- Farthest from the ideal worst solution (A⁻)

**Steps**:
1. **Normalize**: Vector normalization to make criteria comparable
2. **Weight**: Multiply by CRITIC weights
3. **Find ideals**:
   - A⁺ = {max of each criterion} (best possible node)
   - A⁻ = {min of each criterion} (worst possible node)
4. **Compute distances**:
   - D⁺ = Euclidean distance to A⁺
   - D⁻ = Euclidean distance to A⁻
5. **Closeness coefficient**: `C = D⁻ / (D⁺ + D⁻)`
   - Range [0, 1]: higher = better

**Result**: Nodes ranked by closeness coefficient = our critical node ranking.

---

## 6. Evaluation: Targeted Attack Simulation

**How do we know if our ranking is good?**

We simulate "targeted attacks" by removing nodes in order of their ranking:
1. Remove top-k% nodes according to each method
2. Measure network damage:
   - **Largest Connected Component (LCC)**: What fraction of nodes remain connected?
   - **Global Efficiency**: How well can information flow?

**Interpretation**:
- Method that causes **most damage** when its top nodes are removed = **best at identifying critical nodes**
- We compare CRITIC-TOPSIS against single centralities

---

## 7. Mathematical Summary

### Normalization (Min-Max)
```
x'_ij = (x_ij - min_j) / (max_j - min_j)
```

### CRITIC Weight Computation
```
σ_j = std(x'_j)                    # Standard deviation of criterion j
r_jk = corr(x'_j, x'_k)            # Correlation between j and k
C_j = σ_j × Σ_k(1 - r_jk)          # Information content
w_j = C_j / Σ C_j                   # Normalized weight
```

### TOPSIS Ranking
```
r_ij = x_ij / √(Σ x²_ij)           # Vector normalization
v_ij = w_j × r_ij                   # Weighted normalized matrix
A⁺ = {max(v_ij) for each j}        # Ideal best
A⁻ = {min(v_ij) for each j}        # Ideal worst
D⁺_i = √(Σ(v_ij - A⁺_j)²)         # Distance to ideal best
D⁻_i = √(Σ(v_ij - A⁻_j)²)         # Distance to ideal worst
C_i = D⁻_i / (D⁺_i + D⁻_i)        # Closeness coefficient
```

---

## References

1. Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995). Determining objective weights in multiple criteria problems: The CRITIC method.
2. Hwang, C.L., & Yoon, K. (1981). Multiple Attribute Decision Making: Methods and Applications.
3. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
