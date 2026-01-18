"""
Explainable AI Module for Critical Node Detection
==================================================
Generate natural language explanations for why nodes are critical.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from centralities import compute_all_centralities
from critic import compute_critic_weights
from topsis import topsis_rank


CENTRALITY_DESCRIPTIONS = {
    'degree': {
        'high': "has many direct connections ({value:.0f} neighbors)",
        'role': "a hub that connects many nodes directly",
        'analogy': "like a popular person with many friends"
    },
    'betweenness': {
        'high': "bridges different communities (controls {value:.1%} of shortest paths)",
        'role': "a critical bridge between groups",
        'analogy': "like a translator between different departments"
    },
    'closeness': {
        'high': "can reach all nodes quickly (avg distance: {value:.2f})",
        'role': "central to information flow",
        'analogy': "like a news anchor that can spread info fast"
    },
    'eigenvector': {
        'high': "connected to other important nodes (influence score: {value:.3f})",
        'role': "influential through powerful connections",
        'analogy': "like knowing influential people"
    },
    'pagerank': {
        'high': "receives attention from important nodes (PR: {value:.4f})",
        'role': "authoritative in the network",
        'analogy': "like a highly-cited researcher"
    },
    'kshell': {
        'high': "in the network's core (k-shell: {value:.0f})",
        'role': "structurally central to the network backbone",
        'analogy': "like being in the inner circle"
    },
    'hindex': {
        'high': "has many well-connected neighbors (h-index: {value:.0f})",
        'role': "both connected and influential",
        'analogy': "like knowing many important people"
    }
}


def get_percentile_rank(series: pd.Series, value) -> float:
    """Get percentile rank of a value in a series."""
    return (series < value).mean() * 100


def explain_node(node, 
                 df_centrality: pd.DataFrame,
                 weights: pd.Series,
                 topsis_results: pd.DataFrame,
                 verbose: bool = False) -> Dict:
    """
    Generate a detailed explanation for why a node is critical.
    
    Returns:
        Dictionary with natural language explanations
    """
    if node not in df_centrality.index:
        return {'error': f"Node {node} not found"}
    
    node_data = df_centrality.loc[node]
    topsis_rank = int(topsis_results.loc[node, 'rank'])
    topsis_score = topsis_results.loc[node, 'closeness']
    
    # Overall summary
    total_nodes = len(df_centrality)
    percentile = 100 - (topsis_rank / total_nodes * 100)
    
    if topsis_rank <= 3:
        criticality = "HIGHLY CRITICAL"
        emoji = "🔴"
    elif topsis_rank <= 10:
        criticality = "Very Critical"
        emoji = "🟠"
    elif topsis_rank <= total_nodes * 0.2:
        criticality = "Moderately Critical"
        emoji = "🟡"
    else:
        criticality = "Not Critical"
        emoji = "🟢"
    
    # Find top contributing factors
    contributions = []
    for metric in df_centrality.columns:
        value = node_data[metric]
        weight = weights[metric]
        pct = get_percentile_rank(df_centrality[metric], value)
        contributions.append({
            'metric': metric,
            'value': value,
            'weight': weight,
            'percentile': pct,
            'weighted_contribution': weight * (pct / 100)
        })
    
    contributions = sorted(contributions, key=lambda x: x['weighted_contribution'], reverse=True)
    
    # Generate natural language
    top_reasons = []
    for c in contributions[:3]:
        metric = c['metric']
        if metric in CENTRALITY_DESCRIPTIONS:
            desc = CENTRALITY_DESCRIPTIONS[metric]['high'].format(value=c['value'])
            top_reasons.append(f"**{metric.capitalize()}**: {desc}")
    
    # Main explanation
    main_explanation = f"""
{emoji} **Node {node}** is **{criticality}** (Rank #{topsis_rank} of {total_nodes}, top {percentile:.0f}%)

**TOPSIS Score**: {topsis_score:.4f}

**Why this node matters:**
"""
    
    for reason in top_reasons:
        main_explanation += f"\n- {reason}"
    
    # Impact statement
    if topsis_rank <= 5:
        impact = f"\n\n⚠️ **Impact**: Removing this node would likely cause significant network damage."
    elif topsis_rank <= 10:
        impact = f"\n\n⚠️ **Impact**: This node is important for network connectivity."
    else:
        impact = ""
    
    return {
        'node': node,
        'rank': topsis_rank,
        'criticality': criticality,
        'score': topsis_score,
        'percentile': percentile,
        'main_explanation': main_explanation + impact,
        'top_factors': contributions[:3],
        'all_factors': contributions
    }


def explain_top_k(df_centrality: pd.DataFrame,
                  weights: pd.Series,
                  topsis_results: pd.DataFrame,
                  k: int = 5) -> List[Dict]:
    """Generate explanations for top k critical nodes."""
    top_nodes = topsis_results.nsmallest(k, 'rank').index.tolist()
    
    explanations = []
    for node in top_nodes:
        exp = explain_node(node, df_centrality, weights, topsis_results)
        explanations.append(exp)
    
    return explanations


def generate_summary_report(df_centrality: pd.DataFrame,
                            weights: pd.Series,
                            topsis_results: pd.DataFrame) -> str:
    """Generate a natural language summary of the analysis."""
    
    n_nodes = len(df_centrality)
    top_5 = topsis_results.nsmallest(5, 'rank').index.tolist()
    
    # Most influential metric
    top_metric = weights.idxmax()
    top_weight = weights.max()
    
    report = f"""
# Critical Node Analysis Report

## Summary
Analyzed **{n_nodes} nodes** using 7 centrality measures combined with CRITIC-TOPSIS.

## Key Findings

### Most Important Metric
**{top_metric.capitalize()}** has the highest CRITIC weight ({top_weight:.1%}), meaning it provides the most discriminating information for this network.

### Top 5 Critical Nodes
"""
    
    for i, node in enumerate(top_5, 1):
        score = topsis_results.loc[node, 'closeness']
        report += f"{i}. **Node {node}** (Score: {score:.4f})\n"
    
    report += f"""
### Recommendation
Focus protection efforts on nodes {top_5[:3]}. These nodes, if compromised, would cause the greatest disruption to network functionality.
"""
    
    return report


if __name__ == "__main__":
    print("Explainable AI Module Test")
    print("=" * 50)
    
    G = nx.karate_club_graph()
    df = compute_all_centralities(G, verbose=False)
    weights, _ = compute_critic_weights(df, verbose=False)
    results, _ = topsis_rank(df, weights, verbose=False)
    
    exp = explain_node(0, df, weights, results)
    print(exp['main_explanation'])
