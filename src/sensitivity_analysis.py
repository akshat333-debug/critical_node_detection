"""
Sensitivity Analysis Module
============================
Analyze how rankings change with different parameters and methods.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from centralities import compute_all_centralities
from critic import compute_critic_weights, normalize_minmax, normalize_zscore
from topsis import topsis_rank


def compute_ranking_similarity(ranking1: List, ranking2: List, k: int = 10) -> Dict:
    """
    Compare two rankings using multiple similarity metrics.
    
    Args:
        ranking1, ranking2: Lists of ranked nodes
        k: Top-k to consider for overlap
    
    Returns:
        Dictionary with similarity metrics
    """
    # Top-k overlap (Jaccard)
    top_k_1 = set(ranking1[:k])
    top_k_2 = set(ranking2[:k])
    overlap = len(top_k_1 & top_k_2) / k * 100
    
    # Kendall Tau-like for top-k
    common = top_k_1 & top_k_2
    if len(common) >= 2:
        pos1 = {node: i for i, node in enumerate(ranking1) if node in common}
        pos2 = {node: i for i, node in enumerate(ranking2) if node in common}
        
        concordant = 0
        discordant = 0
        common_list = list(common)
        for i in range(len(common_list)):
            for j in range(i+1, len(common_list)):
                a, b = common_list[i], common_list[j]
                if (pos1[a] < pos1[b]) == (pos2[a] < pos2[b]):
                    concordant += 1
                else:
                    discordant += 1
        
        total_pairs = concordant + discordant
        rank_correlation = (concordant - discordant) / total_pairs if total_pairs > 0 else 0
    else:
        rank_correlation = 0
    
    return {
        'top_k_overlap': overlap,
        'rank_correlation': rank_correlation * 100,
        'common_nodes': len(common) if 'common' in dir() else 0
    }


def sensitivity_to_normalization(G: nx.Graph, verbose: bool = False) -> pd.DataFrame:
    """
    Analyze how normalization method affects rankings.
    """
    df = compute_all_centralities(G, verbose=False)
    
    rankings = {}
    for norm in ['minmax', 'zscore']:
        weights, _ = compute_critic_weights(df, normalization=norm, verbose=False)
        results, _ = topsis_rank(df, weights, verbose=False)
        rankings[norm] = results.sort_values('rank').index.tolist()
    
    similarity = compute_ranking_similarity(rankings['minmax'], rankings['zscore'])
    
    return pd.DataFrame([{
        'comparison': 'Min-Max vs Z-Score',
        'top_10_overlap': similarity['top_k_overlap'],
        'rank_correlation': similarity['rank_correlation']
    }])


def sensitivity_to_centrality_removal(G: nx.Graph, verbose: bool = False) -> pd.DataFrame:
    """
    Analyze how removing each centrality affects rankings.
    """
    df = compute_all_centralities(G, verbose=False)
    
    # Baseline with all centralities
    weights_base, _ = compute_critic_weights(df, verbose=False)
    results_base, _ = topsis_rank(df, weights_base, verbose=False)
    ranking_base = results_base.sort_values('rank').index.tolist()
    
    results = []
    for col in df.columns:
        # Remove one centrality
        df_reduced = df.drop(columns=[col])
        weights, _ = compute_critic_weights(df_reduced, verbose=False)
        results_reduced, _ = topsis_rank(df_reduced, weights, verbose=False)
        ranking_reduced = results_reduced.sort_values('rank').index.tolist()
        
        similarity = compute_ranking_similarity(ranking_base, ranking_reduced)
        
        results.append({
            'removed_centrality': col,
            'top_10_overlap': similarity['top_k_overlap'],
            'rank_correlation': similarity['rank_correlation'],
            'impact': 100 - similarity['top_k_overlap']  # Higher = more impact
        })
    
    return pd.DataFrame(results).sort_values('impact', ascending=False)


def sensitivity_to_top_k(G: nx.Graph, k_values: List[int] = None) -> pd.DataFrame:
    """
    Analyze ranking consistency at different k values.
    """
    if k_values is None:
        k_values = [5, 10, 15, 20]
    
    df = compute_all_centralities(G, verbose=False)
    weights, _ = compute_critic_weights(df, verbose=False)
    results, _ = topsis_rank(df, weights, verbose=False)
    topsis_ranking = results.sort_values('rank').index.tolist()
    
    data = []
    # Compare TOPSIS with each single metric at different k
    for metric in df.columns:
        metric_ranking = df[metric].sort_values(ascending=False).index.tolist()
        
        for k in k_values:
            sim = compute_ranking_similarity(topsis_ranking, metric_ranking, k=k)
            data.append({
                'metric': metric,
                'k': k,
                'overlap': sim['top_k_overlap']
            })
    
    return pd.DataFrame(data)


def run_full_sensitivity_analysis(G: nx.Graph) -> Dict[str, pd.DataFrame]:
    """
    Run complete sensitivity analysis.
    """
    return {
        'normalization': sensitivity_to_normalization(G),
        'centrality_impact': sensitivity_to_centrality_removal(G),
        'top_k_stability': sensitivity_to_top_k(G)
    }


if __name__ == "__main__":
    print("Testing Sensitivity Analysis")
    print("=" * 50)
    
    G = nx.karate_club_graph()
    
    results = run_full_sensitivity_analysis(G)
    
    print("\n1. Normalization Sensitivity:")
    print(results['normalization'])
    
    print("\n2. Centrality Impact (removing each):")
    print(results['centrality_impact'])
    
    print("\n3. Top-k Stability (TOPSIS vs single metrics):")
    print(results['top_k_stability'].pivot(index='metric', columns='k', values='overlap'))
