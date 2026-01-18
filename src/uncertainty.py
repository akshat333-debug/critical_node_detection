"""
Uncertainty Quantification Module
==================================
Compute confidence intervals and probability estimates for node rankings.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from centralities import compute_all_centralities
from critic import compute_critic_weights
from topsis import topsis_rank


def bootstrap_rankings(G: nx.Graph,
                       n_bootstrap: int = 100,
                       sample_fraction: float = 0.8) -> Dict:
    """
    Use bootstrap sampling to estimate ranking uncertainty.
    
    For each bootstrap iteration:
    1. Sample edges with replacement
    2. Recompute centralities and rankings
    3. Record each node's rank
    
    Returns distribution of ranks for each node.
    """
    all_ranks = {node: [] for node in G.nodes()}
    edges = list(G.edges())
    
    for i in range(n_bootstrap):
        # Bootstrap sample of edges
        n_sample = int(len(edges) * sample_fraction)
        sample_indices = np.random.choice(len(edges), size=n_sample, replace=True)
        
        # Create bootstrap graph
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        for idx in sample_indices:
            H.add_edge(*edges[idx])
        
        # Compute rankings
        try:
            df = compute_all_centralities(H, verbose=False)
            weights, _ = compute_critic_weights(df, verbose=False)
            results, _ = topsis_rank(df, weights, verbose=False)
            
            for node in G.nodes():
                if node in results.index:
                    all_ranks[node].append(int(results.loc[node, 'rank']))
                else:
                    all_ranks[node].append(len(G.nodes()))
        except:
            # If computation fails, skip this iteration
            continue
    
    return all_ranks


def compute_rank_confidence_intervals(rank_distributions: Dict,
                                      confidence_level: float = 0.95) -> pd.DataFrame:
    """
    Compute confidence intervals for each node's rank.
    """
    results = []
    
    for node, ranks in rank_distributions.items():
        if len(ranks) < 2:
            continue
            
        ranks = np.array(ranks)
        mean_rank = np.mean(ranks)
        std_rank = np.std(ranks)
        
        # Confidence interval
        alpha = 1 - confidence_level
        ci_low = np.percentile(ranks, alpha/2 * 100)
        ci_high = np.percentile(ranks, (1 - alpha/2) * 100)
        
        results.append({
            'node': node,
            'mean_rank': mean_rank,
            'std_rank': std_rank,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'rank_range': ci_high - ci_low
        })
    
    return pd.DataFrame(results).sort_values('mean_rank')


def probability_in_top_k(rank_distributions: Dict,
                         k: int = 10) -> pd.DataFrame:
    """
    Compute probability that each node is in the top-k.
    """
    results = []
    
    for node, ranks in rank_distributions.items():
        if len(ranks) == 0:
            continue
            
        ranks = np.array(ranks)
        prob_top_k = (ranks <= k).mean()
        
        results.append({
            'node': node,
            'prob_top_k': prob_top_k,
            'mean_rank': np.mean(ranks),
            'min_rank': np.min(ranks),
            'max_rank': np.max(ranks)
        })
    
    return pd.DataFrame(results).sort_values('prob_top_k', ascending=False)


def compute_ranking_stability(rank_distributions: Dict) -> Dict:
    """
    Compute overall stability metrics for the ranking.
    """
    stabilities = []
    for node, ranks in rank_distributions.items():
        if len(ranks) > 1:
            stabilities.append(np.std(ranks))
    
    return {
        'mean_rank_std': np.mean(stabilities),
        'max_rank_std': np.max(stabilities),
        'min_rank_std': np.min(stabilities),
        'stable_nodes': sum(1 for s in stabilities if s < 2),
        'unstable_nodes': sum(1 for s in stabilities if s >= 5)
    }


def full_uncertainty_analysis(G: nx.Graph,
                              n_bootstrap: int = 50,
                              top_k: int = 10,
                              confidence: float = 0.95) -> Dict:
    """
    Run complete uncertainty quantification.
    """
    # Bootstrap rankings
    distributions = bootstrap_rankings(G, n_bootstrap)
    
    # Compute metrics
    ci_df = compute_rank_confidence_intervals(distributions, confidence)
    prob_df = probability_in_top_k(distributions, top_k)
    stability = compute_ranking_stability(distributions)
    
    # Top-k with high confidence
    high_conf_top_k = prob_df[prob_df['prob_top_k'] >= 0.9]
    
    return {
        'confidence_intervals': ci_df,
        'top_k_probabilities': prob_df,
        'stability_metrics': stability,
        'high_confidence_critical': high_conf_top_k['node'].tolist(),
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence
    }


if __name__ == "__main__":
    print("Uncertainty Quantification Test")
    print("=" * 50)
    
    G = nx.karate_club_graph()
    
    print("Running bootstrap analysis (50 iterations)...")
    result = full_uncertainty_analysis(G, n_bootstrap=50, top_k=10)
    
    print(f"\nStability metrics: {result['stability_metrics']}")
    print(f"\nHigh-confidence critical nodes (>90%): {result['high_confidence_critical']}")
    print(f"\nTop-10 probability for node 0: {result['top_k_probabilities'][result['top_k_probabilities']['node']==0]['prob_top_k'].values}")
