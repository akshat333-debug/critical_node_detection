"""
Temporal Critical Node Detection Module
========================================
Analyze network evolution over time and predict future critical nodes.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from centralities import compute_all_centralities
from critic import compute_critic_weights
from topsis import topsis_rank


def create_network_snapshot(G: nx.Graph, 
                            remove_fraction: float = 0.1,
                            add_fraction: float = 0.1) -> nx.Graph:
    """
    Create a modified snapshot simulating network evolution.
    Randomly removes and adds edges.
    """
    H = G.copy()
    n_edges = H.number_of_edges()
    
    # Remove some edges
    edges = list(H.edges())
    n_remove = int(n_edges * remove_fraction)
    if n_remove > 0:
        remove_indices = np.random.choice(len(edges), size=n_remove, replace=False)
        for idx in remove_indices:
            H.remove_edge(*edges[idx])
    
    # Add some edges
    nodes = list(H.nodes())
    n_add = int(n_edges * add_fraction)
    added = 0
    attempts = 0
    while added < n_add and attempts < n_add * 10:
        u, v = np.random.choice(nodes, size=2, replace=False)
        if not H.has_edge(u, v):
            H.add_edge(u, v)
            added += 1
        attempts += 1
    
    return H


def generate_temporal_snapshots(G: nx.Graph, 
                                n_snapshots: int = 5,
                                volatility: float = 0.1) -> List[nx.Graph]:
    """
    Generate a sequence of network snapshots simulating temporal evolution.
    
    Args:
        G: Base network
        n_snapshots: Number of time points
        volatility: How much the network changes each step (0.1 = 10%)
    
    Returns:
        List of network snapshots
    """
    snapshots = [G.copy()]
    current = G.copy()
    
    for _ in range(n_snapshots - 1):
        current = create_network_snapshot(current, volatility, volatility)
        snapshots.append(current)
    
    return snapshots


def analyze_temporal_rankings(snapshots: List[nx.Graph], 
                              top_k: int = 10) -> pd.DataFrame:
    """
    Analyze how node rankings change over time.
    
    Returns DataFrame with columns: node, t0_rank, t1_rank, ..., trend, stability
    """
    all_rankings = []
    
    for t, G in enumerate(snapshots):
        df = compute_all_centralities(G, verbose=False)
        weights, _ = compute_critic_weights(df, verbose=False)
        results, _ = topsis_rank(df, weights, verbose=False)
        
        ranking = {node: int(rank) for node, rank in results['rank'].items()}
        all_rankings.append(ranking)
    
    # Build comparison DataFrame
    all_nodes = set()
    for r in all_rankings:
        all_nodes.update(r.keys())
    
    data = []
    for node in all_nodes:
        row = {'node': node}
        ranks = []
        for t, ranking in enumerate(all_rankings):
            rank = ranking.get(node, len(ranking) + 1)
            row[f't{t}_rank'] = rank
            ranks.append(rank)
        
        # Compute trend (negative = becoming more critical)
        if len(ranks) >= 2:
            row['trend'] = ranks[-1] - ranks[0]  # Positive = less critical
        else:
            row['trend'] = 0
        
        # Compute stability (std of ranks)
        row['stability'] = np.std(ranks) if len(ranks) > 1 else 0
        
        # Average rank
        row['avg_rank'] = np.mean(ranks)
        
        data.append(row)
    
    return pd.DataFrame(data).sort_values('avg_rank')


def predict_future_critical(snapshots: List[nx.Graph],
                            top_k: int = 10) -> Dict:
    """
    Predict which nodes will become critical in the future.
    
    Strategy: Look for nodes with improving (decreasing) rank trend
    that are not yet in top-k but on trajectory to enter.
    
    Returns:
        Dictionary with predictions and analysis
    """
    analysis = analyze_temporal_rankings(snapshots, top_k)
    
    # Current top-k (from last snapshot)
    current_top = set(analysis.nsmallest(top_k, f't{len(snapshots)-1}_rank')['node'])
    
    # Rising stars: Not in current top-k but improving rapidly
    not_current = analysis[~analysis['node'].isin(current_top)]
    rising = not_current[not_current['trend'] < 0].nsmallest(5, 'trend')
    
    # Stable critical: In top-k with low variance
    stable = analysis[analysis['node'].isin(current_top)].nsmallest(5, 'stability')
    
    # Declining: Were more critical before but falling
    declining = analysis[analysis['trend'] > 5].nlargest(5, 'trend')
    
    return {
        'current_critical': list(current_top),
        'rising_stars': rising[['node', 'trend', 'avg_rank']].to_dict('records'),
        'stable_critical': stable[['node', 'stability', 'avg_rank']].to_dict('records'),
        'declining': declining[['node', 'trend', 'avg_rank']].to_dict('records'),
        'full_analysis': analysis
    }


def analyze_weight_evolution(snapshots: List[nx.Graph]) -> Dict:
    """
    Track how CRITIC weights evolve across temporal snapshots.
    
    Returns:
        Dictionary with weight timeseries and drift analysis
    """
    weight_history = []  # List of {metric: weight} per snapshot
    
    for t, G in enumerate(snapshots):
        df = compute_all_centralities(G, verbose=False)
        weights, _ = compute_critic_weights(df, verbose=False)
        weight_dict = {'t': t}
        for metric, w in weights.items():
            weight_dict[metric] = w
        weight_history.append(weight_dict)
    
    weight_df = pd.DataFrame(weight_history).set_index('t')
    
    # Compute drift: absolute change from first to last snapshot
    metrics = [c for c in weight_df.columns]
    drift = {}
    for m in metrics:
        first_val = weight_df[m].iloc[0]
        last_val = weight_df[m].iloc[-1]
        drift[m] = {
            'initial': first_val,
            'final': last_val,
            'absolute_change': abs(last_val - first_val),
            'direction': 'increased' if last_val > first_val else 'decreased',
            'significant': abs(last_val - first_val) > 0.03  # >3% change
        }
    
    return {
        'weight_timeseries': weight_df,
        'drift': drift,
        'metrics': metrics
    }


def compute_adaptive_weights(snapshots: List[nx.Graph],
                             decay: float = 0.3) -> pd.Series:
    """
    Compute exponentially-weighted adaptive CRITIC weights.
    
    Recent snapshots are weighted more heavily using exponential decay.
    
    Args:
        snapshots: List of network snapshots (time-ordered)
        decay: Decay factor (0 = equal weight, 1 = only latest matters)
    
    Returns:
        pd.Series of adaptive weights (summing to 1.0)
    """
    n = len(snapshots)
    all_weights = []
    
    for G in snapshots:
        df = compute_all_centralities(G, verbose=False)
        weights, _ = compute_critic_weights(df, verbose=False)
        all_weights.append(weights)
    
    # Exponential time weights: more recent = higher weight
    time_weights = np.array([np.exp(-decay * (n - 1 - t)) for t in range(n)])
    time_weights = time_weights / time_weights.sum()
    
    # Weighted average of CRITIC weights
    metrics = all_weights[0].index
    adaptive = {}
    for m in metrics:
        vals = np.array([w[m] for w in all_weights])
        adaptive[m] = np.dot(vals, time_weights)
    
    result = pd.Series(adaptive)
    result = result / result.sum()  # Normalize to sum to 1
    return result


def temporal_adaptive_summary(G: nx.Graph,
                              n_snapshots: int = 5,
                              volatility: float = 0.1,
                              decay: float = 0.3) -> Dict:
    """
    Full temporal analysis with adaptive weight recalculation.
    
    Combines rank prediction with weight evolution tracking.
    """
    snapshots = generate_temporal_snapshots(G, n_snapshots, volatility)
    predictions = predict_future_critical(snapshots)
    weight_evolution = analyze_weight_evolution(snapshots)
    adaptive_weights = compute_adaptive_weights(snapshots, decay=decay)
    
    # Also compute static weights for comparison
    df_static = compute_all_centralities(G, verbose=False)
    static_weights, _ = compute_critic_weights(df_static, verbose=False)
    
    # Weight comparison
    comparison = {}
    for m in adaptive_weights.index:
        if m in static_weights.index:
            comparison[m] = {
                'static': static_weights[m],
                'adaptive': adaptive_weights[m],
                'diff': adaptive_weights[m] - static_weights[m]
            }
    
    return {
        'n_snapshots': n_snapshots,
        'volatility': volatility,
        'decay': decay,
        **predictions,
        'weight_evolution': weight_evolution,
        'adaptive_weights': adaptive_weights,
        'static_weights': static_weights,
        'weight_comparison': comparison
    }


def temporal_prediction_summary(G: nx.Graph, 
                                n_snapshots: int = 5,
                                volatility: float = 0.1) -> Dict:
    """
    Main entry point for temporal analysis (backward compatible).
    """
    snapshots = generate_temporal_snapshots(G, n_snapshots, volatility)
    predictions = predict_future_critical(snapshots)
    
    return {
        'n_snapshots': n_snapshots,
        'volatility': volatility,
        **predictions
    }


if __name__ == "__main__":
    print("Temporal Critical Node Detection")
    print("=" * 50)
    
    G = nx.karate_club_graph()
    
    # Test adaptive weights
    result = temporal_adaptive_summary(G, n_snapshots=5, volatility=0.15, decay=0.3)
    
    print(f"\nCurrent critical: {result['current_critical'][:5]}")
    print(f"\nRising stars: {result['rising_stars']}")
    print(f"\nAdaptive weights: {result['adaptive_weights'].to_dict()}")
    print(f"\nWeight drift:")
    for m, d in result['weight_evolution']['drift'].items():
        if d['significant']:
            print(f"  ⚠️ {m}: {d['direction']} by {d['absolute_change']:.4f}")
