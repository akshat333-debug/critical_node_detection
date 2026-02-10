"""
Scalability Benchmark Module
=============================
Benchmark CRITIC-TOPSIS pipeline on increasing network sizes
to demonstrate performance on large networks.
"""

import networkx as nx
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from centralities import compute_all_centralities
from critic import compute_critic_weights
from topsis import topsis_rank


def benchmark_single(G: nx.Graph) -> Dict:
    """
    Benchmark CRITIC-TOPSIS pipeline on a single graph.
    
    Returns timing for each stage in seconds.
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    # Stage 1: Centralities
    t0 = time.time()
    df = compute_all_centralities(G, verbose=False)
    centrality_time = time.time() - t0
    
    # Stage 2: CRITIC weights
    t0 = time.time()
    weights, _ = compute_critic_weights(df, verbose=False)
    critic_time = time.time() - t0
    
    # Stage 3: TOPSIS ranking
    t0 = time.time()
    results, _ = topsis_rank(df, weights, verbose=False)
    topsis_time = time.time() - t0
    
    total_time = centrality_time + critic_time + topsis_time
    
    return {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'centrality_time': round(centrality_time, 4),
        'critic_time': round(critic_time, 4),
        'topsis_time': round(topsis_time, 4),
        'total_time': round(total_time, 4),
        'top_node': results.index[0],
        'top_score': round(results['closeness'].iloc[0], 4)
    }


def benchmark_scalability(sizes: Optional[List[int]] = None,
                          model: str = 'barabasi_albert',
                          m: int = 3,
                          progress_callback=None) -> pd.DataFrame:
    """
    Benchmark CRITIC-TOPSIS on increasing network sizes.
    
    Args:
        sizes: List of node counts to test (default: [100, 250, 500, 1000, 2000])
        model: Network model ('barabasi_albert' or 'erdos_renyi')
        m: Edges per new node (for BA) or connectivity param
        progress_callback: Optional callable(current_index, total) for progress
    
    Returns:
        DataFrame with timing results per size
    """
    if sizes is None:
        sizes = [100, 250, 500, 1000, 2000]
    
    results = []
    
    for i, n in enumerate(sizes):
        if progress_callback:
            progress_callback(i, len(sizes))
        
        # Generate network
        if model == 'barabasi_albert':
            G = nx.barabasi_albert_graph(n, min(m, n-1), seed=42)
        else:
            p = min(1.0, 2 * m / n)  # Target avg degree ~ 2*m
            G = nx.erdos_renyi_graph(n, p, seed=42)
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
        
        result = benchmark_single(G)
        results.append(result)
    
    return pd.DataFrame(results)


def benchmark_real_network(G: nx.Graph, name: str = "Network") -> Dict:
    """
    Benchmark a specific real-world network.
    """
    result = benchmark_single(G)
    result['network'] = name
    return result


if __name__ == "__main__":
    print("Scalability Benchmark")
    print("=" * 60)
    
    df = benchmark_scalability(sizes=[100, 250, 500, 1000])
    print(df.to_string(index=False))
    print(f"\nScaling factor (1000 vs 100): {df.iloc[-1]['total_time'] / df.iloc[0]['total_time']:.1f}x")
