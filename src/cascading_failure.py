"""
Cascading Failure Simulation Module
====================================
Simulates how failures spread through a network after initial node removal.

Unlike simple targeted attack (remove nodes sequentially), cascading failure models:
1. Remove initial critical nodes
2. Check if remaining nodes become overloaded (based on load redistribution)
3. Failed nodes cause further failures in cascade
4. Continue until network stabilizes
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import deque


def compute_node_load(G: nx.Graph, node, method: str = 'betweenness') -> float:
    """
    Compute load on a node based on centrality measure.
    
    Args:
        G: Network graph
        node: Node to compute load for
        method: 'betweenness' or 'degree'
    
    Returns:
        Load value for the node
    """
    if method == 'betweenness':
        bc = nx.betweenness_centrality(G)
        return bc.get(node, 0)
    else:  # degree
        return G.degree(node) / (G.number_of_nodes() - 1)


def compute_all_loads(G: nx.Graph, method: str = 'betweenness') -> Dict:
    """Compute load for all nodes."""
    if method == 'betweenness':
        return nx.betweenness_centrality(G)
    else:
        n = G.number_of_nodes() - 1 if G.number_of_nodes() > 1 else 1
        return {node: G.degree(node) / n for node in G.nodes()}


def simulate_cascading_failure(G: nx.Graph,
                                initial_failures: List,
                                capacity_factor: float = 1.2,
                                load_method: str = 'betweenness',
                                max_iterations: int = 100,
                                verbose: bool = False) -> Dict:
    """
    Simulate cascading failure after initial node removals.
    
    Model: Each node has capacity = initial_load * capacity_factor
    When a node's load exceeds capacity, it fails and redistributes load.
    
    Args:
        G: Original network graph
        initial_failures: List of initially failed nodes
        capacity_factor: How much extra load a node can handle (1.0 = no tolerance)
        load_method: 'betweenness' or 'degree'
        max_iterations: Maximum cascade iterations
        verbose: Print progress
    
    Returns:
        Dictionary with cascade results
    """
    # Create working copy
    H = G.copy()
    
    # Compute initial loads and capacities
    initial_loads = compute_all_loads(H, load_method)
    capacities = {node: load * capacity_factor for node, load in initial_loads.items()}
    
    # Track failures
    all_failed = set(initial_failures)
    cascade_history = [{'iteration': 0, 'failed': list(initial_failures), 'total_failed': len(initial_failures)}]
    
    # Remove initial failures
    H.remove_nodes_from(initial_failures)
    
    if verbose:
        print(f"Initial failures: {len(initial_failures)} nodes")
    
    # Cascade loop
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        
        if H.number_of_nodes() == 0:
            break
        
        # Compute new loads after failures
        current_loads = compute_all_loads(H, load_method)
        
        # Find overloaded nodes
        new_failures = []
        for node in H.nodes():
            if node in capacities:
                if current_loads.get(node, 0) > capacities[node]:
                    new_failures.append(node)
        
        if not new_failures:
            # Cascade stopped
            break
        
        # Remove overloaded nodes
        all_failed.update(new_failures)
        H.remove_nodes_from(new_failures)
        
        cascade_history.append({
            'iteration': iteration,
            'failed': new_failures,
            'total_failed': len(all_failed)
        })
        
        if verbose:
            print(f"  Iteration {iteration}: {len(new_failures)} new failures, total: {len(all_failed)}")
    
    # Compute final metrics
    original_size = G.number_of_nodes()
    surviving_nodes = original_size - len(all_failed)
    
    # Largest component in surviving network
    if H.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(H), key=len)
        lcc_size = len(largest_cc)
    else:
        lcc_size = 0
    
    return {
        'original_nodes': original_size,
        'initial_failures': len(initial_failures),
        'cascade_iterations': iteration,
        'total_failures': len(all_failed),
        'surviving_nodes': surviving_nodes,
        'survival_rate': surviving_nodes / original_size,
        'lcc_size': lcc_size,
        'lcc_fraction': lcc_size / original_size,
        'cascade_history': cascade_history,
        'failed_nodes': list(all_failed)
    }


def compare_cascade_methods(G: nx.Graph,
                            rankings: Dict[str, List],
                            initial_fraction: float = 0.05,
                            capacity_factor: float = 1.2,
                            verbose: bool = False) -> pd.DataFrame:
    """
    Compare cascading failure impact of different ranking methods.
    
    Args:
        G: Network graph
        rankings: Dict of method_name -> node ranking list
        initial_fraction: Fraction of nodes to initially remove
        capacity_factor: Node capacity tolerance
        verbose: Print progress
    
    Returns:
        DataFrame comparing methods
    """
    n_initial = max(1, int(G.number_of_nodes() * initial_fraction))
    
    results = []
    for method, ranking in rankings.items():
        initial_failures = ranking[:n_initial]
        
        cascade_result = simulate_cascading_failure(
            G, initial_failures, capacity_factor, 
            verbose=False
        )
        
        results.append({
            'method': method,
            'initial_removed': n_initial,
            'cascade_failures': cascade_result['total_failures'] - n_initial,
            'total_failures': cascade_result['total_failures'],
            'survival_rate': cascade_result['survival_rate'],
            'lcc_fraction': cascade_result['lcc_fraction'],
            'cascade_iterations': cascade_result['cascade_iterations']
        })
        
        if verbose:
            print(f"{method}: {cascade_result['total_failures']} total failures ({cascade_result['cascade_iterations']} iterations)")
    
    return pd.DataFrame(results).sort_values('total_failures', ascending=False)


def cascade_over_fractions(G: nx.Graph,
                           ranking: List,
                           fractions: List[float] = None,
                           capacity_factor: float = 1.2) -> pd.DataFrame:
    """
    Run cascading failure for different initial removal fractions.
    
    Returns DataFrame with results for each fraction.
    """
    if fractions is None:
        fractions = [0.02, 0.05, 0.10, 0.15, 0.20]
    
    results = []
    for frac in fractions:
        n_initial = max(1, int(G.number_of_nodes() * frac))
        initial_failures = ranking[:n_initial]
        
        cascade_result = simulate_cascading_failure(
            G, initial_failures, capacity_factor, verbose=False
        )
        
        results.append({
            'initial_fraction': frac,
            'initial_removed': n_initial,
            'total_failures': cascade_result['total_failures'],
            'cascade_multiplier': cascade_result['total_failures'] / n_initial if n_initial > 0 else 0,
            'survival_rate': cascade_result['survival_rate'],
            'lcc_fraction': cascade_result['lcc_fraction']
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test cascading failure
    import sys
    sys.path.insert(0, '.')
    from centralities import compute_all_centralities
    from critic import compute_critic_weights
    from topsis import topsis_rank
    from evaluation import get_ranking_from_topsis
    
    print("Testing Cascading Failure Simulation")
    print("=" * 50)
    
    G = nx.karate_club_graph()
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Get TOPSIS ranking
    df = compute_all_centralities(G, verbose=False)
    weights, _ = compute_critic_weights(df, verbose=False)
    results, _ = topsis_rank(df, weights, verbose=False)
    ranking = get_ranking_from_topsis(results)
    
    # Simulate cascade
    result = simulate_cascading_failure(
        G, ranking[:3], capacity_factor=1.1, verbose=True
    )
    
    print(f"\nResults:")
    print(f"  Initial failures: {result['initial_failures']}")
    print(f"  Cascade iterations: {result['cascade_iterations']}")
    print(f"  Total failures: {result['total_failures']}")
    print(f"  Survival rate: {result['survival_rate']:.2%}")
    print(f"  LCC fraction: {result['lcc_fraction']:.2%}")
