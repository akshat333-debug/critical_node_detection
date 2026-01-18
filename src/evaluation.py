"""
Evaluation Module - Targeted attack simulations for critical node detection.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


def get_largest_component_size(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    return len(max(nx.connected_components(G), key=len))


def get_largest_component_fraction(G: nx.Graph, original_size: int) -> float:
    if original_size == 0:
        return 0
    return get_largest_component_size(G) / original_size


def compute_global_efficiency(G: nx.Graph) -> float:
    return nx.global_efficiency(G)


def compute_average_path_length(G: nx.Graph) -> float:
    if G.number_of_nodes() < 2:
        return float('inf')
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc)
    if G.number_of_nodes() < 2:
        return float('inf')
    return nx.average_shortest_path_length(G)


def simulate_targeted_attack(G: nx.Graph, node_ranking: List,
                              fractions: Optional[List[float]] = None,
                              verbose: bool = True) -> pd.DataFrame:
    if fractions is None:
        fractions = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    original_size = G.number_of_nodes()
    results = [{
        'fraction_removed': 0.0, 'nodes_removed': 0,
        'lcc_size': get_largest_component_size(G), 'lcc_fraction': 1.0,
        'efficiency': compute_global_efficiency(G),
        'avg_path_length': compute_average_path_length(G)
    }]
    
    G_copy = G.copy()
    nodes_removed = 0
    
    for frac in fractions:
        target_removed = int(frac * original_size)
        while nodes_removed < target_removed and nodes_removed < len(node_ranking):
            node = node_ranking[nodes_removed]
            if node in G_copy:
                G_copy.remove_node(node)
            nodes_removed += 1
        
        if G_copy.number_of_nodes() == 0:
            results.append({'fraction_removed': frac, 'nodes_removed': nodes_removed,
                           'lcc_size': 0, 'lcc_fraction': 0.0, 'efficiency': 0.0,
                           'avg_path_length': float('inf')})
        else:
            results.append({
                'fraction_removed': frac, 'nodes_removed': nodes_removed,
                'lcc_size': get_largest_component_size(G_copy),
                'lcc_fraction': get_largest_component_fraction(G_copy, original_size),
                'efficiency': compute_global_efficiency(G_copy),
                'avg_path_length': compute_average_path_length(G_copy)
            })
        if verbose:
            print(f"  Removed {frac*100:.0f}%: LCC={results[-1]['lcc_fraction']:.3f}")
    
    return pd.DataFrame(results)


def compare_attack_methods(G: nx.Graph, rankings: Dict[str, List],
                           fractions: Optional[List[float]] = None,
                           verbose: bool = True) -> Dict[str, pd.DataFrame]:
    results = {}
    for method, ranking in rankings.items():
        if verbose:
            print(f"\nAttack using {method}...")
        results[method] = simulate_targeted_attack(G, ranking, fractions, verbose)
    return results


def compute_attack_effectiveness(attack_results: Dict[str, pd.DataFrame],
                                  metric: str = 'lcc_fraction') -> pd.DataFrame:
    effectiveness = []
    for method, df in attack_results.items():
        x, y = df['fraction_removed'].values, df[metric].values
        auc = np.trapezoid(y, x)
        max_area = x.max()
        eff = 1 - (auc / max_area if max_area > 0 else 0)
        effectiveness.append({'method': method, 'auc': auc, 'effectiveness': eff})
    return pd.DataFrame(effectiveness).sort_values('effectiveness', ascending=False)


def get_ranking_from_centrality(df: pd.DataFrame, column: str) -> List:
    return df[column].sort_values(ascending=False).index.tolist()


def get_ranking_from_topsis(topsis_results: pd.DataFrame) -> List:
    return topsis_results.sort_values('closeness', ascending=False).index.tolist()
