"""
Adversarial Robustness Module
==============================
Test if an attacker can fool the critical node detection system.
Evaluate robustness against strategic manipulations.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from centralities import compute_all_centralities
from critic import compute_critic_weights
from topsis import topsis_rank


def add_decoy_edges(G: nx.Graph, 
                    target_node, 
                    n_edges: int = 5) -> nx.Graph:
    """
    Adversarial attack: Add edges to make a non-critical node appear critical.
    """
    H = G.copy()
    nodes = [n for n in H.nodes() if n != target_node and not H.has_edge(target_node, n)]
    
    # Connect to high-degree nodes to boost centrality
    degrees = {n: H.degree(n) for n in nodes}
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
    
    added = 0
    for node in sorted_nodes:
        if added >= n_edges:
            break
        if not H.has_edge(target_node, node):
            H.add_edge(target_node, node)
            added += 1
    
    return H


def remove_protective_edges(G: nx.Graph,
                            target_node,
                            n_edges: int = 3) -> nx.Graph:
    """
    Adversarial attack: Remove edges to hide a critical node.
    Remove edges to low-degree neighbors to minimize detection.
    """
    H = G.copy()
    neighbors = list(H.neighbors(target_node))
    
    if len(neighbors) <= 1:
        return H  # Can't remove without disconnecting
    
    # Remove edges to nodes with lowest degree (least impact on network)
    degrees = {n: H.degree(n) for n in neighbors}
    sorted_nodes = sorted(degrees, key=degrees.get)
    
    removed = 0
    for node in sorted_nodes:
        if removed >= n_edges:
            break
        if H.degree(target_node) > 1:  # Keep at least one edge
            H.remove_edge(target_node, node)
            removed += 1
    
    return H


def sybil_attack(G: nx.Graph,
                 n_sybils: int = 5,
                 target_node=None) -> Tuple[nx.Graph, List]:
    """
    Sybil attack: Add fake nodes to boost a target's importance.
    """
    H = G.copy()
    # Determine new node strategy
    existing_nodes = set(H.nodes())
    is_integer_graph = all(isinstance(n, int) for n in existing_nodes)
    
    max_node_val = -1
    if is_integer_graph and existing_nodes:
        max_node_val = max(existing_nodes)

    sybil_nodes = []
    for i in range(n_sybils):
        if is_integer_graph:
            sybil = max_node_val + i + 1
        else:
            # Generate unique string ID
            sybil = f"sybil_{i}"
            while sybil in existing_nodes:
                sybil = f"sybil_{i}_{np.random.randint(1000)}"
        
        sybil_nodes.append(sybil)
        H.add_node(sybil)
        
        if target_node is not None:
            # Connect sybil to target
            H.add_edge(sybil, target_node)
        
        # Also connect to random existing nodes
        existing = [n for n in G.nodes() if n != target_node]
        if existing:
            random_target = np.random.choice(existing)
            H.add_edge(sybil, random_target)
    
    return H, sybil_nodes


def evaluate_attack_success(G_original: nx.Graph,
                            G_attacked: nx.Graph,
                            target_node,
                            attack_type: str) -> Dict:
    """
    Evaluate if an attack successfully changed the target's rank.
    """
    # Original ranking
    df_orig = compute_all_centralities(G_original, verbose=False)
    weights_orig, _ = compute_critic_weights(df_orig, verbose=False)
    results_orig, _ = topsis_rank(df_orig, weights_orig, verbose=False)
    
    # Attacked ranking
    df_atk = compute_all_centralities(G_attacked, verbose=False)
    weights_atk, _ = compute_critic_weights(df_atk, verbose=False)
    results_atk, _ = topsis_rank(df_atk, weights_atk, verbose=False)
    
    orig_rank = int(results_orig.loc[target_node, 'rank'])
    atk_rank = int(results_atk.loc[target_node, 'rank']) if target_node in results_atk.index else -1
    
    rank_change = orig_rank - atk_rank  # Positive = became more critical
    
    return {
        'attack_type': attack_type,
        'target_node': target_node,
        'original_rank': orig_rank,
        'attacked_rank': atk_rank,
        'rank_change': rank_change,
        'success': abs(rank_change) >= 3,  # Significant change
        'attack_direction': 'promotion' if rank_change > 0 else 'demotion'
    }


def test_robustness(G: nx.Graph,
                    target_node=None,
                    n_trials: int = 5) -> Dict:
    """
    Comprehensive robustness testing against multiple attack types.
    """
    if target_node is None:
        # Pick a mid-ranked node for testing
        df = compute_all_centralities(G, verbose=False)
        weights, _ = compute_critic_weights(df, verbose=False)
        results, _ = topsis_rank(df, weights, verbose=False)
        mid_idx = len(results) // 2
        target_node = results.iloc[mid_idx].name
    
    attack_results = []
    
    # Test edge addition attack
    for n_edges in [3, 5, 10]:
        G_attacked = add_decoy_edges(G, target_node, n_edges)
        result = evaluate_attack_success(G, G_attacked, target_node, f'add_{n_edges}_edges')
        result['perturbation_size'] = n_edges
        attack_results.append(result)
    
    # Test edge removal attack
    for n_edges in [1, 2, 3]:
        G_attacked = remove_protective_edges(G, target_node, n_edges)
        result = evaluate_attack_success(G, G_attacked, target_node, f'remove_{n_edges}_edges')
        result['perturbation_size'] = n_edges
        attack_results.append(result)
    
    # Test Sybil attack
    for n_sybils in [3, 5]:
        G_attacked, sybils = sybil_attack(G, n_sybils, target_node)
        result = evaluate_attack_success(G, G_attacked, target_node, f'sybil_{n_sybils}')
        result['perturbation_size'] = n_sybils
        attack_results.append(result)
    
    # Aggregate results
    df_results = pd.DataFrame(attack_results)
    
    successful_attacks = df_results[df_results['success']]
    vulnerability_score = len(successful_attacks) / len(df_results) * 100
    
    return {
        'target_node': target_node,
        'attack_results': df_results,
        'vulnerability_score': vulnerability_score,
        'most_effective_attack': df_results.loc[df_results['rank_change'].abs().idxmax()].to_dict() if len(df_results) > 0 else None,
        'robustness_grade': 'A' if vulnerability_score < 20 else 'B' if vulnerability_score < 40 else 'C' if vulnerability_score < 60 else 'D'
    }


def full_adversarial_analysis(G: nx.Graph) -> Dict:
    """
    Complete adversarial robustness analysis.
    """
    # Get top-5 critical nodes
    df = compute_all_centralities(G, verbose=False)
    weights, _ = compute_critic_weights(df, verbose=False)
    results, _ = topsis_rank(df, weights, verbose=False)
    
    top_5 = results.nsmallest(5, 'rank').index.tolist()
    
    # Test robustness of top nodes
    robustness_results = []
    for node in top_5[:3]:  # Test top 3
        result = test_robustness(G, node)
        robustness_results.append({
            'node': node,
            'original_rank': int(results.loc[node, 'rank']),
            'vulnerability_score': result['vulnerability_score'],
            'robustness_grade': result['robustness_grade']
        })
    
    avg_vulnerability = np.mean([r['vulnerability_score'] for r in robustness_results])
    
    return {
        'overall_vulnerability': avg_vulnerability,
        'overall_grade': 'A' if avg_vulnerability < 20 else 'B' if avg_vulnerability < 40 else 'C',
        'node_robustness': robustness_results,
        'recommendations': get_robustness_recommendations(avg_vulnerability)
    }


def get_robustness_recommendations(vulnerability: float) -> List[str]:
    """Generate recommendations based on vulnerability score."""
    recs = []
    
    if vulnerability > 50:
        recs.append("⚠️ HIGH VULNERABILITY: Consider using ensemble methods")
        recs.append("Consider removing low-variance centralities")
    
    if vulnerability > 30:
        recs.append("Use temporal analysis to detect sudden ranking changes")
        recs.append("Monitor for unusual edge additions")
    
    recs.append("Regularly re-validate rankings against attack simulations")
    
    return recs


if __name__ == "__main__":
    print("Adversarial Robustness Test")
    print("=" * 50)
    
    G = nx.karate_club_graph()
    result = full_adversarial_analysis(G)
    
    print(f"Overall vulnerability: {result['overall_vulnerability']:.1f}%")
    print(f"Robustness grade: {result['overall_grade']}")
    print(f"\nNode robustness:")
    for r in result['node_robustness']:
        print(f"  Node {r['node']}: {r['robustness_grade']} ({r['vulnerability_score']:.1f}% vulnerable)")
    print(f"\nRecommendations: {result['recommendations']}")
