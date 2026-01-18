"""
Main Pipeline - Complete CRITIC-TOPSIS Critical Node Detection Experiment
=========================================================================

This script runs the complete experiment pipeline:
1. Load a network
2. Compute all centrality measures
3. Apply CRITIC weighting
4. Apply TOPSIS ranking
5. Simulate targeted attacks
6. Generate visualizations and results
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime

from data_loading import (load_karate_club, load_les_miserables, 
                          load_florentine_families, load_dolphins,
                          load_usair, load_power_grid, load_football,
                          create_barabasi_albert, get_network_info)
from centralities import compute_all_centralities
from critic import compute_critic_weights, explain_weights
from topsis import topsis_rank, get_critical_nodes, compare_with_single_metrics
from evaluation import (compare_attack_methods, compute_attack_effectiveness,
                        get_ranking_from_centrality, get_ranking_from_topsis)
from visualization import (plot_attack_curves, plot_centrality_heatmap,
                           plot_ranking_comparison, plot_network_with_critical_nodes,
                           plot_weight_distribution, create_summary_figure)


def run_experiment(G: nx.Graph, 
                   network_name: str,
                   results_dir: str = None,
                   attack_fractions: list = None,
                   top_k: int = 10,
                   verbose: bool = True) -> dict:
    """
    Run complete CRITIC-TOPSIS experiment on a network.
    
    Args:
        G: NetworkX graph
        network_name: Name for saving results
        results_dir: Directory to save results (None = don't save)
        attack_fractions: Fractions for attack simulation
        top_k: Number of top critical nodes to identify
        verbose: Print progress
    
    Returns:
        dict: All experiment results
    """
    if attack_fractions is None:
        attack_fractions = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    results = {'network_name': network_name}
    
    # Setup results directory
    if results_dir:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        network_dir = results_path / network_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        network_dir.mkdir(exist_ok=True)
    else:
        network_dir = None
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {network_name}")
    print(f"{'='*60}")
    
    # Network info
    info = get_network_info(G)
    results['network_info'] = info
    if verbose:
        print(f"\nNetwork: {info['nodes']} nodes, {info['edges']} edges")
        print(f"Density: {info['density']:.4f}, Avg clustering: {info['avg_clustering']:.4f}")
    
    # Step 1: Compute centralities
    print("\n[1/5] Computing centralities...")
    df_centrality = compute_all_centralities(G, verbose=verbose)
    results['centralities'] = df_centrality
    
    # Step 2: CRITIC weights
    print("\n[2/5] Computing CRITIC weights...")
    weights, critic_details = compute_critic_weights(df_centrality, verbose=verbose)
    results['weights'] = weights
    results['critic_details'] = critic_details
    
    if verbose:
        print(f"\nWeights: {weights.round(3).to_dict()}")
    
    # Step 3: TOPSIS ranking
    print("\n[3/5] Performing TOPSIS ranking...")
    topsis_results, topsis_details = topsis_rank(df_centrality, weights, verbose=verbose)
    results['topsis_results'] = topsis_results
    
    critical_nodes = get_critical_nodes(topsis_results, top_k)
    results['critical_nodes'] = critical_nodes
    
    if verbose:
        print(f"\nTop {top_k} critical nodes (CRITIC-TOPSIS): {critical_nodes}")
    
    # Step 4: Attack simulation
    print("\n[4/5] Simulating targeted attacks...")
    
    # Create rankings for comparison
    rankings = {'CRITIC-TOPSIS': get_ranking_from_topsis(topsis_results)}
    for col in ['degree', 'betweenness', 'closeness', 'pagerank']:
        rankings[col] = get_ranking_from_centrality(df_centrality, col)
    
    attack_results = compare_attack_methods(G, rankings, attack_fractions, verbose=verbose)
    results['attack_results'] = attack_results
    
    effectiveness = compute_attack_effectiveness(attack_results)
    results['effectiveness'] = effectiveness
    
    if verbose:
        print("\n" + "="*40)
        print("ATTACK EFFECTIVENESS (higher = better):")
        print(effectiveness.to_string(index=False))
    
    # Step 5: Visualizations
    print("\n[5/5] Generating visualizations...")
    
    if network_dir:
        # Save attack curves
        fig = plot_attack_curves(attack_results, 
                                title=f'{network_name} - Targeted Attack',
                                save_path=network_dir / 'attack_curves.png')
        
        # Save centrality heatmap
        fig = plot_centrality_heatmap(df_centrality,
                                      title=f'{network_name} - Centralities',
                                      save_path=network_dir / 'centrality_heatmap.png')
        
        # Save weight distribution
        fig = plot_weight_distribution(weights,
                                       title=f'{network_name} - CRITIC Weights',
                                       save_path=network_dir / 'weights.png')
        
        # Save network visualization (for small networks)
        if G.number_of_nodes() <= 150:
            fig = plot_network_with_critical_nodes(G, critical_nodes,
                                                   title=f'{network_name} - Critical Nodes',
                                                   save_path=network_dir / 'network.png')
        
        # Save summary figure
        fig = create_summary_figure(attack_results, effectiveness,
                                   save_path=network_dir / 'summary.png')
        
        # Save data
        df_centrality.to_csv(network_dir / 'centralities.csv')
        topsis_results.to_csv(network_dir / 'topsis_ranking.csv')
        effectiveness.to_csv(network_dir / 'effectiveness.csv', index=False)
        weights.to_csv(network_dir / 'critic_weights.csv')
        
        print(f"  Results saved to: {network_dir}")
    
    import matplotlib.pyplot as plt
    plt.close('all')
    
    return results


def run_all_experiments(results_dir: str = None, include_large: bool = False) -> dict:
    """Run experiments on all benchmark networks.
    
    Args:
        results_dir: Directory to save results
        include_large: If True, include larger networks (USAir, Power Grid)
    """
    
    if results_dir is None:
        results_dir = Path(__file__).parent.parent / 'results'
    
    all_results = {}
    
    # Core networks to test
    networks = [
        ('Karate Club', load_karate_club()),
        ('Les Miserables', load_les_miserables()),
        ('Florentine Families', load_florentine_families()),
        ('Dolphins', load_dolphins()),
        ('Football', load_football()),
        ('Barabasi-Albert (100)', create_barabasi_albert(100, 3)),
    ]
    
    # Add larger networks if requested
    if include_large:
        networks.extend([
            ('USAir', load_usair()),
            ('Power Grid', load_power_grid()),
        ])
    
    for name, G in networks:
        try:
            results = run_experiment(G, name, results_dir)
            all_results[name] = results
        except Exception as e:
            print(f"ERROR in {name}: {e}")
    
    # Generate comparison summary
    print("\n" + "="*60)
    print("OVERALL COMPARISON")
    print("="*60)
    
    summary_data = []
    for name, results in all_results.items():
        eff = results['effectiveness']
        topsis_eff = eff[eff['method'] == 'CRITIC-TOPSIS']['effectiveness'].values[0]
        best_single = eff[eff['method'] != 'CRITIC-TOPSIS']['effectiveness'].max()
        
        summary_data.append({
            'Network': name,
            'Nodes': results['network_info']['nodes'],
            'Edges': results['network_info']['edges'],
            'TOPSIS_Eff': topsis_eff,
            'Best_Single': best_single,
            'TOPSIS_Wins': 'Yes' if topsis_eff >= best_single else 'No'
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    if results_dir:
        summary_df.to_csv(Path(results_dir) / 'overall_summary.csv', index=False)
    
    return all_results


if __name__ == "__main__":
    print("=" * 60)
    print("CRITICAL NODE DETECTION: CRITIC-TOPSIS FRAMEWORK")
    print("=" * 60)
    
    # Run on all benchmark networks
    results = run_all_experiments()
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)
