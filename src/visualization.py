"""
Visualization Module - Plots for critical node detection analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


def setup_style():
    """Set up matplotlib style for academic plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
        'legend.fontsize': 11, 'figure.figsize': (10, 6), 'figure.dpi': 100
    })


def plot_attack_curves(attack_results: Dict[str, pd.DataFrame],
                       metric: str = 'lcc_fraction',
                       title: str = 'Targeted Attack Comparison',
                       save_path: Optional[str] = None) -> plt.Figure:
    """Plot node removal curves comparing different ranking methods."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(attack_results)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h']
    
    for i, (method, df) in enumerate(attack_results.items()):
        ax.plot(df['fraction_removed'] * 100, df[metric],
                label=method, color=colors[i], marker=markers[i % len(markers)],
                linewidth=2, markersize=6)
    
    ax.set_xlabel('Fraction of Nodes Removed (%)')
    ylabel = {'lcc_fraction': 'Largest Component (fraction)', 
              'efficiency': 'Global Efficiency'}.get(metric, metric)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.set_xlim(-1, None)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_centrality_heatmap(df: pd.DataFrame, title: str = 'Centrality Measures',
                            top_k: int = 15, save_path: Optional[str] = None) -> plt.Figure:
    """Plot heatmap of centrality values for top nodes."""
    setup_style()
    
    # Get top nodes by average normalized centrality
    df_norm = (df - df.min()) / (df.max() - df.min() + 1e-10)
    top_nodes = df_norm.mean(axis=1).nlargest(top_k).index
    df_top = df_norm.loc[top_nodes]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_top, annot=True, fmt='.2f', cmap='YlOrRd',
                cbar_kws={'label': 'Normalized Value'}, ax=ax)
    ax.set_title(f'{title} (Top {top_k} Nodes)')
    ax.set_xlabel('Centrality Measure')
    ax.set_ylabel('Node')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_ranking_comparison(df: pd.DataFrame, topsis_results: pd.DataFrame,
                            k: int = 10, save_path: Optional[str] = None) -> plt.Figure:
    """Bar chart comparing top-k nodes across methods."""
    setup_style()
    
    topsis_top = topsis_results.nsmallest(k, 'rank').index.tolist()
    rankings = {'CRITIC-TOPSIS': topsis_top}
    for col in df.columns:
        rankings[col] = df[col].nlargest(k).index.tolist()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    methods = list(rankings.keys())
    x = np.arange(k)
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        nodes = rankings[method]
        ax.bar(x + i * width, [1] * k, width, label=method, alpha=0.8)
        for j, node in enumerate(nodes):
            ax.text(x[j] + i * width, 0.5, str(node), ha='center', va='center',
                   fontsize=8, rotation=90)
    
    ax.set_xlabel('Rank Position')
    ax.set_ylabel('')
    ax.set_title(f'Top {k} Nodes by Different Methods')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([f'#{i+1}' for i in range(k)])
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.set_ylim(0, 1.2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_network_with_critical_nodes(G: nx.Graph, critical_nodes: List,
                                      title: str = 'Network with Critical Nodes',
                                      save_path: Optional[str] = None) -> plt.Figure:
    """Visualize network highlighting critical nodes."""
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 10))
    
    pos = nx.spring_layout(G, k=2/np.sqrt(G.number_of_nodes()), seed=42)
    
    # Draw non-critical nodes
    other_nodes = [n for n in G.nodes() if n not in critical_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_color='lightblue',
                          node_size=200, alpha=0.6, ax=ax)
    
    # Draw critical nodes
    nx.draw_networkx_nodes(G, pos, nodelist=critical_nodes, node_color='red',
                          node_size=500, alpha=0.9, ax=ax)
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    
    # Labels for critical nodes only
    labels = {n: str(n) for n in critical_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', ax=ax)
    
    ax.set_title(title)
    ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_weight_distribution(weights: pd.Series, title: str = 'CRITIC Weights',
                              save_path: Optional[str] = None) -> plt.Figure:
    """Bar chart of CRITIC weights."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    weights_sorted = weights.sort_values(ascending=True)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(weights_sorted)))
    
    bars = ax.barh(weights_sorted.index, weights_sorted.values, color=colors)
    ax.set_xlabel('Weight')
    ax.set_title(title)
    
    for bar, val in zip(bars, weights_sorted.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=10)
    
    ax.set_xlim(0, weights_sorted.max() * 1.15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def create_summary_figure(attack_results: Dict[str, pd.DataFrame],
                          effectiveness: pd.DataFrame,
                          save_path: Optional[str] = None) -> plt.Figure:
    """Create summary figure with attack curves and effectiveness bars."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Attack curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(attack_results)))
    for i, (method, df) in enumerate(attack_results.items()):
        ax1.plot(df['fraction_removed'] * 100, df['lcc_fraction'],
                label=method, color=colors[i], linewidth=2, marker='o')
    ax1.set_xlabel('Nodes Removed (%)')
    ax1.set_ylabel('Largest Component (fraction)')
    ax1.set_title('Targeted Attack Comparison')
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    
    # Effectiveness bars
    eff_sorted = effectiveness.sort_values('effectiveness')
    colors2 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(eff_sorted)))
    ax2.barh(eff_sorted['method'], eff_sorted['effectiveness'], color=colors2)
    ax2.set_xlabel('Attack Effectiveness')
    ax2.set_title('Method Comparison')
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
