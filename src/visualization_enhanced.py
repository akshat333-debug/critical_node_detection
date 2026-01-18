"""
Enhanced Visualization Module
=============================
Publication-quality plots for critical node detection analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


def setup_publication_style():
    """Set up matplotlib style for publication-quality plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_multi_network_comparison(all_results: Dict, 
                                   metric: str = 'effectiveness',
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Create bar chart comparing method effectiveness across all networks.
    
    Args:
        all_results: Dict of network_name -> experiment results
        metric: 'effectiveness' or 'auc'
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    setup_publication_style()
    
    # Prepare data
    data = []
    for network_name, results in all_results.items():
        eff = results['effectiveness']
        for _, row in eff.iterrows():
            data.append({
                'Network': network_name,
                'Method': row['method'],
                metric.capitalize(): row[metric]
            })
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    methods = df['Method'].unique()
    networks = df['Network'].unique()
    x = np.arange(len(networks))
    width = 0.15
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    for i, method in enumerate(methods):
        method_data = df[df['Method'] == method]
        values = [method_data[method_data['Network'] == n][metric.capitalize()].values[0] 
                  if len(method_data[method_data['Network'] == n]) > 0 else 0
                  for n in networks]
        bars = ax.bar(x + i * width, values, width, label=method, color=colors[i])
    
    ax.set_xlabel('Network')
    ax.set_ylabel(f'{metric.capitalize()}')
    ax.set_title(f'Method Comparison Across Networks')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(networks, rotation=45, ha='right')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_weight_comparison(all_results: Dict, 
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Heatmap showing CRITIC weights across different networks.
    
    Args:
        all_results: Dict of network_name -> experiment results
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    setup_publication_style()
    
    # Collect weights
    weight_data = {}
    for network_name, results in all_results.items():
        weight_data[network_name] = results['weights']
    
    df = pd.DataFrame(weight_data).T
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'CRITIC Weight'}, ax=ax)
    ax.set_title('CRITIC Weights Across Networks')
    ax.set_xlabel('Centrality Measure')
    ax.set_ylabel('Network')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_ranking_overlap(df_centrality: pd.DataFrame,
                         topsis_results: pd.DataFrame,
                         k_values: List[int] = [5, 10, 15],
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Show overlap between TOPSIS and single-metric rankings at different k.
    
    Args:
        df_centrality: Centrality DataFrame
        topsis_results: TOPSIS results
        k_values: List of k values to compare
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    setup_publication_style()
    
    metrics = list(df_centrality.columns)
    
    # Calculate overlap
    overlap_data = []
    for k in k_values:
        topsis_top = set(topsis_results.nsmallest(k, 'rank').index.tolist())
        for metric in metrics:
            metric_top = set(df_centrality[metric].nlargest(k).index.tolist())
            overlap = len(topsis_top & metric_top) / k * 100
            overlap_data.append({'k': k, 'Metric': metric, 'Overlap (%)': overlap})
    
    df = pd.DataFrame(overlap_data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, k in enumerate(k_values):
        k_data = df[df['k'] == k]
        values = [k_data[k_data['Metric'] == m]['Overlap (%)'].values[0] for m in metrics]
        ax.bar(x + i * width, values, width, label=f'Top-{k}')
    
    ax.set_xlabel('Centrality Measure')
    ax.set_ylabel('Overlap with TOPSIS (%)')
    ax.set_title('Ranking Overlap: Single Metrics vs CRITIC-TOPSIS')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend(title='Top-k')
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_attack_curves_enhanced(attack_results: Dict[str, pd.DataFrame],
                                 network_name: str = '',
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Enhanced attack curves with confidence annotations.
    
    Args:
        attack_results: Results from compare_attack_methods
        network_name: Name of the network for title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    setup_publication_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color scheme
    colors = {
        'CRITIC-TOPSIS': '#e41a1c',  # Red
        'degree': '#377eb8',          # Blue
        'betweenness': '#4daf4a',     # Green
        'closeness': '#984ea3',       # Purple
        'pagerank': '#ff7f00',        # Orange
    }
    
    markers = {'CRITIC-TOPSIS': 'o', 'degree': 's', 'betweenness': '^', 
               'closeness': 'D', 'pagerank': 'v'}
    
    # Left plot: LCC curves
    for method, df in attack_results.items():
        color = colors.get(method, '#999999')
        marker = markers.get(method, 'o')
        ax1.plot(df['fraction_removed'] * 100, df['lcc_fraction'],
                label=method, color=color, marker=marker,
                linewidth=2.5, markersize=8, markeredgecolor='white', markeredgewidth=1)
    
    ax1.set_xlabel('Nodes Removed (%)')
    ax1.set_ylabel('Largest Connected Component (fraction)')
    ax1.set_title(f'Network Fragmentation Under Attack')
    ax1.legend(loc='upper right')
    ax1.set_xlim(-1, 32)
    ax1.set_ylim(0, 1.05)
    ax1.fill_between([0, 30], [0.5, 0.5], [0, 0], alpha=0.1, color='red', 
                     label='Critical zone')
    
    # Right plot: Efficiency curves
    for method, df in attack_results.items():
        if 'efficiency' in df.columns:
            color = colors.get(method, '#999999')
            marker = markers.get(method, 'o')
            ax2.plot(df['fraction_removed'] * 100, df['efficiency'],
                    label=method, color=color, marker=marker,
                    linewidth=2.5, markersize=8, markeredgecolor='white', markeredgewidth=1)
    
    ax2.set_xlabel('Nodes Removed (%)')
    ax2.set_ylabel('Global Efficiency')
    ax2.set_title(f'Network Efficiency Under Attack')
    ax2.legend(loc='upper right')
    ax2.set_xlim(-1, 32)
    ax2.set_ylim(0, None)
    
    plt.suptitle(f'{network_name}' if network_name else '', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_topsis_scores_distribution(topsis_results: pd.DataFrame,
                                     network_name: str = '',
                                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Distribution of TOPSIS closeness scores.
    
    Args:
        topsis_results: TOPSIS results DataFrame
        network_name: Network name for title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    setup_publication_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(topsis_results['closeness'], bins=20, color='steelblue', 
             edgecolor='white', alpha=0.8)
    ax1.axvline(topsis_results['closeness'].mean(), color='red', 
                linestyle='--', linewidth=2, label='Mean')
    ax1.axvline(topsis_results['closeness'].median(), color='orange', 
                linestyle='--', linewidth=2, label='Median')
    ax1.set_xlabel('TOPSIS Closeness Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Node Importance Scores')
    ax1.legend()
    
    # Top nodes bar chart
    top_20 = topsis_results.nsmallest(20, 'rank')
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, 20))[::-1]
    ax2.barh(range(20), top_20['closeness'].values, color=colors)
    ax2.set_yticks(range(20))
    ax2.set_yticklabels(top_20.index)
    ax2.invert_yaxis()
    ax2.set_xlabel('TOPSIS Closeness Score')
    ax2.set_title('Top 20 Critical Nodes')
    
    plt.suptitle(f'{network_name}' if network_name else '', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_results_dashboard(G: nx.Graph,
                              df_centrality: pd.DataFrame,
                              weights: pd.Series,
                              topsis_results: pd.DataFrame,
                              attack_results: Dict[str, pd.DataFrame],
                              effectiveness: pd.DataFrame,
                              network_name: str,
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive results dashboard.
    
    Args:
        Various experiment results
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    setup_publication_style()
    
    fig = plt.figure(figsize=(16, 12))
    
    # Layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Network visualization (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    pos = nx.spring_layout(G, k=2/np.sqrt(G.number_of_nodes()), seed=42)
    critical = topsis_results.nsmallest(10, 'rank').index.tolist()
    other = [n for n in G.nodes() if n not in critical]
    nx.draw_networkx_nodes(G, pos, nodelist=other, node_color='lightblue',
                          node_size=50, alpha=0.5, ax=ax1)
    nx.draw_networkx_nodes(G, pos, nodelist=critical, node_color='red',
                          node_size=150, alpha=0.9, ax=ax1)
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax1)
    ax1.set_title('Network (Critical Nodes in Red)')
    ax1.axis('off')
    
    # 2. CRITIC Weights (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    weights_sorted = weights.sort_values()
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(weights)))
    ax2.barh(weights_sorted.index, weights_sorted.values, color=colors)
    ax2.set_xlabel('Weight')
    ax2.set_title('CRITIC Weights')
    
    # 3. Effectiveness comparison (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    eff_sorted = effectiveness.sort_values('effectiveness')
    colors = ['red' if m == 'CRITIC-TOPSIS' else 'steelblue' 
              for m in eff_sorted['method']]
    ax3.barh(eff_sorted['method'], eff_sorted['effectiveness'], color=colors)
    ax3.set_xlabel('Attack Effectiveness')
    ax3.set_title('Method Comparison')
    ax3.set_xlim(0, 1)
    
    # 4. Attack curves (middle, spanning 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    cmap = {'CRITIC-TOPSIS': 'red', 'degree': 'blue', 'betweenness': 'green',
            'closeness': 'purple', 'pagerank': 'orange'}
    for method, df in attack_results.items():
        ax4.plot(df['fraction_removed'] * 100, df['lcc_fraction'],
                label=method, color=cmap.get(method, 'gray'), linewidth=2, marker='o')
    ax4.set_xlabel('Nodes Removed (%)')
    ax4.set_ylabel('LCC Fraction')
    ax4.set_title('Targeted Attack Simulation')
    ax4.legend(loc='upper right')
    ax4.set_ylim(0, 1.05)
    
    # 5. Correlation heatmap (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    corr = df_centrality.corr()
    sns.heatmap(corr, annot=True, fmt='.1f', cmap='RdYlBu_r', center=0,
                square=True, ax=ax5, cbar=False, annot_kws={'size': 8})
    ax5.set_title('Centrality Correlations')
    
    # 6. TOPSIS score distribution (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.hist(topsis_results['closeness'], bins=15, color='steelblue', 
             edgecolor='white')
    ax6.set_xlabel('Closeness Score')
    ax6.set_ylabel('Frequency')
    ax6.set_title('TOPSIS Score Distribution')
    
    # 7. Top nodes table (bottom middle/right)
    ax7 = fig.add_subplot(gs[2, 1:])
    top_10 = topsis_results.nsmallest(10, 'rank')[['closeness', 'rank']].round(4)
    cell_text = [[str(idx)] + [f'{v:.4f}' for v in row] for idx, row in top_10.iterrows()]
    table = ax7.table(cellText=cell_text, 
                      colLabels=['Node', 'Closeness', 'Rank'],
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax7.axis('off')
    ax7.set_title('Top 10 Critical Nodes')
    
    plt.suptitle(f'{network_name} - CRITIC-TOPSIS Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("Enhanced visualization module loaded.")
    print("Use create_results_dashboard() for comprehensive visualizations.")
