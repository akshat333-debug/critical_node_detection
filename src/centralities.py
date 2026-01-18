"""
Centralities Module
===================
Compute multiple centrality measures for all nodes in a network.

Centrality Measures:
1. Degree Centrality - Number of direct connections (local influence)
2. Betweenness Centrality - How often a node lies on shortest paths (bridge role)
3. Closeness Centrality - How quickly a node can reach all others (global accessibility)
4. Eigenvector Centrality - Importance based on connections to important nodes
5. PageRank - Random walk-based importance (like Google's algorithm)
6. K-Shell (Core Number) - Position in nested core structure (network backbone)
7. H-Index - Balance between degree and neighbor connectivity
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def compute_degree_centrality(G: nx.Graph) -> Dict[int, float]:
    """
    Compute degree centrality for all nodes.
    
    Degree centrality = (number of neighbors) / (max possible neighbors)
    High degree = many direct connections, local hub.
    
    Args:
        G: NetworkX graph
    
    Returns:
        dict: Node -> degree centrality value
    """
    return nx.degree_centrality(G)


def compute_betweenness_centrality(G: nx.Graph, normalized: bool = True) -> Dict[int, float]:
    """
    Compute betweenness centrality for all nodes.
    
    Betweenness = fraction of all shortest paths that pass through this node.
    High betweenness = bridge between communities, controls information flow.
    
    Note: O(n*m) complexity, can be slow for large graphs.
    
    Args:
        G: NetworkX graph
        normalized: Whether to normalize by 2/((n-1)(n-2))
    
    Returns:
        dict: Node -> betweenness centrality value
    """
    return nx.betweenness_centrality(G, normalized=normalized)


def compute_closeness_centrality(G: nx.Graph) -> Dict[int, float]:
    """
    Compute closeness centrality for all nodes.
    
    Closeness = 1 / (average distance to all other nodes)
    High closeness = can quickly reach everyone in the network.
    
    For disconnected graphs, uses only reachable nodes.
    
    Args:
        G: NetworkX graph
    
    Returns:
        dict: Node -> closeness centrality value
    """
    return nx.closeness_centrality(G)


def compute_eigenvector_centrality(G: nx.Graph, max_iter: int = 1000, 
                                    tol: float = 1e-6) -> Dict[int, float]:
    """
    Compute eigenvector centrality for all nodes.
    
    A node is important if connected to other important nodes.
    Computed as the principal eigenvector of the adjacency matrix.
    
    Args:
        G: NetworkX graph
        max_iter: Maximum iterations for power method
        tol: Convergence tolerance
    
    Returns:
        dict: Node -> eigenvector centrality value
    """
    try:
        return nx.eigenvector_centrality(G, max_iter=max_iter, tol=tol)
    except nx.PowerIterationFailedConvergence:
        # Fallback to numpy-based computation
        return nx.eigenvector_centrality_numpy(G)


def compute_pagerank(G: nx.Graph, alpha: float = 0.85, 
                     max_iter: int = 100, tol: float = 1e-6) -> Dict[int, float]:
    """
    Compute PageRank for all nodes.
    
    Similar to eigenvector centrality but with damping factor.
    Models a random walker who sometimes jumps to random nodes.
    
    Args:
        G: NetworkX graph
        alpha: Damping factor (probability of following edges)
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        dict: Node -> PageRank value
    """
    return nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol)


def compute_kshell(G: nx.Graph) -> Dict[int, int]:
    """
    Compute k-shell (core number) for all nodes.
    
    K-shell decomposition: iteratively remove nodes with degree < k.
    The k-shell of a node is the highest k for which it remains in the network.
    Higher k-shell = deeper in the network core, more "essential".
    
    Args:
        G: NetworkX graph
    
    Returns:
        dict: Node -> k-shell value (integer)
    """
    return nx.core_number(G)


def compute_hindex(G: nx.Graph) -> Dict[int, int]:
    """
    Compute H-index centrality for all nodes.
    
    H-index of a node = largest h such that the node has at least h neighbors
    with degree >= h.
    
    Captures both local connectivity and importance of neighbors.
    A node needs well-connected neighbors, not just many neighbors.
    
    Args:
        G: NetworkX graph
    
    Returns:
        dict: Node -> H-index value (integer)
    """
    h_index = {}
    degrees = dict(G.degree())
    
    for node in G.nodes():
        # Get degrees of all neighbors
        neighbor_degrees = sorted([degrees[n] for n in G.neighbors(node)], reverse=True)
        
        if not neighbor_degrees:
            h_index[node] = 0
            continue
        
        # Find h: largest h such that at least h neighbors have degree >= h
        h = 0
        for i, deg in enumerate(neighbor_degrees):
            if deg >= i + 1:
                h = i + 1
            else:
                break
        h_index[node] = h
    
    return h_index


def compute_all_centralities(G: nx.Graph, verbose: bool = True) -> pd.DataFrame:
    """
    Compute all centrality measures for all nodes in the graph.
    
    This is the main function that produces the multi-attribute decision matrix
    used as input to CRITIC-TOPSIS.
    
    Args:
        G: NetworkX graph
        verbose: Whether to print progress
    
    Returns:
        pd.DataFrame: DataFrame with nodes as index, centralities as columns
            Columns: degree, betweenness, closeness, eigenvector, pagerank, kshell, hindex
    """
    if verbose:
        print(f"Computing centralities for {G.number_of_nodes()} nodes...")
    
    nodes = list(G.nodes())
    
    # Compute each centrality
    if verbose:
        print("  - Degree centrality...")
    degree = compute_degree_centrality(G)
    
    if verbose:
        print("  - Betweenness centrality...")
    betweenness = compute_betweenness_centrality(G)
    
    if verbose:
        print("  - Closeness centrality...")
    closeness = compute_closeness_centrality(G)
    
    if verbose:
        print("  - Eigenvector centrality...")
    eigenvector = compute_eigenvector_centrality(G)
    
    if verbose:
        print("  - PageRank...")
    pagerank = compute_pagerank(G)
    
    if verbose:
        print("  - K-shell...")
    kshell = compute_kshell(G)
    
    if verbose:
        print("  - H-index...")
    hindex = compute_hindex(G)
    
    # Build DataFrame
    data = {
        'degree': [degree[n] for n in nodes],
        'betweenness': [betweenness[n] for n in nodes],
        'closeness': [closeness[n] for n in nodes],
        'eigenvector': [eigenvector[n] for n in nodes],
        'pagerank': [pagerank[n] for n in nodes],
        'kshell': [kshell[n] for n in nodes],
        'hindex': [hindex[n] for n in nodes],
    }
    
    df = pd.DataFrame(data, index=nodes)
    df.index.name = 'node'
    
    if verbose:
        print("  Done!")
    
    return df


def get_top_nodes_by_centrality(df: pd.DataFrame, centrality: str, k: int = 10) -> List:
    """
    Get top-k nodes by a specific centrality measure.
    
    Args:
        df: DataFrame from compute_all_centralities
        centrality: Column name (e.g., 'degree', 'betweenness')
        k: Number of top nodes to return
    
    Returns:
        list: Node IDs of top-k nodes
    """
    return df[centrality].nlargest(k).index.tolist()


def compare_rankings(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Compare top-k nodes across different centrality measures.
    
    Useful for seeing which nodes are consistently important vs.
    which are important only by certain measures.
    
    Args:
        df: DataFrame from compute_all_centralities
        k: Number of top nodes to compare
    
    Returns:
        pd.DataFrame: Top-k nodes by each centrality
    """
    rankings = {}
    for col in df.columns:
        rankings[col] = get_top_nodes_by_centrality(df, col, k)
    
    return pd.DataFrame(rankings)


if __name__ == "__main__":
    # Test the module
    print("Testing centralities module...\n")
    
    # Load a test network
    G = nx.karate_club_graph()
    print(f"Network: Karate Club ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)\n")
    
    # Compute all centralities
    df = compute_all_centralities(G)
    
    print("\nCentrality matrix (first 10 nodes):")
    print(df.head(10).round(4))
    
    print("\nTop 5 nodes by each centrality:")
    rankings = compare_rankings(df, k=5)
    print(rankings)
    
    print("\nCorrelation between centralities:")
    print(df.corr().round(3))
