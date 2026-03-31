import pytest
import networkx as nx
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.centralities import compute_all_centralities
from src.critic import compute_critic_weights
from src.topsis import topsis_rank

def test_extreme_density_complete_graph():
    """
    Test mathematically extreme graph condition: Complete Graph K_n.
    Every node is connected to every other node. All centralities are identical.
    Standard deviation is 0, correlation is undefined.
    CRITIC should fallback to equal weighting safely without dividing by zero.
    """
    G = nx.complete_graph(10)
    df_centrality = compute_all_centralities(G, verbose=False)
    
    weights, details = compute_critic_weights(df_centrality, verbose=False)
    
    # Expect uniform weights because info content is identically 0
    expected_weight = 1.0 / len(df_centrality.columns)
    for col in df_centrality.columns:
        assert np.isclose(weights[col], expected_weight), f"Expected uniform weight {expected_weight}, got {weights[col]}"

def test_disconnected_network():
    """
    Test mathematically extreme graph condition: Empty Graph (no edges).
    Testing TOPSIS specifically: if D+ and D- are both 0, it should not throw ZeroDivisionError.
    """
    G = nx.empty_graph(5)
    df_centrality = compute_all_centralities(G, verbose=False)
    
    weights, _ = compute_critic_weights(df_centrality, verbose=False)
    results, _ = topsis_rank(df_centrality, weights, verbose=False)
    
    # In a completely disconnected graph, all nodes are identically poor.
    # So their rank must be identical 
    assert len(results['rank'].unique()) == 1, "All nodes in a disconnected graph should tie with the exact same rank"
