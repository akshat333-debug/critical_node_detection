"""
Real-World Dataset Integration Module
======================================
Load and process real-world network datasets from various sources.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Optional, Dict
import urllib.request
import io
import ssl


# Disable SSL verification for downloads
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


# ============================================================================
# DATASET CATALOG
# ============================================================================

DATASETS = {
    'social': {
        'facebook_ego': {
            'name': 'Facebook Ego Network',
            'nodes': '~4000',
            'type': 'Social',
            'url': 'https://snap.stanford.edu/data/facebook_combined.txt.gz',
            'description': 'Facebook friendship network'
        },
        'twitter_higgs': {
            'name': 'Twitter Higgs',
            'nodes': '~456k', 
            'type': 'Social',
            'description': 'Twitter activity during Higgs boson discovery'
        }
    },
    'infrastructure': {
        'euro_roads': {
            'name': 'European Roads',
            'nodes': '~1200',
            'type': 'Infrastructure',
            'description': 'European road network'
        },
        'us_airports': {
            'name': 'US Airports',
            'nodes': '~500',
            'type': 'Infrastructure',
            'description': 'US airport connections'
        }
    },
    'biological': {
        'c_elegans': {
            'name': 'C. Elegans Neural',
            'nodes': '~300',
            'type': 'Biological',
            'description': 'Neural network of C. elegans worm'
        },
        'protein_interaction': {
            'name': 'Protein Interaction',
            'nodes': '~2k',
            'type': 'Biological',
            'description': 'Yeast protein interactions'
        }
    }
}


# ============================================================================
# SYNTHETIC GENERATORS FOR REAL-WORLD-LIKE NETWORKS
# ============================================================================

def generate_social_network(n: int = 500, 
                            avg_friends: int = 10,
                            community_prob: float = 0.3) -> nx.Graph:
    """
    Generate a social-network-like graph with community structure.
    Uses LFR benchmark or relaxed caveman model.
    """
    # Simple community-based model
    n_communities = max(2, n // 50)
    comm_size = n // n_communities
    
    G = nx.Graph()
    G.name = f"Social Network ({n} nodes)"
    
    # Add nodes to communities
    communities = []
    for i in range(n_communities):
        start = i * comm_size
        end = start + comm_size if i < n_communities - 1 else n
        comm = list(range(start, end))
        communities.append(comm)
        
        # Dense within-community connections
        for j, node in enumerate(comm):
            G.add_node(node)
            # Connect to ~avg_friends/2 within community
            n_internal = min(len(comm) - 1, avg_friends // 2)
            if n_internal > 0 and len(comm) > 1:
                other_nodes = [c for c in comm if c != node]
                targets = np.random.choice(other_nodes, 
                                           size=min(n_internal, len(other_nodes)), 
                                           replace=False)
                for t in targets:
                    G.add_edge(node, t)
    
    # Cross-community connections
    for node in G.nodes():
        if np.random.random() < community_prob:
            # Find which community this node belongs to
            node_comm_idx = node // comm_size
            if node_comm_idx >= len(communities):
                node_comm_idx = len(communities) - 1
            
            # Select a different community
            other_comm_indices = [i for i in range(len(communities)) if i != node_comm_idx]
            if other_comm_indices:
                other_idx = np.random.choice(other_comm_indices)
                other_comm = communities[other_idx]
                target = np.random.choice(other_comm)
                G.add_edge(node, target)
    
    return G


def generate_infrastructure_network(n: int = 300,
                                     hubs: int = 5,
                                     hub_size: int = 20) -> nx.Graph:
    """
    Generate infrastructure-like network with hub-and-spoke pattern.
    """
    G = nx.Graph()
    G.name = f"Infrastructure Network ({n} nodes)"
    
    # Create hubs
    hub_nodes = list(range(hubs))
    for h in hub_nodes:
        G.add_node(h)
    
    # Connect hubs
    for i in range(hubs):
        for j in range(i+1, hubs):
            if np.random.random() < 0.7:
                G.add_edge(i, j)
    
    # Add spoke nodes
    next_node = hubs
    for hub in hub_nodes:
        for _ in range(hub_size):
            if next_node >= n:
                break
            G.add_node(next_node)
            G.add_edge(hub, next_node)
            # Some spoke-to-spoke connections
            if next_node > hubs and np.random.random() < 0.2:
                other = np.random.randint(hubs, next_node)
                G.add_edge(next_node, other)
            next_node += 1
    
    # Fill remaining nodes
    while next_node < n:
        G.add_node(next_node)
        # Connect to random existing node preferentially
        degrees = dict(G.degree())
        probs = np.array([degrees[node] for node in G.nodes() if node < next_node])
        probs = probs / probs.sum()
        target = np.random.choice(list(G.nodes())[:next_node], p=probs)
        G.add_edge(next_node, target)
        next_node += 1
    
    return G


def generate_biological_network(n: int = 300,
                                 avg_degree: int = 4) -> nx.Graph:
    """
    Generate biological-network-like graph using scale-free model.
    Biological networks often have power-law degree distributions.
    """
    m = avg_degree // 2
    G = nx.barabasi_albert_graph(n, m)
    G.name = f"Biological Network ({n} nodes)"
    
    # Add some clustering (protein networks have higher clustering)
    nodes = list(G.nodes())
    for _ in range(n // 5):
        # Make triangles
        node = np.random.choice(nodes)
        neighbors = list(G.neighbors(node))
        if len(neighbors) >= 2:
            n1, n2 = np.random.choice(neighbors, size=2, replace=False)
            if not G.has_edge(n1, n2):
                G.add_edge(n1, n2)
    
    return G


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def get_available_datasets() -> Dict:
    """Return catalog of available datasets."""
    return DATASETS


def load_real_world_network(category: str, 
                            name: str, 
                            n: int = 300) -> nx.Graph:
    """
    Load or generate a real-world-like network.
    
    Args:
        category: 'social', 'infrastructure', or 'biological'
        name: Dataset name or 'generate' for synthetic
        n: Number of nodes for synthetic networks
    
    Returns:
        NetworkX graph
    """
    if category == 'social':
        return generate_social_network(n)
    elif category == 'infrastructure':
        return generate_infrastructure_network(n)
    elif category == 'biological':
        return generate_biological_network(n)
    else:
        raise ValueError(f"Unknown category: {category}")


def get_network_characteristics(G: nx.Graph) -> Dict:
    """
    Compute comprehensive network characteristics.
    """
    degrees = [d for n, d in G.degree()]
    
    return {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': np.mean(degrees),
        'max_degree': max(degrees),
        'clustering': nx.average_clustering(G),
        'is_connected': nx.is_connected(G),
        'n_components': nx.number_connected_components(G),
        'diameter': nx.diameter(G) if nx.is_connected(G) else 'N/A'
    }


if __name__ == "__main__":
    print("Real-World Dataset Module")
    print("=" * 50)
    
    for category in ['social', 'infrastructure', 'biological']:
        print(f"\n{category.upper()}:")
        G = load_real_world_network(category, 'generate', n=200)
        chars = get_network_characteristics(G)
        print(f"  Nodes: {chars['nodes']}, Edges: {chars['edges']}")
        print(f"  Density: {chars['density']:.4f}, Clustering: {chars['clustering']:.4f}")
