"""
Data Loading Module
===================
Functions to load synthetic and real benchmark networks for critical node detection experiments.

Supported Networks:
- Karate Club (34 nodes, social network)
- Dolphins (62 nodes, social network)
- Les Miserables (77 nodes, character co-occurrence)
- Football (115 nodes, game schedule)
- Power Grid (4941 nodes, infrastructure - requires download)
- USAir (332 nodes, airline routes - requires download)
"""

import networkx as nx
import numpy as np
import os
from pathlib import Path


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent.parent / "data"


# ============================================================================
# BUILT-IN NETWORKS (Available directly from NetworkX)
# ============================================================================

def load_karate_club() -> nx.Graph:
    """
    Load Zachary's Karate Club network.
    
    A social network of friendships between 34 members of a karate club.
    Classic benchmark for community detection and node importance.
    
    Returns:
        nx.Graph: Undirected graph with 34 nodes and 78 edges
    """
    G = nx.karate_club_graph()
    G.name = "Karate Club"
    return G


def load_les_miserables() -> nx.Graph:
    """
    Load Les Miserables character co-occurrence network.
    
    Nodes are characters from Victor Hugo's novel, edges represent
    co-appearances in the same chapter.
    
    Returns:
        nx.Graph: Undirected graph with 77 nodes and 254 edges
    """
    G = nx.les_miserables_graph()
    G.name = "Les Miserables"
    return G


def load_florentine_families() -> nx.Graph:
    """
    Load Florentine Families marriage network.
    
    Historical marriage ties between Renaissance Florentine families.
    Good for testing centrality measures on small networks.
    
    Returns:
        nx.Graph: Undirected graph with 15 nodes and 20 edges
    """
    G = nx.florentine_families_graph()
    G.name = "Florentine Families"
    return G


# ============================================================================
# DOWNLOADABLE NETWORKS (From network repositories)
# ============================================================================

def load_dolphins() -> nx.Graph:
    """
    Load the Dolphins social network.
    
    Association network of 62 bottlenose dolphins in New Zealand.
    Edges represent frequent associations between dolphins.
    
    Returns:
        nx.Graph: Undirected graph with ~62 nodes and ~159 edges
    """
    # URL for dolphins dataset (GML format)
    url = "http://www-personal.umich.edu/~mejn/netdata/dolphins.zip"
    data_path = get_data_dir() / "real_networks" / "dolphins.gml"
    
    if data_path.exists():
        return nx.read_gml(data_path)
    
    # Try to download
    try:
        import urllib.request
        import zipfile
        import io
        
        print(f"Downloading dolphins network...")
        with urllib.request.urlopen(url, timeout=30) as response:
            zip_data = io.BytesIO(response.read())
        
        with zipfile.ZipFile(zip_data) as zf:
            for name in zf.namelist():
                if name.endswith('.gml'):
                    data_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as gml_file:
                        content = gml_file.read().decode('utf-8')
                        data_path.write_text(content)
                    break
        
        G = nx.read_gml(data_path)
        G.name = "Dolphins"
        return G
        
    except Exception as e:
        print(f"Could not download dolphins network: {e}")
        print("Creating synthetic alternative...")
        return _create_synthetic_dolphins()


def _create_synthetic_dolphins() -> nx.Graph:
    """Create a synthetic network similar to dolphins for fallback."""
    G = nx.connected_watts_strogatz_graph(62, 4, 0.3, seed=42)
    G.name = "Dolphins (synthetic)"
    return G


def load_football() -> nx.Graph:
    """
    Load the American College Football network.
    
    Network of American football games between Division IA colleges
    during Fall 2000 season. 115 nodes (teams), ~613 edges (games).
    
    Returns:
        nx.Graph: Undirected graph
    """
    url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"
    data_path = get_data_dir() / "real_networks" / "football.gml"
    
    if data_path.exists():
        return nx.read_gml(data_path)
    
    try:
        import urllib.request
        import zipfile
        import io
        
        print(f"Downloading football network...")
        with urllib.request.urlopen(url, timeout=30) as response:
            zip_data = io.BytesIO(response.read())
        
        with zipfile.ZipFile(zip_data) as zf:
            for name in zf.namelist():
                if name.endswith('.gml'):
                    data_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as gml_file:
                        content = gml_file.read().decode('utf-8')
                        data_path.write_text(content)
                    break
        
        G = nx.read_gml(data_path)
        G.name = "Football"
        return G
        
    except Exception as e:
        print(f"Could not download football network: {e}")
        G = nx.connected_watts_strogatz_graph(115, 6, 0.3, seed=42)
        G.name = "Football (synthetic)"
        return G


def load_power_grid() -> nx.Graph:
    """
    Load the Western US Power Grid network.
    
    High-voltage power grid of the Western United States.
    4941 nodes (generators, transformers, substations), 6594 edges (power lines).
    
    Returns:
        nx.Graph: Undirected graph
    """
    url = "http://www-personal.umich.edu/~mejn/netdata/power.zip"
    data_path = get_data_dir() / "real_networks" / "power.gml"
    
    if data_path.exists():
        return nx.read_gml(data_path)
    
    try:
        import urllib.request
        import zipfile
        import io
        
        print(f"Downloading power grid network...")
        with urllib.request.urlopen(url, timeout=30) as response:
            zip_data = io.BytesIO(response.read())
        
        with zipfile.ZipFile(zip_data) as zf:
            for name in zf.namelist():
                if name.endswith('.gml'):
                    data_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as gml_file:
                        content = gml_file.read().decode('utf-8')
                        data_path.write_text(content)
                    break
        
        G = nx.read_gml(data_path)
        G.name = "Power Grid"
        return G
        
    except Exception as e:
        print(f"Could not download power grid network: {e}")
        # Create smaller synthetic version
        G = nx.powerlaw_cluster_graph(500, 3, 0.1, seed=42)
        G.name = "Power Grid (synthetic)"
        return G


# ============================================================================
# SYNTHETIC NETWORKS (For controlled experiments)
# ============================================================================

def create_barabasi_albert(n: int = 100, m: int = 3, seed: int = 42) -> nx.Graph:
    """
    Create a Barabási-Albert scale-free network.
    
    Args:
        n: Number of nodes
        m: Number of edges to attach from new node to existing nodes
        seed: Random seed for reproducibility
    
    Returns:
        nx.Graph: Scale-free network with power-law degree distribution
    """
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    G.name = f"Barabasi-Albert (n={n}, m={m})"
    return G


def create_erdos_renyi(n: int = 100, p: float = 0.05, seed: int = 42) -> nx.Graph:
    """
    Create an Erdős-Rényi random network.
    
    Args:
        n: Number of nodes
        p: Probability of edge creation
        seed: Random seed for reproducibility
    
    Returns:
        nx.Graph: Random network
    """
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    # Ensure connected
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    G.name = f"Erdos-Renyi (n={n}, p={p})"
    return G


def create_watts_strogatz(n: int = 100, k: int = 4, p: float = 0.3, seed: int = 42) -> nx.Graph:
    """
    Create a Watts-Strogatz small-world network.
    
    Args:
        n: Number of nodes
        k: Each node is joined with its k nearest neighbors
        p: Probability of rewiring each edge
        seed: Random seed for reproducibility
    
    Returns:
        nx.Graph: Small-world network
    """
    G = nx.connected_watts_strogatz_graph(n, k, p, seed=seed)
    G.name = f"Watts-Strogatz (n={n}, k={k}, p={p})"
    return G


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_network_info(G: nx.Graph) -> dict:
    """
    Get basic information about a network.
    
    Args:
        G: NetworkX graph
    
    Returns:
        dict: Network statistics
    """
    return {
        "name": getattr(G, 'name', 'Unknown'),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_connected": nx.is_connected(G),
        "avg_clustering": nx.average_clustering(G),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes()
    }


def load_all_benchmark_networks() -> dict:
    """
    Load all available benchmark networks.
    
    Returns:
        dict: Dictionary mapping network names to graphs
    """
    networks = {}
    
    loaders = [
        ("karate", load_karate_club),
        ("florentine", load_florentine_families),
        ("les_miserables", load_les_miserables),
        ("dolphins", load_dolphins),
        ("football", load_football),
    ]
    
    for name, loader in loaders:
        try:
            networks[name] = loader()
            print(f"✓ Loaded {name}")
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
    
    return networks


if __name__ == "__main__":
    # Test loading networks
    print("Testing network loading...\n")
    
    # Test built-in networks
    for loader in [load_karate_club, load_les_miserables, load_florentine_families]:
        G = loader()
        info = get_network_info(G)
        print(f"{info['name']}: {info['nodes']} nodes, {info['edges']} edges")
    
    print("\nTesting synthetic networks...")
    for creator in [create_barabasi_albert, create_erdos_renyi, create_watts_strogatz]:
        G = creator()
        info = get_network_info(G)
        print(f"{info['name']}: {info['nodes']} nodes, {info['edges']} edges")
