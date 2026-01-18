#!/usr/bin/env python3
"""
Test script to verify all required libraries are installed correctly.
Run: python test_installation.py
"""

def test_imports():
    """Test that all required libraries can be imported."""
    print("Testing library imports...\n")
    
    libraries = [
        ("networkx", "nx"),
        ("numpy", "np"),
        ("pandas", "pd"),
        ("scipy", None),
        ("matplotlib.pyplot", "plt"),
        ("seaborn", "sns"),
        ("sklearn", None),
    ]
    
    all_success = True
    
    for lib, alias in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {lib:20} - version {version}")
        except ImportError as e:
            print(f"✗ {lib:20} - FAILED: {e}")
            all_success = False
    
    print("\n" + "=" * 50)
    
    if all_success:
        print("SUCCESS: All libraries installed correctly!")
        print("\nQuick NetworkX test:")
        import networkx as nx
        G = nx.karate_club_graph()
        print(f"  - Loaded Karate Club graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"  - Example degree centrality: {list(nx.degree_centrality(G).items())[:3]}")
    else:
        print("FAILED: Some libraries are missing. Please install them.")
    
    return all_success

if __name__ == "__main__":
    test_imports()
