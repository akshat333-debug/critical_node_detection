"""
Domain-Specific Weighting Module
=================================
Pre-trained weight profiles for different network domains.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import networkx as nx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from centralities import compute_all_centralities
from critic import compute_critic_weights
from topsis import topsis_rank


# Pre-defined weight profiles based on domain characteristics
DOMAIN_WEIGHTS = {
    'social': {
        'name': 'Social Network',
        'description': 'Optimized for social networks (Facebook, Twitter, etc.)',
        'weights': {
            'degree': 0.12,
            'betweenness': 0.10,
            'closeness': 0.08,
            'eigenvector': 0.25,  # Influential connections matter
            'pagerank': 0.25,     # Authority matters
            'kshell': 0.10,
            'hindex': 0.10
        },
        'rationale': 'Social networks prioritize influence (eigenvector, pagerank) over pure connectivity.'
    },
    'infrastructure': {
        'name': 'Infrastructure Network',
        'description': 'Optimized for infrastructure (power grids, roads, airports)',
        'weights': {
            'degree': 0.15,
            'betweenness': 0.30,  # Bridges are critical
            'closeness': 0.20,   # Reachability matters
            'eigenvector': 0.10,
            'pagerank': 0.05,
            'kshell': 0.10,
            'hindex': 0.10
        },
        'rationale': 'Infrastructure networks prioritize bottlenecks (betweenness) and accessibility (closeness).'
    },
    'biological': {
        'name': 'Biological Network',
        'description': 'Optimized for biological networks (protein, neural, metabolic)',
        'weights': {
            'degree': 0.20,       # Hubs are important
            'betweenness': 0.15,
            'closeness': 0.10,
            'eigenvector': 0.15,
            'pagerank': 0.10,
            'kshell': 0.20,       # Core structure matters
            'hindex': 0.10
        },
        'rationale': 'Biological networks balance hub importance (degree) with core structure (kshell).'
    },
    'communication': {
        'name': 'Communication Network',
        'description': 'Optimized for communication networks (email, internet)',
        'weights': {
            'degree': 0.10,
            'betweenness': 0.25,  # Routing bottlenecks
            'closeness': 0.25,   # Latency matters
            'eigenvector': 0.10,
            'pagerank': 0.15,
            'kshell': 0.10,
            'hindex': 0.05
        },
        'rationale': 'Communication networks prioritize routing efficiency and low latency.'
    },
    'citation': {
        'name': 'Citation Network',
        'description': 'Optimized for citation/reference networks',
        'weights': {
            'degree': 0.15,
            'betweenness': 0.10,
            'closeness': 0.05,
            'eigenvector': 0.20,
            'pagerank': 0.35,     # PageRank was designed for this!
            'kshell': 0.05,
            'hindex': 0.10
        },
        'rationale': 'Citation networks are best analyzed with PageRank, designed for web/citation graphs.'
    }
}


def get_available_domains() -> Dict:
    """Return available domain profiles."""
    return {k: {'name': v['name'], 'description': v['description']} 
            for k, v in DOMAIN_WEIGHTS.items()}


def get_domain_weights(domain: str) -> pd.Series:
    """Get pre-defined weights for a domain."""
    if domain not in DOMAIN_WEIGHTS:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_WEIGHTS.keys())}")
    
    return pd.Series(DOMAIN_WEIGHTS[domain]['weights'])


def detect_network_domain(G: nx.Graph) -> Dict:
    """
    Attempt to automatically detect the most suitable domain for a network.
    Based on structural properties.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)
    clustering = nx.average_clustering(G)
    
    # Degree distribution analysis
    degrees = [d for n, d in G.degree()]
    degree_variance = np.var(degrees)
    max_degree = max(degrees)
    
    # Heuristic scoring
    scores = {}
    
    # Social: high clustering, moderate density
    scores['social'] = clustering * 2 + (0.5 if density < 0.1 else 0)
    
    # Infrastructure: low clustering, low density, high betweenness variance
    scores['infrastructure'] = (1 - clustering) + (1 if density < 0.05 else 0)
    
    # Biological: power-law-ish degree, moderate clustering
    scores['biological'] = (degree_variance / (max_degree + 1)) * 0.1 + clustering
    
    # Communication: moderate everything
    scores['communication'] = 1 - abs(density - 0.1) - abs(clustering - 0.3)
    
    # Citation: likely directed originally, moderate clustering
    scores['citation'] = clustering * 0.5 + (1 if degree_variance > 10 else 0)
    
    best_domain = max(scores, key=scores.get)
    
    return {
        'suggested_domain': best_domain,
        'confidence': scores[best_domain] / sum(scores.values()) if sum(scores.values()) > 0 else 0,
        'all_scores': scores,
        'network_properties': {
            'nodes': n,
            'edges': m,
            'density': density,
            'clustering': clustering,
            'degree_variance': degree_variance
        }
    }


def domain_aware_analysis(G: nx.Graph,
                          domain: Optional[str] = None,
                          auto_detect: bool = True) -> Dict:
    """
    Run critical node analysis with domain-specific weights.
    
    Args:
        G: Network graph
        domain: Domain name (social, infrastructure, biological, etc.)
        auto_detect: If True and domain is None, auto-detect domain
    
    Returns:
        Analysis results with domain-specific weights
    """
    # Detect domain if not specified
    if domain is None and auto_detect:
        detection = detect_network_domain(G)
        domain = detection['suggested_domain']
        detection_info = detection
    else:
        detection_info = None
    
    # Get domain weights
    domain_weights = get_domain_weights(domain)
    
    # Compute centralities
    df = compute_all_centralities(G, verbose=False)
    
    # Also compute CRITIC weights for comparison
    critic_weights, _ = compute_critic_weights(df, verbose=False)
    
    # Run TOPSIS with domain weights
    # Filter weights to only include columns present in df
    available_weights = domain_weights[domain_weights.index.isin(df.columns)]
    available_weights = available_weights / available_weights.sum()  # Renormalize
    
    results, _ = topsis_rank(df, available_weights, verbose=False)
    
    # Also run with CRITIC for comparison
    results_critic, _ = topsis_rank(df, critic_weights, verbose=False)
    
    # Compare rankings
    domain_top10 = set(results.nsmallest(10, 'rank').index)
    critic_top10 = set(results_critic.nsmallest(10, 'rank').index)
    overlap = len(domain_top10 & critic_top10) / 10
    
    return {
        'domain': domain,
        'domain_info': DOMAIN_WEIGHTS[domain],
        'domain_weights': available_weights.to_dict(),
        'critic_weights': critic_weights.to_dict(),
        'results': results,
        'top_10_critical': results.nsmallest(10, 'rank').index.tolist(),
        'comparison_with_critic': {
            'overlap': overlap,
            'unique_to_domain': list(domain_top10 - critic_top10),
            'unique_to_critic': list(critic_top10 - domain_top10)
        },
        'detection': detection_info
    }


if __name__ == "__main__":
    print("Domain-Specific Weighting Test")
    print("=" * 50)
    
    G = nx.karate_club_graph()
    
    # Auto-detect
    detection = detect_network_domain(G)
    print(f"Detected domain: {detection['suggested_domain']} ({detection['confidence']:.1%} confidence)")
    
    # Run analysis
    result = domain_aware_analysis(G, domain='social')
    print(f"\nSocial domain top-10: {result['top_10_critical']}")
    print(f"Overlap with CRITIC: {result['comparison_with_critic']['overlap']:.0%}")
