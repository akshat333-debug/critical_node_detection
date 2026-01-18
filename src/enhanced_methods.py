"""
Enhanced CRITIC-TOPSIS Module
=============================
Improved implementations with multiple normalization methods,
adaptive weighting, and hybrid approaches.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from centralities import compute_all_centralities
from critic import normalize_minmax, compute_correlation_matrix
from topsis import normalize_vector, compute_ideal_solutions, compute_distances, compute_closeness_coefficient


# ============================================================================
# ENHANCED NORMALIZATION METHODS
# ============================================================================

def normalize_max(df: pd.DataFrame) -> pd.DataFrame:
    """Max normalization: x' = x / max(x)."""
    return df / df.max()


def normalize_sum(df: pd.DataFrame) -> pd.DataFrame:
    """Sum normalization: x' = x / sum(x)."""
    return df / df.sum()


def normalize_log(df: pd.DataFrame) -> pd.DataFrame:
    """Log normalization for handling skewed distributions."""
    df_log = df.copy()
    for col in df.columns:
        df_log[col] = np.log1p(df[col])
    return normalize_minmax(df_log)


# ============================================================================
# ENHANCED WEIGHTING METHODS
# ============================================================================

def compute_entropy_weights(df: pd.DataFrame, verbose: bool = False) -> pd.Series:
    """
    Entropy-based objective weighting.
    
    Higher entropy = less information = lower weight.
    Lower entropy = more discriminating = higher weight.
    """
    # Normalize to proportions
    df_norm = df / df.sum()
    df_norm = df_norm.replace(0, 1e-10)  # Avoid log(0)
    
    # Compute entropy
    n = len(df)
    k = 1 / np.log(n)
    
    entropy = {}
    for col in df.columns:
        p = df_norm[col]
        e = -k * (p * np.log(p)).sum()
        entropy[col] = e
    
    entropy = pd.Series(entropy)
    
    # Weight = 1 - entropy (normalized)
    weights = 1 - entropy
    weights = weights / weights.sum()
    
    if verbose:
        print(f"Entropy weights: {weights.round(4).to_dict()}")
    
    return weights


def compute_stddev_weights(df: pd.DataFrame, verbose: bool = False) -> pd.Series:
    """
    Standard deviation-based weighting.
    
    Higher std = more discriminating = higher weight.
    """
    df_norm = normalize_minmax(df)
    std_devs = df_norm.std()
    weights = std_devs / std_devs.sum()
    
    if verbose:
        print(f"StdDev weights: {weights.round(4).to_dict()}")
    
    return weights


def compute_critic_enhanced(df: pd.DataFrame, 
                            correlation_power: float = 1.0,
                            verbose: bool = False) -> pd.Series:
    """
    Enhanced CRITIC with adjustable correlation influence.
    
    Args:
        df: Centrality DataFrame
        correlation_power: Exponent for (1-r) term. Higher = more emphasis on uncorrelated criteria.
        verbose: Print details
    
    Returns:
        pd.Series: Weights
    """
    df_norm = normalize_minmax(df)
    std_devs = df_norm.std()
    corr_matrix = compute_correlation_matrix(df_norm)
    
    info_content = {}
    for col in df.columns:
        conflict_sum = 0
        for other_col in df.columns:
            corr_val = corr_matrix.loc[col, other_col]
            if pd.isna(corr_val):
                corr_val = 0
            conflict_sum += (1 - abs(corr_val)) ** correlation_power
        info_content[col] = std_devs[col] * conflict_sum
    
    info_content = pd.Series(info_content)
    
    if info_content.sum() == 0:
        weights = pd.Series({col: 1.0/len(df.columns) for col in df.columns})
    else:
        weights = info_content / info_content.sum()
    
    if verbose:
        print(f"Enhanced CRITIC weights (power={correlation_power}): {weights.round(4).to_dict()}")
    
    return weights


def compute_hybrid_weights(df: pd.DataFrame,
                           method_weights: Dict[str, float] = None,
                           verbose: bool = False) -> pd.Series:
    """
    Combine multiple weighting methods.
    
    Args:
        df: Centrality DataFrame
        method_weights: Dict mapping method name to its weight in combination
        verbose: Print details
    
    Returns:
        pd.Series: Hybrid weights
    """
    if method_weights is None:
        method_weights = {
            'critic': 0.4,
            'entropy': 0.3,
            'stddev': 0.3
        }
    
    all_weights = {}
    
    if 'critic' in method_weights:
        all_weights['critic'] = compute_critic_enhanced(df, verbose=False)
    if 'entropy' in method_weights:
        all_weights['entropy'] = compute_entropy_weights(df, verbose=False)
    if 'stddev' in method_weights:
        all_weights['stddev'] = compute_stddev_weights(df, verbose=False)
    
    # Combine
    hybrid = pd.Series({col: 0 for col in df.columns})
    for method, w in all_weights.items():
        hybrid += method_weights[method] * w
    
    hybrid = hybrid / hybrid.sum()
    
    if verbose:
        print(f"Hybrid weights: {hybrid.round(4).to_dict()}")
    
    return hybrid


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_top_centralities(df: pd.DataFrame, 
                            n: int = 5,
                            selection_method: str = 'variance') -> List[str]:
    """
    Select top-n most informative centralities.
    
    Args:
        df: Centrality DataFrame
        n: Number of centralities to keep
        selection_method: 'variance' or 'correlation'
    
    Returns:
        List of selected column names
    """
    df_norm = normalize_minmax(df)
    
    if selection_method == 'variance':
        # Select by highest variance
        variances = df_norm.var().sort_values(ascending=False)
        return variances.head(n).index.tolist()
    
    elif selection_method == 'correlation':
        # Select by lowest average correlation (most unique)
        corr = df_norm.corr().abs()
        avg_corr = corr.mean().sort_values()
        return avg_corr.head(n).index.tolist()
    
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")


def remove_redundant_centralities(df: pd.DataFrame, 
                                    threshold: float = 0.9) -> pd.DataFrame:
    """
    Remove highly correlated centralities (keep the one with higher variance).
    
    Args:
        df: Centrality DataFrame
        threshold: Correlation threshold above which to remove
    
    Returns:
        DataFrame with redundant columns removed
    """
    df_norm = normalize_minmax(df)
    corr = df_norm.corr().abs()
    variances = df_norm.var()
    
    to_remove = set()
    for i, col1 in enumerate(df.columns):
        if col1 in to_remove:
            continue
        for col2 in df.columns[i+1:]:
            if col2 in to_remove:
                continue
            if corr.loc[col1, col2] > threshold:
                # Remove the one with lower variance
                if variances[col1] < variances[col2]:
                    to_remove.add(col1)
                else:
                    to_remove.add(col2)
    
    return df.drop(columns=list(to_remove))


# ============================================================================
# ENHANCED TOPSIS
# ============================================================================

def topsis_with_threshold(df: pd.DataFrame,
                          weights: pd.Series,
                          threshold: float = 0.5) -> Tuple[pd.DataFrame, List]:
    """
    TOPSIS with hard threshold for critical node identification.
    
    Nodes with closeness above threshold are marked as critical.
    
    Args:
        df: Centrality DataFrame
        weights: Criterion weights
        threshold: Closeness threshold for critical classification
    
    Returns:
        Tuple of (full results, critical node list)
    """
    # Standard TOPSIS
    df_norm = normalize_vector(df)
    df_weighted = df_norm * weights
    
    ideal_best, ideal_worst = compute_ideal_solutions(df_weighted)
    dist_best, dist_worst = compute_distances(df_weighted, ideal_best, ideal_worst)
    closeness = compute_closeness_coefficient(dist_best, dist_worst)
    
    results = pd.DataFrame({
        'closeness': closeness,
        'dist_to_best': dist_best,
        'dist_to_worst': dist_worst
    })
    results['rank'] = results['closeness'].rank(ascending=False).astype(int)
    results['is_critical'] = results['closeness'] >= threshold
    
    critical_nodes = results[results['is_critical']].index.tolist()
    
    return results.sort_values('rank'), critical_nodes


def topsis_ensemble(df: pd.DataFrame,
                    weight_methods: List[str] = ['critic', 'entropy', 'stddev'],
                    verbose: bool = False) -> pd.DataFrame:
    """
    Ensemble TOPSIS using multiple weighting methods.
    
    Final ranking is based on average rank across methods.
    
    Args:
        df: Centrality DataFrame
        weight_methods: List of weighting methods to use
        verbose: Print details
    
    Returns:
        DataFrame with ensemble rankings
    """
    all_rankings = {}
    
    for method in weight_methods:
        if method == 'critic':
            weights = compute_critic_enhanced(df)
        elif method == 'entropy':
            weights = compute_entropy_weights(df)
        elif method == 'stddev':
            weights = compute_stddev_weights(df)
        else:
            continue
        
        # Standard TOPSIS
        df_norm = normalize_vector(df)
        df_weighted = df_norm * weights
        ideal_best, ideal_worst = compute_ideal_solutions(df_weighted)
        dist_best, dist_worst = compute_distances(df_weighted, ideal_best, ideal_worst)
        closeness = compute_closeness_coefficient(dist_best, dist_worst)
        
        all_rankings[method] = pd.Series(closeness, index=df.index).rank(ascending=False)
    
    # Compute average rank (Borda-like aggregation)
    rankings_df = pd.DataFrame(all_rankings)
    rankings_df['avg_rank'] = rankings_df.mean(axis=1)
    rankings_df['final_rank'] = rankings_df['avg_rank'].rank().astype(int)
    
    if verbose:
        print(f"Ensemble with {len(weight_methods)} methods:")
        print(rankings_df.nsmallest(10, 'final_rank'))
    
    return rankings_df.sort_values('final_rank')


# ============================================================================
# COMPLETE ENHANCED PIPELINE
# ============================================================================

def run_enhanced_experiment(G, 
                            enhancement: str = 'hybrid_weights',
                            verbose: bool = True) -> Dict:
    """
    Run enhanced CRITIC-TOPSIS experiment.
    
    Args:
        G: NetworkX graph
        enhancement: One of 'hybrid_weights', 'feature_selection', 'ensemble', 'correlation_power'
        verbose: Print progress
    
    Returns:
        Dict with results
    """
    # Compute centralities
    df = compute_all_centralities(G, verbose=False)
    
    if enhancement == 'hybrid_weights':
        weights = compute_hybrid_weights(df, verbose=verbose)
        
    elif enhancement == 'feature_selection':
        # Remove redundant, keep top 5
        df = remove_redundant_centralities(df, threshold=0.85)
        selected = select_top_centralities(df, n=5, selection_method='variance')
        df = df[selected]
        weights = compute_critic_enhanced(df, verbose=verbose)
        
    elif enhancement == 'ensemble':
        rankings = topsis_ensemble(df, verbose=verbose)
        return {'rankings': rankings, 'centralities': df}
        
    elif enhancement == 'correlation_power':
        # Increase emphasis on unique criteria
        weights = compute_critic_enhanced(df, correlation_power=2.0, verbose=verbose)
    
    else:
        weights = compute_critic_enhanced(df, verbose=verbose)
    
    # TOPSIS
    df_norm = normalize_vector(df)
    df_weighted = df_norm * weights
    ideal_best, ideal_worst = compute_ideal_solutions(df_weighted)
    dist_best, dist_worst = compute_distances(df_weighted, ideal_best, ideal_worst)
    closeness = compute_closeness_coefficient(dist_best, dist_worst)
    
    results = pd.DataFrame({
        'closeness': closeness,
        'rank': pd.Series(closeness).rank(ascending=False).astype(int)
    }, index=df.index)
    
    return {
        'results': results.sort_values('rank'),
        'weights': weights,
        'centralities': df
    }


if __name__ == "__main__":
    import networkx as nx
    
    print("Testing Enhanced CRITIC-TOPSIS Methods")
    print("=" * 60)
    
    G = nx.karate_club_graph()
    
    for enhancement in ['hybrid_weights', 'feature_selection', 'ensemble', 'correlation_power']:
        print(f"\n{enhancement.upper()}")
        print("-" * 40)
        result = run_enhanced_experiment(G, enhancement=enhancement, verbose=True)
        
        if 'results' in result:
            top_5 = result['results'].head(5).index.tolist()
            print(f"Top 5 nodes: {top_5}")
        elif 'rankings' in result:
            top_5 = result['rankings'].head(5).index.tolist()
            print(f"Top 5 nodes: {top_5}")
