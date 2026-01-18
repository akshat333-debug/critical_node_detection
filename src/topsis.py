"""
TOPSIS Module (Technique for Order of Preference by Similarity to Ideal Solution)
==================================================================================

TOPSIS ranks alternatives (nodes) by their closeness to an ideal best solution
and distance from an ideal worst solution.

Steps:
1. Normalize the decision matrix
2. Apply weights to get weighted normalized matrix
3. Identify ideal best (A+) and ideal worst (A-) for each criterion
4. Calculate distance of each alternative to A+ and A-
5. Calculate closeness coefficient: C = D- / (D+ + D-)
6. Rank alternatives by closeness (higher = better)

All our centrality measures are "benefit" criteria (higher is better),
so ideal best = max values, ideal worst = min values.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def normalize_vector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vector normalization for TOPSIS.
    
    Formula: r_ij = x_ij / sqrt(Σ x_ij²)
    
    This ensures the sum of squares of each column equals 1.
    
    Args:
        df: Decision matrix (alternatives x criteria)
    
    Returns:
        pd.DataFrame: Vector-normalized matrix
    """
    df_norm = df.copy()
    for col in df.columns:
        norm_factor = np.sqrt((df[col] ** 2).sum())
        if norm_factor > 0:
            df_norm[col] = df[col] / norm_factor
        else:
            df_norm[col] = 0
    return df_norm


def apply_weights(df_normalized: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """
    Apply weights to normalized matrix.
    
    Formula: v_ij = w_j * r_ij
    
    Args:
        df_normalized: Normalized decision matrix
        weights: Weight for each criterion (should sum to 1)
    
    Returns:
        pd.DataFrame: Weighted normalized matrix
    """
    df_weighted = df_normalized.copy()
    for col in df_normalized.columns:
        df_weighted[col] = df_normalized[col] * weights[col]
    return df_weighted


def compute_ideal_solutions(df_weighted: pd.DataFrame, 
                            criteria_types: Optional[dict] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Identify ideal best (A+) and ideal worst (A-) solutions.
    
    For benefit criteria (higher = better): A+ = max, A- = min
    For cost criteria (lower = better): A+ = min, A- = max
    
    Args:
        df_weighted: Weighted normalized matrix
        criteria_types: Dict mapping criterion name to 'benefit' or 'cost'
                       Default: all criteria are 'benefit'
    
    Returns:
        Tuple of (ideal_best, ideal_worst) as pd.Series
    """
    if criteria_types is None:
        # All centralities are benefit criteria (higher = more important)
        criteria_types = {col: 'benefit' for col in df_weighted.columns}
    
    ideal_best = pd.Series(index=df_weighted.columns, dtype=float)
    ideal_worst = pd.Series(index=df_weighted.columns, dtype=float)
    
    for col in df_weighted.columns:
        if criteria_types.get(col, 'benefit') == 'benefit':
            ideal_best[col] = df_weighted[col].max()
            ideal_worst[col] = df_weighted[col].min()
        else:
            ideal_best[col] = df_weighted[col].min()
            ideal_worst[col] = df_weighted[col].max()
    
    return ideal_best, ideal_worst


def compute_distances(df_weighted: pd.DataFrame, 
                      ideal_best: pd.Series, 
                      ideal_worst: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Euclidean distance of each alternative to ideal solutions.
    
    D+ = sqrt(Σ (v_ij - v_j+)²)  (distance to ideal best)
    D- = sqrt(Σ (v_ij - v_j-)²)  (distance to ideal worst)
    
    Args:
        df_weighted: Weighted normalized matrix
        ideal_best: Ideal best solution
        ideal_worst: Ideal worst solution
    
    Returns:
        Tuple of (distance_to_best, distance_to_worst) as pd.Series
    """
    # Distance to ideal best
    dist_best = np.sqrt(((df_weighted - ideal_best) ** 2).sum(axis=1))
    dist_best.name = 'dist_to_best'
    
    # Distance to ideal worst
    dist_worst = np.sqrt(((df_weighted - ideal_worst) ** 2).sum(axis=1))
    dist_worst.name = 'dist_to_worst'
    
    return dist_best, dist_worst


def compute_closeness_coefficient(dist_best: pd.Series, 
                                    dist_worst: pd.Series) -> pd.Series:
    """
    Calculate closeness coefficient for each alternative.
    
    C_i = D-_i / (D+_i + D-_i)
    
    Range: [0, 1]
    - C = 1: Alternative is the ideal best
    - C = 0: Alternative is the ideal worst
    - Higher C = better alternative
    
    Args:
        dist_best: Distance to ideal best
        dist_worst: Distance to ideal worst
    
    Returns:
        pd.Series: Closeness coefficient for each alternative
    """
    total_dist = dist_best + dist_worst
    # Avoid division by zero
    closeness = np.where(total_dist > 0, dist_worst / total_dist, 0)
    return pd.Series(closeness, index=dist_best.index, name='closeness')


def topsis_rank(df: pd.DataFrame, 
                weights: pd.Series, 
                verbose: bool = True) -> Tuple[pd.DataFrame, dict]:
    """
    Complete TOPSIS ranking of alternatives.
    
    Args:
        df: Decision matrix (alternatives x criteria)
              For us: nodes x centrality measures
        weights: Weights for each criterion (from CRITIC or user-defined)
        verbose: Whether to print progress
    
    Returns:
        Tuple containing:
        - pd.DataFrame: Results with columns [closeness, rank] plus distances
        - dict: Intermediate results
    """
    if verbose:
        print("Performing TOPSIS ranking...")
    
    # Step 1: Vector normalization
    df_normalized = normalize_vector(df)
    if verbose:
        print("  - Applied vector normalization")
    
    # Step 2: Apply weights
    df_weighted = apply_weights(df_normalized, weights)
    if verbose:
        print("  - Applied CRITIC weights")
    
    # Step 3: Compute ideal solutions
    ideal_best, ideal_worst = compute_ideal_solutions(df_weighted)
    if verbose:
        print("  - Computed ideal best and worst solutions")
    
    # Step 4: Compute distances
    dist_best, dist_worst = compute_distances(df_weighted, ideal_best, ideal_worst)
    if verbose:
        print("  - Computed distances to ideal solutions")
    
    # Step 5: Compute closeness coefficient
    closeness = compute_closeness_coefficient(dist_best, dist_worst)
    if verbose:
        print("  - Computed closeness coefficients")
    
    # Step 6: Create results DataFrame
    results = pd.DataFrame({
        'closeness': closeness,
        'dist_to_best': dist_best,
        'dist_to_worst': dist_worst
    })
    
    # Add rank (1 = best)
    results['rank'] = results['closeness'].rank(ascending=False).astype(int)
    results = results.sort_values('rank')
    
    if verbose:
        print(f"  - Ranking complete. Top node: {results.index[0]}")
    
    details = {
        'normalized': df_normalized,
        'weighted': df_weighted,
        'ideal_best': ideal_best,
        'ideal_worst': ideal_worst
    }
    
    return results, details


def get_critical_nodes(results: pd.DataFrame, k: int = 10) -> List:
    """
    Get top-k critical nodes from TOPSIS results.
    
    Args:
        results: TOPSIS results DataFrame (must have 'rank' column)
        k: Number of top nodes to return
    
    Returns:
        list: Node IDs of top-k critical nodes
    """
    return results.nsmallest(k, 'rank').index.tolist()


def compare_with_single_metrics(df_centrality: pd.DataFrame, 
                                 topsis_results: pd.DataFrame,
                                 k: int = 10) -> pd.DataFrame:
    """
    Compare TOPSIS ranking with single-metric rankings.
    
    Shows overlap between TOPSIS top-k and each centrality's top-k.
    
    Args:
        df_centrality: Original centrality DataFrame
        topsis_results: TOPSIS results
        k: Number of top nodes to compare
    
    Returns:
        pd.DataFrame: Comparison table
    """
    topsis_topk = set(get_critical_nodes(topsis_results, k))
    
    comparison = []
    for col in df_centrality.columns:
        metric_topk = set(df_centrality[col].nlargest(k).index.tolist())
        overlap = len(topsis_topk & metric_topk)
        comparison.append({
            'metric': col,
            'overlap_with_topsis': overlap,
            'overlap_percent': overlap / k * 100
        })
    
    return pd.DataFrame(comparison)


if __name__ == "__main__":
    # Test the TOPSIS module
    print("Testing TOPSIS module...\n")
    
    import sys
    sys.path.insert(0, str(__file__).rsplit('/', 1)[0])
    from centralities import compute_all_centralities
    from critic import compute_critic_weights
    import networkx as nx
    
    # Load test network
    G = nx.karate_club_graph()
    print(f"Network: Karate Club ({G.number_of_nodes()} nodes)\n")
    
    # Compute centralities
    df = compute_all_centralities(G, verbose=False)
    
    # Compute CRITIC weights
    weights, _ = compute_critic_weights(df, verbose=False)
    print(f"CRITIC weights: {weights.round(4).to_dict()}\n")
    
    # Perform TOPSIS ranking
    results, details = topsis_rank(df, weights)
    
    print("\nTop 10 critical nodes (CRITIC-TOPSIS):")
    print(results.head(10).round(4))
    
    print("\nBottom 5 nodes:")
    print(results.tail(5).round(4))
    
    print("\nComparison with single metrics (top-10 overlap):")
    comparison = compare_with_single_metrics(df, results, k=10)
    print(comparison)
