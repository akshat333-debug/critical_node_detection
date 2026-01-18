"""
CRITIC Module (CRiteria Importance Through Intercriteria Correlation)
=====================================================================

CRITIC is an objective weighting method for multi-criteria decision making.
It determines weights based on:
1. Standard deviation (contrast intensity) - higher variance = more discriminating power
2. Inter-criteria correlation - less correlated criteria are more informative

Key idea: A criterion is more important if it:
- Has high variance (distinguishes well between alternatives)
- Has low correlation with other criteria (provides unique information)

Weight formula:
    w_j = C_j / Σ C_j
where:
    C_j = σ_j * Σ(1 - r_jk)  (information content)
    σ_j = standard deviation of criterion j
    r_jk = correlation between criteria j and k
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


def normalize_minmax(df: pd.DataFrame) -> pd.DataFrame:
    """
    Min-max normalization: scale each column to [0, 1].
    
    Formula: x_normalized = (x - min) / (max - min)
    
    Args:
        df: DataFrame with nodes as rows, criteria as columns
    
    Returns:
        pd.DataFrame: Normalized DataFrame (same shape)
    """
    df_norm = df.copy()
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val - min_val > 0:
            df_norm[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            # All values are the same - set to 0.5
            df_norm[col] = 0.5
    return df_norm


def normalize_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score (standard) normalization.
    
    Formula: x_normalized = (x - mean) / std
    
    Args:
        df: DataFrame with nodes as rows, criteria as columns
    
    Returns:
        pd.DataFrame: Normalized DataFrame (same shape)
    """
    df_norm = df.copy()
    for col in df.columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 0:
            df_norm[col] = (df[col] - mean_val) / std_val
        else:
            df_norm[col] = 0
    return df_norm


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix between all criteria.
    
    Args:
        df: Normalized DataFrame
    
    Returns:
        pd.DataFrame: Correlation matrix (criteria x criteria)
    """
    return df.corr()


def compute_critic_weights(df: pd.DataFrame, 
                           normalization: str = 'minmax',
                           verbose: bool = True) -> Tuple[pd.Series, Dict]:
    """
    Compute CRITIC weights for multi-attribute decision making.
    
    Steps:
    1. Normalize the data (min-max or z-score)
    2. Compute standard deviation of each criterion
    3. Compute correlation matrix
    4. Calculate information content: C_j = σ_j * Σ(1 - r_jk)
    5. Normalize to get weights: w_j = C_j / Σ C_j
    
    Args:
        df: DataFrame with nodes as rows, criteria (centralities) as columns
        normalization: 'minmax' or 'zscore'
        verbose: Whether to print intermediate results
    
    Returns:
        Tuple containing:
        - pd.Series: Weights for each criterion (sums to 1)
        - dict: Intermediate results (normalized df, std, corr, info_content)
    """
    if verbose:
        print("Computing CRITIC weights...")
    
    # Step 1: Normalize
    if normalization == 'minmax':
        df_norm = normalize_minmax(df)
    else:
        df_norm = normalize_zscore(df)
    
    if verbose:
        print(f"  - Applied {normalization} normalization")
    
    # Step 2: Compute standard deviation for each criterion
    std_devs = df_norm.std()
    
    if verbose:
        print(f"  - Standard deviations: {std_devs.round(4).to_dict()}")
    
    # Step 3: Compute correlation matrix
    corr_matrix = compute_correlation_matrix(df_norm)
    
    if verbose:
        print("  - Correlation matrix computed")
    
    # Step 4: Compute information content for each criterion
    # C_j = σ_j * Σ_k(1 - r_jk)
    n_criteria = len(df.columns)
    info_content = {}
    
    for col in df.columns:
        # Sum of (1 - correlation) with all other criteria
        # Handle NaN correlations (occur when std=0) by treating them as 0 correlation
        conflict_sum = 0
        for other_col in df.columns:
            corr_val = corr_matrix.loc[col, other_col]
            if pd.isna(corr_val):
                corr_val = 0  # Treat NaN as no correlation
            conflict_sum += (1 - corr_val)
        info_content[col] = std_devs[col] * conflict_sum
    
    info_content = pd.Series(info_content)
    
    # Handle case where all info_content is 0 or NaN
    if info_content.sum() == 0 or pd.isna(info_content.sum()):
        # Fall back to equal weights
        weights = pd.Series({col: 1.0 / len(df.columns) for col in df.columns})
        if verbose:
            print("  - Warning: Using equal weights (insufficient variance)")
    else:
        if verbose:
            print(f"  - Information content: {info_content.round(4).to_dict()}")
        
        # Step 5: Normalize to get weights
        total_info = info_content.sum()
        weights = info_content / total_info
    
    if verbose:
        print(f"  - Final weights: {weights.round(4).to_dict()}")
        print(f"  - Weights sum: {weights.sum():.6f}")
    
    # Return weights and intermediate results
    details = {
        'normalized_df': df_norm,
        'std_devs': std_devs,
        'correlation_matrix': corr_matrix,
        'info_content': info_content
    }
    
    return weights, details


def explain_weights(weights: pd.Series, details: Dict) -> str:
    """
    Generate a human-readable explanation of the CRITIC weights.
    
    Args:
        weights: CRITIC weights
        details: Details dict from compute_critic_weights
    
    Returns:
        str: Explanation text
    """
    explanation = []
    explanation.append("CRITIC Weight Analysis")
    explanation.append("=" * 50)
    
    # Sort by weight
    sorted_weights = weights.sort_values(ascending=False)
    
    explanation.append("\nCriteria ranked by importance:")
    for i, (criterion, weight) in enumerate(sorted_weights.items(), 1):
        std = details['std_devs'][criterion]
        info = details['info_content'][criterion]
        explanation.append(
            f"  {i}. {criterion}: weight={weight:.4f} "
            f"(std={std:.4f}, info={info:.4f})"
        )
    
    explanation.append("\nInterpretation:")
    top_criterion = sorted_weights.index[0]
    bottom_criterion = sorted_weights.index[-1]
    
    explanation.append(
        f"  - '{top_criterion}' has highest weight because it has "
        f"high variance and/or low correlation with other criteria."
    )
    explanation.append(
        f"  - '{bottom_criterion}' has lowest weight because it has "
        f"low variance and/or high correlation with other criteria."
    )
    
    return "\n".join(explanation)


if __name__ == "__main__":
    # Test the CRITIC module
    print("Testing CRITIC module...\n")
    
    import sys
    sys.path.insert(0, str(__file__).rsplit('/', 1)[0])
    from centralities import compute_all_centralities
    import networkx as nx
    
    # Load test network
    G = nx.karate_club_graph()
    print(f"Network: Karate Club ({G.number_of_nodes()} nodes)\n")
    
    # Compute centralities
    df = compute_all_centralities(G, verbose=False)
    
    print("Input centrality matrix (first 5 rows):")
    print(df.head().round(4))
    print()
    
    # Compute CRITIC weights
    weights, details = compute_critic_weights(df)
    
    print("\n" + explain_weights(weights, details))
    
    print("\nCorrelation matrix:")
    print(details['correlation_matrix'].round(3))
