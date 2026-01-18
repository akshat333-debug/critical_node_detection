"""
Unit Tests for Critical Node Detection Modules
==============================================

Run with: python -m pytest tests/ -v
Or: python tests/test_all.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import unittest
import numpy as np
import pandas as pd
import networkx as nx

from centralities import (
    compute_degree_centrality,
    compute_betweenness_centrality,
    compute_closeness_centrality,
    compute_eigenvector_centrality,
    compute_pagerank,
    compute_kshell,
    compute_hindex,
    compute_all_centralities
)

from critic import (
    normalize_minmax,
    normalize_zscore,
    compute_correlation_matrix,
    compute_critic_weights
)

from topsis import (
    normalize_vector,
    apply_weights,
    compute_ideal_solutions,
    compute_distances,
    compute_closeness_coefficient,
    topsis_rank,
    get_critical_nodes
)

from evaluation import (
    get_largest_component_size,
    get_largest_component_fraction,
    compute_global_efficiency,
    simulate_targeted_attack,
    compare_attack_methods,
    compute_attack_effectiveness,
    get_ranking_from_centrality,
    get_ranking_from_topsis
)

from data_loading import (
    load_karate_club,
    load_florentine_families,
    create_barabasi_albert,
    get_network_info
)


class TestCentralities(unittest.TestCase):
    """Tests for centrality measures."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test network."""
        cls.G = nx.karate_club_graph()
    
    def test_degree_centrality(self):
        """Test degree centrality computation."""
        dc = compute_degree_centrality(self.G)
        self.assertEqual(len(dc), self.G.number_of_nodes())
        self.assertTrue(all(0 <= v <= 1 for v in dc.values()))
        # Node 0 and 33 should have high degree
        self.assertGreater(dc[0], 0.3)
        self.assertGreater(dc[33], 0.3)
    
    def test_betweenness_centrality(self):
        """Test betweenness centrality computation."""
        bc = compute_betweenness_centrality(self.G)
        self.assertEqual(len(bc), self.G.number_of_nodes())
        self.assertTrue(all(0 <= v <= 1 for v in bc.values()))
    
    def test_closeness_centrality(self):
        """Test closeness centrality computation."""
        cc = compute_closeness_centrality(self.G)
        self.assertEqual(len(cc), self.G.number_of_nodes())
        self.assertTrue(all(0 <= v <= 1 for v in cc.values()))
    
    def test_eigenvector_centrality(self):
        """Test eigenvector centrality computation."""
        ec = compute_eigenvector_centrality(self.G)
        self.assertEqual(len(ec), self.G.number_of_nodes())
        self.assertTrue(all(v >= 0 for v in ec.values()))
    
    def test_pagerank(self):
        """Test PageRank computation."""
        pr = compute_pagerank(self.G)
        self.assertEqual(len(pr), self.G.number_of_nodes())
        self.assertAlmostEqual(sum(pr.values()), 1.0, places=5)
    
    def test_kshell(self):
        """Test k-shell computation."""
        ks = compute_kshell(self.G)
        self.assertEqual(len(ks), self.G.number_of_nodes())
        self.assertTrue(all(v >= 0 for v in ks.values()))
    
    def test_hindex(self):
        """Test H-index computation."""
        hi = compute_hindex(self.G)
        self.assertEqual(len(hi), self.G.number_of_nodes())
        self.assertTrue(all(v >= 0 for v in hi.values()))
    
    def test_compute_all_centralities(self):
        """Test computing all centralities returns proper DataFrame."""
        df = compute_all_centralities(self.G, verbose=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), self.G.number_of_nodes())
        self.assertEqual(len(df.columns), 7)
        expected_cols = ['degree', 'betweenness', 'closeness', 
                        'eigenvector', 'pagerank', 'kshell', 'hindex']
        self.assertEqual(list(df.columns), expected_cols)


class TestCRITIC(unittest.TestCase):
    """Tests for CRITIC weighting method."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.G = nx.karate_club_graph()
        cls.df = compute_all_centralities(cls.G, verbose=False)
    
    def test_normalize_minmax(self):
        """Test min-max normalization."""
        df_norm = normalize_minmax(self.df)
        for col in df_norm.columns:
            self.assertAlmostEqual(df_norm[col].min(), 0.0, places=5)
            self.assertAlmostEqual(df_norm[col].max(), 1.0, places=5)
    
    def test_normalize_zscore(self):
        """Test z-score normalization."""
        df_norm = normalize_zscore(self.df)
        for col in df_norm.columns:
            self.assertAlmostEqual(df_norm[col].mean(), 0.0, places=5)
            self.assertAlmostEqual(df_norm[col].std(), 1.0, places=1)
    
    def test_correlation_matrix(self):
        """Test correlation matrix computation."""
        corr = compute_correlation_matrix(self.df)
        self.assertEqual(corr.shape, (7, 7))
        # Diagonal should be 1
        for i in range(7):
            self.assertAlmostEqual(corr.iloc[i, i], 1.0, places=5)
    
    def test_critic_weights(self):
        """Test CRITIC weight computation."""
        weights, details = compute_critic_weights(self.df, verbose=False)
        self.assertIsInstance(weights, pd.Series)
        self.assertEqual(len(weights), 7)
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        self.assertTrue(all(w >= 0 for w in weights.values))
        self.assertIn('normalized_df', details)
        self.assertIn('std_devs', details)
        self.assertIn('correlation_matrix', details)


class TestTOPSIS(unittest.TestCase):
    """Tests for TOPSIS ranking method."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.G = nx.karate_club_graph()
        cls.df = compute_all_centralities(cls.G, verbose=False)
        cls.weights, _ = compute_critic_weights(cls.df, verbose=False)
    
    def test_normalize_vector(self):
        """Test vector normalization."""
        df_norm = normalize_vector(self.df)
        for col in df_norm.columns:
            # Sum of squares should be 1
            self.assertAlmostEqual((df_norm[col] ** 2).sum(), 1.0, places=5)
    
    def test_apply_weights(self):
        """Test weight application."""
        df_norm = normalize_vector(self.df)
        df_weighted = apply_weights(df_norm, self.weights)
        # Weighted values should be smaller than normalized
        self.assertTrue((df_weighted.values <= df_norm.values).all())
    
    def test_ideal_solutions(self):
        """Test ideal solution computation."""
        df_norm = normalize_vector(self.df)
        df_weighted = apply_weights(df_norm, self.weights)
        ideal_best, ideal_worst = compute_ideal_solutions(df_weighted)
        
        for col in df_weighted.columns:
            self.assertEqual(ideal_best[col], df_weighted[col].max())
            self.assertEqual(ideal_worst[col], df_weighted[col].min())
    
    def test_closeness_coefficient(self):
        """Test closeness coefficient computation."""
        dist_best = pd.Series([1.0, 0.5, 0.0])
        dist_worst = pd.Series([0.0, 0.5, 1.0])
        closeness = compute_closeness_coefficient(dist_best, dist_worst)
        
        np.testing.assert_array_almost_equal(closeness, [0.0, 0.5, 1.0])
    
    def test_topsis_rank(self):
        """Test complete TOPSIS ranking."""
        results, details = topsis_rank(self.df, self.weights, verbose=False)
        
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), self.G.number_of_nodes())
        self.assertIn('closeness', results.columns)
        self.assertIn('rank', results.columns)
        
        # Ranks should be 1 to n
        self.assertEqual(results['rank'].min(), 1)
        self.assertEqual(results['rank'].max(), self.G.number_of_nodes())
    
    def test_get_critical_nodes(self):
        """Test getting top-k critical nodes."""
        results, _ = topsis_rank(self.df, self.weights, verbose=False)
        top5 = get_critical_nodes(results, k=5)
        
        self.assertEqual(len(top5), 5)
        # Top nodes should have rank 1-5
        self.assertEqual(set(results.loc[top5, 'rank'].values), {1, 2, 3, 4, 5})


class TestEvaluation(unittest.TestCase):
    """Tests for evaluation and attack simulation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.G = nx.karate_club_graph()
    
    def test_largest_component_size(self):
        """Test LCC size computation."""
        size = get_largest_component_size(self.G)
        self.assertEqual(size, self.G.number_of_nodes())
    
    def test_largest_component_fraction(self):
        """Test LCC fraction computation."""
        frac = get_largest_component_fraction(self.G, self.G.number_of_nodes())
        self.assertAlmostEqual(frac, 1.0)
    
    def test_global_efficiency(self):
        """Test global efficiency computation."""
        eff = compute_global_efficiency(self.G)
        self.assertTrue(0 <= eff <= 1)
    
    def test_simulate_attack(self):
        """Test attack simulation."""
        ranking = list(self.G.nodes())
        results = simulate_targeted_attack(
            self.G, ranking, 
            fractions=[0.1, 0.2], 
            verbose=False
        )
        
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 3)  # 0%, 10%, 20%
        self.assertEqual(results.iloc[0]['lcc_fraction'], 1.0)
    
    def test_attack_effectiveness(self):
        """Test attack effectiveness computation."""
        df = pd.DataFrame({
            'fraction_removed': [0.0, 0.1, 0.2, 0.3],
            'lcc_fraction': [1.0, 0.8, 0.5, 0.2]
        })
        
        eff = compute_attack_effectiveness({'method1': df})
        self.assertEqual(len(eff), 1)
        self.assertTrue(0 <= eff.iloc[0]['effectiveness'] <= 1)


class TestDataLoading(unittest.TestCase):
    """Tests for data loading functions."""
    
    def test_load_karate_club(self):
        """Test loading Karate Club network."""
        G = load_karate_club()
        self.assertEqual(G.number_of_nodes(), 34)
        self.assertEqual(G.number_of_edges(), 78)
    
    def test_load_florentine(self):
        """Test loading Florentine Families network."""
        G = load_florentine_families()
        self.assertEqual(G.number_of_nodes(), 15)
    
    def test_create_synthetic(self):
        """Test creating synthetic networks."""
        G = create_barabasi_albert(50, 2)
        self.assertEqual(G.number_of_nodes(), 50)
        self.assertTrue(nx.is_connected(G))
    
    def test_network_info(self):
        """Test network info extraction."""
        G = load_karate_club()
        info = get_network_info(G)
        
        self.assertIn('name', info)
        self.assertIn('nodes', info)
        self.assertIn('edges', info)
        self.assertIn('density', info)
        self.assertEqual(info['nodes'], 34)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline."""
    
    def test_complete_pipeline(self):
        """Test running complete CRITIC-TOPSIS pipeline."""
        # Load network
        G = nx.karate_club_graph()
        
        # Compute centralities
        df = compute_all_centralities(G, verbose=False)
        self.assertEqual(len(df), 34)
        
        # CRITIC weights
        weights, _ = compute_critic_weights(df, verbose=False)
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        
        # TOPSIS ranking
        results, _ = topsis_rank(df, weights, verbose=False)
        self.assertEqual(len(results), 34)
        
        # Attack simulation
        rankings = {
            'TOPSIS': get_ranking_from_topsis(results),
            'degree': get_ranking_from_centrality(df, 'degree')
        }
        attack_results = compare_attack_methods(
            G, rankings, [0.1, 0.2], verbose=False
        )
        
        # Effectiveness
        eff = compute_attack_effectiveness(attack_results)
        self.assertEqual(len(eff), 2)


if __name__ == '__main__':
    print("Running Critical Node Detection Unit Tests...")
    print("=" * 60)
    unittest.main(verbosity=2)
