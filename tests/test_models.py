"""
Unit tests for portfolio optimization modules.
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import GeometricBrownianMotion, VaRCalculator
from optimization import PortfolioOptimizer


class TestGBM(unittest.TestCase):
    """Test Geometric Brownian Motion model."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500)
        returns = pd.DataFrame(
            np.random.randn(500, 5) * 0.01,
            index=dates,
            columns=['Stock1', 'Stock2', 'Stock3', 'Stock4', 'Stock5']
        )
        self.gbm = GeometricBrownianMotion(returns)
    
    def test_parameter_estimation(self):
        """Test parameter estimation."""
        params = self.gbm.estimate_parameters()
        
        self.assertIsNotNone(params['mu'])
        self.assertIsNotNone(params['sigma'])
        self.assertIsNotNone(params['correlation'])
        
        self.assertEqual(len(params['mu']), 5)
        self.assertEqual(len(params['sigma']), 5)
        self.assertEqual(params['correlation'].shape, (5, 5))
    
    def test_simulate_paths(self):
        """Test path simulation."""
        self.gbm.estimate_parameters()
        
        S0 = np.array([100, 200, 150, 120, 180])
        paths = self.gbm.simulate_paths(S0, T=252, n_paths=100, use_sobol=False)
        
        self.assertEqual(paths.shape, (100, 253, 5))
        self.assertTrue(np.all(paths[:, 0, :] == S0))
        self.assertTrue(np.all(paths > 0))


class TestVaRCalculator(unittest.TestCase):
    """Test VaR calculator."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.returns = np.random.randn(1000) * 0.01
        self.var_calc = VaRCalculator([0.95, 0.99])
    
    def test_historical_var(self):
        """Test historical VaR calculation."""
        var_95 = self.var_calc.historical_var(self.returns, 0.95)
        var_99 = self.var_calc.historical_var(self.returns, 0.99)
        
        self.assertLess(var_99, var_95)  # 99% VaR should be more extreme
        self.assertLess(var_95, 0)  # VaR should be negative
    
    def test_parametric_var(self):
        """Test parametric VaR calculation."""
        mean = self.returns.mean()
        std = self.returns.std()
        
        var = self.var_calc.parametric_var(mean, std, 0.95)
        self.assertIsInstance(var, float)
    
    def test_conditional_var(self):
        """Test CVaR calculation."""
        cvar = self.var_calc.conditional_var(self.returns, 0.95)
        var = self.var_calc.historical_var(self.returns, 0.95)
        
        self.assertLess(cvar, var)  # CVaR should be more extreme than VaR


class TestPortfolioOptimizer(unittest.TestCase):
    """Test portfolio optimizer."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.expected_returns = np.array([0.10, 0.12, 0.08, 0.15, 0.09])
        cov = np.random.randn(5, 5) * 0.01
        self.covariance_matrix = cov @ cov.T  # Ensure positive semi-definite
        self.optimizer = PortfolioOptimizer(
            self.expected_returns,
            self.covariance_matrix,
            risk_free_rate=0.05
        )
    
    def test_max_sharpe_optimization(self):
        """Test maximum Sharpe ratio optimization."""
        result = self.optimizer.optimize_max_sharpe(min_weight=0.0, max_weight=1.0)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result['weights']), 5)
        self.assertAlmostEqual(result['weights'].sum(), 1.0, places=5)
        self.assertTrue(np.all(result['weights'] >= 0))
        self.assertTrue(np.all(result['weights'] <= 1))
    
    def test_mean_variance_optimization(self):
        """Test mean-variance optimization."""
        result = self.optimizer.optimize_mean_variance(
            target_return=0.10,
            min_weight=0.0,
            max_weight=1.0
        )
        
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['weights'].sum(), 1.0, places=5)
    
    def test_efficient_frontier(self):
        """Test efficient frontier calculation."""
        frontier = self.optimizer.efficient_frontier(n_points=10)
        
        self.assertGreater(len(frontier), 0)
        self.assertTrue('return' in frontier.columns)
        self.assertTrue('volatility' in frontier.columns)
        self.assertTrue('sharpe_ratio' in frontier.columns)


if __name__ == '__main__':
    unittest.main()
