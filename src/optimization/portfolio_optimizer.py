"""
Portfolio optimization module.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import cvxpy as cp
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Portfolio optimization using various methods.
    Supports mean-variance, minimum variance, and maximum Sharpe ratio.
    """
    
    def __init__(self, expected_returns: np.ndarray, 
                 covariance_matrix: np.ndarray,
                 risk_free_rate: float = 0.07):
        """
        Initialize optimizer.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            risk_free_rate: Risk-free rate for Sharpe ratio
        """
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
    
    def optimize_mean_variance(self, target_return: Optional[float] = None,
                              min_weight: float = 0.0,
                              max_weight: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Mean-variance optimization.
        
        Args:
            target_return: Target portfolio return (None for minimum variance)
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            
        Returns:
            Dictionary with optimal weights and metrics
        """
        logger.info("Running mean-variance optimization")
        
        # Define optimization variables
        w = cp.Variable(self.n_assets)
        
        # Objective: minimize portfolio variance
        portfolio_variance = cp.quad_form(w, self.covariance_matrix)
        objective = cp.Minimize(portfolio_variance)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= min_weight,  # No shorting (unless min_weight < 0)
            w <= max_weight   # Maximum position size
        ]
        
        # Add target return constraint if specified
        if target_return is not None:
            portfolio_return = w @ self.expected_returns
            constraints.append(portfolio_return >= target_return)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            logger.warning(f"Optimization status: {problem.status}")
            return None
        
        # Extract results
        optimal_weights = w.value
        portfolio_return = optimal_weights @ self.expected_returns
        portfolio_std = np.sqrt(optimal_weights @ self.covariance_matrix @ optimal_weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        
        return {
            'weights': optimal_weights,
            'return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio
        }
    
    def optimize_max_sharpe(self, min_weight: float = 0.0,
                           max_weight: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Maximize Sharpe ratio.
        
        Args:
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            
        Returns:
            Dictionary with optimal weights and metrics
        """
        logger.info("Optimizing for maximum Sharpe ratio")
        
        def neg_sharpe(weights):
            """Negative Sharpe ratio (for minimization)."""
            portfolio_return = weights @ self.expected_returns
            portfolio_std = np.sqrt(weights @ self.covariance_matrix @ weights)
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
            return -sharpe
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        
        # Initial guess (equal weights)
        w0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = minimize(
            neg_sharpe,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            return None
        
        optimal_weights = result.x
        portfolio_return = optimal_weights @ self.expected_returns
        portfolio_std = np.sqrt(optimal_weights @ self.covariance_matrix @ optimal_weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        
        return {
            'weights': optimal_weights,
            'return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio
        }
    
    def efficient_frontier(self, n_points: int = 50,
                          min_weight: float = 0.0,
                          max_weight: float = 1.0) -> pd.DataFrame:
        """
        Calculate efficient frontier.
        
        Args:
            n_points: Number of points on the frontier
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            
        Returns:
            DataFrame with frontier portfolios
        """
        logger.info(f"Calculating efficient frontier with {n_points} points")
        
        # Find minimum and maximum possible returns
        min_var_portfolio = self.optimize_mean_variance(
            target_return=None, min_weight=min_weight, max_weight=max_weight
        )
        
        # Maximum return portfolio (all in highest return asset, respecting constraints)
        max_ret = self.expected_returns.max()
        
        # Generate target returns
        target_returns = np.linspace(
            min_var_portfolio['return'],
            max_ret * 0.95,  # Stay slightly below max to ensure feasibility
            n_points
        )
        
        frontier_portfolios = []
        
        for target_ret in target_returns:
            portfolio = self.optimize_mean_variance(
                target_return=target_ret,
                min_weight=min_weight,
                max_weight=max_weight
            )
            
            if portfolio is not None:
                frontier_portfolios.append({
                    'return': portfolio['return'],
                    'volatility': portfolio['volatility'],
                    'sharpe_ratio': portfolio['sharpe_ratio'],
                    'weights': portfolio['weights']
                })
        
        frontier_df = pd.DataFrame(frontier_portfolios)
        
        logger.info(f"Efficient frontier calculated with {len(frontier_df)} valid points")
        
        return frontier_df
    
    def optimize_risk_parity(self) -> Dict[str, np.ndarray]:
        """
        Risk parity optimization (equal risk contribution).
        
        Returns:
            Dictionary with optimal weights and metrics
        """
        logger.info("Optimizing for risk parity")
        
        def risk_parity_objective(weights):
            """Objective: minimize difference in risk contributions."""
            portfolio_variance = weights @ self.covariance_matrix @ weights
            marginal_contrib = self.covariance_matrix @ weights
            risk_contrib = weights * marginal_contrib / np.sqrt(portfolio_variance)
            
            # Minimize sum of squared differences from equal risk
            target_risk = 1.0 / self.n_assets
            return np.sum((risk_contrib - target_risk) ** 2)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = tuple((0.0, 1.0) for _ in range(self.n_assets))
        w0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(
            risk_parity_objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        optimal_weights = result.x
        portfolio_return = optimal_weights @ self.expected_returns
        portfolio_std = np.sqrt(optimal_weights @ self.covariance_matrix @ optimal_weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        
        return {
            'weights': optimal_weights,
            'return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio
        }
    
    def black_litterman(self, views: Dict[int, float], 
                       view_confidence: float = 0.25) -> Dict[str, np.ndarray]:
        """
        Black-Litterman model for incorporating views.
        
        Args:
            views: Dictionary mapping asset index to expected return view
            view_confidence: Confidence in views (0-1)
            
        Returns:
            Dictionary with optimal weights using adjusted returns
        """
        logger.info("Applying Black-Litterman model")
        
        # Market equilibrium returns (reverse optimization)
        market_weights = np.ones(self.n_assets) / self.n_assets
        risk_aversion = 2.5
        pi = risk_aversion * self.covariance_matrix @ market_weights
        
        # View matrix
        P = np.zeros((len(views), self.n_assets))
        Q = np.zeros(len(views))
        
        for i, (asset_idx, view_return) in enumerate(views.items()):
            P[i, asset_idx] = 1
            Q[i] = view_return
        
        # Uncertainty in views
        omega = view_confidence * np.eye(len(views))
        
        # Black-Litterman formula
        tau = 0.025  # Scaling factor
        
        M_inv = np.linalg.inv(tau * self.covariance_matrix)
        BL_return = np.linalg.inv(M_inv + P.T @ np.linalg.inv(omega) @ P) @ \
                   (M_inv @ pi + P.T @ np.linalg.inv(omega) @ Q)
        
        # Update expected returns
        original_returns = self.expected_returns.copy()
        self.expected_returns = BL_return
        
        # Optimize with updated returns
        portfolio = self.optimize_max_sharpe()
        
        # Restore original returns
        self.expected_returns = original_returns
        
        return portfolio


def calculate_portfolio_metrics(weights: np.ndarray,
                               returns: np.ndarray,
                               risk_free_rate: float = 0.07) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio metrics.
    
    Args:
        weights: Portfolio weights
        returns: Return matrix (n_samples, n_assets)
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary of portfolio metrics
    """
    portfolio_returns = returns @ weights
    
    metrics = {
        'mean_return': portfolio_returns.mean() * 252,  # Annualized
        'volatility': portfolio_returns.std() * np.sqrt(252),  # Annualized
        'sharpe_ratio': (portfolio_returns.mean() * 252 - risk_free_rate) / \
                       (portfolio_returns.std() * np.sqrt(252)),
        'max_drawdown': calculate_max_drawdown(portfolio_returns),
        'sortino_ratio': calculate_sortino_ratio(portfolio_returns, risk_free_rate),
        'calmar_ratio': (portfolio_returns.mean() * 252) / \
                       abs(calculate_max_drawdown(portfolio_returns))
    }
    
    return metrics


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float) -> float:
    """Calculate Sortino ratio (downside risk adjusted)."""
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf
    
    downside_std = downside_returns.std() * np.sqrt(252)
    return (returns.mean() * 252 - risk_free_rate) / downside_std
