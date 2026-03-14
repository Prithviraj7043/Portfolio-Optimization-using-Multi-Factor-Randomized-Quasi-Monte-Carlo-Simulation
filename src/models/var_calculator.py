"""
Value at Risk (VaR) calculation module.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class VaRCalculator:
    """Calculate Value at Risk using various methods."""
    
    def __init__(self, confidence_levels: List[float] = [0.90, 0.95, 0.99]):
        """
        Initialize VaR calculator.
        
        Args:
            confidence_levels: List of confidence levels for VaR
        """
        self.confidence_levels = confidence_levels
    
    def historical_var(self, returns: np.ndarray, 
                       confidence_level: float = 0.95) -> float:
        """
        Calculate VaR using historical simulation method.
        
        Args:
            returns: Array of portfolio returns
            confidence_level: Confidence level for VaR
            
        Returns:
            VaR value (negative number representing potential loss)
        """
        alpha = 1 - confidence_level
        var = np.percentile(returns, alpha * 100)
        return var
    
    def parametric_var(self, mean_return: float, std_return: float,
                      confidence_level: float = 0.95) -> float:
        """
        Calculate VaR using parametric (variance-covariance) method.
        
        Args:
            mean_return: Mean portfolio return
            std_return: Standard deviation of portfolio returns
            confidence_level: Confidence level for VaR
            
        Returns:
            VaR value
        """
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(alpha)
        var = mean_return + z_score * std_return
        return var
    
    def monte_carlo_var(self, simulated_returns: np.ndarray,
                       confidence_level: float = 0.95) -> float:
        """
        Calculate VaR using Monte Carlo simulation.
        
        Args:
            simulated_returns: Array of simulated portfolio returns
            confidence_level: Confidence level for VaR
            
        Returns:
            VaR value
        """
        return self.historical_var(simulated_returns, confidence_level)
    
    def conditional_var(self, returns: np.ndarray,
                       confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional VaR (CVaR or Expected Shortfall).
        
        Args:
            returns: Array of portfolio returns
            confidence_level: Confidence level
            
        Returns:
            CVaR value (average loss beyond VaR)
        """
        var = self.historical_var(returns, confidence_level)
        cvar = returns[returns <= var].mean()
        return cvar
    
    def calculate_portfolio_var(self, portfolio_returns: np.ndarray,
                               method: str = 'historical') -> Dict[str, float]:
        """
        Calculate VaR at multiple confidence levels.
        
        Args:
            portfolio_returns: Array of portfolio returns
            method: VaR calculation method ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            Dictionary with VaR values at different confidence levels
        """
        var_results = {}
        
        for cl in self.confidence_levels:
            if method == 'historical':
                var = self.historical_var(portfolio_returns, cl)
            elif method == 'parametric':
                mean_ret = portfolio_returns.mean()
                std_ret = portfolio_returns.std()
                var = self.parametric_var(mean_ret, std_ret, cl)
            elif method == 'monte_carlo':
                var = self.monte_carlo_var(portfolio_returns, cl)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Also calculate CVaR
            cvar = self.conditional_var(portfolio_returns, cl)
            
            var_results[f'VaR_{int(cl*100)}'] = var
            var_results[f'CVaR_{int(cl*100)}'] = cvar
        
        return var_results
    
    def calculate_var_from_paths(self, price_paths: np.ndarray, 
                                 weights: np.ndarray,
                                 initial_value: float = 1.0,
                                 method: str = 'historical') -> Dict[str, float]:
        """
        Calculate VaR from simulated price paths.
        
        Args:
            price_paths: Array of shape (n_paths, T+1, n_stocks)
            weights: Portfolio weights
            initial_value: Initial portfolio value
            method: VaR calculation method
            
        Returns:
            Dictionary with VaR metrics
        """
        # Calculate portfolio values over time
        portfolio_values = (price_paths * weights).sum(axis=2) * initial_value
        
        # Calculate terminal returns
        terminal_returns = (portfolio_values[:, -1] - initial_value) / initial_value
        
        # Calculate VaR
        var_results = self.calculate_portfolio_var(terminal_returns, method)
        
        # Add summary statistics
        var_results['mean_return'] = terminal_returns.mean()
        var_results['std_return'] = terminal_returns.std()
        var_results['min_return'] = terminal_returns.min()
        var_results['max_return'] = terminal_returns.max()
        
        return var_results
    
    def calculate_incremental_var(self, portfolio_returns: np.ndarray,
                                  individual_returns: np.ndarray,
                                  weights: np.ndarray,
                                  confidence_level: float = 0.95) -> np.ndarray:
        """
        Calculate incremental VaR for each asset.
        
        Args:
            portfolio_returns: Portfolio returns
            individual_returns: Array of individual asset returns (n_samples, n_assets)
            weights: Current portfolio weights
            confidence_level: Confidence level
            
        Returns:
            Array of incremental VaR for each asset
        """
        base_var = self.historical_var(portfolio_returns, confidence_level)
        
        incremental_vars = np.zeros(len(weights))
        
        for i in range(len(weights)):
            # Increase weight by small amount
            delta = 0.01
            new_weights = weights.copy()
            new_weights[i] += delta
            new_weights = new_weights / new_weights.sum()  # Renormalize
            
            # Calculate new portfolio returns
            new_portfolio_returns = (individual_returns * new_weights).sum(axis=1)
            new_var = self.historical_var(new_portfolio_returns, confidence_level)
            
            # Incremental VaR
            incremental_vars[i] = (new_var - base_var) / delta
        
        return incremental_vars
    
    def var_summary(self, portfolio_returns: np.ndarray,
                   method: str = 'historical') -> pd.DataFrame:
        """
        Generate comprehensive VaR summary.
        
        Args:
            portfolio_returns: Portfolio returns
            method: VaR calculation method
            
        Returns:
            DataFrame with VaR summary statistics
        """
        results = []
        
        for cl in self.confidence_levels:
            if method == 'historical':
                var = self.historical_var(portfolio_returns, cl)
            elif method == 'parametric':
                mean_ret = portfolio_returns.mean()
                std_ret = portfolio_returns.std()
                var = self.parametric_var(mean_ret, std_ret, cl)
            else:
                var = self.monte_carlo_var(portfolio_returns, cl)
            
            cvar = self.conditional_var(portfolio_returns, cl)
            
            results.append({
                'Confidence_Level': f'{int(cl*100)}%',
                'VaR': var,
                'CVaR': cvar,
                'VaR_Dollar': var * 100,  # Assuming $100 initial investment
                'CVaR_Dollar': cvar * 100
            })
        
        summary_df = pd.DataFrame(results)
        
        logger.info("VaR Summary:")
        logger.info(f"\n{summary_df.to_string()}")
        
        return summary_df
