"""
Geometric Brownian Motion model - numerically stable version.
Handles edge cases and prevents NaN/Inf values in simulations.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class GeometricBrownianMotion:
    """GBM model for stock price simulation with numerical stability."""
    
    def __init__(self, stock_returns: pd.DataFrame, 
                 external_factors: Optional[pd.DataFrame] = None,
                 estimation_window: int = 252):
        """Initialize GBM model."""
        self.stock_returns = stock_returns
        self.external_factors = external_factors
        self.estimation_window = estimation_window
        
        self.mu = None
        self.sigma = None
        self.correlation_matrix = None
        self.factor_loadings = None
        
    def estimate_parameters(self) -> Dict[str, np.ndarray]:
        """Estimate GBM parameters from historical data."""
        logger.info("Estimating GBM parameters")
        
        # Use last estimation_window days
        recent_returns = self.stock_returns.tail(self.estimation_window)
        
        # Calculate drift - annualized mean return
        self.mu = recent_returns.mean().values * 252
        
        # Calculate volatility - annualized standard deviation
        self.sigma = recent_returns.std().values * np.sqrt(252)
        
        # Ensure no zero volatility
        self.sigma = np.maximum(self.sigma, 0.01)  # Minimum 1% volatility
        
        # Calculate correlation matrix
        corr_matrix = recent_returns.corr().values
        
        # Ensure correlation matrix is positive semi-definite
        self.correlation_matrix = self._fix_correlation_matrix(corr_matrix)
        
        # Estimate factor loadings if external factors provided
        if self.external_factors is not None:
            self.factor_loadings = self._estimate_factor_loadings(recent_returns)
        
        logger.info(f"Estimated parameters for {len(self.mu)} stocks")
        logger.info(f"Mean annual return: {self.mu.mean():.4f}")
        logger.info(f"Mean annual volatility: {self.sigma.mean():.4f}")
        
        return {
            'mu': self.mu,
            'sigma': self.sigma,
            'correlation': self.correlation_matrix,
            'factor_loadings': self.factor_loadings
        }
    
    def _fix_correlation_matrix(self, corr: np.ndarray) -> np.ndarray:
        """Fix correlation matrix to be positive semi-definite."""
        # Replace NaN and inf
        corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure diagonal is 1
        np.fill_diagonal(corr, 1.0)
        
        # Make symmetric
        corr = (corr + corr.T) / 2
        
        # Check if positive semi-definite
        eigenvalues = np.linalg.eigvalsh(corr)
        
        if eigenvalues.min() < 0:
            logger.warning("Correlation matrix not positive semi-definite. Fixing...")
            # Add small value to diagonal to make it positive definite
            epsilon = abs(eigenvalues.min()) + 0.01
            corr = corr + np.eye(len(corr)) * epsilon
            # Rescale to keep correlations in [-1, 1]
            scaling = np.sqrt(np.diag(corr))
            corr = corr / np.outer(scaling, scaling)
            np.fill_diagonal(corr, 1.0)
        
        return corr
    
    def _estimate_factor_loadings(self, returns: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Estimate factor loadings using regression."""
        from sklearn.linear_model import LinearRegression
        
        factor_loadings = {}
        
        factor_cols = [col for col in self.external_factors.columns 
                      if 'Returns' in col or 'Change' in col]
        
        factors = self.external_factors[factor_cols].tail(self.estimation_window)
        
        common_dates = returns.index.intersection(factors.index)
        returns_aligned = returns.loc[common_dates]
        factors_aligned = factors.loc[common_dates].fillna(0)
        
        for stock in returns.columns:
            y = returns_aligned[stock].values.reshape(-1, 1)
            X = factors_aligned.values
            
            mask = ~np.isnan(y.flatten()) & ~np.isnan(X).any(axis=1) & ~np.isinf(y.flatten())
            if mask.sum() < 30:
                factor_loadings[stock] = np.zeros(len(factor_cols))
                continue
            
            y_clean = y[mask]
            X_clean = X[mask]
            
            try:
                model = LinearRegression()
                model.fit(X_clean, y_clean)
                factor_loadings[stock] = model.coef_.flatten()
            except:
                factor_loadings[stock] = np.zeros(len(factor_cols))
        
        return factor_loadings
    
    def simulate_paths(self, S0: np.ndarray, T: int, n_paths: int,
                      use_sobol: bool = True, 
                      factor_shocks: Optional[np.ndarray] = None,
                      random_state: Optional[int] = None) -> np.ndarray:
        """Simulate stock price paths using GBM with numerical stability."""
        if self.mu is None or self.sigma is None:
            raise ValueError("Parameters not estimated. Call estimate_parameters() first.")
        
        n_stocks = len(S0)
        dt = 1/252  # Daily time step
        
        # Ensure S0 has no zeros or negatives
        S0_safe = np.maximum(S0, 1e-6)
        
        # Generate random shocks
        if use_sobol:
            shocks = self._generate_sobol_shocks(n_paths, T, n_stocks, random_state)
        else:
            np.random.seed(random_state)
            shocks = np.random.randn(n_paths, T, n_stocks)
        
        # Apply correlation structure
        try:
            L = np.linalg.cholesky(self.correlation_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Cholesky decomposition failed. Using identity correlation.")
            L = np.eye(n_stocks)
        
        correlated_shocks = np.zeros_like(shocks)
        for i in range(n_paths):
            for t in range(T):
                correlated_shocks[i, t, :] = L @ shocks[i, t, :]
        
        # Add factor impact if available
        if factor_shocks is not None and self.factor_loadings is not None:
            factor_impact = self._apply_factor_shocks(factor_shocks, n_paths, T)
            correlated_shocks += factor_impact
        
        # Simulate paths with numerical stability
        paths = np.zeros((n_paths, T+1, n_stocks))
        paths[:, 0, :] = S0_safe
        
        for t in range(T):
            drift = (self.mu - 0.5 * self.sigma**2) * dt
            diffusion = self.sigma * np.sqrt(dt) * correlated_shocks[:, t, :]
            
            # Prevent extreme movements
            diffusion = np.clip(diffusion, -3, 3)  # Limit to ±3 sigma moves
            
            # Update paths
            paths[:, t+1, :] = paths[:, t, :] * np.exp(drift + diffusion)
            
            # Prevent numerical issues
            paths[:, t+1, :] = np.maximum(paths[:, t+1, :], 1e-6)  # No zeros
            paths[:, t+1, :] = np.minimum(paths[:, t+1, :], S0_safe * 100)  # Cap at 100x initial
        
        # Final check for NaN/Inf
        if np.any(np.isnan(paths)) or np.any(np.isinf(paths)):
            logger.warning("NaN or Inf values detected in paths. Replacing with valid values.")
            paths = np.nan_to_num(paths, nan=S0_safe, posinf=S0_safe*10, neginf=S0_safe*0.1)
        
        return paths
    
    def _generate_sobol_shocks(self, n_paths: int, T: int, n_stocks: int,
                               random_state: Optional[int] = None) -> np.ndarray:
        """Generate quasi-random shocks using Sobol sequences."""
        from scipy.stats import qmc
        
        dimensions = T * n_stocks
        
        sampler = qmc.Sobol(d=dimensions, scramble=True, seed=random_state)
        
        n_samples = int(np.ceil(np.log2(n_paths)))
        n_samples = 2**n_samples
        
        sobol_samples = sampler.random(n=n_samples)[:n_paths]
        
        # Transform to normal with clipping
        normal_samples = stats.norm.ppf(np.clip(sobol_samples, 0.001, 0.999))
        
        shocks = normal_samples.reshape(n_paths, T, n_stocks)
        
        return shocks
    
    def _apply_factor_shocks(self, factor_shocks: np.ndarray, 
                            n_paths: int, T: int) -> np.ndarray:
        """Apply factor shocks to simulation."""
        n_stocks = len(self.mu)
        factor_impact = np.zeros((n_paths, T, n_stocks))
        
        for i, stock in enumerate(self.stock_returns.columns):
            if stock in self.factor_loadings:
                loadings = self.factor_loadings[stock]
                for t in range(T):
                    impact = np.dot(loadings, factor_shocks[t, :])
                    factor_impact[:, t, i] = np.clip(impact, -0.5, 0.5)  # Limit factor impact
        
        return factor_impact
    
    def calculate_returns(self, paths: np.ndarray) -> np.ndarray:
        """Calculate returns from price paths with numerical stability."""
        # Ensure no zeros
        paths_safe = np.maximum(paths, 1e-10)
        
        # Calculate log returns
        returns = np.diff(np.log(paths_safe), axis=1)
        
        # Replace any remaining NaN/Inf
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.5, neginf=-0.5)
        
        return returns