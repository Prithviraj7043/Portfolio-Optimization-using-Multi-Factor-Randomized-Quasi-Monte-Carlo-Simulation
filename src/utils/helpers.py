"""
Utility functions for portfolio optimization.
"""

import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured logger
    """
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    
    logging.basicConfig(
        level=level,
        format=log_config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Add file handler if configured
    if log_config.get('log_to_file', False):
        log_dir = Path(config['data']['logs_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / 'portfolio_optimization.log')
        file_handler.setFormatter(logging.Formatter(log_config.get('log_format')))
        logging.getLogger().addHandler(file_handler)
    
    return logging.getLogger(__name__)


def create_output_directories(config: Dict[str, Any]) -> None:
    """
    Create output directories if they don't exist.
    
    Args:
        config: Configuration dictionary
    """
    directories = [
        config['data']['output_dir'],
        config['data']['plots_dir'],
        config['data']['logs_dir']
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def save_results(results: Dict[str, Any], filename: str, config: Dict[str, Any]) -> None:
    """
    Save results to file.
    
    Args:
        results: Results dictionary
        filename: Output filename
        config: Configuration dictionary
    """
    output_path = Path(config['data']['output_dir']) / filename
    
    if filename.endswith('.csv'):
        # Convert to DataFrame if needed
        if isinstance(results, dict):
            df = pd.DataFrame(results)
        else:
            df = results
        df.to_csv(output_path, index=False)
    elif filename.endswith('.npy'):
        np.save(output_path, results)
    elif filename.endswith('.npz'):
        np.savez(output_path, **results)
    else:
        # Save as pickle
        pd.to_pickle(results, output_path)
    
    logging.info(f"Results saved to {output_path}")


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate returns from prices.
    
    Args:
        prices: Price DataFrame
        
    Returns:
        Returns DataFrame
    """
    returns = prices.pct_change().dropna()
    return returns


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log returns from prices.
    
    Args:
        prices: Price DataFrame
        
    Returns:
        Log returns DataFrame
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def annualize_metrics(mean_return: float, std_return: float, 
                     periods_per_year: int = 252) -> tuple:
    """
    Annualize return metrics.
    
    Args:
        mean_return: Mean return
        std_return: Standard deviation of returns
        periods_per_year: Number of periods per year
        
    Returns:
        Tuple of (annualized_return, annualized_volatility)
    """
    annualized_return = mean_return * periods_per_year
    annualized_volatility = std_return * np.sqrt(periods_per_year)
    return annualized_return, annualized_volatility


def format_weights(weights: np.ndarray, asset_names: list, 
                   threshold: float = 0.001) -> pd.DataFrame:
    """
    Format portfolio weights for display.
    
    Args:
        weights: Array of portfolio weights
        asset_names: List of asset names
        threshold: Minimum weight to display
        
    Returns:
        DataFrame with formatted weights
    """
    weights_df = pd.DataFrame({
        'Asset': asset_names,
        'Weight': weights,
        'Weight_Pct': weights * 100
    })
    
    # Filter out very small weights
    weights_df = weights_df[weights_df['Weight'] >= threshold]
    weights_df = weights_df.sort_values('Weight', ascending=False)
    
    return weights_df


def print_portfolio_summary(weights: np.ndarray, 
                          expected_return: float,
                          volatility: float,
                          sharpe_ratio: float,
                          var_metrics: Dict[str, float],
                          asset_names: list) -> None:
    """
    Print formatted portfolio summary.
    
    Args:
        weights: Portfolio weights
        expected_return: Expected portfolio return
        volatility: Portfolio volatility
        sharpe_ratio: Sharpe ratio
        var_metrics: VaR metrics dictionary
        asset_names: List of asset names
    """
    print("\n" + "="*60)
    print("PORTFOLIO OPTIMIZATION SUMMARY")
    print("="*60)
    
    print("\nPortfolio Metrics:")
    print(f"  Expected Annual Return: {expected_return:.2%}")
    print(f"  Annual Volatility:      {volatility:.2%}")
    print(f"  Sharpe Ratio:           {sharpe_ratio:.4f}")
    
    print("\nValue at Risk:")
    for key, value in var_metrics.items():
        if 'VaR' in key or 'CVaR' in key:
            print(f"  {key}: {value:.2%}")
    
    print("\nTop 10 Portfolio Allocations:")
    weights_df = format_weights(weights, asset_names)
    print(weights_df.head(10).to_string(index=False))
    
    print("\n" + "="*60 + "\n")


def estimate_correlation_with_factors(stock_returns: pd.DataFrame,
                                      factor_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate correlation between stocks and external factors.
    
    Args:
        stock_returns: Stock returns DataFrame
        factor_returns: Factor returns DataFrame
        
    Returns:
        Correlation matrix
    """
    # Align data
    common_dates = stock_returns.index.intersection(factor_returns.index)
    stock_aligned = stock_returns.loc[common_dates]
    factor_aligned = factor_returns.loc[common_dates]
    
    # Calculate correlations
    correlations = pd.DataFrame(
        index=stock_returns.columns,
        columns=factor_returns.columns
    )
    
    for stock in stock_returns.columns:
        for factor in factor_returns.columns:
            corr = stock_aligned[stock].corr(factor_aligned[factor])
            correlations.loc[stock, factor] = corr
    
    return correlations.astype(float)


def plot_style_setup(config: Dict[str, Any]) -> None:
    """
    Setup plotting style.
    
    Args:
        config: Configuration dictionary
    """
    plot_config = config.get('plotting', {})
    
    # Set style
    style = plot_config.get('style', 'seaborn-v0_8')
    try:
        plt.style.use(style)
    except:
        plt.style.use('seaborn-v0_8-darkgrid')
    
    # Set default figure size
    figsize = plot_config.get('figsize', [12, 8])
    plt.rcParams['figure.figsize'] = figsize
    
    # Set DPI
    dpi = plot_config.get('dpi', 300)
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    
    # Other aesthetics
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
