"""
Main script - works with both standard and per-security data loaders.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import load_config, setup_logging, create_output_directories, save_results

logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    
    config = load_config('config/config.yaml')
    setup_logging(config)
    create_output_directories(config)
    
    logger.info("="*80)
    logger.info("PORTFOLIO OPTIMIZATION WITH MONTE CARLO SIMULATION")
    logger.info("="*80)
    
    # ====================================================================
    # 1. Load Data
    # ====================================================================
    logger.info("\n1. Loading data...")
    
    from data_processing.data_loader_per_security import DataLoader
    
    data_loader = DataLoader(
        excel_path=config['data']['excel_file'],
        clean_data_sheet=config['data']['clean_data_sheet'],
        stock_data_sheet=config['data']['stock_data_sheet']
    )
    
    data = data_loader.load_all_data()
    
    # Handle different data loader return structures
    if 'stock_prices' in data:
        # Standard data loader
        stock_prices = data['stock_prices']
        indicators = data['indicators']
    elif 'per_security_data' in data:
        # Per-security data loader
        per_security_data = data['per_security_data']
        stock_names = data['stock_names']
        
        # Build combined stock prices DataFrame from per-security data
        # Use the common date range (intersection of all securities)
        common_start = max(d['first_date'] for d in per_security_data.values())
        common_end = min(d['last_date'] for d in per_security_data.values())
        
        logger.info(f"Using common date range: {common_start.date()} to {common_end.date()}")
        
        stock_prices_dict = {}
        for name in stock_names:
            prices = per_security_data[name]['prices']
            stock_prices_dict[name] = prices.loc[common_start:common_end]
        
        stock_prices = pd.DataFrame(stock_prices_dict)
        
        # Use indicators from first security (they should all have same indicators for common dates)
        first_stock = stock_names[0]
        indicators = per_security_data[first_stock]['indicators'].loc[common_start:common_end]
    else:
        raise ValueError(f"Unknown data structure returned: {data.keys()}")
    
    stock_names = stock_prices.columns.tolist()
    
    logger.info(f"Loaded {len(stock_prices)} days of data")
    logger.info(f"Number of stocks: {len(stock_names)}")
    logger.info(f"Date range: {stock_prices.index[0]} to {stock_prices.index[-1]}")
    
    # Check RBI events
    if 'RBI_Repo_Change' in indicators.columns:
        rbi_events = indicators['RBI_Repo_Change'][indicators['RBI_Repo_Change'].abs() > 1e-8]
        logger.info(f"\nRBI Repo Events detected: {len(rbi_events)}")
        if len(rbi_events) > 0:
            logger.info(f"  Sample dates: {list(rbi_events.head(5).index.date)}")
            logger.info(f"  Sample magnitudes: {rbi_events.head(5).values}")
    
    # Calculate returns
    stock_returns = np.log(stock_prices / stock_prices.shift(1)).dropna()
    
    # ====================================================================
    # 2. Initialize Models
    # ====================================================================
    logger.info("\n2. Initializing models...")
    
    from models.gbm_fixed import GeometricBrownianMotion
    from simulation.monte_carlo import MonteCarloSimulator, EventSimulator
    
    # GBM
    gbm = GeometricBrownianMotion(
        stock_returns=stock_returns,
        external_factors=indicators if config['gbm']['include_external_factors'] else None,
        estimation_window=config['gbm']['estimation_window']
    )
    
    params = gbm.estimate_parameters()
    
    logger.info(f"GBM parameters:")
    logger.info(f"  Mean drift: {params['mu'].mean():.4f}")
    logger.info(f"  Mean volatility: {params['sigma'].mean():.4f}")
    logger.info(f"  Volatility range: [{params['sigma'].min():.4f}, {params['sigma'].max():.4f}]")
    
    # Event Simulator
    event_simulator = None
    if config['events']['include_repo_rate_events']:
        event_simulator = EventSimulator(
            event_data=indicators,
            event_col=config['events']['repo_change_column']
        )
    
    # ====================================================================
    # 3. Run Monte Carlo Simulation
    # ====================================================================
    logger.info("\n3. Running Monte Carlo simulation...")
    
    simulator = MonteCarloSimulator(
        gbm_model=gbm,
        event_simulator=event_simulator
    )
    
    # Initial prices
    S0 = stock_prices.iloc[-1].values
    
    # Simulation parameters
    T = config['simulation']['time_horizon_days']
    n_paths = config['simulation']['num_simulations']
    n_replications = config['simulation']['num_replications']
    use_sobol = config['simulation']['use_sobol']
    parallel = config['simulation']['parallel']
    n_jobs = config['simulation']['num_cores']
    random_state = config['simulation']['seed']
    
    logger.info(f"Parameters: {n_paths} paths × {n_replications} reps = {n_paths * n_replications} total")
    
    # Run replications
    replication_results = simulator.run_multiple_replications(
        S0=S0,
        T=T,
        n_paths=n_paths,
        n_replications=n_replications,
        use_sobol=use_sobol,
        include_events=config['events']['include_repo_rate_events'],
        parallel=parallel,
        n_jobs=n_jobs,
        random_state=random_state
    )
    
    # Aggregate results
    aggregated_results = simulator.aggregate_replications(replication_results)
    
    logger.info(f"Total simulation paths: {aggregated_results['n_total_paths']}")
    
    # ====================================================================
    # 4. Portfolio Optimization
    # ====================================================================
    logger.info("\n4. Optimizing portfolio...")
    
    from optimization.portfolio_optimizer import PortfolioOptimizer
    
    # Calculate expected returns and covariance from simulations
    simulated_returns = aggregated_results['returns']
    path_avg_returns = simulated_returns.mean(axis=1)
    
    expected_returns = path_avg_returns.mean(axis=0) * 252  # Annualize
    covariance_matrix = np.cov(path_avg_returns.T) * 252    # Annualize
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=config['market']['risk_free_rate']
    )
    
    # Optimize for maximum Sharpe ratio
    optimal_portfolio = optimizer.optimize_max_sharpe(
        min_weight=config['optimization']['constraints']['min_weight'],
        max_weight=config['optimization']['constraints']['max_weight']
    )
    
    if optimal_portfolio is None:
        logger.error("Optimization failed!")
        return
    
    logger.info("Optimal portfolio found!")
    
    optimal_weights = optimal_portfolio['weights']
    
    logger.info(f"\nExpected return: {optimal_portfolio['return']*100:.2f}%")
    logger.info(f"Volatility: {optimal_portfolio['volatility']*100:.2f}%")
    logger.info(f"Sharpe ratio: {optimal_portfolio['sharpe_ratio']:.4f}")
    
    logger.info("\nTop allocations:")
    sorted_idx = np.argsort(optimal_weights)[::-1]
    for i in sorted_idx[:10]:
        if optimal_weights[i] > 0.001:
            logger.info(f"  {stock_names[i]:20s}: {optimal_weights[i]*100:6.2f}%")
    
    # Calculate efficient frontier
    logger.info("\nCalculating efficient frontier...")
    frontier_df = optimizer.efficient_frontier(
        n_points=config['optimization']['num_frontier_points'],
        min_weight=config['optimization']['constraints']['min_weight'],
        max_weight=config['optimization']['constraints']['max_weight']
    )
    
    # ====================================================================
    # 5. Calculate VaR
    # ====================================================================
    logger.info("\n5. Calculating Value at Risk...")
    
    from models.var_calculator import VaRCalculator
    
    var_calculator = VaRCalculator(
        confidence_levels=config['var']['confidence_levels']
    )
    
    var_results = var_calculator.calculate_var_from_paths(
        price_paths=aggregated_results['paths'],
        weights=optimal_weights,
        initial_value=1.0,
        method=config['var']['method']
    )
    
    logger.info(f"VaR 95%: {var_results.get('VaR_95', 0)*100:.2f}%")
    logger.info(f"VaR 99%: {var_results.get('VaR_99', 0)*100:.2f}%")
    
    # ====================================================================
    # 6. Generate Visualizations
    # ====================================================================
    logger.info("\n6. Creating visualizations...")
    
    try:
        from utils.visualization_complete import EnhancedPortfolioVisualizer
        visualizer = EnhancedPortfolioVisualizer(config)
        
        # Individual stock plots
        logger.info("Creating individual stock plots...")
        visualizer.plot_all_individual_stocks(
            paths=aggregated_results['paths'][:5000],
            asset_names=stock_names,
            S0=S0
        )
        
        # Portfolio evolution
        logger.info("Creating portfolio evolution plot...")
        visualizer.plot_optimized_portfolio_evolution(
            paths=aggregated_results['paths'][:5000],
            weights=optimal_weights,
            asset_names=stock_names,
            initial_value=100000
        )
        
        # Standard plots
        logger.info("Creating standard plots...")
        visualizer.plot_efficient_frontier(frontier_df, optimal_portfolio)
        visualizer.plot_portfolio_weights(optimal_weights, stock_names)
        
        portfolio_returns = (aggregated_results['returns'] * optimal_weights).sum(axis=2)
        final_returns = portfolio_returns[:, -1]
        visualizer.plot_return_distribution(final_returns)
        visualizer.plot_var_analysis(var_results)
        
        convergence_df = simulator.calculate_convergence_metrics(replication_results, optimal_weights)
        visualizer.plot_convergence(convergence_df)
        visualizer.plot_correlation_matrix(params['correlation'], stock_names)
        
        logger.info("[OK] All visualizations created")
        
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        import traceback
        traceback.print_exc()
    
    # ====================================================================
    # 7. Save Results
    # ====================================================================
    logger.info("\n7. Saving results...")
    
    results_dict = {
        'optimal_weights': optimal_weights,
        'stock_names': stock_names,
        'expected_return': optimal_portfolio['return'],
        'volatility': optimal_portfolio['volatility'],
        'sharpe_ratio': optimal_portfolio['sharpe_ratio'],
        'var_results': var_results
    }
    
    save_results(results_dict, 'optimal_portfolio.npz', config)
    save_results(frontier_df, 'efficient_frontier.csv', config)
    save_results(convergence_df, 'convergence_metrics.csv', config)
    
    # Save weights as CSV
    weights_df = pd.DataFrame({
        'Stock': stock_names,
        'Weight_%': optimal_weights * 100,
        'Expected_Return_%': expected_returns * 100,
        'Volatility_%': np.sqrt(np.diag(covariance_matrix)) * 100
    }).sort_values('Weight_%', ascending=False)
    
    save_results(weights_df, 'optimal_weights.csv', config)
    
    logger.info("\n" + "="*80)
    logger.info("PORTFOLIO OPTIMIZATION COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"Results saved to: {config['data']['output_dir']}")
    logger.info(f"Plots saved to: {config['data']['plots_dir']}")
    logger.info("="*80)


if __name__ == "__main__":
    main()