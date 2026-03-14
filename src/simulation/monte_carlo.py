"""
Monte Carlo simulation engine with proper event handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.stats import qmc
import logging
from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EventSimulator:
    """Handle random events (e.g., RBI repo rate changes) with proper detection."""
    
    def __init__(self, event_data: pd.DataFrame, 
                 event_col: str = 'RBI_Repo_Change'):
        """
        Initialize event simulator.
        
        Args:
            event_data: DataFrame with historical event data
            event_col: Column name for event changes
        """
        self.event_data = event_data
        self.event_col = event_col
        
        # Check if event column exists
        if event_col not in event_data.columns:
            logger.warning(f"Event column '{event_col}' not found in data.")
            logger.warning(f"Available columns: {list(event_data.columns)}")
            self.events_available = False
            self._set_no_events()
            return
        
        # Extract the event column and clean it
        event_series = pd.to_numeric(event_data[event_col], errors='coerce')
        event_series = event_series.dropna()
        
        # Find non-zero events (actual rate changes)
        non_zero_events = event_series[event_series.abs() > 1e-8]  # Small threshold for floating point
        
        if len(non_zero_events) == 0:
            logger.warning(f"No non-zero events found in '{event_col}'.")
            logger.warning("This means RBI repo rate was constant in the data period.")
            logger.warning("Event simulation will be disabled.")
            self.events_available = False
            self._set_no_events()
            return
        
        # Events found - initialize properly
        self.events_available = True
        self.event_dates = non_zero_events.index
        self.event_values = non_zero_events.values
        
        # Calculate statistics
        total_days = len(event_data)
        n_events = len(self.event_dates)
        self.event_probability = n_events / total_days
        self.event_mean = self.event_values.mean()
        self.event_std = self.event_values.std()
        
        logger.info(f"Event simulator initialized: {n_events} events over {total_days} days")
        logger.info(f"Event probability: {self.event_probability:.6f} ({self.event_probability*100:.2f}%)")
        logger.info(f"Event magnitude - mean: {self.event_mean:.6f}, std: {self.event_std:.6f}")
        logger.info(f"Event range: [{self.event_values.min():.6f}, {self.event_values.max():.6f}]")
        
        # Show sample events
        logger.info(f"Sample events detected:")
        for i, (date, value) in enumerate(zip(self.event_dates[:5], self.event_values[:5])):
            logger.info(f"  {date.date() if hasattr(date, 'date') else date}: {value:+.6f} ({value*100:+.2f}%)")
        
        if n_events > 5:
            logger.info(f"  ... and {n_events - 5} more events")
    
    def _set_no_events(self):
        """Set default values when no events available."""
        self.event_dates = pd.DatetimeIndex([])
        self.event_values = np.array([])
        self.event_probability = 0.0
        self.event_mean = 0.0
        self.event_std = 0.0
    
    def has_events(self) -> bool:
        """Check if events are available."""
        return self.events_available
    
    def generate_events(self, T: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random events for simulation period.
        
        Args:
            T: Number of time steps
            random_state: Random seed
            
        Returns:
            Array of shape (T,) with event magnitudes (0 for no event)
        """
        if not self.events_available:
            return np.zeros(T)
        
        np.random.seed(random_state)
        
        # Determine which days have events (Bernoulli process)
        event_occurs = np.random.random(T) < self.event_probability
        
        # Generate event magnitudes
        events = np.zeros(T)
        n_events = event_occurs.sum()
        
        if n_events > 0:
            # Sample from historical event distribution with replacement
            event_magnitudes = np.random.choice(self.event_values, size=n_events, replace=True)
            events[event_occurs] = event_magnitudes
        
        return events
    
    def apply_event_impact(self, prices: np.ndarray, events: np.ndarray,
                          sensitivity: np.ndarray) -> np.ndarray:
        """
        Apply event impacts to price paths.
        
        Args:
            prices: Price paths of shape (n_paths, T+1, n_stocks)
            events: Event array of shape (T,)
            sensitivity: Stock sensitivity to events (n_stocks,)
            
        Returns:
            Adjusted price paths
        """
        if not self.events_available:
            return prices
        
        n_paths, T_plus_1, n_stocks = prices.shape
        T = T_plus_1 - 1
        
        adjusted_prices = prices.copy()
        
        for t in range(T):
            if abs(events[t]) > 1e-8:  # Event occurred
                # Calculate impact: shock proportional to event and sensitivity
                # Negative relationship: rate cut → positive for stocks
                shock = -events[t] * sensitivity
                
                # Multiplicative impact
                impact_factor = 1 + shock
                
                # Clip to prevent unrealistic moves
                impact_factor = np.clip(impact_factor, 0.7, 1.3)
                
                # Apply to all future time steps (persistent impact)
                adjusted_prices[:, t+1:, :] *= impact_factor
        
        return adjusted_prices


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for portfolio optimization.
    Supports quasi-Monte Carlo (Sobol sequences) and parallel processing.
    """
    
    def __init__(self, gbm_model, event_simulator: Optional[EventSimulator] = None):
        """
        Initialize simulator.
        
        Args:
            gbm_model: Geometric Brownian Motion model
            event_simulator: Event simulator for random shocks
        """
        self.gbm_model = gbm_model
        self.event_simulator = event_simulator
    
    def run_single_simulation(self, S0: np.ndarray, T: int, n_paths: int,
                             use_sobol: bool = True,
                             include_events: bool = True,
                             random_state: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Run a single simulation replication.
        
        Args:
            S0: Initial stock prices
            T: Number of time steps
            n_paths: Number of simulation paths
            use_sobol: Use Sobol sequences for quasi-Monte Carlo
            include_events: Include random events
            random_state: Random seed
            
        Returns:
            Dictionary with simulation results
        """
        # Generate price paths
        paths = self.gbm_model.simulate_paths(
            S0=S0,
            T=T,
            n_paths=n_paths,
            use_sobol=use_sobol,
            random_state=random_state
        )
        
        # Apply events if enabled and available
        if include_events and self.event_simulator is not None:
            if self.event_simulator.has_events():
                # Generate events
                events = self.event_simulator.generate_events(T, random_state)
                
                # Estimate stock sensitivity to events
                # Simple approach: small random sensitivity
                n_stocks = len(S0)
                sensitivity = np.random.uniform(0.01, 0.1, n_stocks)
                
                # Apply event impacts
                paths = self.event_simulator.apply_event_impact(paths, events, sensitivity)
        
        # Calculate returns
        returns = self.gbm_model.calculate_returns(paths)
        
        return {
            'paths': paths,
            'returns': returns,
            'final_prices': paths[:, -1, :]
        }
    
    def run_multiple_replications(self, S0: np.ndarray, T: int, 
                                 n_paths: int, n_replications: int,
                                 use_sobol: bool = True,
                                 include_events: bool = True,
                                 parallel: bool = True,
                                 n_jobs: int = -1,
                                 random_state: Optional[int] = None) -> List[Dict[str, np.ndarray]]:
        """
        Run multiple independent simulation replications.
        
        Args:
            S0: Initial stock prices
            T: Number of time steps
            n_paths: Number of paths per replication
            n_replications: Number of independent replications
            use_sobol: Use Sobol sequences
            include_events: Include random events
            parallel: Use parallel processing
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Base random seed
            
        Returns:
            List of simulation results for each replication
        """
        logger.info(f"Running {n_replications} replications with {n_paths} paths each")
        
        # Generate independent seeds for each replication
        if random_state is not None:
            np.random.seed(random_state)
        seeds = np.random.randint(0, 2**31, n_replications)
        
        if parallel and n_replications > 1:
            logger.info(f"Using parallel processing with {n_jobs} jobs")
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.run_single_simulation)(
                    S0, T, n_paths, use_sobol, include_events, seed
                ) for seed in tqdm(seeds, desc="Replications")
            )
        else:
            results = []
            for seed in tqdm(seeds, desc="Replications"):
                result = self.run_single_simulation(
                    S0, T, n_paths, use_sobol, include_events, seed
                )
                results.append(result)
        
        logger.info(f"Completed {n_replications} replications")
        
        return results
    
    def aggregate_replications(self, replication_results: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Aggregate results from multiple replications.
        
        Args:
            replication_results: List of simulation results
            
        Returns:
            Aggregated results
        """
        logger.info("Aggregating replication results")
        
        # Combine all paths
        all_paths = np.concatenate([r['paths'] for r in replication_results], axis=0)
        all_returns = np.concatenate([r['returns'] for r in replication_results], axis=0)
        all_final_prices = np.concatenate([r['final_prices'] for r in replication_results], axis=0)
        
        # Calculate statistics across replications
        final_prices_by_rep = np.array([r['final_prices'].mean(axis=0) 
                                        for r in replication_results])
        
        return {
            'paths': all_paths,
            'returns': all_returns,
            'final_prices': all_final_prices,
            'mean_final_prices': all_final_prices.mean(axis=0),
            'std_final_prices': all_final_prices.std(axis=0),
            'final_prices_by_replication': final_prices_by_rep,
            'n_total_paths': len(all_paths),
            'n_replications': len(replication_results)
        }
    
    def calculate_convergence_metrics(self, replication_results: List[Dict[str, np.ndarray]],
                                     weights: np.ndarray) -> pd.DataFrame:
        """
        Calculate convergence metrics across replications.
        
        Args:
            replication_results: List of simulation results
            weights: Portfolio weights
            
        Returns:
            DataFrame with convergence metrics
        """
        n_replications = len(replication_results)
        
        metrics = []
        cumulative_returns = []
        
        for i, result in enumerate(replication_results):
            # Calculate portfolio returns for this replication
            portfolio_values = (result['final_prices'] * weights).sum(axis=1)
            portfolio_return = portfolio_values.mean()
            
            cumulative_returns.append(portfolio_return)
            
            # Running statistics
            running_mean = np.mean(cumulative_returns)
            running_std = np.std(cumulative_returns)
            
            metrics.append({
                'replication': i + 1,
                'portfolio_return': portfolio_return,
                'running_mean': running_mean,
                'running_std': running_std,
                'std_error': running_std / np.sqrt(i + 1)
            })
        
        convergence_df = pd.DataFrame(metrics)
        
        return convergence_df