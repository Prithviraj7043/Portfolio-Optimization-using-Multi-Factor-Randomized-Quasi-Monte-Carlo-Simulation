# Quick Start Guide

## Installation

1. **Clone or download the project**

2. **Install dependencies**:
```bash
cd portfolio_optimization
pip install -r requirements.txt
```

## Running the Full Pipeline

### Option 1: Command Line
```bash
python main.py
```

This will:
- Load and process data
- Estimate GBM parameters
- Run Monte Carlo simulations (100 replications × 10,000 paths each)
- Optimize portfolio weights
- Calculate VaR metrics
- Generate visualizations
- Save results to `outputs/` directory

### Option 2: Jupyter Notebook
```bash
jupyter notebook notebooks/portfolio_analysis.ipynb
```

The notebook provides interactive analysis with visualization.

## Configuration

Edit `config/config.yaml` to customize:

### Simulation Parameters
```yaml
simulation:
  num_simulations: 10000     # Paths per replication
  num_replications: 100      # Number of replications
  time_horizon_days: 252     # 1 year of trading days
  use_sobol: true           # Quasi-Monte Carlo
  parallel: true            # Parallel processing
```

### Optimization Constraints
```yaml
optimization:
  constraints:
    min_weight: 0.0         # No shorting
    max_weight: 0.4         # Max 40% per stock
```

### VaR Settings
```yaml
var:
  confidence_levels: [0.90, 0.95, 0.99]
  method: "historical"
```

## Understanding the Output

### Results Directory Structure
```
outputs/
├── results/
│   ├── optimal_portfolio.npz      # Optimal weights and metrics
│   ├── efficient_frontier.csv     # Efficient frontier points
│   ├── var_summary.csv           # VaR analysis
│   └── convergence_metrics.csv   # Simulation convergence
├── plots/
│   ├── efficient_frontier.png
│   ├── portfolio_weights.png
│   ├── price_paths.png
│   ├── return_distribution.png
│   ├── var_analysis.png
│   ├── convergence.png
│   └── correlation_matrix.png
└── logs/
    └── portfolio_optimization.log
```

### Key Metrics

**Optimal Portfolio Metrics**:
- **Expected Annual Return**: Portfolio's expected return (annualized)
- **Annual Volatility**: Portfolio risk (standard deviation)
- **Sharpe Ratio**: Risk-adjusted return metric
- **VaR (90%, 95%, 99%)**: Potential losses at different confidence levels
- **CVaR**: Expected loss beyond VaR threshold

## Example Usage Patterns

### 1. Basic Optimization
```python
from src.data_processing import DataLoader
from src.models import GeometricBrownianMotion
from src.optimization import PortfolioOptimizer

# Load data
loader = DataLoader('data/raw/Book1__1_.xlsx')
data = loader.load_all_data()

# Calculate returns
returns = data['stock_prices'].pct_change().dropna()

# Estimate parameters
gbm = GeometricBrownianMotion(returns)
params = gbm.estimate_parameters()

# Optimize
optimizer = PortfolioOptimizer(
    expected_returns=params['mu'],
    covariance_matrix=params['correlation'] * params['sigma'][:, np.newaxis] * params['sigma']
)
optimal = optimizer.optimize_max_sharpe()
```

### 2. Custom Simulation
```python
from src.simulation import MonteCarloSimulator

simulator = MonteCarloSimulator(gbm)

result = simulator.run_single_simulation(
    S0=current_prices,
    T=252,
    n_paths=5000,
    use_sobol=True
)
```

### 3. VaR Analysis
```python
from src.models import VaRCalculator

var_calc = VaRCalculator([0.95, 0.99])
var_results = var_calc.calculate_var_from_paths(
    price_paths=result['paths'],
    weights=optimal_weights,
    initial_value=100000  # $100k portfolio
)
```

## Advanced Features

### Including External Factors
The system automatically incorporates external market indicators:
- NIFTY50 returns
- S&P500 returns
- USDINR exchange rate
- Brent Crude oil prices
- FII/DII activity
- India 10-year bond yields
- India VIX
- US Dollar Index

### Random Events (RBI Repo Rate)
Events are automatically included based on historical frequency and magnitude.

### Multiple Optimization Methods
```python
# Maximum Sharpe Ratio
max_sharpe = optimizer.optimize_max_sharpe()

# Minimum Variance
min_var = optimizer.optimize_mean_variance(target_return=None)

# Risk Parity
risk_parity = optimizer.optimize_risk_parity()

# Efficient Frontier
frontier = optimizer.efficient_frontier(n_points=50)
```

## Performance Tips

1. **Parallel Processing**: Enable `parallel: true` for faster simulations
2. **Sobol Sequences**: Use `use_sobol: true` for better convergence
3. **Adjust Replications**: More replications = more stable results but longer runtime
4. **Sample Size**: 10,000 paths per replication is a good balance

## Troubleshooting

### Memory Issues
Reduce `num_simulations` or `num_replications` in config.

### Optimization Fails
- Check for NaN values in data
- Relax constraints (increase `max_weight`)
- Verify covariance matrix is positive semi-definite

### Slow Performance
- Enable parallel processing
- Reduce number of replications for initial testing
- Use fewer frontier points

## Testing

Run unit tests:
```bash
python -m pytest tests/
```

Or:
```bash
python -m unittest discover tests/
```

## Next Steps

1. Review the generated plots in `outputs/plots/`
2. Examine optimal weights in `outputs/results/optimal_portfolio.npz`
3. Analyze VaR metrics for risk assessment
4. Adjust configuration and re-run for different scenarios
5. Use the Jupyter notebook for interactive exploration

## Support

For issues or questions:
1. Check the log file: `outputs/logs/portfolio_optimization.log`
2. Review the configuration: `config/config.yaml`
3. Examine the data structure in the Excel file
