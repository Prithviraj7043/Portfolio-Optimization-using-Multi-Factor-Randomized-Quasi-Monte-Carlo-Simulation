# Portfolio Optimization Project - Complete Summary

## Project Overview

This is a comprehensive portfolio optimization system using **Quasi-Monte Carlo simulation** with **Geometric Brownian Motion (GBM)** to simulate stock prices and optimize portfolio weights. The system incorporates multiple external market indicators and random events (RBI repo rate changes) to create realistic simulations.

## Key Features

### 1. Quasi-Monte Carlo Simulation
- Uses **Sobol sequences** for better convergence than traditional Monte Carlo
- Supports multiple independent replications for statistical robustness
- Parallel processing for faster computation
- Default: 100 replications × 10,000 paths = 1,000,000 total simulations

### 2. Geometric Brownian Motion (GBM)
- Models stock price evolution: dS = μS dt + σS dW
- Incorporates correlation structure between stocks
- Estimates drift (μ) and volatility (σ) from historical data
- Factor model integration with external indicators

### 3. External Market Indicators
The system automatically incorporates:
- **NIFTY50 returns** - Indian market index
- **S&P500 returns** - US market index
- **USDINR** - Currency exchange rate
- **Brent Crude Oil** - Commodity prices
- **FII activity** - Foreign institutional investor flows
- **DII activity** - Domestic institutional investor flows
- **India 10-year bond yields** - Interest rate proxy
- **India VIX** - Market volatility
- **US Dollar Index** - Global currency strength

### 4. Random Events
- **RBI Repo Rate Changes** integrated as discrete random events
- Historical frequency and magnitude distribution
- Event impact on stock prices based on sensitivity

### 5. Portfolio Optimization
Multiple optimization strategies:
- **Maximum Sharpe Ratio** - Best risk-adjusted returns
- **Minimum Variance** - Lowest risk portfolio
- **Risk Parity** - Equal risk contribution
- **Efficient Frontier** - Full risk-return spectrum
- **Black-Litterman** - Bayesian approach with views

### 6. Value at Risk (VaR)
Comprehensive risk analysis:
- **Historical VaR** - Empirical quantiles
- **Parametric VaR** - Normal distribution assumption
- **Monte Carlo VaR** - Simulation-based
- **CVaR (Conditional VaR)** - Expected shortfall
- Multiple confidence levels: 90%, 95%, 99%

### 7. Visualization Suite
Automatic generation of:
- Efficient frontier plot
- Portfolio weight allocation charts
- Simulated price path evolution
- Return distribution analysis
- VaR analysis charts
- Convergence diagnostics
- Correlation heatmaps

## Technical Architecture

### Directory Structure
```
portfolio_optimization/
├── data/                          # Data directory
│   └── raw/                       # Contains Book1__1_.xlsx
├── src/                           # Source code
│   ├── data_processing/           # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── data_loader.py        # Excel data loader
│   ├── models/                    # Financial models
│   │   ├── __init__.py
│   │   ├── gbm.py                # Geometric Brownian Motion
│   │   └── var_calculator.py    # VaR calculations
│   ├── simulation/                # Monte Carlo engine
│   │   ├── __init__.py
│   │   └── monte_carlo.py        # Simulation & event handling
│   ├── optimization/              # Portfolio optimization
│   │   ├── __init__.py
│   │   └── portfolio_optimizer.py
│   └── utils/                     # Utilities
│       ├── __init__.py
│       ├── helpers.py            # Helper functions
│       └── visualization.py      # Plotting functions
├── config/
│   └── config.yaml               # Configuration parameters
├── outputs/
│   ├── results/                  # Numerical results
│   ├── plots/                    # Visualizations
│   └── logs/                     # Log files
├── tests/
│   ├── __init__.py
│   └── test_models.py           # Unit tests
├── notebooks/
│   └── portfolio_analysis.ipynb # Interactive analysis
├── main.py                       # Main execution script
├── requirements.txt              # Dependencies
├── setup.py                      # Package setup
├── README.md                     # Project documentation
├── QUICKSTART.md                 # Quick start guide
└── .gitignore                    # Git ignore rules
```

## Mathematical Framework

### Geometric Brownian Motion
Stock price evolution:
```
dS_t = μ S_t dt + σ S_t dW_t
```
Where:
- S_t: Stock price at time t
- μ: Drift (expected return)
- σ: Volatility (standard deviation)
- W_t: Wiener process (Brownian motion)

### Portfolio Optimization
Maximize Sharpe Ratio:
```
max (w^T μ - r_f) / √(w^T Σ w)
subject to:
  sum(w) = 1
  w_min ≤ w_i ≤ w_max
```
Where:
- w: Portfolio weights
- μ: Expected returns
- Σ: Covariance matrix
- r_f: Risk-free rate

### Value at Risk
VaR at confidence level α:
```
VaR_α = inf{x : P(L ≤ x) ≥ α}
```
Where L is the portfolio loss distribution.

## Data Requirements

### Stock Data (Stock Data sheet)
- Historical stock prices
- Date, Close price, Returns for each stock
- Format: Groups of 3 columns per stock

### External Indicators (Clean Data sheet)
- NIFTY50: Price, Returns
- India VIX: Price, Returns
- FII/DII: Net investment values
- USDINR: Price, Returns
- 10Y Yield: Price, Returns
- Brent Crude: Price, Returns
- S&P500: Price, Returns
- US Dollar Index: Price, Returns
- RBI Repo Rate: Rate, Changes

## Configuration Options

### Critical Parameters to Adjust

**Simulation Scale**:
```yaml
num_simulations: 10000      # Paths per replication
num_replications: 100       # Number of replications
```

**Time Horizon**:
```yaml
time_horizon_days: 252      # 1 year = 252 trading days
```

**Portfolio Constraints**:
```yaml
min_weight: 0.0            # No short selling
max_weight: 0.4            # Max 40% in single stock
```

**Risk Settings**:
```yaml
risk_free_rate: 0.07       # 7% annual
confidence_levels: [0.90, 0.95, 0.99]
```

## Performance Characteristics

### Computational Complexity
- **Single simulation**: O(T × N × M) where:
  - T = time steps (252)
  - N = number of stocks (~17)
  - M = number of paths (10,000)
  
- **Total**: 100 replications × 10,000 paths = 1,000,000 simulations
- **Runtime**: ~5-15 minutes (depending on hardware, with parallel processing)

### Memory Requirements
- ~2-4 GB RAM for default configuration
- Scales with: num_simulations × num_replications × num_stocks

### Convergence
- **Sobol sequences**: Faster convergence (O(1/n)) vs random (O(1/√n))
- **Multiple replications**: Provide confidence intervals and stability
- **Typical convergence**: <1% variance after 50 replications

## Output Interpretation

### Optimal Portfolio Results
```
Expected Annual Return: 15.23%
Annual Volatility: 18.45%
Sharpe Ratio: 0.446
VaR (95%): -4.23%
CVaR (95%): -6.12%
```

**Interpretation**:
- Portfolio expected to return 15.23% annually
- Risk (volatility) is 18.45%
- Risk-adjusted return (Sharpe) is 0.446
- 95% confidence: won't lose more than 4.23% (VaR)
- If losses exceed VaR, expected loss is 6.12% (CVaR)

### Portfolio Weights
Top allocations show which stocks contribute most to optimal portfolio.
Diversification is automatic through optimization constraints.

## Advanced Usage

### Custom Event Handling
Modify `EventSimulator` to include other events:
- Earnings announcements
- Policy changes
- Market crashes
- Regulatory changes

### Factor Models
Extend `GeometricBrownianMotion` to:
- Add custom factors
- Use different factor models (APT, Fama-French)
- Incorporate macroeconomic variables

### Alternative Optimization
Implement in `PortfolioOptimizer`:
- CVaR optimization
- Robust optimization
- Multi-period optimization
- Transaction costs

## Validation and Testing

### Unit Tests
Run: `python -m pytest tests/`

Tests cover:
- GBM parameter estimation
- Path simulation correctness
- VaR calculations
- Optimization convergence
- Data loading integrity

### Backtesting
Compare optimized portfolio against:
- Equal-weight benchmark
- Market index (NIFTY50)
- Historical performance

## Common Issues and Solutions

### Issue 1: Optimization Fails
**Cause**: Singular covariance matrix or infeasible constraints
**Solution**: 
- Increase estimation window
- Relax constraints (increase max_weight)
- Remove highly correlated assets

### Issue 2: Slow Performance
**Cause**: Too many simulations or serial processing
**Solution**:
- Enable parallel processing: `parallel: true`
- Reduce replications for testing
- Use fewer paths per replication

### Issue 3: Memory Error
**Cause**: Large simulation arrays
**Solution**:
- Reduce num_simulations or num_replications
- Process in batches
- Use smaller time horizon

### Issue 4: Unrealistic Results
**Cause**: Poor parameter estimation or data quality
**Solution**:
- Check for outliers in data
- Increase estimation window
- Verify external factor alignment
- Review event frequency

## Future Enhancements

### Potential Extensions
1. **Real-time data integration** - Live market feeds
2. **Machine learning** - Predict returns/volatility
3. **Regime switching** - Different market states
4. **Multi-asset classes** - Bonds, commodities, FX
5. **Transaction costs** - Realistic trading costs
6. **Rebalancing strategies** - Dynamic weight adjustments
7. **Stress testing** - Extreme scenario analysis
8. **ESG factors** - Environmental, social, governance

### Performance Optimizations
1. **GPU acceleration** - CuPy for array operations
2. **Cython compilation** - Speed up critical loops
3. **Distributed computing** - Dask or Ray
4. **Streaming processing** - Handle larger datasets
5. **Caching** - Memoize expensive calculations

## References and Theory

### Key Concepts
- **Markowitz Portfolio Theory** (1952)
- **Capital Asset Pricing Model** (CAPM)
- **Efficient Market Hypothesis**
- **Monte Carlo Methods in Finance**
- **Quasi-Monte Carlo Theory**

### Recommended Reading
1. "Options, Futures, and Other Derivatives" - John Hull
2. "Quantitative Risk Management" - McNeil, Frey, Embrechts
3. "Monte Carlo Methods in Financial Engineering" - Paul Glasserman
4. "Portfolio Selection" - Harry Markowitz
5. "The Black-Litterman Model" - He & Litterman

## Support and Maintenance

### Documentation
- README.md - Overview and installation
- QUICKSTART.md - Getting started guide
- This document - Complete technical reference
- Code comments - Inline documentation

### Logging
All operations logged to `outputs/logs/portfolio_optimization.log`
Log levels: DEBUG, INFO, WARNING, ERROR

### Version Control
- Use Git for tracking changes
- Tag releases: v1.0.0, v1.1.0, etc.
- Document breaking changes

## License

MIT License - Free for academic and commercial use

## Contact

For questions, issues, or contributions:
- Check logs for debugging
- Review configuration settings
- Consult QUICKSTART.md
- Examine sample notebook

---

**Project Status**: Production-ready
**Last Updated**: January 2026
**Python Version**: 3.8+
**Dependencies**: See requirements.txt
