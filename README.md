# Portfolio Optimization using Quasi-Monte Carlo Simulation

## Project Overview
This project implements portfolio optimization using Geometric Brownian Motion (GBM) with quasi-Monte Carlo simulation. The system incorporates multiple market indicators and random events (RBI repo rate changes) to optimize stock portfolios with Value at Risk (VaR) analysis.

## Project Structure
```
portfolio_optimization/
├── data/                          # Data directory
│   └── raw/                       # Raw data files
├── src/                           # Source code
│   ├── data_processing/           # Data loading and preprocessing
│   ├── models/                    # Financial models (GBM, VaR)
│   ├── simulation/                # Monte Carlo simulation engine
│   ├── optimization/              # Portfolio optimization algorithms
│   └── utils/                     # Utility functions
├── config/                        # Configuration files
├── outputs/                       # Output directory
│   ├── results/                   # Simulation results
│   ├── plots/                     # Visualizations
│   └── logs/                      # Log files
├── tests/                         # Unit tests
└── notebooks/                     # Jupyter notebooks for analysis

```

## Features
- **Quasi-Monte Carlo Simulation**: Uses low-discrepancy sequences (Sobol) for better convergence
- **Geometric Brownian Motion**: Models stock price evolution
- **Multi-factor Model**: Incorporates external indicators (Nifty50, S&P500, USDINR, Brent Crude, FII/DII activity, bond yields, India VIX, US Dollar Index)
- **Random Events**: RBI repo rate changes integrated as discrete events
- **VaR Calculation**: Value at Risk analysis at multiple confidence levels
- **Portfolio Optimization**: Efficient frontier and optimal portfolio weights
- **Multiple Replications**: Parallel processing of simulation replications

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Run the complete pipeline
python main.py

# Or run individual components
python src/data_processing/load_data.py
python src/simulation/monte_carlo.py
python src/optimization/portfolio_optimizer.py
```

## Configuration
Edit `config/config.yaml` to adjust:
- Number of simulations per replication
- Number of replications
- Time horizon
- Risk-free rate
- VaR confidence levels
- Optimization constraints

## Data Requirements
- Stock historical prices (Stock Data sheet)
- External indicators (Clean Data sheet)
- RBI repo rate history (for event simulation)

## Output
- Optimized portfolio weights
- Expected returns and volatility
- VaR metrics
- Efficient frontier plots
- Simulation convergence analysis
