# QuantOpt

A comprehensive machine learning project for portfolio optimization using PySpark and PyTorch, implementing Modern Portfolio Theory (MPT), Deep Reinforcement Learning, and Monte Carlo simulations.

## 🚀 Features

### 1. Data Preprocessing
- **Adjustments**: Handle stock splits and dividends
- **Returns Calculation**: Compute log returns and simple returns
- **Missing Data**: Handle missing trading days and data gaps
- **PySpark Integration**: Scalable data processing for large datasets

### 2. Feature Engineering
- **Technical Indicators**: MACD, RSI, Bollinger Bands, Moving Averages
- **Risk Metrics**: Volatility, Sharpe ratio, Sortino ratio, VaR, CVaR
- **Market Sentiment**: News sentiment analysis using transformers
- **Rolling Statistics**: Multi-window rolling calculations

### 3. Modeling
- **Baseline MPT**: Modern Portfolio Theory optimizer with multiple objectives
- **Deep RL**: DDPG and PPO algorithms for dynamic portfolio rebalancing
- **Monte Carlo**: Scenario testing and risk analysis

### 4. Evaluation
- **Backtesting**: Out-of-sample performance evaluation
- **Risk Metrics**: Comprehensive risk assessment
- **Stress Testing**: 2008-like crash scenario analysis
- **Benchmark Comparison**: Performance vs S&P 500

## 📁 Project Structure

```
mms-finance/
├── src/
│   ├── preprocessing/          # Data preprocessing modules
│   ├── features/              # Feature engineering modules
│   ├── models/                # ML models (MPT, Deep RL, Monte Carlo)
│   ├── evaluation/            # Backtesting and evaluation
│   ├── utils/                 # Utility functions
│   └── main.py               # Main pipeline orchestrator
├── config/
│   └── config.yaml           # Configuration file
├── data/
│   ├── raw/                  # Raw data files
│   ├── processed/            # Processed data files
│   └── external/             # External data sources
├── models/                   # Trained model artifacts
├── results/                  # Results and outputs
├── logs/                     # Log files
├── tests/                    # Unit tests
├── notebooks/                # Jupyter notebooks
├── requirements.txt          # Python dependencies
└── setup.py                 # Package setup
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd mms-finance
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install TA-Lib** (optional, for advanced technical indicators):
```bash
# On macOS
brew install ta-lib
pip install TA-Lib

# On Ubuntu/Debian
sudo apt-get install libta-lib-dev
pip install TA-Lib

# On Windows
# Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

- **Data sources**: Asset tickers, date ranges, API keys
- **Feature engineering**: Technical indicators, rolling windows
- **Models**: Optimization methods, hyperparameters
- **Evaluation**: Backtesting periods, risk metrics

### Key Configuration Sections

```yaml
# Data Configuration
data:
  assets:
    stocks: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    etfs: ["SPY", "QQQ", "VTI"]
    crypto: ["BTC-USD", "ETH-USD"]
    bonds: ["GS10", "GS2"]
  start_date: "2018-01-01"
  end_date: "2025-01-01"

# Model Configuration
models:
  baseline:
    optimization_method: "max_sharpe"  # max_sharpe, min_variance, max_return
    rebalance_frequency: "monthly"
    transaction_costs: 0.001
```

## 🚀 Quick Start

### 1. Download Data
```bash
python portfolio_ml_ready_dataset.py
```

### 2. Run Full Pipeline
```bash
python src/main.py
```

### 3. Run Data Pipeline Only
```bash
python src/main.py --data-only
```

### 4. Run Model Pipeline Only
```bash
python src/main.py --model-only
```

### 5. Skip Sentiment Analysis
```bash
python src/main.py --no-sentiment
```

## 📊 Usage Examples

### Basic Portfolio Optimization
```python
from src.models import MPTOptimizer
from src.preprocessing import DataProcessor
import pandas as pd

# Load configuration
config = {...}  # Your configuration

# Initialize components
data_processor = DataProcessor(config)
mpt_optimizer = MPTOptimizer(config)

# Load and process data
data = pd.read_csv("data/raw/ml_ready_assets.csv")
processed_data = data_processor.process_assets_data(data)

# Prepare returns data
returns_data = processed_data.pivot(index='Date', columns='Ticker', values='Return')

# Optimize portfolio
expected_returns = mpt_optimizer.calculate_expected_returns(returns_data)
covariance_matrix = mpt_optimizer.calculate_covariance_matrix(returns_data)
result = mpt_optimizer.optimize_portfolio(expected_returns, covariance_matrix)

print(f"Optimal weights: {result['weights']}")
print(f"Expected return: {result['expected_return']:.4f}")
print(f"Volatility: {result['volatility']:.4f}")
print(f"Sharpe ratio: {result['sharpe_ratio']:.4f}")
```

### Feature Engineering
```python
from src.features import FeatureEngineer

# Initialize feature engineer
feature_engineer = FeatureEngineer(config)

# Engineer features
engineered_data = feature_engineer.engineer_features(processed_data)

# Get feature importance
importance = feature_engineer.get_feature_importance(engineered_data, 'Return')
print("Top 10 most important features:")
for feature, score in list(importance.items())[:10]:
    print(f"{feature}: {score:.4f}")
```

### Backtesting
```python
# Backtest portfolio
backtest_results = mpt_optimizer.backtest_portfolio(returns_data)

# Calculate performance metrics
metrics = mpt_optimizer.calculate_portfolio_metrics(returns_data, result['weights'])

print("Portfolio Performance:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

## 🔧 Advanced Features

### Deep Reinforcement Learning
```python
from src.models import DeepRLPortfolio

# Initialize Deep RL model
rl_model = DeepRLPortfolio(config)

# Train the model
rl_model.train(engineered_data)

# Get portfolio actions
actions = rl_model.predict(engineered_data)
```

### Monte Carlo Simulation
```python
from src.models import MonteCarloSimulator

# Initialize Monte Carlo simulator
mc_simulator = MonteCarloSimulator(config)

# Run simulations
scenarios = mc_simulator.run_simulations(returns_data, n_simulations=10000)

# Analyze results
risk_metrics = mc_simulator.analyze_scenarios(scenarios)
```

### Sentiment Analysis
```python
from src.features import SentimentAnalyzer

# Initialize sentiment analyzer
sentiment_analyzer = SentimentAnalyzer(config)

# Analyze sentiment for tickers
tickers = ["AAPL", "MSFT", "GOOGL"]
headlines = sentiment_analyzer.scrape_news_headlines(tickers)
sentiment_scores = sentiment_analyzer.calculate_sentiment_scores(headlines)
```

## 📈 Results

The pipeline generates several output files:

- **`results/portfolio_weights.csv`**: Optimal portfolio weights
- **`results/portfolio_metrics.csv`**: Performance metrics
- **`results/backtest_results.csv`**: Historical performance
- **`results/expected_returns.csv`**: Expected returns for assets
- **`results/covariance_matrix.csv`**: Asset covariance matrix

### Key Metrics

- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Annual return / Maximum drawdown
- **Sortino Ratio**: Downside risk-adjusted returns
- **VaR/CVaR**: Value at Risk and Conditional VaR

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=src/
```

## 📝 Logging

Logs are automatically generated in the `logs/` directory:

- **`mms_finance.log`**: Main application logs
- **`preprocessing.log`**: Data preprocessing logs
- **`models.log`**: Model training and inference logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PySpark**: Distributed data processing
- **PyTorch**: Deep learning framework
- **TA-Lib**: Technical analysis library
- **Transformers**: Sentiment analysis models
- **CVXPY**: Convex optimization
- **Stable-Baselines3**: Reinforcement learning algorithms



For questions and support:

- Create an issue in the repository


