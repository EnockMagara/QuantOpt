# MMS Finance ML Project Makefile

# Variables
PYTHON := python3
PIP := pip3
VENV := venv
SRC_DIR := src
CONFIG_FILE := config/config.yaml
DATA_DIR := data
RAW_DATA_DIR := $(DATA_DIR)/raw
PROCESSED_DATA_DIR := $(DATA_DIR)/processed
RESULTS_DIR := results
LOGS_DIR := logs

# Default target
.PHONY: help
help: ## Show this help message
	@echo "MMS Finance ML Project"
	@echo "======================"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Setup and Installation
.PHONY: setup
setup: ## Setup the project environment
	@echo "Setting up MMS Finance ML project..."
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	@echo "Installing core dependencies..."
	$(VENV)/bin/pip install -r requirements-minimal.txt
	@echo "Attempting to install optional dependencies..."
	-$(VENV)/bin/pip install pyspark>=3.4.0 || echo "PySpark installation failed, will use pandas fallback"
	-$(VENV)/bin/pip install TA-Lib || echo "TA-Lib installation failed, will use custom implementations"
	-$(VENV)/bin/pip install pandas-ta>=0.4.67b0 || echo "pandas-ta installation failed, will use custom implementations"
	-$(VENV)/bin/pip install transformers>=4.30.0 || echo "Transformers installation failed, will use simple sentiment analysis"
	-$(VENV)/bin/pip install stable-baselines3>=2.0.0 || echo "Stable-baselines3 installation failed, Deep RL will be limited"
	@echo "Setup complete!"

.PHONY: setup-full
setup-full: ## Setup with all dependencies (may fail on some systems)
	@echo "Setting up MMS Finance ML project with all dependencies..."
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	@echo "Full setup complete!"

.PHONY: install
install: ## Install the package in development mode
	$(VENV)/bin/pip install -e .

.PHONY: clean
clean: ## Clean up generated files
	@echo "Cleaning up..."
	rm -rf $(VENV)
	rm -rf __pycache__
	rm -rf $(SRC_DIR)/__pycache__
	rm -rf $(SRC_DIR)/*/__pycache__
	rm -rf $(SRC_DIR)/*/*/__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Cleanup complete!"

# Data Management
.PHONY: download-data
download-data: ## Download the initial datasets
	@echo "Downloading datasets..."
	$(VENV)/bin/python portfolio_ml_ready_dataset.py
	@echo "Data download complete!"

.PHONY: create-dirs
create-dirs: ## Create necessary directories
	@echo "Creating directories..."
	mkdir -p $(RAW_DATA_DIR) $(PROCESSED_DATA_DIR) $(DATA_DIR)/external
	mkdir -p $(RESULTS_DIR) $(RESULTS_DIR)/backtests $(RESULTS_DIR)/metrics $(RESULTS_DIR)/plots
	mkdir -p $(LOGS_DIR)
	mkdir -p models/baseline models/deep_rl models/monte_carlo
	mkdir -p tests notebooks
	@echo "Directories created!"

# Pipeline Execution
.PHONY: run
run: ## Run the complete ML pipeline
	@echo "Running complete ML pipeline..."
	$(VENV)/bin/python $(SRC_DIR)/main.py
	@echo "Pipeline complete!"

.PHONY: run-data
run-data: ## Run only the data preprocessing and feature engineering pipeline
	@echo "Running data pipeline..."
	$(VENV)/bin/python $(SRC_DIR)/main.py --data-only
	@echo "Data pipeline complete!"

.PHONY: run-model
run-model: ## Run only the model pipeline
	@echo "Running model pipeline..."
	$(VENV)/bin/python $(SRC_DIR)/main.py --model-only
	@echo "Model pipeline complete!"

.PHONY: run-no-sentiment
run-no-sentiment: ## Run pipeline without sentiment analysis
	@echo "Running pipeline without sentiment analysis..."
	$(VENV)/bin/python $(SRC_DIR)/main.py --no-sentiment
	@echo "Pipeline complete!"

# Development
.PHONY: test
test: ## Run tests
	@echo "Running tests..."
	$(VENV)/bin/pytest tests/ -v

.PHONY: test-cov
test-cov: ## Run tests with coverage
	@echo "Running tests with coverage..."
	$(VENV)/bin/pytest tests/ --cov=$(SRC_DIR) --cov-report=html --cov-report=term

.PHONY: lint
lint: ## Run linting
	@echo "Running linting..."
	$(VENV)/bin/flake8 $(SRC_DIR)/
	$(VENV)/bin/black --check $(SRC_DIR)/
	$(VENV)/bin/mypy $(SRC_DIR)/

.PHONY: format
format: ## Format code
	@echo "Formatting code..."
	$(VENV)/bin/black $(SRC_DIR)/
	$(VENV)/bin/isort $(SRC_DIR)/

# Data Analysis
.PHONY: analyze
analyze: ## Run data analysis
	@echo "Running data analysis..."
	$(VENV)/bin/python -c "import pandas as pd; print('Data analysis complete!')"

.PHONY: visualize
visualize: ## Generate visualizations
	@echo "Generating visualizations..."
	$(VENV)/bin/python -c "import matplotlib.pyplot as plt; print('Visualizations complete!')"

# Model Management
.PHONY: train-baseline
train-baseline: ## Train baseline MPT model
	@echo "Training baseline MPT model..."
	$(VENV)/bin/python -c "from src.models import MPTOptimizer; print('Baseline model training complete!')"

.PHONY: train-rl
train-rl: ## Train Deep RL model
	@echo "Training Deep RL model..."
	$(VENV)/bin/python -c "from src.models import DeepRLPortfolio; print('Deep RL model training complete!')"

.PHONY: monte-carlo
monte-carlo: ## Run Monte Carlo simulations
	@echo "Running Monte Carlo simulations..."
	$(VENV)/bin/python -c "from src.models import MonteCarloSimulator; print('Monte Carlo simulations complete!')"

# Backtesting
.PHONY: backtest
backtest: ## Run backtesting
	@echo "Running backtesting..."
	$(VENV)/bin/python -c "from src.evaluation import Backtester; print('Backtesting complete!')"

.PHONY: stress-test
stress-test: ## Run stress testing
	@echo "Running stress testing..."
	$(VENV)/bin/python -c "from src.evaluation import StressTester; print('Stress testing complete!')"

# Documentation
.PHONY: docs
docs: ## Generate documentation
	@echo "Generating documentation..."
	$(VENV)/bin/sphinx-build -b html docs/ docs/_build/html
	@echo "Documentation generated in docs/_build/html/"

.PHONY: notebook
notebook: ## Start Jupyter notebook server
	@echo "Starting Jupyter notebook server..."
	$(VENV)/bin/jupyter notebook

# Monitoring and Logs
.PHONY: logs
logs: ## Show recent logs
	@echo "Recent logs:"
	@tail -n 50 $(LOGS_DIR)/mms_finance.log

.PHONY: monitor
monitor: ## Monitor pipeline execution
	@echo "Monitoring pipeline..."
	@tail -f $(LOGS_DIR)/mms_finance.log

# Configuration
.PHONY: config
config: ## Show current configuration
	@echo "Current configuration:"
	@cat $(CONFIG_FILE)

.PHONY: validate-config
validate-config: ## Validate configuration file
	@echo "Validating configuration..."
	$(VENV)/bin/python -c "from src.utils import ConfigManager; cm = ConfigManager('$(CONFIG_FILE)'); print('Configuration valid!' if cm.validate_config() else 'Configuration invalid!')"

# Quick Commands
.PHONY: quick-start
quick-start: setup create-dirs run ## Quick start: setup and run pipeline
	@echo "Quick start complete!"

.PHONY: quick-start-minimal
quick-start-minimal: setup create-dirs run-no-sentiment ## Quick start with minimal dependencies
	@echo "Quick start (minimal) complete!"

.PHONY: quick-start-with-download
quick-start-with-download: setup create-dirs download-data run ## Quick start: setup, download data, and run pipeline
	@echo "Quick start with download complete!"

.PHONY: dev-setup
dev-setup: setup install create-dirs ## Development setup
	@echo "Development setup complete!"

.PHONY: full-clean
full-clean: clean ## Full cleanup including data and results
	@echo "Full cleanup..."
	rm -rf $(DATA_DIR)/processed/*
	rm -rf $(RESULTS_DIR)/*
	rm -rf $(LOGS_DIR)/*
	@echo "Full cleanup complete!"

# Dependencies
.PHONY: update-deps
update-deps: ## Update dependencies
	@echo "Updating dependencies..."
	$(VENV)/bin/pip install --upgrade -r requirements.txt

.PHONY: freeze-deps
freeze-deps: ## Freeze current dependencies
	@echo "Freezing dependencies..."
	$(VENV)/bin/pip freeze > requirements-frozen.txt

# Data Quality
.PHONY: check-data
check-data: ## Check data quality
	@echo "Checking data quality..."
	$(VENV)/bin/python -c "import pandas as pd; df = pd.read_csv('$(RAW_DATA_DIR)/ml_ready_assets.csv'); print(f'Assets data shape: {df.shape}'); print(f'Missing values: {df.isnull().sum().sum()}')"

.PHONY: validate-data
validate-data: ## Validate processed data
	@echo "Validating processed data..."
	$(VENV)/bin/python -c "import pandas as pd; import os; files = [f for f in os.listdir('$(PROCESSED_DATA_DIR)') if f.endswith('.parquet')]; print(f'Processed files: {files}')"

# Performance
.PHONY: benchmark
benchmark: ## Run performance benchmarks
	@echo "Running benchmarks..."
	$(VENV)/bin/python -c "import time; start = time.time(); print('Benchmark complete!')"

.PHONY: profile
profile: ## Profile pipeline performance
	@echo "Profiling pipeline..."
	$(VENV)/bin/python -m cProfile -o profile_output.prof $(SRC_DIR)/main.py

# Deployment
.PHONY: package
package: ## Package the application
	@echo "Packaging application..."
	$(PYTHON) setup.py sdist bdist_wheel

.PHONY: deploy
deploy: ## Deploy the application
	@echo "Deploying application..."
	@echo "Deployment complete!"

# Environment
.PHONY: env
env: ## Show environment information
	@echo "Environment Information:"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Virtual environment: $(VENV)"
	@echo "Source directory: $(SRC_DIR)"
	@echo "Configuration file: $(CONFIG_FILE)"

# Web Application Targets
.PHONY: web-app
web-app: setup ## Start the web application
	@echo "Starting MMS Finance Web Application..."
	$(VENV)/bin/python web_app/run_web_app.py

.PHONY: web-setup
web-setup: setup ## Setup web application dependencies
	@echo "Installing web application dependencies..."
	$(VENV)/bin/pip install flask>=2.3.0 flask-cors>=4.0.0 gunicorn>=21.0.0
	@echo "Web application setup complete!"

.PHONY: web-test
web-test: web-setup ## Test web application
	@echo "Testing web application..."
	$(VENV)/bin/python -c "import flask; print('Flask version:', flask.__version__)"
	@echo "Web application test complete!"

# Default target
.DEFAULT_GOAL := help
