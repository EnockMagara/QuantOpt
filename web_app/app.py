#!/usr/bin/env python3
"""
MMS Finance Web Application
A web interface for testing the MMS Finance ML model with user data
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config_manager import ConfigManager
from models.baseline import MPTOptimizer
from features.feature_engineer import FeatureEngineer
from preprocessing.data_processor import DataProcessor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mms-finance-web-app-2024'

# Global variables for model components
config_manager = None
mpt_optimizer = None
feature_engineer = None
data_processor = None

def initialize_models():
    """Initialize the MMS Finance models"""
    global config_manager, mpt_optimizer, feature_engineer, data_processor
    
    try:
        # Load configuration
        config_manager = ConfigManager('../config/config.yaml')
        config = config_manager.config
        
        # Initialize components
        mpt_optimizer = MPTOptimizer(config)
        feature_engineer = FeatureEngineer(config)
        data_processor = DataProcessor(config)
        
        print("✅ Models initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': mpt_optimizer is not None
    })

@app.route('/api/optimize', methods=['POST'])
def optimize_portfolio():
    """Optimize portfolio based on user input"""
    try:
        data = request.get_json()
        
        # Extract user inputs
        tickers = data.get('tickers', [])
        start_date = data.get('start_date', '2020-01-01')
        end_date = data.get('end_date', '2024-01-01')
        risk_tolerance = data.get('risk_tolerance', 'medium')  # low, medium, high
        max_position = data.get('max_position', 0.3)  # Maximum position size
        risk_free_rate = data.get('risk_free_rate', 0.04)  # Risk-free rate
        optimization_method = data.get('optimization_method', 'max_sharpe')  # Optimization method
        rebalancing_frequency = data.get('rebalancing_frequency', 'monthly')  # Rebalancing frequency
        transaction_costs = data.get('transaction_costs', 0.001)  # Transaction costs
        min_weight = data.get('min_weight', 0.01)  # Minimum weight threshold
        lookback_period = data.get('lookback_period', 252)  # Lookback period in days
        
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400
        
        # Download data
        print(f"Downloading data for {len(tickers)} tickers...")
        stock_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        
        if stock_data.empty:
            return jsonify({'error': 'No data found for the specified tickers and date range'}), 400
        
        # Process data
        processed_data = []
        for ticker in tickers:
            if ticker in stock_data.columns.get_level_values(0):
                ticker_data = stock_data[ticker].copy()
                ticker_data['Ticker'] = ticker
                ticker_data = ticker_data.reset_index()
                processed_data.append(ticker_data)
        
        if not processed_data:
            return jsonify({'error': 'No valid data found for processing'}), 400
        
        # Combine all ticker data
        combined_data = pd.concat(processed_data, ignore_index=True)
        
        # Calculate returns
        combined_data = combined_data.sort_values(['Ticker', 'Date'])
        combined_data['Return'] = combined_data.groupby('Ticker')['Close'].pct_change()
        combined_data = combined_data.dropna()
        
        # Calculate expected returns and covariance matrix
        returns_matrix = combined_data.pivot(index='Date', columns='Ticker', values='Return')
        expected_returns = returns_matrix.mean() * 252  # Annualized
        covariance_matrix = returns_matrix.cov() * 252  # Annualized
        
        # Adjust risk tolerance
        if risk_tolerance == 'low':
            max_position = min(max_position, 0.15)
        elif risk_tolerance == 'high':
            max_position = min(max_position, 0.5)
        
        # Optimize portfolio
        print(f"Optimizing portfolio using {optimization_method} method...")
        result = mpt_optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            optimization_method=optimization_method,
            max_position=max_position,
            min_weight=min_weight,
            risk_free_rate=risk_free_rate
        )
        
        # Calculate portfolio metrics
        portfolio_return = (expected_returns * result['weights']).sum()
        portfolio_volatility = np.sqrt(result['weights'].T @ covariance_matrix @ result['weights'])
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Prepare response
        response = {
            'success': True,
            'portfolio_weights': result['weights'].to_dict(),
            'portfolio_metrics': {
                'expected_return': float(portfolio_return),
                'volatility': float(portfolio_volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'optimization_status': result.get('status', 'optimal')
            },
            'data_summary': {
                'tickers': tickers,
                'date_range': f"{start_date} to {end_date}",
                'data_points': len(combined_data),
                'risk_tolerance': risk_tolerance,
                'max_position': max_position,
                'risk_free_rate': risk_free_rate,
                'optimization_method': optimization_method,
                'rebalancing_frequency': rebalancing_frequency,
                'transaction_costs': transaction_costs,
                'min_weight': min_weight,
                'lookback_period': lookback_period
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in portfolio optimization: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def backtest_portfolio():
    """Run backtest on optimized portfolio"""
    try:
        data = request.get_json()
        
        # Extract inputs
        tickers = data.get('tickers', [])
        weights = data.get('weights', {})
        start_date = data.get('start_date', '2020-01-01')
        end_date = data.get('end_date', '2024-01-01')
        
        if not tickers or not weights:
            return jsonify({'error': 'Tickers and weights required for backtest'}), 400
        
        # Download data
        stock_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        
        if stock_data.empty:
            return jsonify({'error': 'No data found for backtest'}), 400
        
        # Calculate portfolio returns
        portfolio_returns = []
        dates = []
        
        for ticker in tickers:
            if ticker in stock_data.columns.get_level_values(0):
                ticker_data = stock_data[ticker]['Close']
                returns = ticker_data.pct_change().dropna()
                
                if len(portfolio_returns) == 0:
                    portfolio_returns = returns * weights.get(ticker, 0)
                    dates = returns.index
                else:
                    portfolio_returns += returns * weights.get(ticker, 0)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1
        
        # Calculate metrics
        total_return = cumulative_returns.iloc[-1]
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / (1 + peak)
        max_drawdown = drawdown.min()
        
        # Prepare backtest data
        backtest_data = []
        for i, (date, ret, cum_ret) in enumerate(zip(dates, portfolio_returns, cumulative_returns)):
            backtest_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'return': float(ret),
                'cumulative_return': float(cum_ret)
            })
        
        response = {
            'success': True,
            'backtest_data': backtest_data,
            'metrics': {
                'total_return': float(total_return),
                'annualized_return': float(annualized_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in backtest: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/suggestions')
def get_ticker_suggestions():
    """Get popular ticker suggestions"""
    popular_tickers = {
        'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC'],
        'etfs': ['SPY', 'QQQ', 'VTI', 'VOO', 'ARKK', 'XLF', 'XLK', 'XLE', 'XLI', 'XLV'],
        'crypto': ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD'],
        'bonds': ['^TNX', '^FVX', '^TYX', 'TLT', 'IEF', 'SHY']
    }
    return jsonify(popular_tickers)

@app.route('/api/optimization-methods')
def get_optimization_methods():
    """Get available optimization methods with descriptions"""
    methods = {
        'max_sharpe': {
            'name': 'Maximum Sharpe Ratio',
            'description': 'Optimizes for the highest risk-adjusted return (return per unit of risk)',
            'best_for': 'Balanced portfolios seeking optimal risk-return tradeoff'
        },
        'min_volatility': {
            'name': 'Minimum Volatility',
            'description': 'Minimizes portfolio volatility while maintaining diversification',
            'best_for': 'Conservative investors prioritizing capital preservation'
        },
        'max_return': {
            'name': 'Maximum Return',
            'description': 'Maximizes expected portfolio return within risk constraints',
            'best_for': 'Aggressive investors willing to accept higher volatility'
        },
        'efficient_frontier': {
            'name': 'Efficient Frontier',
            'description': 'Finds optimal portfolios along the efficient frontier curve',
            'best_for': 'Advanced users seeking frontier optimization'
        }
    }
    return jsonify(methods)

if __name__ == '__main__':
    print("🚀 Starting MMS Finance Web Application...")
    
    # Initialize models
    if initialize_models():
        print("✅ All systems ready!")
        print("🌐 Web application will be available at: http://localhost:8080")
        print("📊 API endpoints available at: http://localhost:8080/api/")
        app.run(debug=True, host='0.0.0.0', port=8080)
    else:
        print("❌ Failed to initialize models. Exiting...")
        sys.exit(1)
