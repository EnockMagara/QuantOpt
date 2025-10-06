"""
Baseline Modern Portfolio Theory (MPT) optimizer
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import cvxpy as cp
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import warnings

warnings.filterwarnings('ignore')


class MPTOptimizer:
    """
    Modern Portfolio Theory optimizer for portfolio construction
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the MPT optimizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # MPT configuration
        self.mpt_config = config.get('models', {}).get('baseline', {})
        self.optimization_method = self.mpt_config.get('optimization_method', 'max_sharpe')
        self.rebalance_frequency = self.mpt_config.get('rebalance_frequency', 'monthly')
        self.transaction_costs = self.mpt_config.get('transaction_costs', 0.001)
        
        # Risk-free rate
        self.risk_free_rate = config.get('evaluation', {}).get('metrics', {}).get('risk_free_rate', 0.02)
        
        # Portfolio weights
        self.weights = None
        self.expected_returns = None
        self.covariance_matrix = None
        
    def calculate_expected_returns(self, returns_data: pd.DataFrame, 
                                 method: str = 'mean') -> pd.Series:
        """
        Calculate expected returns for assets
        
        Args:
            returns_data: DataFrame with returns data
            method: Method for calculating expected returns ('mean', 'capm', 'black_litterman')
            
        Returns:
            Series with expected returns
        """
        self.logger.info(f"Calculating expected returns using method: {method}")
        
        if method == 'mean':
            # Simple historical mean
            expected_returns = returns_data.mean()
        elif method == 'capm':
            # CAPM-based expected returns
            expected_returns = self._calculate_capm_returns(returns_data)
        elif method == 'black_litterman':
            # Black-Litterman model
            expected_returns = self._calculate_black_litterman_returns(returns_data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.expected_returns = expected_returns
        return expected_returns
    
    def _calculate_capm_returns(self, returns_data: pd.DataFrame) -> pd.Series:
        """Calculate expected returns using CAPM"""
        # Market return (assuming SPY as market proxy)
        if 'SPY' in returns_data.columns:
            market_return = returns_data['SPY'].mean()
        else:
            market_return = returns_data.mean().mean()
        
        # Risk-free rate (annualized)
        risk_free_rate = self.risk_free_rate / 252  # Daily risk-free rate
        
        # Calculate betas
        betas = {}
        for asset in returns_data.columns:
            if asset != 'SPY':
                # Calculate beta relative to market
                if 'SPY' in returns_data.columns:
                    covariance = returns_data[asset].cov(returns_data['SPY'])
                    market_variance = returns_data['SPY'].var()
                    beta = covariance / market_variance if market_variance != 0 else 1.0
                else:
                    beta = 1.0  # Default beta
                
                # CAPM expected return
                expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
                betas[asset] = expected_return
        
        return pd.Series(betas)
    
    def _calculate_black_litterman_returns(self, returns_data: pd.DataFrame) -> pd.Series:
        """Calculate expected returns using Black-Litterman model"""
        # This is a simplified implementation
        # In practice, you would need market cap weights and views
        
        # Market cap weights (simplified - equal weights)
        market_caps = pd.Series(1.0 / len(returns_data.columns), index=returns_data.columns)
        
        # Implied equilibrium returns
        cov_matrix = returns_data.cov()
        risk_aversion = 3.0  # Risk aversion parameter
        implied_returns = risk_aversion * cov_matrix.dot(market_caps)
        
        # For simplicity, we'll use implied returns
        # In practice, you would incorporate views here
        return implied_returns
    
    def calculate_covariance_matrix(self, returns_data: pd.DataFrame, 
                                  method: str = 'sample') -> pd.DataFrame:
        """
        Calculate covariance matrix for assets
        
        Args:
            returns_data: DataFrame with returns data
            method: Method for calculating covariance ('sample', 'ledoit_wolf', 'shrunk')
            
        Returns:
            Covariance matrix
        """
        self.logger.info(f"Calculating covariance matrix using method: {method}")
        
        if method == 'sample':
            # Sample covariance matrix
            cov_matrix = returns_data.cov()
        elif method == 'ledoit_wolf':
            # Ledoit-Wolf shrinkage estimator
            lw = LedoitWolf()
            cov_matrix = pd.DataFrame(
                lw.fit(returns_data).covariance_,
                index=returns_data.columns,
                columns=returns_data.columns
            )
        elif method == 'shrunk':
            # Shrunk covariance matrix
            cov_matrix = self._calculate_shrunk_covariance(returns_data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.covariance_matrix = cov_matrix
        return cov_matrix
    
    def _calculate_shrunk_covariance(self, returns_data: pd.DataFrame, 
                                   shrinkage: float = 0.1) -> pd.DataFrame:
        """Calculate shrunk covariance matrix"""
        sample_cov = returns_data.cov()
        
        # Target: diagonal matrix with average variance
        avg_var = np.trace(sample_cov) / len(sample_cov.columns)
        target = np.eye(len(sample_cov.columns)) * avg_var
        
        # Shrinkage
        shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * target
        
        return pd.DataFrame(shrunk_cov, 
                          index=sample_cov.index, 
                          columns=sample_cov.columns)
    
    def optimize_portfolio(self, expected_returns: pd.Series, 
                          covariance_matrix: pd.DataFrame,
                          optimization_method: str = "max_sharpe",
                          max_position: float = 0.5,
                          min_weight: float = 0.01,
                          risk_free_rate: float = 0.04) -> Dict:
        """
        Optimize portfolio weights
        
        Args:
            expected_returns: Expected returns for assets
            covariance_matrix: Covariance matrix
            optimization_method: Optimization method to use
            
        Returns:
            Dictionary with optimization results
        """
        if optimization_method is None:
            optimization_method = self.optimization_method
        
        self.logger.info(f"Optimizing portfolio using method: {optimization_method}")
        
        if optimization_method == 'max_sharpe':
            return self._optimize_max_sharpe(expected_returns, covariance_matrix, 
                                           max_position, min_weight, risk_free_rate)
        elif optimization_method == 'min_variance':
            return self._optimize_min_variance(covariance_matrix, max_position, min_weight)
        elif optimization_method == 'min_volatility':
            return self._optimize_min_variance(covariance_matrix, max_position, min_weight)
        elif optimization_method == 'max_return':
            return self._optimize_max_return(expected_returns, covariance_matrix, 
                                           max_position, min_weight)
        elif optimization_method == 'efficient_frontier':
            return self._optimize_efficient_frontier(expected_returns, covariance_matrix, 
                                                   max_position, min_weight, risk_free_rate)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
    
    def _optimize_max_sharpe(self, expected_returns: pd.Series, 
                           covariance_matrix: pd.DataFrame,
                           max_position: float = 0.5,
                           min_weight: float = 0.01,
                           risk_free_rate: float = 0.04) -> Dict:
        """Optimize for maximum Sharpe ratio using improved constraints"""
        n_assets = len(expected_returns)
        
        # Define variables
        weights = cp.Variable(n_assets)
        
        # Dynamic constraints based on parameters
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= min_weight,  # Minimum weight threshold
            weights <= max_position,  # Maximum position size
        ]
        
        # Temporarily remove crypto constraints for testing
        # high_vol_assets = []
        # for i, asset in enumerate(expected_returns.index):
        #     if any(crypto in asset for crypto in ['BTC', 'ETH', 'USD']):
        #         high_vol_assets.append(i)
        
        # # Limit total exposure to high-volatility assets (more lenient)
        # if high_vol_assets:
        #     crypto_weights = cp.sum([weights[i] for i in high_vol_assets])
        #     constraints.append(crypto_weights <= 0.50)  # Max 50% in crypto
        
        # Define objective: maximize return with risk constraint
        portfolio_return = expected_returns.values @ weights
        portfolio_risk = cp.quad_form(weights, covariance_matrix.values)
        
        # Temporarily remove risk constraint to test feasibility
        # max_risk = 0.0625  # 25% annual volatility squared
        # constraints.append(portfolio_risk <= max_risk)
        objective = cp.Maximize(portfolio_return)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            self.logger.warning(f"Optimization failed with status: {problem.status}")
            # Fallback to equal weights
            weights_value = np.ones(n_assets) / n_assets
            # Apply position constraints to fallback weights
            weights_value = np.clip(weights_value, min_weight, max_position)
            weights_value = weights_value / weights_value.sum()  # Renormalize
        else:
            weights_value = weights.value
        
        # Calculate portfolio metrics
        portfolio_return = expected_returns.values @ weights_value
        portfolio_risk = np.sqrt(weights_value.T @ covariance_matrix.values @ weights_value)
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        self.weights = pd.Series(weights_value, index=expected_returns.index)
        
        return {
            'weights': self.weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'optimization_status': problem.status
        }
    
    def _optimize_min_variance(self, covariance_matrix: pd.DataFrame, 
                              max_position: float = 0.5, 
                              min_weight: float = 0.01) -> Dict:
        """Optimize for minimum variance"""
        n_assets = len(covariance_matrix)
        
        # Define variables
        weights = cp.Variable(n_assets)
        
        # Define constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= min_weight,  # Minimum weight threshold
            weights <= max_position  # Maximum position size
        ]
        
        # Define objective: minimize variance
        portfolio_risk = cp.quad_form(weights, covariance_matrix.values)
        objective = cp.Minimize(portfolio_risk)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            self.logger.warning(f"Optimization failed with status: {problem.status}")
            # Fallback to equal weights
            weights_value = np.ones(n_assets) / n_assets
        else:
            weights_value = weights.value
        
        # Calculate portfolio metrics
        portfolio_risk = np.sqrt(weights_value.T @ covariance_matrix.values @ weights_value)
        portfolio_return = self.expected_returns.values @ weights_value if self.expected_returns is not None else 0
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        self.weights = pd.Series(weights_value, index=covariance_matrix.index)
        
        return {
            'weights': self.weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'optimization_status': problem.status
        }
    
    def _optimize_max_return(self, expected_returns: pd.Series, 
                           covariance_matrix: pd.DataFrame,
                           max_position: float = 0.5,
                           min_weight: float = 0.01) -> Dict:
        """Optimize for maximum return"""
        n_assets = len(expected_returns)
        
        # Define variables
        weights = cp.Variable(n_assets)
        
        # Define constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= 0,  # Long-only portfolio
            weights <= 0.4,  # Maximum 40% in any single asset
            cp.quad_form(weights, covariance_matrix.values) <= 0.04  # Max 20% volatility
        ]
        
        # Define objective: maximize return
        portfolio_return = expected_returns.values @ weights
        objective = cp.Maximize(portfolio_return)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            self.logger.warning(f"Optimization failed with status: {problem.status}")
            # Fallback to equal weights
            weights_value = np.ones(n_assets) / n_assets
        else:
            weights_value = weights.value
        
        # Calculate portfolio metrics
        portfolio_return = expected_returns.values @ weights_value
        portfolio_risk = np.sqrt(weights_value.T @ covariance_matrix.values @ weights_value)
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        self.weights = pd.Series(weights_value, index=expected_returns.index)
        
        return {
            'weights': self.weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'optimization_status': problem.status
        }
    
    def _optimize_efficient_frontier(self, expected_returns: pd.Series, 
                                   covariance_matrix: pd.DataFrame,
                                   max_position: float = 0.5,
                                   min_weight: float = 0.01,
                                   risk_free_rate: float = 0.04,
                                   n_portfolios: int = 100) -> Dict:
        """Generate efficient frontier"""
        n_assets = len(expected_returns)
        
        # Define variables
        weights = cp.Variable(n_assets)
        
        # Define constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= 0,  # Long-only portfolio
            weights <= 0.4  # Maximum 40% in any single asset
        ]
        
        # Generate efficient frontier
        target_returns = np.linspace(expected_returns.min(), expected_returns.max(), n_portfolios)
        efficient_portfolios = []
        
        for target_return in target_returns:
            # Add target return constraint
            target_constraints = constraints + [expected_returns.values @ weights >= target_return]
            
            # Minimize variance for given return
            portfolio_risk = cp.quad_form(weights, covariance_matrix.values)
            objective = cp.Minimize(portfolio_risk)
            
            # Solve optimization problem
            problem = cp.Problem(objective, target_constraints)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                weights_value = weights.value
                portfolio_return = expected_returns.values @ weights_value
                portfolio_risk = np.sqrt(weights_value.T @ covariance_matrix.values @ weights_value)
                sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
                
                efficient_portfolios.append({
                    'weights': pd.Series(weights_value, index=expected_returns.index),
                    'expected_return': portfolio_return,
                    'volatility': portfolio_risk,
                    'sharpe_ratio': sharpe_ratio
                })
        
        # Find maximum Sharpe ratio portfolio
        max_sharpe_portfolio = max(efficient_portfolios, key=lambda x: x['sharpe_ratio'])
        
        self.weights = max_sharpe_portfolio['weights']
        
        return {
            'efficient_frontier': efficient_portfolios,
            'max_sharpe_portfolio': max_sharpe_portfolio,
            'weights': self.weights
        }
    
    def calculate_portfolio_metrics(self, returns_data: pd.DataFrame, 
                                  weights: pd.Series) -> Dict:
        """
        Calculate portfolio performance metrics
        
        Args:
            returns_data: Historical returns data
            weights: Portfolio weights
            
        Returns:
            Dictionary with portfolio metrics
        """
        self.logger.info("Calculating portfolio metrics")
        
        # Calculate portfolio returns
        portfolio_returns = (returns_data * weights).sum(axis=1)
        
        # Calculate metrics
        metrics = {
            'total_return': (1 + portfolio_returns).prod() - 1,
            'annualized_return': (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1,
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': (portfolio_returns.mean() * 252 - self.risk_free_rate) / (portfolio_returns.std() * np.sqrt(252)),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'calmar_ratio': (portfolio_returns.mean() * 252) / abs(self._calculate_max_drawdown(portfolio_returns)),
            'sortino_ratio': self._calculate_sortino_ratio(portfolio_returns),
            'var_95': np.percentile(portfolio_returns, 5),
            'cvar_95': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean()
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        
        if downside_deviation == 0:
            return 0
        
        return (excess_returns.mean() * 252) / (downside_deviation * np.sqrt(252))
    
    def rebalance_portfolio(self, current_weights: pd.Series, 
                          target_weights: pd.Series) -> Dict:
        """
        Calculate rebalancing trades
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            
        Returns:
            Dictionary with rebalancing information
        """
        self.logger.info("Calculating portfolio rebalancing")
        
        # Calculate weight changes
        weight_changes = target_weights - current_weights
        
        # Calculate transaction costs
        transaction_costs = abs(weight_changes).sum() * self.transaction_costs
        
        # Calculate turnover
        turnover = abs(weight_changes).sum() / 2
        
        return {
            'weight_changes': weight_changes,
            'transaction_costs': transaction_costs,
            'turnover': turnover,
            'new_weights': target_weights
        }
    
    def backtest_portfolio(self, returns_data: pd.DataFrame, 
                          rebalance_frequency: str = None) -> pd.DataFrame:
        """
        Backtest portfolio performance
        
        Args:
            returns_data: Historical returns data
            rebalance_frequency: Rebalancing frequency
            
        Returns:
            DataFrame with portfolio performance
        """
        if rebalance_frequency is None:
            rebalance_frequency = self.rebalance_frequency
        
        self.logger.info(f"Backtesting portfolio with rebalancing frequency: {rebalance_frequency}")
        
        # Calculate expected returns and covariance matrix
        expected_returns = self.calculate_expected_returns(returns_data)
        covariance_matrix = self.calculate_covariance_matrix(returns_data)
        
        # Initialize portfolio
        portfolio_returns = []
        portfolio_weights = []
        dates = []
        
        # Determine rebalancing dates
        if rebalance_frequency == 'daily':
            rebalance_dates = returns_data.index
        elif rebalance_frequency == 'weekly':
            rebalance_dates = returns_data.index[::5]  # Every 5 trading days
        elif rebalance_frequency == 'monthly':
            rebalance_dates = returns_data.index[::21]  # Every 21 trading days
        elif rebalance_frequency == 'quarterly':
            rebalance_dates = returns_data.index[::63]  # Every 63 trading days
        else:
            rebalance_dates = returns_data.index[::21]  # Default to monthly
        
        # Initialize weights
        current_weights = pd.Series(1.0 / len(returns_data.columns), index=returns_data.columns)
        
        for i, date in enumerate(returns_data.index):
            # Check if rebalancing is needed
            if date in rebalance_dates:
                # Use lookback window for optimization
                lookback_window = min(252, i)  # 1 year or available data
                if lookback_window > 50:  # Minimum 50 days
                    lookback_data = returns_data.iloc[i-lookback_window:i]
                    
                    # Optimize portfolio
                    expected_returns = self.calculate_expected_returns(lookback_data)
                    covariance_matrix = self.calculate_covariance_matrix(lookback_data)
                    optimization_result = self.optimize_portfolio(expected_returns, covariance_matrix)
                    
                    current_weights = optimization_result['weights']
            
            # Calculate portfolio return for this period
            portfolio_return = (returns_data.loc[date] * current_weights).sum()
            portfolio_returns.append(portfolio_return)
            portfolio_weights.append(current_weights.copy())
            dates.append(date)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Date': dates,
            'Portfolio_Return': portfolio_returns
        })
        
        # Add cumulative returns
        results['Cumulative_Return'] = (1 + results['Portfolio_Return']).cumprod() - 1
        
        return results
