"""
Risk metrics calculator for financial data
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, List, Optional
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, when, lag, sum as spark_sum, mean, stddev, min as spark_min, max as spark_max
from pyspark.sql.window import Window
import logging


class RiskMetrics:
    """
    Calculates various risk metrics for financial data
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the risk metrics calculator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.risk_free_rate = config.get('evaluation', {}).get('metrics', {}).get('risk_free_rate', 0.02)
        self.rolling_windows = config.get('features', {}).get('rolling', {}).get('windows', [5, 10, 21, 63])
    
    def calculate_volatility(self, data: Union[pd.DataFrame, SparkDataFrame], 
                           windows: List[int] = None) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate rolling volatility for different windows
        
        Args:
            data: Data containing returns
            windows: List of window sizes for volatility calculation
            
        Returns:
            Data with volatility metrics
        """
        if windows is None:
            windows = self.rolling_windows
        
        self.logger.info(f"Calculating volatility for windows: {windows}")
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_volatility_pandas(data, windows)
        else:
            return self._calculate_volatility_spark(data, windows)
    
    def _calculate_volatility_pandas(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Calculate volatility using pandas"""
        for window in windows:
            # Rolling volatility (annualized)
            data[f'Volatility_{window}'] = (
                data.groupby('Ticker')['Return'].rolling(window).std() * np.sqrt(252)
            ).values
            
            # Rolling volatility of volatility
            data[f'Vol_Vol_{window}'] = (
                data.groupby('Ticker')[f'Volatility_{window}'].rolling(window).std()
            ).values
        
        return data
    
    def _calculate_volatility_spark(self, data: SparkDataFrame, windows: List[int]) -> SparkDataFrame:
        """Calculate volatility using PySpark"""
        for window in windows:
            rolling_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-window+1, 0)
            
            # Rolling volatility (annualized)
            data = data.withColumn(f'Volatility_{window}', 
                                 stddev("Return").over(rolling_window) * np.sqrt(252))
            
            # Rolling volatility of volatility
            vol_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-window+1, 0)
            data = data.withColumn(f'Vol_Vol_{window}', 
                                 stddev(f'Volatility_{window}').over(vol_window))
        
        return data
    
    def calculate_sharpe_ratio(self, data: Union[pd.DataFrame, SparkDataFrame], 
                             windows: List[int] = None) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate rolling Sharpe ratio for different windows
        
        Args:
            data: Data containing returns
            windows: List of window sizes for Sharpe ratio calculation
            
        Returns:
            Data with Sharpe ratio metrics
        """
        if windows is None:
            windows = self.rolling_windows
        
        self.logger.info(f"Calculating Sharpe ratio for windows: {windows}")
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_sharpe_ratio_pandas(data, windows)
        else:
            return self._calculate_sharpe_ratio_spark(data, windows)
    
    def _calculate_sharpe_ratio_pandas(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Calculate Sharpe ratio using pandas"""
        daily_rf_rate = self.risk_free_rate / 252
        
        for window in windows:
            # Rolling mean excess return
            excess_return = data['Return'] - daily_rf_rate
            rolling_excess_return = data.groupby('Ticker')[excess_return].rolling(window).mean()
            
            # Rolling volatility
            rolling_volatility = data.groupby('Ticker')['Return'].rolling(window).std()
            
            # Sharpe ratio
            data[f'Sharpe_Ratio_{window}'] = (rolling_excess_return / rolling_volatility).values
        
        return data
    
    def _calculate_sharpe_ratio_spark(self, data: SparkDataFrame, windows: List[int]) -> SparkDataFrame:
        """Calculate Sharpe ratio using PySpark"""
        daily_rf_rate = self.risk_free_rate / 252
        
        for window in windows:
            rolling_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-window+1, 0)
            
            # Excess return
            data = data.withColumn("Excess_Return", col("Return") - daily_rf_rate)
            
            # Rolling mean excess return
            data = data.withColumn(f'Excess_Return_Mean_{window}', 
                                 mean("Excess_Return").over(rolling_window))
            
            # Rolling volatility
            data = data.withColumn(f'Volatility_{window}', 
                                 stddev("Return").over(rolling_window))
            
            # Sharpe ratio
            data = data.withColumn(f'Sharpe_Ratio_{window}', 
                                 col(f'Excess_Return_Mean_{window}') / col(f'Volatility_{window}'))
        
        return data
    
    def calculate_sortino_ratio(self, data: Union[pd.DataFrame, SparkDataFrame], 
                              windows: List[int] = None) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate rolling Sortino ratio for different windows
        
        Args:
            data: Data containing returns
            windows: List of window sizes for Sortino ratio calculation
            
        Returns:
            Data with Sortino ratio metrics
        """
        if windows is None:
            windows = self.rolling_windows
        
        self.logger.info(f"Calculating Sortino ratio for windows: {windows}")
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_sortino_ratio_pandas(data, windows)
        else:
            return self._calculate_sortino_ratio_spark(data, windows)
    
    def _calculate_sortino_ratio_pandas(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Calculate Sortino ratio using pandas"""
        daily_rf_rate = self.risk_free_rate / 252
        
        for window in windows:
            # Rolling mean excess return
            excess_return = data['Return'] - daily_rf_rate
            rolling_excess_return = data.groupby('Ticker')[excess_return].rolling(window).mean()
            
            # Downside deviation (only negative returns)
            negative_returns = data['Return'].where(data['Return'] < 0)
            downside_deviation = data.groupby('Ticker')[negative_returns].rolling(window).std()
            
            # Sortino ratio
            data[f'Sortino_Ratio_{window}'] = (rolling_excess_return / downside_deviation).values
        
        return data
    
    def _calculate_sortino_ratio_spark(self, data: SparkDataFrame, windows: List[int]) -> SparkDataFrame:
        """Calculate Sortino ratio using PySpark"""
        daily_rf_rate = self.risk_free_rate / 252
        
        for window in windows:
            rolling_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-window+1, 0)
            
            # Excess return
            data = data.withColumn("Excess_Return", col("Return") - daily_rf_rate)
            
            # Rolling mean excess return
            data = data.withColumn(f'Excess_Return_Mean_{window}', 
                                 mean("Excess_Return").over(rolling_window))
            
            # Downside deviation (only negative returns)
            data = data.withColumn("Negative_Return", 
                                 when(col("Return") < 0, col("Return")).otherwise(0))
            data = data.withColumn(f'Downside_Deviation_{window}', 
                                 stddev("Negative_Return").over(rolling_window))
            
            # Sortino ratio
            data = data.withColumn(f'Sortino_Ratio_{window}', 
                                 col(f'Excess_Return_Mean_{window}') / col(f'Downside_Deviation_{window}'))
        
        return data
    
    def calculate_max_drawdown(self, data: Union[pd.DataFrame, SparkDataFrame], 
                             windows: List[int] = None) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate rolling maximum drawdown for different windows
        
        Args:
            data: Data containing returns
            windows: List of window sizes for max drawdown calculation
            
        Returns:
            Data with maximum drawdown metrics
        """
        if windows is None:
            windows = self.rolling_windows
        
        self.logger.info(f"Calculating maximum drawdown for windows: {windows}")
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_max_drawdown_pandas(data, windows)
        else:
            return self._calculate_max_drawdown_spark(data, windows)
    
    def _calculate_max_drawdown_pandas(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Calculate maximum drawdown using pandas"""
        for window in windows:
            # Calculate rolling maximum drawdown
            def calculate_drawdown(returns):
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                return drawdown.min()
            
            data[f'Max_Drawdown_{window}'] = (
                data.groupby('Ticker')['Return'].rolling(window).apply(calculate_drawdown)
            ).values
        
        return data
    
    def _calculate_max_drawdown_spark(self, data: SparkDataFrame, windows: List[int]) -> SparkDataFrame:
        """Calculate maximum drawdown using PySpark"""
        # This is a simplified implementation for PySpark
        # A more sophisticated approach would require custom UDFs
        
        for window in windows:
            rolling_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-window+1, 0)
            
            # Calculate cumulative returns
            data = data.withColumn(f'Cumulative_Return_{window}', 
                                 spark_sum("Return").over(rolling_window))
            
            # Calculate running maximum
            data = data.withColumn(f'Running_Max_{window}', 
                                 max(f'Cumulative_Return_{window}').over(rolling_window))
            
            # Calculate drawdown
            data = data.withColumn(f'Drawdown_{window}', 
                                 (col(f'Cumulative_Return_{window}') - col(f'Running_Max_{window}')) / 
                                 col(f'Running_Max_{window}'))
            
            # Calculate maximum drawdown
            data = data.withColumn(f'Max_Drawdown_{window}', 
                                 min(f'Drawdown_{window}').over(rolling_window))
        
        return data
    
    def calculate_var_cvar(self, data: Union[pd.DataFrame, SparkDataFrame], 
                          confidence_levels: List[float] = [0.05, 0.1, 0.9, 0.95],
                          windows: List[int] = None) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR)
        
        Args:
            data: Data containing returns
            confidence_levels: List of confidence levels for VaR/CVaR
            windows: List of window sizes for VaR/CVaR calculation
            
        Returns:
            Data with VaR and CVaR metrics
        """
        if windows is None:
            windows = self.rolling_windows
        
        self.logger.info(f"Calculating VaR and CVaR for confidence levels: {confidence_levels}")
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_var_cvar_pandas(data, confidence_levels, windows)
        else:
            return self._calculate_var_cvar_spark(data, confidence_levels, windows)
    
    def _calculate_var_cvar_pandas(self, data: pd.DataFrame, confidence_levels: List[float], windows: List[int]) -> pd.DataFrame:
        """Calculate VaR and CVaR using pandas"""
        for window in windows:
            for conf_level in confidence_levels:
                # VaR calculation
                def calculate_var(returns):
                    return np.percentile(returns, conf_level * 100)
                
                def calculate_cvar(returns):
                    var = np.percentile(returns, conf_level * 100)
                    return returns[returns <= var].mean()
                
                data[f'VaR_{conf_level}_{window}'] = (
                    data.groupby('Ticker')['Return'].rolling(window).apply(calculate_var)
                ).values
                
                data[f'CVaR_{conf_level}_{window}'] = (
                    data.groupby('Ticker')['Return'].rolling(window).apply(calculate_cvar)
                ).values
        
        return data
    
    def _calculate_var_cvar_spark(self, data: SparkDataFrame, confidence_levels: List[float], windows: List[int]) -> SparkDataFrame:
        """Calculate VaR and CVaR using PySpark"""
        # This is a simplified implementation for PySpark
        # A more sophisticated approach would require custom UDFs for percentile calculations
        
        for window in windows:
            rolling_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-window+1, 0)
            
            for conf_level in confidence_levels:
                # Simplified VaR using standard deviation approximation
                data = data.withColumn(f'VaR_{conf_level}_{window}', 
                                     mean("Return").over(rolling_window) + 
                                     1.645 * stddev("Return").over(rolling_window))  # 95% confidence
                
                # Simplified CVaR (same as VaR for this implementation)
                data = data.withColumn(f'CVaR_{conf_level}_{window}', 
                                     col(f'VaR_{conf_level}_{window}'))
        
        return data
    
    def calculate_beta(self, data: Union[pd.DataFrame, SparkDataFrame], 
                      benchmark_col: str = 'SPY_Return', windows: List[int] = None) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate rolling beta relative to benchmark
        
        Args:
            data: Data containing returns and benchmark returns
            benchmark_col: Column name for benchmark returns
            windows: List of window sizes for beta calculation
            
        Returns:
            Data with beta metrics
        """
        if windows is None:
            windows = self.rolling_windows
        
        self.logger.info(f"Calculating beta for windows: {windows}")
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_beta_pandas(data, benchmark_col, windows)
        else:
            return self._calculate_beta_spark(data, benchmark_col, windows)
    
    def _calculate_beta_pandas(self, data: pd.DataFrame, benchmark_col: str, windows: List[int]) -> pd.DataFrame:
        """Calculate beta using pandas"""
        for window in windows:
            def calculate_beta(returns, benchmark):
                if len(returns) < 2 or len(benchmark) < 2:
                    return np.nan
                covariance = np.cov(returns, benchmark)[0, 1]
                benchmark_variance = np.var(benchmark)
                if benchmark_variance == 0:
                    return np.nan
                return covariance / benchmark_variance
            
            data[f'Beta_{window}'] = (
                data.groupby('Ticker').rolling(window).apply(
                    lambda x: calculate_beta(x['Return'], x[benchmark_col])
                )
            ).values
        
        return data
    
    def _calculate_beta_spark(self, data: SparkDataFrame, benchmark_col: str, windows: List[int]) -> SparkDataFrame:
        """Calculate beta using PySpark"""
        # This is a simplified implementation for PySpark
        # A more sophisticated approach would require custom UDFs for covariance calculations
        
        for window in windows:
            rolling_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-window+1, 0)
            
            # Simplified beta calculation using correlation
            data = data.withColumn(f'Beta_{window}', 
                                 col("Return") / col(benchmark_col))  # Simplified approximation
        
        return data
    
    def calculate_all_risk_metrics(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate all risk metrics
        
        Args:
            data: Data containing returns
            
        Returns:
            Data with all risk metrics
        """
        self.logger.info("Calculating all risk metrics")
        
        # Calculate all risk metrics
        data = self.calculate_volatility(data)
        data = self.calculate_sharpe_ratio(data)
        data = self.calculate_sortino_ratio(data)
        data = self.calculate_max_drawdown(data)
        data = self.calculate_var_cvar(data)
        
        # Calculate beta if benchmark data is available
        if 'SPY_Return' in data.columns:
            data = self.calculate_beta(data)
        
        return data
