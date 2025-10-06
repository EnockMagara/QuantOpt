"""
Returns calculator for financial data
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, List
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, lag, when, isnan, isnull, log, exp
from pyspark.sql.window import Window
import logging


class ReturnsCalculator:
    """
    Calculates various types of returns for financial data
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the returns calculator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.use_log_returns = config.get('preprocessing', {}).get('use_log_returns', True)
    
    def calculate_returns(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate returns for the data
        
        Args:
            data: Data containing price information
            
        Returns:
            Data with returns calculated
        """
        self.logger.info("Calculating returns")
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_returns_pandas(data)
        else:
            return self._calculate_returns_spark(data)
    
    def _calculate_returns_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns using pandas"""
        # Use adjusted close prices if available, otherwise use close
        price_col = 'Close_Adj' if 'Close_Adj' in data.columns else 'Close'
        
        if self.use_log_returns:
            # Log returns
            data['Return'] = np.log(data[price_col] / data[price_col].shift(1))
        else:
            # Simple returns
            data['Return'] = data[price_col].pct_change()
        
        # Calculate additional return metrics
        data['Return_Abs'] = abs(data['Return'])
        data['Return_Squared'] = data['Return'] ** 2
        
        return data
    
    def _calculate_returns_spark(self, data: SparkDataFrame) -> SparkDataFrame:
        """Calculate returns using PySpark"""
        # Use adjusted close prices if available, otherwise use close
        price_col = 'Close_Adj' if 'Close_Adj' in data.columns else 'Close'
        
        # Define window for lag calculations
        window_spec = Window.partitionBy("Ticker").orderBy("Date")
        
        if self.use_log_returns:
            # Log returns
            data = data.withColumn("Return", 
                                 log(col(price_col) / lag(col(price_col), 1).over(window_spec)))
        else:
            # Simple returns
            data = data.withColumn("Return", 
                                 (col(price_col) - lag(col(price_col), 1).over(window_spec)) / 
                                 lag(col(price_col), 1).over(window_spec))
        
        # Calculate additional return metrics
        data = data.withColumn("Return_Abs", abs(col("Return")))
        data = data.withColumn("Return_Squared", col("Return") ** 2)
        
        return data
    
    def calculate_yield_changes(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate yield changes for bond data
        
        Args:
            data: Bond data containing yield information
            
        Returns:
            Data with yield changes calculated
        """
        self.logger.info("Calculating yield changes")
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_yield_changes_pandas(data)
        else:
            return self._calculate_yield_changes_spark(data)
    
    def _calculate_yield_changes_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate yield changes using pandas"""
        # Absolute yield change
        data['Yield_Change'] = data['Yield'].diff()
        
        # Percentage yield change
        data['Yield_Change_Pct'] = data['Yield'].pct_change()
        
        # Log yield change
        data['Yield_Change_Log'] = np.log(data['Yield'] / data['Yield'].shift(1))
        
        return data
    
    def _calculate_yield_changes_spark(self, data: SparkDataFrame) -> SparkDataFrame:
        """Calculate yield changes using PySpark"""
        # Define window for lag calculations
        window_spec = Window.partitionBy("Ticker").orderBy("Date")
        
        # Absolute yield change
        data = data.withColumn("Yield_Change", 
                             col("Yield") - lag(col("Yield"), 1).over(window_spec))
        
        # Percentage yield change
        data = data.withColumn("Yield_Change_Pct", 
                             (col("Yield") - lag(col("Yield"), 1).over(window_spec)) / 
                             lag(col("Yield"), 1).over(window_spec))
        
        # Log yield change
        data = data.withColumn("Yield_Change_Log", 
                             log(col("Yield") / lag(col("Yield"), 1).over(window_spec)))
        
        return data
    
    def calculate_rolling_returns(self, data: Union[pd.DataFrame, SparkDataFrame], 
                                windows: List[int] = [5, 10, 21, 63]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate rolling returns for different windows
        
        Args:
            data: Data with returns
            windows: List of window sizes
            
        Returns:
            Data with rolling returns
        """
        self.logger.info(f"Calculating rolling returns for windows: {windows}")
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_rolling_returns_pandas(data, windows)
        else:
            return self._calculate_rolling_returns_spark(data, windows)
    
    def _calculate_rolling_returns_pandas(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Calculate rolling returns using pandas"""
        for window in windows:
            # Rolling mean return
            data[f'Return_Rolling_Mean_{window}'] = data.groupby('Ticker')['Return'].rolling(window).mean().values
            
            # Rolling volatility
            data[f'Return_Rolling_Std_{window}'] = data.groupby('Ticker')['Return'].rolling(window).std().values
            
            # Rolling Sharpe ratio (assuming risk-free rate = 0 for simplicity)
            data[f'Return_Rolling_Sharpe_{window}'] = (
                data[f'Return_Rolling_Mean_{window}'] / data[f'Return_Rolling_Std_{window}']
            )
        
        return data
    
    def _calculate_rolling_returns_spark(self, data: SparkDataFrame, windows: List[int]) -> SparkDataFrame:
        """Calculate rolling returns using PySpark"""
        from pyspark.sql.functions import mean, stddev
        
        for window in windows:
            # Define rolling window
            rolling_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-window+1, 0)
            
            # Rolling mean return
            data = data.withColumn(f'Return_Rolling_Mean_{window}', 
                                 mean("Return").over(rolling_window))
            
            # Rolling volatility
            data = data.withColumn(f'Return_Rolling_Std_{window}', 
                                 stddev("Return").over(rolling_window))
            
            # Rolling Sharpe ratio
            data = data.withColumn(f'Return_Rolling_Sharpe_{window}', 
                                 col(f'Return_Rolling_Mean_{window}') / col(f'Return_Rolling_Std_{window}'))
        
        return data
    
    def calculate_cumulative_returns(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate cumulative returns
        
        Args:
            data: Data with returns
            
        Returns:
            Data with cumulative returns
        """
        self.logger.info("Calculating cumulative returns")
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_cumulative_returns_pandas(data)
        else:
            return self._calculate_cumulative_returns_spark(data)
    
    def _calculate_cumulative_returns_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate cumulative returns using pandas"""
        if self.use_log_returns:
            # For log returns, cumulative return is sum of log returns
            data['Cumulative_Return'] = data.groupby('Ticker')['Return'].cumsum()
        else:
            # For simple returns, cumulative return is product of (1 + returns)
            data['Cumulative_Return'] = (1 + data.groupby('Ticker')['Return']).cumprod() - 1
        
        return data
    
    def _calculate_cumulative_returns_spark(self, data: SparkDataFrame) -> SparkDataFrame:
        """Calculate cumulative returns using PySpark"""
        from pyspark.sql.functions import sum as spark_sum, product
        
        window_spec = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(Window.unboundedPreceding, 0)
        
        if self.use_log_returns:
            # For log returns, cumulative return is sum of log returns
            data = data.withColumn("Cumulative_Return", 
                                 spark_sum("Return").over(window_spec))
        else:
            # For simple returns, cumulative return is product of (1 + returns)
            data = data.withColumn("Cumulative_Return", 
                                 product(1 + col("Return")).over(window_spec) - 1)
        
        return data
    
    def calculate_risk_metrics(self, data: Union[pd.DataFrame, SparkDataFrame], 
                             risk_free_rate: float = 0.02) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate risk metrics like Sharpe ratio, Sortino ratio, etc.
        
        Args:
            data: Data with returns
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Data with risk metrics
        """
        self.logger.info("Calculating risk metrics")
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_risk_metrics_pandas(data, risk_free_rate)
        else:
            return self._calculate_risk_metrics_spark(data, risk_free_rate)
    
    def _calculate_risk_metrics_pandas(self, data: pd.DataFrame, risk_free_rate: float) -> pd.DataFrame:
        """Calculate risk metrics using pandas"""
        # Daily risk-free rate
        daily_rf_rate = risk_free_rate / 252
        
        # Excess returns
        data['Excess_Return'] = data['Return'] - daily_rf_rate
        
        # Rolling Sharpe ratio
        data['Sharpe_Ratio'] = (
            data.groupby('Ticker')['Excess_Return'].rolling(21).mean() / 
            data.groupby('Ticker')['Return'].rolling(21).std()
        ).values
        
        # Rolling Sortino ratio (downside deviation)
        negative_returns = data['Return'].where(data['Return'] < 0)
        data['Sortino_Ratio'] = (
            data.groupby('Ticker')['Excess_Return'].rolling(21).mean() / 
            data.groupby('Ticker')['Return'].rolling(21).apply(lambda x: x[x < 0].std())
        ).values
        
        return data
    
    def _calculate_risk_metrics_spark(self, data: SparkDataFrame, risk_free_rate: float) -> SparkDataFrame:
        """Calculate risk metrics using PySpark"""
        from pyspark.sql.functions import mean, stddev, when, col
        
        # Daily risk-free rate
        daily_rf_rate = risk_free_rate / 252
        
        # Excess returns
        data = data.withColumn("Excess_Return", col("Return") - daily_rf_rate)
        
        # Rolling window for 21 days
        rolling_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-20, 0)
        
        # Rolling Sharpe ratio
        data = data.withColumn("Sharpe_Ratio", 
                             mean("Excess_Return").over(rolling_window) / 
                             stddev("Return").over(rolling_window))
        
        # Rolling Sortino ratio (simplified - would need custom UDF for downside deviation)
        data = data.withColumn("Sortino_Ratio", 
                             mean("Excess_Return").over(rolling_window) / 
                             stddev(when(col("Return") < 0, col("Return")).otherwise(0)).over(rolling_window))
        
        return data
