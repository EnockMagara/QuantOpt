"""
Technical indicators calculator for financial data
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, List, Optional
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, when, lag, sum as spark_sum, mean, stddev, max as spark_max, min as spark_min
from pyspark.sql.window import Window
import logging

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.info("TA-Lib not available, using custom implementations")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logging.info("pandas-ta not available, using custom implementations")


class TechnicalIndicators:
    """
    Calculates various technical indicators for financial data
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the technical indicators calculator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.technical_config = config.get('features', {}).get('technical', {})
    
    def calculate_macd(self, data: Union[pd.DataFrame, SparkDataFrame], 
                      fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Data containing price information
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            Data with MACD indicators
        """
        self.logger.info(f"Calculating MACD with periods: {fast_period}, {slow_period}, {signal_period}")
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_macd_pandas(data, fast_period, slow_period, signal_period)
        else:
            return self._calculate_macd_spark(data, fast_period, slow_period, signal_period)
    
    def _calculate_macd_pandas(self, data: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int) -> pd.DataFrame:
        """Calculate MACD using pandas"""
        price_col = 'Close_Adj' if 'Close_Adj' in data.columns else 'Close'
        
        if TALIB_AVAILABLE:
            # Use TA-Lib for more accurate calculations
            for ticker in data['Ticker'].unique():
                ticker_data = data[data['Ticker'] == ticker]
                macd, signal, histogram = talib.MACD(ticker_data[price_col].values, 
                                                   fastperiod=fast_period, 
                                                   slowperiod=slow_period, 
                                                   signalperiod=signal_period)
                
                mask = data['Ticker'] == ticker
                data.loc[mask, 'MACD'] = macd
                data.loc[mask, 'MACD_Signal'] = signal
                data.loc[mask, 'MACD_Histogram'] = histogram
        else:
            # Custom implementation
            for ticker in data['Ticker'].unique():
                ticker_data = data[data['Ticker'] == ticker].copy()
                
                # Calculate EMAs
                ema_fast = ticker_data[price_col].ewm(span=fast_period).mean()
                ema_slow = ticker_data[price_col].ewm(span=slow_period).mean()
                
                # MACD line
                macd = ema_fast - ema_slow
                
                # Signal line
                signal = macd.ewm(span=signal_period).mean()
                
                # Histogram
                histogram = macd - signal
                
                mask = data['Ticker'] == ticker
                data.loc[mask, 'MACD'] = macd.values
                data.loc[mask, 'MACD_Signal'] = signal.values
                data.loc[mask, 'MACD_Histogram'] = histogram.values
        
        return data
    
    def _calculate_macd_spark(self, data: SparkDataFrame, fast_period: int, slow_period: int, signal_period: int) -> SparkDataFrame:
        """Calculate MACD using PySpark"""
        price_col = 'Close_Adj' if 'Close_Adj' in data.columns else 'Close'
        
        # Define window for calculations
        window_spec = Window.partitionBy("Ticker").orderBy("Date")
        
        # Calculate EMAs (simplified using rolling averages as proxy)
        data = data.withColumn("EMA_Fast", 
                             mean(col(price_col)).over(Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-fast_period+1, 0)))
        data = data.withColumn("EMA_Slow", 
                             mean(col(price_col)).over(Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-slow_period+1, 0)))
        
        # MACD line
        data = data.withColumn("MACD", col("EMA_Fast") - col("EMA_Slow"))
        
        # Signal line
        data = data.withColumn("MACD_Signal", 
                             mean(col("MACD")).over(Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-signal_period+1, 0)))
        
        # Histogram
        data = data.withColumn("MACD_Histogram", col("MACD") - col("MACD_Signal"))
        
        return data
    
    def calculate_rsi(self, data: Union[pd.DataFrame, SparkDataFrame], period: int = 14) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate RSI (Relative Strength Index)
        
        Args:
            data: Data containing price information
            period: RSI calculation period
            
        Returns:
            Data with RSI indicator
        """
        self.logger.info(f"Calculating RSI with period: {period}")
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_rsi_pandas(data, period)
        else:
            return self._calculate_rsi_spark(data, period)
    
    def _calculate_rsi_pandas(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """Calculate RSI using pandas"""
        price_col = 'Close_Adj' if 'Close_Adj' in data.columns else 'Close'
        
        if TALIB_AVAILABLE:
            # Use TA-Lib for more accurate calculations
            for ticker in data['Ticker'].unique():
                ticker_data = data[data['Ticker'] == ticker]
                rsi = talib.RSI(ticker_data[price_col].values, timeperiod=period)
                
                mask = data['Ticker'] == ticker
                data.loc[mask, 'RSI'] = rsi
        else:
            # Custom implementation
            for ticker in data['Ticker'].unique():
                ticker_data = data[data['Ticker'] == ticker].copy()
                
                # Calculate price changes
                delta = ticker_data[price_col].diff()
                
                # Separate gains and losses
                gains = delta.where(delta > 0, 0)
                losses = -delta.where(delta < 0, 0)
                
                # Calculate average gains and losses
                avg_gains = gains.rolling(window=period).mean()
                avg_losses = losses.rolling(window=period).mean()
                
                # Calculate RSI
                rs = avg_gains / avg_losses
                rsi = 100 - (100 / (1 + rs))
                
                mask = data['Ticker'] == ticker
                data.loc[mask, 'RSI'] = rsi.values
        
        return data
    
    def _calculate_rsi_spark(self, data: SparkDataFrame, period: int) -> SparkDataFrame:
        """Calculate RSI using PySpark"""
        price_col = 'Close_Adj' if 'Close_Adj' in data.columns else 'Close'
        
        # Define window for calculations
        window_spec = Window.partitionBy("Ticker").orderBy("Date")
        
        # Calculate price changes
        data = data.withColumn("Price_Change", col(price_col) - lag(col(price_col), 1).over(window_spec))
        
        # Separate gains and losses
        data = data.withColumn("Gains", when(col("Price_Change") > 0, col("Price_Change")).otherwise(0))
        data = data.withColumn("Losses", when(col("Price_Change") < 0, -col("Price_Change")).otherwise(0))
        
        # Calculate average gains and losses
        rolling_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-period+1, 0)
        data = data.withColumn("Avg_Gains", mean("Gains").over(rolling_window))
        data = data.withColumn("Avg_Losses", mean("Losses").over(rolling_window))
        
        # Calculate RSI
        data = data.withColumn("RS", col("Avg_Gains") / col("Avg_Losses"))
        data = data.withColumn("RSI", 100 - (100 / (1 + col("RS"))))
        
        return data
    
    def calculate_bollinger_bands(self, data: Union[pd.DataFrame, SparkDataFrame], 
                                 period: int = 20, std_dev: float = 2.0) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate Bollinger Bands
        
        Args:
            data: Data containing price information
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Data with Bollinger Bands
        """
        self.logger.info(f"Calculating Bollinger Bands with period: {period}, std_dev: {std_dev}")
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_bollinger_pandas(data, period, std_dev)
        else:
            return self._calculate_bollinger_spark(data, period, std_dev)
    
    def _calculate_bollinger_pandas(self, data: pd.DataFrame, period: int, std_dev: float) -> pd.DataFrame:
        """Calculate Bollinger Bands using pandas"""
        price_col = 'Close_Adj' if 'Close_Adj' in data.columns else 'Close'
        
        if TALIB_AVAILABLE:
            # Use TA-Lib for more accurate calculations
            for ticker in data['Ticker'].unique():
                ticker_data = data[data['Ticker'] == ticker]
                upper, middle, lower = talib.BBANDS(ticker_data[price_col].values, 
                                                  timeperiod=period, 
                                                  nbdevup=std_dev, 
                                                  nbdevdn=std_dev)
                
                mask = data['Ticker'] == ticker
                data.loc[mask, 'BB_Upper'] = upper
                data.loc[mask, 'BB_Middle'] = middle
                data.loc[mask, 'BB_Lower'] = lower
        else:
            # Custom implementation
            for ticker in data['Ticker'].unique():
                ticker_data = data[data['Ticker'] == ticker].copy()
                
                # Calculate moving average
                middle = ticker_data[price_col].rolling(window=period).mean()
                
                # Calculate standard deviation
                std = ticker_data[price_col].rolling(window=period).std()
                
                # Calculate upper and lower bands
                upper = middle + (std * std_dev)
                lower = middle - (std * std_dev)
                
                mask = data['Ticker'] == ticker
                data.loc[mask, 'BB_Upper'] = upper.values
                data.loc[mask, 'BB_Middle'] = middle.values
                data.loc[mask, 'BB_Lower'] = lower.values
        
        # Calculate Bollinger Band position
        data['BB_Position'] = (data[price_col] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        return data
    
    def _calculate_bollinger_spark(self, data: SparkDataFrame, period: int, std_dev: float) -> SparkDataFrame:
        """Calculate Bollinger Bands using PySpark"""
        price_col = 'Close_Adj' if 'Close_Adj' in data.columns else 'Close'
        
        # Define rolling window
        rolling_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-period+1, 0)
        
        # Calculate moving average and standard deviation
        data = data.withColumn("BB_Middle", mean(col(price_col)).over(rolling_window))
        data = data.withColumn("BB_Std", stddev(col(price_col)).over(rolling_window))
        
        # Calculate upper and lower bands
        data = data.withColumn("BB_Upper", col("BB_Middle") + (col("BB_Std") * std_dev))
        data = data.withColumn("BB_Lower", col("BB_Middle") - (col("BB_Std") * std_dev))
        
        # Calculate Bollinger Band position
        data = data.withColumn("BB_Position", 
                             (col(price_col) - col("BB_Lower")) / (col("BB_Upper") - col("BB_Lower")))
        
        return data
    
    def calculate_moving_averages(self, data: Union[pd.DataFrame, SparkDataFrame], 
                                 periods: List[int] = [5, 10, 20, 50, 200]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate moving averages for different periods
        
        Args:
            data: Data containing price information
            periods: List of periods for moving averages
            
        Returns:
            Data with moving averages
        """
        self.logger.info(f"Calculating moving averages for periods: {periods}")
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_moving_averages_pandas(data, periods)
        else:
            return self._calculate_moving_averages_spark(data, periods)
    
    def _calculate_moving_averages_pandas(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate moving averages using pandas"""
        price_col = 'Close_Adj' if 'Close_Adj' in data.columns else 'Close'
        
        for period in periods:
            data[f'MA_{period}'] = data.groupby('Ticker')[price_col].rolling(window=period).mean().values
            
            # Calculate price relative to moving average
            data[f'Price_MA_{period}_Ratio'] = data[price_col] / data[f'MA_{period}']
        
        return data
    
    def _calculate_moving_averages_spark(self, data: SparkDataFrame, periods: List[int]) -> SparkDataFrame:
        """Calculate moving averages using PySpark"""
        price_col = 'Close_Adj' if 'Close_Adj' in data.columns else 'Close'
        
        for period in periods:
            rolling_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-period+1, 0)
            
            data = data.withColumn(f'MA_{period}', mean(col(price_col)).over(rolling_window))
            data = data.withColumn(f'Price_MA_{period}_Ratio', col(price_col) / col(f'MA_{period}'))
        
        return data
    
    def calculate_volume_indicators(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate volume-based indicators
        
        Args:
            data: Data containing volume information
            
        Returns:
            Data with volume indicators
        """
        self.logger.info("Calculating volume indicators")
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_volume_indicators_pandas(data)
        else:
            return self._calculate_volume_indicators_spark(data)
    
    def _calculate_volume_indicators_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume indicators using pandas"""
        if 'Volume' not in data.columns:
            self.logger.warning("Volume column not found, skipping volume indicators")
            return data
        
        # Volume moving averages
        for period in [10, 20, 50]:
            data[f'Volume_MA_{period}'] = data.groupby('Ticker')['Volume'].rolling(window=period).mean().values
        
        # Volume ratio
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_20']
        
        # Price-Volume trend
        data['PVT'] = (data['Return'] * data['Volume']).groupby(data['Ticker']).cumsum()
        
        # On-Balance Volume (simplified)
        data['OBV'] = data.groupby('Ticker').apply(
            lambda x: (x['Volume'] * np.sign(x['Return'])).cumsum()
        ).values
        
        return data
    
    def _calculate_volume_indicators_spark(self, data: SparkDataFrame) -> SparkDataFrame:
        """Calculate volume indicators using PySpark"""
        if 'Volume' not in data.columns:
            self.logger.warning("Volume column not found, skipping volume indicators")
            return data
        
        # Volume moving averages
        for period in [10, 20, 50]:
            rolling_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-period+1, 0)
            data = data.withColumn(f'Volume_MA_{period}', mean("Volume").over(rolling_window))
        
        # Volume ratio
        data = data.withColumn("Volume_Ratio", col("Volume") / col("Volume_MA_20"))
        
        # Price-Volume trend (simplified)
        window_spec = Window.partitionBy("Ticker").orderBy("Date")
        data = data.withColumn("PVT", spark_sum(col("Return") * col("Volume")).over(window_spec))
        
        return data
    
    def calculate_all_indicators(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate all technical indicators
        
        Args:
            data: Data containing price and volume information
            
        Returns:
            Data with all technical indicators
        """
        self.logger.info("Calculating all technical indicators")
        
        # Get configuration
        macd_config = self.technical_config.get('macd', {})
        rsi_config = self.technical_config.get('rsi', {})
        bb_config = self.technical_config.get('bollinger_bands', {})
        
        # Calculate indicators
        data = self.calculate_macd(data, 
                                 macd_config.get('fast_period', 12),
                                 macd_config.get('slow_period', 26),
                                 macd_config.get('signal_period', 9))
        
        data = self.calculate_rsi(data, rsi_config.get('period', 14))
        
        data = self.calculate_bollinger_bands(data,
                                            bb_config.get('period', 20),
                                            bb_config.get('std_dev', 2.0))
        
        data = self.calculate_moving_averages(data, [5, 10, 20, 50, 200])
        
        data = self.calculate_volume_indicators(data)
        
        return data
