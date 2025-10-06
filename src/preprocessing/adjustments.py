"""
Adjustments processor for handling stock splits and dividends
"""

import pandas as pd
import numpy as np
from typing import Dict, Union
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, when, lag, sum as spark_sum
from pyspark.sql.window import Window
import logging


class AdjustmentsProcessor:
    """
    Handles adjustments for stock splits and dividends
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the adjustments processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def adjust_for_splits(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Adjust prices for stock splits
        
        Args:
            data: Data containing OHLCV information
            
        Returns:
            Data with split-adjusted prices
        """
        self.logger.info("Adjusting prices for stock splits")
        
        if isinstance(data, pd.DataFrame):
            return self._adjust_splits_pandas(data)
        else:
            return self._adjust_splits_spark(data)
    
    def _adjust_splits_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adjust splits using pandas"""
        # Check if Adj Close column exists
        if 'Adj Close' in data.columns:
            # Calculate split ratio from adjusted close vs close
            data['Split_Ratio'] = data['Adj Close'] / data['Close']
        else:
            # If no Adj Close, assume no splits (ratio = 1)
            data['Split_Ratio'] = 1.0
        
        # Apply split adjustment to OHLCV
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            data[f'{col}_Split_Adj'] = data[col] * data['Split_Ratio']
        
        # Adjust volume (divide by split ratio)
        data['Volume_Split_Adj'] = data['Volume'] / data['Split_Ratio']
        
        return data
    
    def _adjust_splits_spark(self, data: SparkDataFrame) -> SparkDataFrame:
        """Adjust splits using PySpark"""
        # Calculate split ratio
        data = data.withColumn("Split_Ratio", col("Adj Close") / col("Close"))
        
        # Apply split adjustment to OHLCV
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col_name in price_columns:
            data = data.withColumn(f'{col_name}_Split_Adj', col(col_name) * col("Split_Ratio"))
        
        # Adjust volume
        data = data.withColumn("Volume_Split_Adj", col("Volume") / col("Split_Ratio"))
        
        return data
    
    def adjust_for_dividends(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Adjust prices for dividends
        
        Args:
            data: Data containing OHLCV information
            
        Returns:
            Data with dividend-adjusted prices
        """
        self.logger.info("Adjusting prices for dividends")
        
        if isinstance(data, pd.DataFrame):
            return self._adjust_dividends_pandas(data)
        else:
            return self._adjust_dividends_spark(data)
    
    def _adjust_dividends_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adjust dividends using pandas"""
        # Check if Adj Close column exists
        if 'Adj Close' in data.columns:
            # Calculate dividend adjustment factor
            data['Dividend_Adj_Factor'] = data['Adj Close'] / data['Close']
        else:
            # If no Adj Close, assume no dividends (factor = 1)
            data['Dividend_Adj_Factor'] = 1.0
        
        # Apply dividend adjustment to OHLCV
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            data[f'{col}_Dividend_Adj'] = data[col] * data['Dividend_Adj_Factor']
        
        return data
    
    def _adjust_dividends_spark(self, data: SparkDataFrame) -> SparkDataFrame:
        """Adjust dividends using PySpark"""
        # Calculate dividend adjustment factor
        data = data.withColumn("Dividend_Adj_Factor", col("Adj Close") / col("Close"))
        
        # Apply dividend adjustment to OHLCV
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col_name in price_columns:
            data = data.withColumn(f'{col_name}_Dividend_Adj', col(col_name) * col("Dividend_Adj_Factor"))
        
        return data
    
    def calculate_adjustment_factors(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate total adjustment factors for both splits and dividends
        
        Args:
            data: Data containing OHLCV information
            
        Returns:
            Data with total adjustment factors
        """
        self.logger.info("Calculating total adjustment factors")
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_factors_pandas(data)
        else:
            return self._calculate_factors_spark(data)
    
    def _calculate_factors_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate adjustment factors using pandas"""
        # Check if Adj Close column exists
        if 'Adj Close' in data.columns:
            # Total adjustment factor
            data['Total_Adj_Factor'] = data['Adj Close'] / data['Close']
        else:
            # If no Adj Close, assume no adjustments (factor = 1)
            data['Total_Adj_Factor'] = 1.0
        
        # Split adjustment factor
        data['Split_Adj_Factor'] = data['Split_Ratio'] if 'Split_Ratio' in data.columns else 1.0
        
        # Dividend adjustment factor
        data['Dividend_Adj_Factor'] = data['Total_Adj_Factor'] / data['Split_Adj_Factor']
        
        return data
    
    def _calculate_factors_spark(self, data: SparkDataFrame) -> SparkDataFrame:
        """Calculate adjustment factors using PySpark"""
        # Total adjustment factor
        data = data.withColumn("Total_Adj_Factor", col("Adj Close") / col("Close"))
        
        # Split adjustment factor
        data = data.withColumn("Split_Adj_Factor", 
                             when(col("Split_Ratio").isNull(), 1.0).otherwise(col("Split_Ratio")))
        
        # Dividend adjustment factor
        data = data.withColumn("Dividend_Adj_Factor", 
                             col("Total_Adj_Factor") / col("Split_Adj_Factor"))
        
        return data
    
    def apply_adjustments(self, data: Union[pd.DataFrame, SparkDataFrame], 
                         use_adj_close: bool = True) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Apply all adjustments to the data
        
        Args:
            data: Raw data
            use_adj_close: Whether to use adjusted close prices
            
        Returns:
            Fully adjusted data
        """
        self.logger.info("Applying all adjustments to data")
        
        if use_adj_close and 'Adj Close' in data.columns:
            # Use adjusted close prices as the base
            if isinstance(data, pd.DataFrame):
                data['Close_Adj'] = data['Adj Close']
                data['Open_Adj'] = data['Open'] * (data['Adj Close'] / data['Close'])
                data['High_Adj'] = data['High'] * (data['Adj Close'] / data['Close'])
                data['Low_Adj'] = data['Low'] * (data['Adj Close'] / data['Close'])
            else:
                data = data.withColumn("Close_Adj", col("Adj Close"))
                data = data.withColumn("Open_Adj", col("Open") * (col("Adj Close") / col("Close")))
                data = data.withColumn("High_Adj", col("High") * (col("Adj Close") / col("Close")))
                data = data.withColumn("Low_Adj", col("Low") * (col("Adj Close") / col("Close")))
        elif use_adj_close and 'Adj Close' not in data.columns:
            # If Adj Close not available, use Close as is
            if isinstance(data, pd.DataFrame):
                data['Close_Adj'] = data['Close']
                data['Open_Adj'] = data['Open']
                data['High_Adj'] = data['High']
                data['Low_Adj'] = data['Low']
            else:
                data = data.withColumn("Close_Adj", col("Close"))
                data = data.withColumn("Open_Adj", col("Open"))
                data = data.withColumn("High_Adj", col("High"))
                data = data.withColumn("Low_Adj", col("Low"))
        else:
            # Apply manual adjustments
            data = self.adjust_for_splits(data)
            data = self.adjust_for_dividends(data)
            
            if isinstance(data, pd.DataFrame):
                data['Close_Adj'] = data['Close_Dividend_Adj']
                data['Open_Adj'] = data['Open_Dividend_Adj']
                data['High_Adj'] = data['High_Dividend_Adj']
                data['Low_Adj'] = data['Low_Dividend_Adj']
            else:
                data = data.withColumn("Close_Adj", col("Close_Dividend_Adj"))
                data = data.withColumn("Open_Adj", col("Open_Dividend_Adj"))
                data = data.withColumn("High_Adj", col("High_Dividend_Adj"))
                data = data.withColumn("Low_Adj", col("Low_Dividend_Adj"))
        
        return data
