"""
Missing data handler for financial data
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, List, Optional
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, when, isnan, isnull, coalesce, last, first
from pyspark.sql.window import Window
import logging


class MissingDataHandler:
    """
    Handles missing data in financial datasets
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the missing data handler
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.strategy = config.get('preprocessing', {}).get('missing_data_strategy', 'forward_fill')
    
    def handle_missing_data(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Handle missing data in the dataset
        
        Args:
            data: Data with potential missing values
            
        Returns:
            Data with missing values handled
        """
        self.logger.info(f"Handling missing data using strategy: {self.strategy}")
        
        if isinstance(data, pd.DataFrame):
            return self._handle_missing_data_pandas(data)
        else:
            return self._handle_missing_data_spark(data)
    
    def _handle_missing_data_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data using pandas"""
        # Log missing data statistics
        missing_stats = data.isnull().sum()
        self.logger.info(f"Missing data statistics:\n{missing_stats}")
        
        # Handle missing data based on strategy
        if self.strategy == 'forward_fill':
            data = data.fillna(method='ffill')
        elif self.strategy == 'backward_fill':
            data = data.fillna(method='bfill')
        elif self.strategy == 'interpolate':
            data = data.interpolate(method='linear')
        elif self.strategy == 'drop':
            data = data.dropna()
        elif self.strategy == 'zero_fill':
            data = data.fillna(0)
        else:
            self.logger.warning(f"Unknown strategy: {self.strategy}, using forward_fill")
            data = data.fillna(method='ffill')
        
        # Handle remaining missing values (e.g., at the beginning of time series)
        data = data.fillna(method='bfill')
        
        return data
    
    def _handle_missing_data_spark(self, data: SparkDataFrame) -> SparkDataFrame:
        """Handle missing data using PySpark"""
        # Get column names
        columns = data.columns
        
        # Handle missing data based on strategy
        if self.strategy == 'forward_fill':
            data = self._forward_fill_spark(data, columns)
        elif self.strategy == 'backward_fill':
            data = self._backward_fill_spark(data, columns)
        elif self.strategy == 'interpolate':
            data = self._interpolate_spark(data, columns)
        elif self.strategy == 'drop':
            data = data.dropna()
        elif self.strategy == 'zero_fill':
            data = data.fillna(0)
        else:
            self.logger.warning(f"Unknown strategy: {self.strategy}, using forward_fill")
            data = self._forward_fill_spark(data, columns)
        
        return data
    
    def _forward_fill_spark(self, data: SparkDataFrame, columns: List[str]) -> SparkDataFrame:
        """Forward fill missing values using PySpark"""
        window_spec = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(Window.unboundedPreceding, 0)
        
        for col_name in columns:
            if col_name not in ['Date', 'Ticker']:
                data = data.withColumn(col_name, 
                                     last(col_name, ignorenulls=True).over(window_spec))
        
        return data
    
    def _backward_fill_spark(self, data: SparkDataFrame, columns: List[str]) -> SparkDataFrame:
        """Backward fill missing values using PySpark"""
        window_spec = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(0, Window.unboundedFollowing)
        
        for col_name in columns:
            if col_name not in ['Date', 'Ticker']:
                data = data.withColumn(col_name, 
                                     first(col_name, ignorenulls=True).over(window_spec))
        
        return data
    
    def _interpolate_spark(self, data: SparkDataFrame, columns: List[str]) -> SparkDataFrame:
        """Interpolate missing values using PySpark (simplified linear interpolation)"""
        # For PySpark, we'll use a combination of forward and backward fill
        # as a proxy for linear interpolation
        data = self._forward_fill_spark(data, columns)
        data = self._backward_fill_spark(data, columns)
        
        return data
    
    def detect_missing_patterns(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Dict:
        """
        Detect patterns in missing data
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary with missing data patterns
        """
        self.logger.info("Detecting missing data patterns")
        
        if isinstance(data, pd.DataFrame):
            return self._detect_patterns_pandas(data)
        else:
            return self._detect_patterns_spark(data)
    
    def _detect_patterns_pandas(self, data: pd.DataFrame) -> Dict:
        """Detect missing data patterns using pandas"""
        patterns = {}
        
        # Overall missing data statistics
        patterns['total_missing'] = data.isnull().sum().sum()
        patterns['missing_percentage'] = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
        
        # Missing data by column
        patterns['missing_by_column'] = data.isnull().sum().to_dict()
        
        # Missing data by ticker
        if 'Ticker' in data.columns:
            patterns['missing_by_ticker'] = data.groupby('Ticker').apply(lambda x: x.isnull().sum().sum()).to_dict()
        
        # Missing data by date
        if 'Date' in data.columns:
            patterns['missing_by_date'] = data.groupby('Date').apply(lambda x: x.isnull().sum().sum()).to_dict()
        
        return patterns
    
    def _detect_patterns_spark(self, data: SparkDataFrame) -> Dict:
        """Detect missing data patterns using PySpark"""
        patterns = {}
        
        # Get column names
        columns = data.columns
        
        # Count total missing values
        total_rows = data.count()
        total_cells = total_rows * len(columns)
        
        missing_counts = {}
        for col_name in columns:
            if col_name not in ['Date', 'Ticker']:
                missing_count = data.filter(col(col_name).isNull()).count()
                missing_counts[col_name] = missing_count
        
        patterns['total_missing'] = sum(missing_counts.values())
        patterns['missing_percentage'] = (patterns['total_missing'] / total_cells) * 100
        patterns['missing_by_column'] = missing_counts
        
        return patterns
    
    def handle_trading_days(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Handle missing trading days (weekends, holidays)
        
        Args:
            data: Data with potential missing trading days
            
        Returns:
            Data with missing trading days handled
        """
        self.logger.info("Handling missing trading days")
        
        if isinstance(data, pd.DataFrame):
            return self._handle_trading_days_pandas(data)
        else:
            return self._handle_trading_days_spark(data)
    
    def _handle_trading_days_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing trading days using pandas"""
        # Convert Date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data['Date']):
            data['Date'] = pd.to_datetime(data['Date'])
        
        # Create a complete date range for each ticker
        complete_data = []
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].copy()
            
            # Create complete date range (business days only)
            start_date = ticker_data['Date'].min()
            end_date = ticker_data['Date'].max()
            complete_dates = pd.bdate_range(start=start_date, end=end_date, freq='B')
            
            # Reindex to complete date range
            ticker_data = ticker_data.set_index('Date').reindex(complete_dates)
            ticker_data['Ticker'] = ticker
            ticker_data = ticker_data.reset_index()
            ticker_data = ticker_data.rename(columns={'index': 'Date'})
            
            complete_data.append(ticker_data)
        
        # Combine all tickers
        data = pd.concat(complete_data, ignore_index=True)
        
        # Fill missing values for new trading days
        data = self._handle_missing_data_pandas(data)
        
        return data
    
    def _handle_trading_days_spark(self, data: SparkDataFrame) -> SparkDataFrame:
        """Handle missing trading days using PySpark"""
        # This is a simplified version for PySpark
        # In practice, you might want to use a more sophisticated approach
        
        # For now, we'll just handle missing data without creating new trading days
        # as PySpark doesn't have built-in business day functionality
        data = self._handle_missing_data_spark(data)
        
        return data
    
    def validate_data_quality(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Dict:
        """
        Validate data quality after handling missing data
        
        Args:
            data: Processed data
            
        Returns:
            Dictionary with data quality metrics
        """
        self.logger.info("Validating data quality")
        
        if isinstance(data, pd.DataFrame):
            return self._validate_quality_pandas(data)
        else:
            return self._validate_quality_spark(data)
    
    def _validate_quality_pandas(self, data: pd.DataFrame) -> Dict:
        """Validate data quality using pandas"""
        quality_metrics = {}
        
        # Check for remaining missing values
        quality_metrics['remaining_missing'] = data.isnull().sum().sum()
        quality_metrics['missing_percentage'] = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
        
        # Check for infinite values
        quality_metrics['infinite_values'] = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
        
        # Check for negative prices (if applicable)
        price_columns = ['Open', 'High', 'Low', 'Close', 'Open_Adj', 'High_Adj', 'Low_Adj', 'Close_Adj']
        negative_prices = 0
        for col in price_columns:
            if col in data.columns:
                negative_prices += (data[col] < 0).sum()
        quality_metrics['negative_prices'] = negative_prices
        
        # Check for zero volumes (if applicable)
        if 'Volume' in data.columns:
            quality_metrics['zero_volumes'] = (data['Volume'] == 0).sum()
        
        return quality_metrics
    
    def _validate_quality_spark(self, data: SparkDataFrame) -> Dict:
        """Validate data quality using PySpark"""
        quality_metrics = {}
        
        # Count total rows
        total_rows = data.count()
        
        # Check for remaining missing values
        columns = data.columns
        remaining_missing = 0
        for col_name in columns:
            if col_name not in ['Date', 'Ticker']:
                missing_count = data.filter(col(col_name).isNull()).count()
                remaining_missing += missing_count
        
        quality_metrics['remaining_missing'] = remaining_missing
        quality_metrics['missing_percentage'] = (remaining_missing / (total_rows * len(columns))) * 100
        
        return quality_metrics
