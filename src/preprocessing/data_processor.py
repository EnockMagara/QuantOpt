"""
Main data processor for financial data preprocessing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.functions import col, when, isnan, isnull, coalesce
import logging

from .adjustments import AdjustmentsProcessor
from .returns_calculator import ReturnsCalculator
from .missing_data_handler import MissingDataHandler


class DataProcessor:
    """
    Main data processor that orchestrates all preprocessing steps
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the data processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize sub-processors
        self.adjustments_processor = AdjustmentsProcessor(config)
        self.returns_calculator = ReturnsCalculator(config)
        self.missing_data_handler = MissingDataHandler(config)
        
        # Initialize Spark session
        self.spark = self._init_spark()
    
    def _init_spark(self) -> Optional[SparkSession]:
        """Initialize Spark session"""
        try:
            spark_config = self.config.get('spark', {})
            
            return SparkSession.builder \
                .appName(spark_config.get('app_name', 'MMS Finance ML')) \
                .master(spark_config.get('master', 'local[*]')) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .getOrCreate()
        except Exception as e:
            self.logger.warning(f"Failed to initialize Spark session: {e}")
            self.logger.info("Falling back to pandas-only processing")
            return None
    
    def process_assets_data(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Process assets data through all preprocessing steps
        
        Args:
            data: Raw assets data
            
        Returns:
            Processed data
        """
        self.logger.info("Starting assets data processing")
        
        # Convert to Spark DataFrame if needed and Spark is available
        if isinstance(data, pd.DataFrame) and self.spark is not None:
            data = self.spark.createDataFrame(data)
        
        # Step 1: Handle missing data
        data = self.missing_data_handler.handle_missing_data(data)
        
        # Step 2: Apply adjustments (splits and dividends)
        if self.config.get('preprocessing', {}).get('adjust_splits', True):
            data = self.adjustments_processor.adjust_for_splits(data)
        
        if self.config.get('preprocessing', {}).get('adjust_dividends', True):
            data = self.adjustments_processor.adjust_for_dividends(data)
        
        # Step 3: Calculate returns
        data = self.returns_calculator.calculate_returns(data)
        
        # Step 4: Add additional features
        data = self._add_basic_features(data)
        
        self.logger.info("Assets data processing completed")
        return data
    
    def process_bonds_data(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Process bonds data through all preprocessing steps
        
        Args:
            data: Raw bonds data
            
        Returns:
            Processed data
        """
        self.logger.info("Starting bonds data processing")
        
        # Convert to Spark DataFrame if needed and Spark is available
        if isinstance(data, pd.DataFrame) and self.spark is not None:
            data = self.spark.createDataFrame(data)
        
        # Step 1: Handle missing data
        data = self.missing_data_handler.handle_missing_data(data)
        
        # Step 2: Calculate yield changes (if not already present)
        if 'Yield_Change' not in data.columns:
            data = self.returns_calculator.calculate_yield_changes(data)
        
        # Step 3: Add lagged features (if not already present)
        if 'Yield_Change_Lag1' not in data.columns:
            data = self._add_lagged_features(data, 'Yield_Change', lags=[1, 2, 3, 4, 5])
        
        self.logger.info("Bonds data processing completed")
        return data
    
    def _add_basic_features(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """Add basic features to the data"""
        if isinstance(data, pd.DataFrame):
            return self._add_basic_features_pandas(data)
        else:
            return self._add_basic_features_spark(data)
    
    def _add_basic_features_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic features using pandas"""
        # Add lagged returns
        for lag_days in range(1, 6):
            data[f'Return_Lag{lag_days}'] = data.groupby('Ticker')['Return'].shift(lag_days)
        
        # Add rolling volatility (21-day window)
        data['Volatility'] = data.groupby('Ticker')['Return'].rolling(21).std().values
        
        # Add rolling mean return
        data['Rolling_Mean_Return'] = data.groupby('Ticker')['Return'].rolling(21).mean().values
        
        # Add rolling Sharpe ratio
        data['Rolling_Sharpe'] = data['Rolling_Mean_Return'] / data['Volatility']
        
        # Add cumulative returns
        data['Cumulative_Return'] = (1 + data['Return']).groupby(data['Ticker']).cumprod() - 1
        
        return data
    
    def _add_basic_features_spark(self, data: SparkDataFrame) -> SparkDataFrame:
        """Add basic features using PySpark"""
        from pyspark.sql.functions import lag, stddev, mean, max as spark_max, min as spark_min
        from pyspark.sql.window import Window
        
        # Define window for rolling calculations
        window_spec = Window.partitionBy("Ticker").orderBy("Date")
        
        # Add lagged returns
        for lag_days in range(1, 6):
            data = data.withColumn(f"Return_Lag{lag_days}", lag("Return", lag_days).over(window_spec))
        
        # Add rolling volatility (21-day window)
        volatility_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-20, 0)
        data = data.withColumn("Volatility", stddev("Return").over(volatility_window))
        
        # Add rolling mean return
        data = data.withColumn("Rolling_Mean_Return", mean("Return").over(volatility_window))
        
        # Add rolling Sharpe ratio
        data = data.withColumn("Rolling_Sharpe", 
                             col("Rolling_Mean_Return") / col("Volatility"))
        
        # Add cumulative returns
        data = data.withColumn("Cumulative_Return", 
                             (1 + col("Return")).over(window_spec) - 1)
        
        return data
    
    def _add_lagged_features(self, data: Union[pd.DataFrame, SparkDataFrame], column: str, lags: List[int]) -> Union[pd.DataFrame, SparkDataFrame]:
        """Add lagged features for a given column"""
        if isinstance(data, pd.DataFrame):
            return self._add_lagged_features_pandas(data, column, lags)
        else:
            return self._add_lagged_features_spark(data, column, lags)
    
    def _add_lagged_features_pandas(self, data: pd.DataFrame, column: str, lags: List[int]) -> pd.DataFrame:
        """Add lagged features using pandas"""
        for lag_days in lags:
            data[f'{column}_Lag{lag_days}'] = data.groupby('Ticker')[column].shift(lag_days)
        return data
    
    def _add_lagged_features_spark(self, data: SparkDataFrame, column: str, lags: List[int]) -> SparkDataFrame:
        """Add lagged features using PySpark"""
        from pyspark.sql.functions import lag
        from pyspark.sql.window import Window
        
        window_spec = Window.partitionBy("Ticker").orderBy("Date")
        
        for lag_days in lags:
            data = data.withColumn(f"{column}_Lag{lag_days}", 
                                 lag(column, lag_days).over(window_spec))
        
        return data
    
    def save_processed_data(self, data: Union[pd.DataFrame, SparkDataFrame], 
                          filepath: str, format: str = "parquet") -> None:
        """
        Save processed data to file
        
        Args:
            data: Processed data
            filepath: Output file path
            format: File format (parquet, csv, json)
        """
        self.logger.info(f"Saving processed data to {filepath}")
        
        if isinstance(data, SparkDataFrame):
            if format.lower() == "parquet":
                data.write.mode("overwrite").parquet(filepath)
            elif format.lower() == "csv":
                data.write.mode("overwrite").option("header", "true").csv(filepath)
            elif format.lower() == "json":
                data.write.mode("overwrite").json(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
        else:
            if format.lower() == "parquet":
                data.to_parquet(filepath, index=False)
            elif format.lower() == "csv":
                data.to_csv(filepath, index=False)
            elif format.lower() == "json":
                data.to_json(filepath, orient="records")
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Data saved successfully to {filepath}")
    
    def load_processed_data(self, filepath: str, format: str = "parquet") -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Load processed data from file
        
        Args:
            filepath: Input file path
            format: File format (parquet, csv, json)
            
        Returns:
            Loaded data
        """
        self.logger.info(f"Loading processed data from {filepath}")
        
        if self.spark is not None:
            # Use Spark to load data
            if format.lower() == "parquet":
                data = self.spark.read.parquet(filepath)
            elif format.lower() == "csv":
                data = self.spark.read.option("header", "true").csv(filepath)
            elif format.lower() == "json":
                data = self.spark.read.json(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
        else:
            # Use pandas to load data
            if format.lower() == "parquet":
                data = pd.read_parquet(filepath)
            elif format.lower() == "csv":
                data = pd.read_csv(filepath)
            elif format.lower() == "json":
                data = pd.read_json(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Data loaded successfully from {filepath}")
        return data
    
    def stop_spark(self):
        """Stop Spark session"""
        if self.spark is not None:
            self.spark.stop()
            self.logger.info("Spark session stopped")
