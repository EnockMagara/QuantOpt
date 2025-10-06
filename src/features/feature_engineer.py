"""
Main feature engineering orchestrator
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, List, Optional
from pyspark.sql import DataFrame as SparkDataFrame
import logging

from .technical_indicators import TechnicalIndicators
from .risk_metrics import RiskMetrics

# Import sentiment analyzer if available
try:
    from .sentiment_analyzer import SentimentAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    SentimentAnalyzer = None


class FeatureEngineer:
    """
    Main feature engineering orchestrator that combines all feature engineering components
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the feature engineer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize feature engineering components
        self.technical_indicators = TechnicalIndicators(config)
        self.risk_metrics = RiskMetrics(config)
        
        # Initialize sentiment analyzer if available
        if SENTIMENT_AVAILABLE:
            self.sentiment_analyzer = SentimentAnalyzer(config)
        else:
            self.sentiment_analyzer = None
        
        # Feature configuration
        self.features_config = config.get('features', {})
        self.rolling_config = self.features_config.get('rolling', {})
    
    def engineer_features(self, data: Union[pd.DataFrame, SparkDataFrame], 
                         include_sentiment: bool = True) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Engineer all features for the dataset
        
        Args:
            data: Dataset to engineer features for
            include_sentiment: Whether to include sentiment analysis
            
        Returns:
            Dataset with all engineered features
        """
        self.logger.info("Starting feature engineering pipeline")
        
        # Step 1: Technical indicators
        self.logger.info("Calculating technical indicators")
        data = self.technical_indicators.calculate_all_indicators(data)
        
        # Step 2: Risk metrics
        self.logger.info("Calculating risk metrics")
        data = self.risk_metrics.calculate_all_risk_metrics(data)
        
        # Step 3: Rolling statistics
        self.logger.info("Calculating rolling statistics")
        data = self._calculate_rolling_statistics(data)
        
        # Step 4: Lagged features
        self.logger.info("Creating lagged features")
        data = self._create_lagged_features(data)
        
        # Step 5: Sentiment analysis (if requested and available)
        if include_sentiment and self.sentiment_analyzer is not None:
            self.logger.info("Performing sentiment analysis")
            if isinstance(data, SparkDataFrame):
                tickers = data.select('Ticker').distinct().rdd.map(lambda x: x[0]).collect()
            else:
                tickers = data['Ticker'].unique().tolist()
            data = self.sentiment_analyzer.process_sentiment_data(data, tickers)
        elif include_sentiment and self.sentiment_analyzer is None:
            self.logger.warning("Sentiment analysis requested but SentimentAnalyzer not available")
        
        # Step 6: Feature interactions
        self.logger.info("Creating feature interactions")
        data = self._create_feature_interactions(data)
        
        # Step 7: Feature scaling and normalization
        self.logger.info("Scaling and normalizing features")
        data = self._scale_features(data)
        
        self.logger.info("Feature engineering pipeline completed")
        return data
    
    def _calculate_rolling_statistics(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """Calculate rolling statistics for various windows"""
        windows = self.rolling_config.get('windows', [5, 10, 21, 63])
        metrics = self.rolling_config.get('metrics', ['mean', 'std', 'min', 'max', 'skew', 'kurtosis'])
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_rolling_statistics_pandas(data, windows, metrics)
        else:
            return self._calculate_rolling_statistics_spark(data, windows, metrics)
    
    def _calculate_rolling_statistics_pandas(self, data: pd.DataFrame, windows: List[int], metrics: List[str]) -> pd.DataFrame:
        """Calculate rolling statistics using pandas"""
        # Define columns to calculate rolling statistics for
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['Date', 'Ticker']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        for window in windows:
            for metric in metrics:
                for col in feature_columns:
                    if metric == 'mean':
                        data[f'{col}_Rolling_Mean_{window}'] = data.groupby('Ticker')[col].rolling(window).mean().values
                    elif metric == 'std':
                        data[f'{col}_Rolling_Std_{window}'] = data.groupby('Ticker')[col].rolling(window).std().values
                    elif metric == 'min':
                        data[f'{col}_Rolling_Min_{window}'] = data.groupby('Ticker')[col].rolling(window).min().values
                    elif metric == 'max':
                        data[f'{col}_Rolling_Max_{window}'] = data.groupby('Ticker')[col].rolling(window).max().values
                    elif metric == 'skew':
                        data[f'{col}_Rolling_Skew_{window}'] = data.groupby('Ticker')[col].rolling(window).skew().values
                    elif metric == 'kurtosis':
                        data[f'{col}_Rolling_Kurtosis_{window}'] = data.groupby('Ticker')[col].rolling(window).kurt().values
        
        return data
    
    def _calculate_rolling_statistics_spark(self, data: SparkDataFrame, windows: List[int], metrics: List[str]) -> SparkDataFrame:
        """Calculate rolling statistics using PySpark"""
        from pyspark.sql.functions import mean, stddev, min as spark_min, max as spark_max, skewness, kurtosis
        
        # Get numeric columns
        numeric_columns = [field.name for field in data.schema.fields 
                          if field.dataType.typeName() in ['double', 'float', 'int', 'long']]
        exclude_columns = ['Date', 'Ticker']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        for window in windows:
            rolling_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-window+1, 0)
            
            for metric in metrics:
                for col_name in feature_columns:
                    if metric == 'mean':
                        data = data.withColumn(f'{col_name}_Rolling_Mean_{window}', 
                                             mean(col_name).over(rolling_window))
                    elif metric == 'std':
                        data = data.withColumn(f'{col_name}_Rolling_Std_{window}', 
                                             stddev(col_name).over(rolling_window))
                    elif metric == 'min':
                        data = data.withColumn(f'{col_name}_Rolling_Min_{window}', 
                                             spark_min(col_name).over(rolling_window))
                    elif metric == 'max':
                        data = data.withColumn(f'{col_name}_Rolling_Max_{window}', 
                                             spark_max(col_name).over(rolling_window))
                    elif metric == 'skew':
                        data = data.withColumn(f'{col_name}_Rolling_Skew_{window}', 
                                             skewness(col_name).over(rolling_window))
                    elif metric == 'kurtosis':
                        data = data.withColumn(f'{col_name}_Rolling_Kurtosis_{window}', 
                                             kurtosis(col_name).over(rolling_window))
        
        return data
    
    def _create_lagged_features(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """Create lagged features for time series analysis"""
        lag_periods = [1, 2, 3, 5, 10, 21]  # 1 day, 2 days, 3 days, 1 week, 2 weeks, 1 month
        
        if isinstance(data, pd.DataFrame):
            return self._create_lagged_features_pandas(data, lag_periods)
        else:
            return self._create_lagged_features_spark(data, lag_periods)
    
    def _create_lagged_features_pandas(self, data: pd.DataFrame, lag_periods: List[int]) -> pd.DataFrame:
        """Create lagged features using pandas"""
        # Define columns to create lags for
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['Date', 'Ticker']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        for lag in lag_periods:
            for col in feature_columns:
                data[f'{col}_Lag_{lag}'] = data.groupby('Ticker')[col].shift(lag)
        
        return data
    
    def _create_lagged_features_spark(self, data: SparkDataFrame, lag_periods: List[int]) -> SparkDataFrame:
        """Create lagged features using PySpark"""
        from pyspark.sql.functions import lag
        
        # Get numeric columns
        numeric_columns = [field.name for field in data.schema.fields 
                          if field.dataType.typeName() in ['double', 'float', 'int', 'long']]
        exclude_columns = ['Date', 'Ticker']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        window_spec = Window.partitionBy("Ticker").orderBy("Date")
        
        for lag in lag_periods:
            for col_name in feature_columns:
                data = data.withColumn(f'{col_name}_Lag_{lag}', 
                                     lag(col_name, lag).over(window_spec))
        
        return data
    
    def _create_feature_interactions(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """Create feature interactions and ratios"""
        if isinstance(data, pd.DataFrame):
            return self._create_feature_interactions_pandas(data)
        else:
            return self._create_feature_interactions_spark(data)
    
    def _create_feature_interactions_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create feature interactions using pandas"""
        # Price-volume interactions
        if 'Volume' in data.columns and 'Close_Adj' in data.columns:
            data['Price_Volume_Interaction'] = data['Close_Adj'] * data['Volume']
            data['Volume_Price_Ratio'] = data['Volume'] / data['Close_Adj']
        
        # Volatility-return interactions
        if 'Volatility_21' in data.columns and 'Return' in data.columns:
            data['Volatility_Return_Interaction'] = data['Volatility_21'] * data['Return']
            data['Return_Volatility_Ratio'] = data['Return'] / data['Volatility_21']
        
        # Technical indicator interactions
        if 'RSI' in data.columns and 'MACD' in data.columns:
            data['RSI_MACD_Interaction'] = data['RSI'] * data['MACD']
        
        # Sentiment-price interactions
        if 'Sentiment_Score' in data.columns and 'Return' in data.columns:
            data['Sentiment_Return_Interaction'] = data['Sentiment_Score'] * data['Return']
        
        return data
    
    def _create_feature_interactions_spark(self, data: SparkDataFrame) -> SparkDataFrame:
        """Create feature interactions using PySpark"""
        from pyspark.sql.functions import col
        
        # Price-volume interactions
        if 'Volume' in data.columns and 'Close_Adj' in data.columns:
            data = data.withColumn('Price_Volume_Interaction', col('Close_Adj') * col('Volume'))
            data = data.withColumn('Volume_Price_Ratio', col('Volume') / col('Close_Adj'))
        
        # Volatility-return interactions
        if 'Volatility_21' in data.columns and 'Return' in data.columns:
            data = data.withColumn('Volatility_Return_Interaction', col('Volatility_21') * col('Return'))
            data = data.withColumn('Return_Volatility_Ratio', col('Return') / col('Volatility_21'))
        
        # Technical indicator interactions
        if 'RSI' in data.columns and 'MACD' in data.columns:
            data = data.withColumn('RSI_MACD_Interaction', col('RSI') * col('MACD'))
        
        # Sentiment-price interactions
        if 'Sentiment_Score' in data.columns and 'Return' in data.columns:
            data = data.withColumn('Sentiment_Return_Interaction', col('Sentiment_Score') * col('Return'))
        
        return data
    
    def _scale_features(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """Scale and normalize features"""
        if isinstance(data, pd.DataFrame):
            return self._scale_features_pandas(data)
        else:
            return self._scale_features_spark(data)
    
    def _scale_features_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale features using pandas"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        # Define columns to scale
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['Date', 'Ticker']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Standard scaling
        scaler = StandardScaler()
        data[feature_columns] = scaler.fit_transform(data[feature_columns])
        
        return data
    
    def _scale_features_spark(self, data: SparkDataFrame) -> SparkDataFrame:
        """Scale features using PySpark"""
        from pyspark.ml.feature import StandardScaler, VectorAssembler
        from pyspark.ml import Pipeline
        
        # Get numeric columns
        numeric_columns = [field.name for field in data.schema.fields 
                          if field.dataType.typeName() in ['double', 'float', 'int', 'long']]
        exclude_columns = ['Date', 'Ticker']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Create vector assembler
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        
        # Create standard scaler
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
        
        # Create pipeline
        pipeline = Pipeline(stages=[assembler, scaler])
        
        # Fit and transform
        model = pipeline.fit(data)
        data = model.transform(data)
        
        return data
    
    def select_features(self, data: Union[pd.DataFrame, SparkDataFrame], 
                       feature_selection_method: str = 'correlation') -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Select most relevant features
        
        Args:
            data: Dataset with all features
            feature_selection_method: Method for feature selection
            
        Returns:
            Dataset with selected features
        """
        self.logger.info(f"Selecting features using method: {feature_selection_method}")
        
        if isinstance(data, pd.DataFrame):
            return self._select_features_pandas(data, feature_selection_method)
        else:
            return self._select_features_spark(data, feature_selection_method)
    
    def _select_features_pandas(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """Select features using pandas"""
        if method == 'correlation':
            # Remove highly correlated features
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            exclude_columns = ['Date', 'Ticker']
            feature_columns = [col for col in numeric_columns if col not in exclude_columns]
            
            # Calculate correlation matrix
            corr_matrix = data[feature_columns].corr().abs()
            
            # Find highly correlated pairs
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Select features to drop
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
            
            # Drop highly correlated features
            data = data.drop(columns=to_drop)
            
            self.logger.info(f"Dropped {len(to_drop)} highly correlated features")
        
        return data
    
    def _select_features_spark(self, data: SparkDataFrame, method: str) -> SparkDataFrame:
        """Select features using PySpark"""
        # For PySpark, we'll implement a simpler feature selection
        # In practice, you might want to use MLlib's feature selection methods
        
        if method == 'variance':
            # Remove low variance features
            from pyspark.ml.feature import VarianceThresholdSelector
            
            # This would require converting to MLlib format
            # For now, we'll return the data as is
            pass
        
        return data
    
    def get_feature_importance(self, data: Union[pd.DataFrame, SparkDataFrame], 
                             target_column: str = 'Return') -> Dict[str, float]:
        """
        Get feature importance scores
        
        Args:
            data: Dataset with features
            target_column: Target column for importance calculation
            
        Returns:
            Dictionary mapping features to importance scores
        """
        self.logger.info("Calculating feature importance")
        
        if isinstance(data, pd.DataFrame):
            return self._get_feature_importance_pandas(data, target_column)
        else:
            return self._get_feature_importance_spark(data, target_column)
    
    def _get_feature_importance_pandas(self, data: pd.DataFrame, target_column: str) -> Dict[str, float]:
        """Get feature importance using pandas"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import mutual_info_regression
        
        # Get feature columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['Date', 'Ticker', target_column]
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Remove rows with missing values
        clean_data = data[feature_columns + [target_column]].dropna()
        
        if len(clean_data) == 0:
            return {}
        
        X = clean_data[feature_columns]
        y = clean_data[target_column]
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Create feature importance dictionary
        feature_importance = dict(zip(feature_columns, mi_scores))
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def _get_feature_importance_spark(self, data: SparkDataFrame, target_column: str) -> Dict[str, float]:
        """Get feature importance using PySpark"""
        # This is a simplified implementation for PySpark
        # In practice, you might want to use MLlib's feature selection methods
        
        # For now, return empty dictionary
        return {}
    
    def save_engineered_features(self, data: Union[pd.DataFrame, SparkDataFrame], 
                               filepath: str, format: str = "parquet") -> None:
        """
        Save engineered features to file
        
        Args:
            data: Dataset with engineered features
            filepath: Output file path
            format: File format (parquet, csv, json)
        """
        self.logger.info(f"Saving engineered features to {filepath}")
        
        if isinstance(data, pd.DataFrame):
            if format.lower() == "parquet":
                data.to_parquet(filepath, index=False)
            elif format.lower() == "csv":
                data.to_csv(filepath, index=False)
            elif format.lower() == "json":
                data.to_json(filepath, orient="records")
            else:
                raise ValueError(f"Unsupported format: {format}")
        else:
            if format.lower() == "parquet":
                data.write.mode("overwrite").parquet(filepath)
            elif format.lower() == "csv":
                data.write.mode("overwrite").option("header", "true").csv(filepath)
            elif format.lower() == "json":
                data.write.mode("overwrite").json(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Engineered features saved successfully to {filepath}")
