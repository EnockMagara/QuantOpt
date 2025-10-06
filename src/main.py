"""
Main pipeline orchestrator for MMS Finance ML project
"""

import os
import sys
import logging
import yaml
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from preprocessing import DataProcessor
from features import FeatureEngineer
from models import MPTOptimizer
from utils.config_manager import ConfigManager
from utils.logger import setup_logger


class MMSFinancePipeline:
    """
    Main pipeline orchestrator for the MMS Finance ML project
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Setup logging
        self.logger = setup_logger(self.config.get('mlops', {}).get('logging', {}))
        
        # Initialize components
        self.data_processor = DataProcessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.mpt_optimizer = MPTOptimizer(self.config)
        
        # Data paths
        self.raw_data_path = self.config['data']['raw_path']
        self.processed_data_path = self.config['data']['processed_path']
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def run_full_pipeline(self, include_sentiment: bool = True) -> Dict[str, Any]:
        """
        Run the complete ML pipeline
        
        Args:
            include_sentiment: Whether to include sentiment analysis
            
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("Starting MMS Finance ML pipeline")
        
        results = {}
        
        try:
            # Step 1: Load and preprocess data
            self.logger.info("Step 1: Loading and preprocessing data")
            processed_data = self._load_and_preprocess_data()
            results['processed_data'] = processed_data
            
            # Step 2: Feature engineering
            self.logger.info("Step 2: Feature engineering")
            engineered_data = self._engineer_features(processed_data, include_sentiment)
            results['engineered_data'] = engineered_data
            
            # Step 3: Baseline model (MPT)
            self.logger.info("Step 3: Running baseline MPT model")
            mpt_results = self._run_baseline_model(engineered_data)
            results['mpt_results'] = mpt_results
            
            # Step 4: Save results
            self.logger.info("Step 4: Saving results")
            self._save_results(results)
            
            self.logger.info("Pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
        
        return results
    
    def _load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess the data"""
        # Try to load cleaned data first
        cleaned_file = os.path.join(self.processed_data_path, "processed_assets_cleaned.parquet")
        processed_file = os.path.join(self.processed_data_path, "processed_assets.parquet")
        
        if os.path.exists(cleaned_file):
            self.logger.info("Loading cleaned processed data")
            return pd.read_parquet(cleaned_file)
        elif os.path.exists(processed_file):
            self.logger.info("Loading existing processed data")
            return pd.read_parquet(processed_file)
        
        # Load existing processed CSV files
        self.logger.info("Loading existing processed CSV files")
        assets_file = os.path.join(self.processed_data_path, "ml_ready_assets.csv")
        bonds_file = os.path.join(self.processed_data_path, "ml_ready_bonds.csv")
        
        if not os.path.exists(assets_file):
            raise FileNotFoundError(f"Assets data not found: {assets_file}")
        
        # Load assets data
        assets_data = pd.read_csv(assets_file)
        assets_data['Date'] = pd.to_datetime(assets_data['Date'])
        
        # Load bonds data if available
        bonds_data = None
        if os.path.exists(bonds_file):
            bonds_data = pd.read_csv(bonds_file)
            bonds_data['Date'] = pd.to_datetime(bonds_data['Date'])
        
        # Process data (minimal processing since data is already processed)
        self.logger.info("Processing assets data")
        processed_assets = self.data_processor.process_assets_data(assets_data)
        
        if bonds_data is not None:
            self.logger.info("Processing bonds data")
            processed_bonds = self.data_processor.process_bonds_data(bonds_data)
            
            # Merge assets and bonds data
            processed_data = self._merge_assets_bonds_data(processed_assets, processed_bonds)
        else:
            processed_data = processed_assets
        
        # Save processed data
        os.makedirs(self.processed_data_path, exist_ok=True)
        processed_data.to_parquet(processed_file, index=False)
        
        return processed_data
    
    def _merge_assets_bonds_data(self, assets_data: pd.DataFrame, 
                                bonds_data: pd.DataFrame) -> pd.DataFrame:
        """Merge assets and bonds data"""
        # Pivot bonds data to have bond yields as columns
        bonds_pivot = bonds_data.pivot(index='Date', columns='Ticker', values='Yield')
        bonds_pivot.columns = [f'{col}_Yield' for col in bonds_pivot.columns]
        bonds_pivot = bonds_pivot.reset_index()
        
        # Merge with assets data
        merged_data = assets_data.merge(bonds_pivot, on='Date', how='left')
        
        return merged_data
    
    def _engineer_features(self, data: pd.DataFrame, include_sentiment: bool = True) -> pd.DataFrame:
        """Engineer features for the data"""
        # Check if engineered data already exists
        engineered_file = os.path.join(self.processed_data_path, "engineered_features.parquet")
        
        if os.path.exists(engineered_file):
            self.logger.info("Loading existing engineered features")
            return pd.read_parquet(engineered_file)
        
        # Engineer features
        engineered_data = self.feature_engineer.engineer_features(data, include_sentiment)
        
        # Save engineered data
        engineered_data.to_parquet(engineered_file, index=False)
        
        return engineered_data
    
    def _run_baseline_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run the baseline MPT model"""
        # Prepare returns data for optimization
        returns_data = self._prepare_returns_data(data)
        
        # Calculate expected returns and covariance matrix
        expected_returns = self.mpt_optimizer.calculate_expected_returns(returns_data)
        covariance_matrix = self.mpt_optimizer.calculate_covariance_matrix(returns_data)
        
        # Optimize portfolio
        optimization_result = self.mpt_optimizer.optimize_portfolio(
            expected_returns, covariance_matrix
        )
        
        # Backtest portfolio
        backtest_results = self.mpt_optimizer.backtest_portfolio(returns_data)
        
        # Calculate portfolio metrics
        portfolio_metrics = self.mpt_optimizer.calculate_portfolio_metrics(
            returns_data, optimization_result['weights']
        )
        
        return {
            'optimization_result': optimization_result,
            'backtest_results': backtest_results,
            'portfolio_metrics': portfolio_metrics,
            'expected_returns': expected_returns,
            'covariance_matrix': covariance_matrix
        }
    
    def _prepare_returns_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare returns data for portfolio optimization"""
        # Get unique tickers
        tickers = data['Ticker'].unique()
        
        # Create returns matrix
        returns_data = {}
        
        for ticker in tickers:
            ticker_data = data[data['Ticker'] == ticker].copy()
            ticker_data = ticker_data.set_index('Date')
            returns_data[ticker] = ticker_data['Return']
        
        # Create DataFrame with returns
        returns_df = pd.DataFrame(returns_data)
        
        # Remove rows with all NaN values
        returns_df = returns_df.dropna(how='all')
        
        return returns_df
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save pipeline results"""
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save MPT results
        mpt_results = results.get('mpt_results', {})
        
        if 'optimization_result' in mpt_results:
            # Save portfolio weights
            weights_df = mpt_results['optimization_result']['weights'].to_frame('Weight')
            weights_df.to_csv(os.path.join(results_dir, "portfolio_weights.csv"))
            
            # Save portfolio metrics
            metrics_df = pd.DataFrame([mpt_results['portfolio_metrics']])
            metrics_df.to_csv(os.path.join(results_dir, "portfolio_metrics.csv"), index=False)
        
        if 'backtest_results' in mpt_results:
            # Save backtest results
            mpt_results['backtest_results'].to_csv(
                os.path.join(results_dir, "backtest_results.csv"), index=False
            )
        
        # Save expected returns and covariance matrix
        if 'expected_returns' in mpt_results:
            mpt_results['expected_returns'].to_csv(
                os.path.join(results_dir, "expected_returns.csv")
            )
        
        if 'covariance_matrix' in mpt_results:
            mpt_results['covariance_matrix'].to_csv(
                os.path.join(results_dir, "covariance_matrix.csv")
            )
        
        self.logger.info(f"Results saved to {results_dir}")
    
    def run_data_pipeline_only(self) -> pd.DataFrame:
        """Run only the data preprocessing and feature engineering pipeline"""
        self.logger.info("Running data pipeline only")
        
        # Load and preprocess data
        processed_data = self._load_and_preprocess_data()
        
        # Engineer features
        engineered_data = self._engineer_features(processed_data, include_sentiment=False)
        
        return engineered_data
    
    def run_model_pipeline_only(self, data: pd.DataFrame = None) -> Dict[str, Any]:
        """Run only the model pipeline"""
        self.logger.info("Running model pipeline only")
        
        if data is None:
            # Load engineered data
            engineered_file = os.path.join(self.processed_data_path, "engineered_features.parquet")
            if not os.path.exists(engineered_file):
                raise FileNotFoundError("Engineered features not found. Run data pipeline first.")
            data = pd.read_parquet(engineered_file)
        
        # Run baseline model
        mpt_results = self._run_baseline_model(data)
        
        # Save results
        self._save_results({'mpt_results': mpt_results})
        
        return mpt_results


def main():
    """Main function to run the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MMS Finance ML Pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--data-only", action="store_true", help="Run only data pipeline")
    parser.add_argument("--model-only", action="store_true", help="Run only model pipeline")
    parser.add_argument("--no-sentiment", action="store_true", help="Skip sentiment analysis")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MMSFinancePipeline(args.config)
    
    try:
        if args.data_only:
            # Run data pipeline only
            data = pipeline.run_data_pipeline_only()
            print("Data pipeline completed successfully")
            print(f"Data shape: {data.shape}")
            
        elif args.model_only:
            # Run model pipeline only
            results = pipeline.run_model_pipeline_only()
            print("Model pipeline completed successfully")
            print(f"Portfolio weights: {results['optimization_result']['weights']}")
            
        else:
            # Run full pipeline
            results = pipeline.run_full_pipeline(include_sentiment=not args.no_sentiment)
            print("Full pipeline completed successfully")
            print(f"Portfolio metrics: {results['mpt_results']['portfolio_metrics']}")
            
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
