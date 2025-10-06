"""
Configuration manager for the MMS Finance ML project
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging


class ConfigManager:
    """
    Manages configuration for the MMS Finance ML project
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the configuration manager
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
    
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
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save_config(self, filepath: Optional[str] = None) -> None:
        """
        Save configuration to file
        
        Args:
            filepath: Path to save configuration (defaults to original path)
        """
        if filepath is None:
            filepath = self.config_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
        
        self.logger.info(f"Configuration saved to {filepath}")
    
    def validate_config(self) -> bool:
        """
        Validate configuration
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_sections = ['data', 'preprocessing', 'features', 'models', 'evaluation']
        
        for section in required_sections:
            if section not in self.config:
                self.logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate data configuration
        data_config = self.config.get('data', {})
        if 'raw_path' not in data_config:
            self.logger.error("Missing 'raw_path' in data configuration")
            return False
        
        if 'processed_path' not in data_config:
            self.logger.error("Missing 'processed_path' in data configuration")
            return False
        
        # Validate assets configuration
        assets_config = data_config.get('assets', {})
        if not assets_config:
            self.logger.error("Missing 'assets' configuration")
            return False
        
        # Validate model configuration
        models_config = self.config.get('models', {})
        if 'baseline' not in models_config:
            self.logger.error("Missing 'baseline' model configuration")
            return False
        
        self.logger.info("Configuration validation passed")
        return True
    
    def get_data_paths(self) -> Dict[str, str]:
        """Get data paths from configuration"""
        data_config = self.config.get('data', {})
        return {
            'raw_path': data_config.get('raw_path', 'data/raw'),
            'processed_path': data_config.get('processed_path', 'data/processed'),
            'external_path': data_config.get('external_path', 'data/external')
        }
    
    def get_asset_tickers(self) -> Dict[str, list]:
        """Get asset tickers from configuration"""
        assets_config = self.config.get('data', {}).get('assets', {})
        return {
            'stocks': assets_config.get('stocks', []),
            'etfs': assets_config.get('etfs', []),
            'crypto': assets_config.get('crypto', []),
            'bonds': assets_config.get('bonds', [])
        }
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.get('models', {}).get(model_name, {})
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration"""
        return self.config.get('features', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration"""
        return self.config.get('evaluation', {})
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables"""
        # Update API keys from environment
        if 'FRED_API_KEY' in os.environ:
            self.set('data.fred_api_key', os.environ['FRED_API_KEY'])
        
        # Update paths from environment
        if 'DATA_RAW_PATH' in os.environ:
            self.set('data.raw_path', os.environ['DATA_RAW_PATH'])
        
        if 'DATA_PROCESSED_PATH' in os.environ:
            self.set('data.processed_path', os.environ['DATA_PROCESSED_PATH'])
        
        # Update model parameters from environment
        if 'RISK_FREE_RATE' in os.environ:
            try:
                risk_free_rate = float(os.environ['RISK_FREE_RATE'])
                self.set('evaluation.metrics.risk_free_rate', risk_free_rate)
            except ValueError:
                self.logger.warning(f"Invalid RISK_FREE_RATE: {os.environ['RISK_FREE_RATE']}")
        
        self.logger.info("Configuration updated from environment variables")
    
    def create_directories(self) -> None:
        """Create necessary directories from configuration"""
        paths = self.get_data_paths()
        
        for path_name, path_value in paths.items():
            os.makedirs(path_value, exist_ok=True)
            self.logger.info(f"Created directory: {path_value}")
        
        # Create other necessary directories
        directories = [
            'logs',
            'models',
            'results',
            'results/backtests',
            'results/metrics',
            'results/plots'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
    
    def get_spark_config(self) -> Dict[str, Any]:
        """Get Spark configuration"""
        return self.config.get('spark', {})
    
    def get_pytorch_config(self) -> Dict[str, Any]:
        """Get PyTorch configuration"""
        return self.config.get('pytorch', {})
    
    def get_mlops_config(self) -> Dict[str, Any]:
        """Get MLOps configuration"""
        return self.config.get('mlops', {})
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment"""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration"""
        return self.get(key) is not None
