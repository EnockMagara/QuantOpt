"""
Feature engineering module for financial data
"""

from .technical_indicators import TechnicalIndicators
from .risk_metrics import RiskMetrics
from .feature_engineer import FeatureEngineer

# Import optional modules if available
try:
    from .sentiment_analyzer import SentimentAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    SentimentAnalyzer = None

__all__ = [
    "TechnicalIndicators",
    "RiskMetrics", 
    "FeatureEngineer"
]

if SENTIMENT_AVAILABLE:
    __all__.append("SentimentAnalyzer")
