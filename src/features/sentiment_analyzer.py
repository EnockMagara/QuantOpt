"""
Sentiment analyzer for financial news and market sentiment
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, List, Optional, Tuple
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, when, isnan, isnull, coalesce, collect_list, avg, stddev
from pyspark.sql.window import Window
import logging
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import time

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available, using simple sentiment analysis")

try:
    import newspaper
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    logging.warning("Newspaper3k not available, using basic web scraping")


class SentimentAnalyzer:
    """
    Analyzes sentiment from financial news and market data
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the sentiment analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.sentiment_config = config.get('features', {}).get('sentiment', {})
        
        # Initialize sentiment analysis model
        self.sentiment_model = None
        self.tokenizer = None
        if TRANSFORMERS_AVAILABLE:
            self._initialize_sentiment_model()
        
        # News sources configuration
        self.news_sources = self.sentiment_config.get('news_sources', ['yahoo', 'reuters', 'bloomberg'])
        self.update_frequency = self.sentiment_config.get('update_frequency', 'daily')
    
    def _initialize_sentiment_model(self):
        """Initialize the sentiment analysis model"""
        model_name = self.sentiment_config.get('model_name', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
        
        try:
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                return_all_scores=True
            )
            self.logger.info(f"Initialized sentiment model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize sentiment model: {e}")
            self.sentiment_model = None
    
    def scrape_news_headlines(self, tickers: List[str], days_back: int = 7) -> Dict[str, List[str]]:
        """
        Scrape news headlines for given tickers
        
        Args:
            tickers: List of ticker symbols
            days_back: Number of days to look back for news
            
        Returns:
            Dictionary mapping tickers to list of headlines
        """
        self.logger.info(f"Scraping news headlines for {len(tickers)} tickers")
        
        headlines = {}
        
        for ticker in tickers:
            ticker_headlines = []
            
            # Scrape from different sources
            for source in self.news_sources:
                try:
                    source_headlines = self._scrape_source_headlines(ticker, source, days_back)
                    ticker_headlines.extend(source_headlines)
                except Exception as e:
                    self.logger.warning(f"Failed to scrape {source} for {ticker}: {e}")
            
            headlines[ticker] = ticker_headlines
            time.sleep(1)  # Rate limiting
        
        return headlines
    
    def _scrape_source_headlines(self, ticker: str, source: str, days_back: int) -> List[str]:
        """Scrape headlines from a specific source"""
        headlines = []
        
        if source == 'yahoo':
            headlines = self._scrape_yahoo_news(ticker, days_back)
        elif source == 'reuters':
            headlines = self._scrape_reuters_news(ticker, days_back)
        elif source == 'bloomberg':
            headlines = self._scrape_bloomberg_news(ticker, days_back)
        
        return headlines
    
    def _scrape_yahoo_news(self, ticker: str, days_back: int) -> List[str]:
        """Scrape news from Yahoo Finance"""
        headlines = []
        
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}/news"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news headlines
            news_items = soup.find_all('h3', class_='Mb(5px)')
            for item in news_items:
                headline = item.get_text().strip()
                if headline and len(headline) > 10:
                    headlines.append(headline)
            
        except Exception as e:
            self.logger.warning(f"Failed to scrape Yahoo news for {ticker}: {e}")
        
        return headlines[:10]  # Limit to 10 headlines
    
    def _scrape_reuters_news(self, ticker: str, days_back: int) -> List[str]:
        """Scrape news from Reuters"""
        headlines = []
        
        try:
            # Reuters search URL
            url = f"https://www.reuters.com/search/news?blob={ticker}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news headlines
            news_items = soup.find_all('h3', class_='search-result-title')
            for item in news_items:
                headline = item.get_text().strip()
                if headline and len(headline) > 10:
                    headlines.append(headline)
            
        except Exception as e:
            self.logger.warning(f"Failed to scrape Reuters news for {ticker}: {e}")
        
        return headlines[:10]  # Limit to 10 headlines
    
    def _scrape_bloomberg_news(self, ticker: str, days_back: int) -> List[str]:
        """Scrape news from Bloomberg"""
        headlines = []
        
        try:
            # Bloomberg search URL
            url = f"https://www.bloomberg.com/search?query={ticker}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news headlines
            news_items = soup.find_all('a', class_='headline')
            for item in news_items:
                headline = item.get_text().strip()
                if headline and len(headline) > 10:
                    headlines.append(headline)
            
        except Exception as e:
            self.logger.warning(f"Failed to scrape Bloomberg news for {ticker}: {e}")
        
        return headlines[:10]  # Limit to 10 headlines
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a given text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if self.sentiment_model is not None:
            return self._analyze_sentiment_transformer(text)
        else:
            return self._analyze_sentiment_simple(text)
    
    def _analyze_sentiment_transformer(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using transformer model"""
        try:
            results = self.sentiment_model(text)
            
            # Convert results to dictionary
            sentiment_scores = {}
            for result in results[0]:
                sentiment_scores[result['label']] = result['score']
            
            return sentiment_scores
        except Exception as e:
            self.logger.warning(f"Failed to analyze sentiment with transformer: {e}")
            return self._analyze_sentiment_simple(text)
    
    def _analyze_sentiment_simple(self, text: str) -> Dict[str, float]:
        """Simple sentiment analysis using keyword matching"""
        # Define positive and negative keywords
        positive_keywords = [
            'bullish', 'positive', 'growth', 'profit', 'gain', 'rise', 'increase',
            'strong', 'outperform', 'buy', 'upgrade', 'beat', 'exceed', 'surge',
            'rally', 'momentum', 'optimistic', 'confident', 'success', 'win'
        ]
        
        negative_keywords = [
            'bearish', 'negative', 'decline', 'loss', 'fall', 'drop', 'decrease',
            'weak', 'underperform', 'sell', 'downgrade', 'miss', 'disappoint', 'crash',
            'plunge', 'concern', 'pessimistic', 'uncertain', 'risk', 'volatile'
        ]
        
        text_lower = text.lower()
        
        # Count positive and negative keywords
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        total_count = positive_count + negative_count
        
        if total_count == 0:
            return {'POSITIVE': 0.5, 'NEGATIVE': 0.5, 'NEUTRAL': 1.0}
        
        positive_score = positive_count / total_count
        negative_score = negative_count / total_count
        neutral_score = 1.0 - abs(positive_score - negative_score)
        
        return {
            'POSITIVE': positive_score,
            'NEGATIVE': negative_score,
            'NEUTRAL': neutral_score
        }
    
    def calculate_sentiment_scores(self, headlines: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate sentiment scores for headlines
        
        Args:
            headlines: Dictionary mapping tickers to headlines
            
        Returns:
            Dictionary mapping tickers to sentiment scores
        """
        self.logger.info("Calculating sentiment scores for headlines")
        
        sentiment_scores = {}
        
        for ticker, ticker_headlines in headlines.items():
            if not ticker_headlines:
                sentiment_scores[ticker] = {'POSITIVE': 0.5, 'NEGATIVE': 0.5, 'NEUTRAL': 1.0}
                continue
            
            # Analyze sentiment for each headline
            ticker_sentiments = []
            for headline in ticker_headlines:
                sentiment = self.analyze_sentiment(headline)
                ticker_sentiments.append(sentiment)
            
            # Calculate average sentiment scores
            avg_sentiment = {
                'POSITIVE': np.mean([s.get('POSITIVE', 0) for s in ticker_sentiments]),
                'NEGATIVE': np.mean([s.get('NEGATIVE', 0) for s in ticker_sentiments]),
                'NEUTRAL': np.mean([s.get('NEUTRAL', 0) for s in ticker_sentiments])
            }
            
            sentiment_scores[ticker] = avg_sentiment
        
        return sentiment_scores
    
    def create_sentiment_features(self, data: Union[pd.DataFrame, SparkDataFrame], 
                                sentiment_scores: Dict[str, Dict[str, float]]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Create sentiment features for the dataset
        
        Args:
            data: Dataset to add sentiment features to
            sentiment_scores: Dictionary mapping tickers to sentiment scores
            
        Returns:
            Dataset with sentiment features
        """
        self.logger.info("Creating sentiment features")
        
        if isinstance(data, pd.DataFrame):
            return self._create_sentiment_features_pandas(data, sentiment_scores)
        else:
            return self._create_sentiment_features_spark(data, sentiment_scores)
    
    def _create_sentiment_features_pandas(self, data: pd.DataFrame, sentiment_scores: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Create sentiment features using pandas"""
        # Add sentiment scores for each ticker
        for ticker in data['Ticker'].unique():
            if ticker in sentiment_scores:
                sentiment = sentiment_scores[ticker]
                mask = data['Ticker'] == ticker
                
                data.loc[mask, 'Sentiment_Positive'] = sentiment.get('POSITIVE', 0.5)
                data.loc[mask, 'Sentiment_Negative'] = sentiment.get('NEGATIVE', 0.5)
                data.loc[mask, 'Sentiment_Neutral'] = sentiment.get('NEUTRAL', 1.0)
                
                # Calculate sentiment score (positive - negative)
                data.loc[mask, 'Sentiment_Score'] = sentiment.get('POSITIVE', 0.5) - sentiment.get('NEGATIVE', 0.5)
            else:
                # Default neutral sentiment
                mask = data['Ticker'] == ticker
                data.loc[mask, 'Sentiment_Positive'] = 0.5
                data.loc[mask, 'Sentiment_Negative'] = 0.5
                data.loc[mask, 'Sentiment_Neutral'] = 1.0
                data.loc[mask, 'Sentiment_Score'] = 0.0
        
        return data
    
    def _create_sentiment_features_spark(self, data: SparkDataFrame, sentiment_scores: Dict[str, Dict[str, float]]) -> SparkDataFrame:
        """Create sentiment features using PySpark"""
        # Create a mapping DataFrame for sentiment scores
        sentiment_data = []
        for ticker, sentiment in sentiment_scores.items():
            sentiment_data.append({
                'Ticker': ticker,
                'Sentiment_Positive': sentiment.get('POSITIVE', 0.5),
                'Sentiment_Negative': sentiment.get('NEGATIVE', 0.5),
                'Sentiment_Neutral': sentiment.get('NEUTRAL', 1.0),
                'Sentiment_Score': sentiment.get('POSITIVE', 0.5) - sentiment.get('NEGATIVE', 0.5)
            })
        
        # Create sentiment DataFrame
        sentiment_df = data.sql_ctx.createDataFrame(sentiment_data)
        
        # Join with main data
        data = data.join(sentiment_df, on='Ticker', how='left')
        
        # Fill missing values with neutral sentiment
        data = data.fillna({
            'Sentiment_Positive': 0.5,
            'Sentiment_Negative': 0.5,
            'Sentiment_Neutral': 1.0,
            'Sentiment_Score': 0.0
        })
        
        return data
    
    def calculate_market_sentiment(self, data: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Calculate market-wide sentiment indicators
        
        Args:
            data: Dataset with sentiment features
            
        Returns:
            Dataset with market sentiment indicators
        """
        self.logger.info("Calculating market sentiment indicators")
        
        if isinstance(data, pd.DataFrame):
            return self._calculate_market_sentiment_pandas(data)
        else:
            return self._calculate_market_sentiment_spark(data)
    
    def _calculate_market_sentiment_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market sentiment using pandas"""
        # Calculate market-wide sentiment by date
        market_sentiment = data.groupby('Date').agg({
            'Sentiment_Score': ['mean', 'std', 'min', 'max'],
            'Sentiment_Positive': 'mean',
            'Sentiment_Negative': 'mean'
        }).reset_index()
        
        # Flatten column names
        market_sentiment.columns = ['Date', 'Market_Sentiment_Mean', 'Market_Sentiment_Std', 
                                  'Market_Sentiment_Min', 'Market_Sentiment_Max',
                                  'Market_Sentiment_Positive', 'Market_Sentiment_Negative']
        
        # Merge back to main data
        data = data.merge(market_sentiment, on='Date', how='left')
        
        # Calculate relative sentiment (ticker sentiment vs market sentiment)
        data['Relative_Sentiment'] = data['Sentiment_Score'] - data['Market_Sentiment_Mean']
        
        return data
    
    def _calculate_market_sentiment_spark(self, data: SparkDataFrame) -> SparkDataFrame:
        """Calculate market sentiment using PySpark"""
        from pyspark.sql.functions import mean, stddev, min as spark_min, max as spark_max
        
        # Calculate market-wide sentiment by date
        market_sentiment = data.groupBy('Date').agg(
            mean('Sentiment_Score').alias('Market_Sentiment_Mean'),
            stddev('Sentiment_Score').alias('Market_Sentiment_Std'),
            spark_min('Sentiment_Score').alias('Market_Sentiment_Min'),
            spark_max('Sentiment_Score').alias('Market_Sentiment_Max'),
            mean('Sentiment_Positive').alias('Market_Sentiment_Positive'),
            mean('Sentiment_Negative').alias('Market_Sentiment_Negative')
        )
        
        # Join back to main data
        data = data.join(market_sentiment, on='Date', how='left')
        
        # Calculate relative sentiment
        data = data.withColumn('Relative_Sentiment', 
                             col('Sentiment_Score') - col('Market_Sentiment_Mean'))
        
        return data
    
    def process_sentiment_data(self, data: Union[pd.DataFrame, SparkDataFrame], 
                             tickers: List[str]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Complete sentiment analysis pipeline
        
        Args:
            data: Dataset to add sentiment features to
            tickers: List of ticker symbols
            
        Returns:
            Dataset with sentiment features
        """
        self.logger.info("Processing sentiment data")
        
        # Scrape news headlines
        headlines = self.scrape_news_headlines(tickers)
        
        # Calculate sentiment scores
        sentiment_scores = self.calculate_sentiment_scores(headlines)
        
        # Create sentiment features
        data = self.create_sentiment_features(data, sentiment_scores)
        
        # Calculate market sentiment
        data = self.calculate_market_sentiment(data)
        
        return data
