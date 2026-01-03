"""
Data Loader Module - Fetch stock market data and financial news.

This module provides functions to:
1. Download stock OHLCV data from Yahoo Finance
2. Fetch company-specific news from NewsAPI or Finnhub
3. Cache data locally to avoid repeated API calls
"""

import os
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
from src.config import NEWS_API_KEY, FINNHUB_API_KEY, CACHE_DIR
from src.utils import save_to_cache, load_from_cache


class StockDataLoader:
    """
    Loader for stock market OHLCV data using yfinance.
    """
    
    def __init__(self, ticker):
        """
        Initialize stock data loader.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')
        """
        self.ticker = ticker.upper()
        self.cache_file = f"{self.ticker}_stock_data.pkl"
    
    def fetch_data(self, start_date, end_date, use_cache=True):
        """
        Fetch stock OHLCV data.
        
        Args:
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            use_cache (bool): Whether to use cached data if available
            
        Returns:
            pd.DataFrame: Stock data with columns [Open, High, Low, Close, Volume]
        """
        # Try loading from cache first
        if use_cache:
            cached_data = load_from_cache(self.cache_file)
            if cached_data is not None:
                # Filter to requested date range
                cached_data.index = pd.to_datetime(cached_data.index)
                mask = (cached_data.index >= start_date) & (cached_data.index <= end_date)
                filtered_data = cached_data[mask]
                
                if len(filtered_data) > 0:
                    print(f"✅ Using cached stock data for {self.ticker}")
                    return filtered_data
        
        # Fetch from Yahoo Finance
        print(f"📥 Fetching stock data for {self.ticker} from {start_date} to {end_date}...")
        
        try:
            stock = yf.Ticker(self.ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            
            # Keep only OHLCV columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Save to cache
            save_to_cache(df, self.cache_file)
            
            print(f"✅ Successfully fetched {len(df)} days of stock data")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching stock data: {e}")
            return pd.DataFrame()
    
    def get_latest_price(self):
        """
        Get the most recent closing price.
        
        Returns:
            float: Latest closing price
        """
        try:
            stock = yf.Ticker(self.ticker)
            data = stock.history(period='1d')
            return data['Close'].iloc[-1]
        except:
            return None


class NewsDataLoader:
    """
    Loader for financial news data from NewsAPI or Finnhub.
    """
    
    def __init__(self, ticker, api_source='newsapi'):
        """
        Initialize news data loader.
        
        Args:
            ticker (str): Stock ticker symbol
            api_source (str): 'newsapi' or 'finnhub'
        """
        self.ticker = ticker.upper()
        self.api_source = api_source.lower()
        self.cache_file = f"{self.ticker}_news_data.pkl"
        
        # Get company name for better search results
        self.company_name = self._get_company_name()
    
    def _get_company_name(self):
        """
        Get company name from ticker.
        
        Returns:
            str: Company name
        """
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info
            return info.get('longName', self.ticker)
        except:
            return self.ticker
    
    def fetch_data(self, start_date, end_date, use_cache=True):
        """
        Fetch news headlines for the stock.
        
        Args:
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            use_cache (bool): Whether to use cached data
            
        Returns:
            pd.DataFrame: News data with columns [date, headline, source]
        """
        # Try loading from cache
        if use_cache:
            cached_data = load_from_cache(self.cache_file)
            if cached_data is not None:
                cached_data['date'] = pd.to_datetime(cached_data['date'])
                mask = (cached_data['date'] >= start_date) & (cached_data['date'] <= end_date)
                filtered_data = cached_data[mask]
                
                if len(filtered_data) > 0:
                    print(f"✅ Using cached news data for {self.ticker}")
                    return filtered_data
        
        # Fetch from API
        print(f"📰 Fetching news for {self.company_name} ({self.ticker})...")
        
        if self.api_source == 'newsapi':
            news_df = self._fetch_from_newsapi(start_date, end_date)
        elif self.api_source == 'finnhub':
            news_df = self._fetch_from_finnhub(start_date, end_date)
        else:
            print(f"❌ Unknown API source: {self.api_source}")
            return pd.DataFrame()
        
        if not news_df.empty:
            save_to_cache(news_df, self.cache_file)
            print(f"✅ Successfully fetched {len(news_df)} news articles")
        
        return news_df
    
    def _fetch_from_newsapi(self, start_date, end_date):
        """
        Fetch news from NewsAPI.
        
        Args:
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            pd.DataFrame: News data
        """
        if not NEWS_API_KEY or NEWS_API_KEY == 'your_newsapi_key_here':
            print("⚠️  NewsAPI key not configured. Using sample data.")
            return self._get_sample_news_data(start_date, end_date)
        
        url = 'https://newsapi.org/v2/everything'
        
        articles = []
        
        # NewsAPI has a 1-month limit per request, so we need to chunk
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        current_start = start
        while current_start < end:
            current_end = min(current_start + timedelta(days=30), end)
            
            params = {
                'q': f'{self.company_name} OR {self.ticker}',
                'from': current_start.strftime('%Y-%m-%d'),
                'to': current_end.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': NEWS_API_KEY
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if data['status'] == 'ok':
                    articles.extend(data['articles'])
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error fetching from NewsAPI: {e}")
                break
            
            current_start = current_end + timedelta(days=1)
        
        if not articles:
            return pd.DataFrame()
        
        # Convert to DataFrame
        news_data = []
        for article in articles:
            news_data.append({
                'date': pd.to_datetime(article['publishedAt']).date(),
                'headline': article['title'],
                'source': article['source']['name']
            })
        
        df = pd.DataFrame(news_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def _fetch_from_finnhub(self, start_date, end_date):
        """
        Fetch news from Finnhub.
        
        Args:
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            pd.DataFrame: News data
        """
        if not FINNHUB_API_KEY or FINNHUB_API_KEY == 'your_finnhub_key_here':
            print("⚠️  Finnhub API key not configured. Using sample data.")
            return self._get_sample_news_data(start_date, end_date)
        
        url = 'https://finnhub.io/api/v1/company-news'
        
        params = {
            'symbol': self.ticker,
            'from': start_date,
            'to': end_date,
            'token': FINNHUB_API_KEY
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            articles = response.json()
            
            if not articles:
                return pd.DataFrame()
            
            # Convert to DataFrame
            news_data = []
            for article in articles:
                news_data.append({
                    'date': pd.to_datetime(article['datetime'], unit='s').date(),
                    'headline': article['headline'],
                    'source': article['source']
                })
            
            df = pd.DataFrame(news_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching from Finnhub: {e}")
            return pd.DataFrame()
    
    def _get_sample_news_data(self, start_date, end_date):
        """
        Generate sample news data for testing without API keys.
        
        Args:
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            pd.DataFrame: Sample news data
        """
        print("📝 Generating sample news data for demonstration...")
        
        # Create sample headlines
        sample_headlines = [
            f"{self.company_name} reports strong quarterly earnings",
            f"{self.company_name} announces new product launch",
            f"Analysts upgrade {self.ticker} stock rating",
            f"{self.company_name} faces regulatory challenges",
            f"{self.ticker} stock reaches new high",
            f"{self.company_name} CEO discusses future plans",
            f"Market volatility affects {self.ticker}",
            f"{self.company_name} expands into new markets",
        ]
        
        # Generate dates
        dates = pd.date_range(start=start_date, end=end_date, freq='3D')
        
        news_data = []
        for i, date in enumerate(dates):
            headline = sample_headlines[i % len(sample_headlines)]
            news_data.append({
                'date': date.date(),
                'headline': headline,
                'source': 'Sample News'
            })
        
        df = pd.DataFrame(news_data)
        df['date'] = pd.to_datetime(df['date'])
        
        return df


def load_stock_and_news(ticker, start_date, end_date, news_api='newsapi', use_cache=True):
    """
    Convenience function to load both stock and news data.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        news_api (str): 'newsapi' or 'finnhub'
        use_cache (bool): Whether to use cached data
        
    Returns:
        tuple: (stock_df, news_df)
    """
    stock_loader = StockDataLoader(ticker)
    news_loader = NewsDataLoader(ticker, api_source=news_api)
    
    stock_df = stock_loader.fetch_data(start_date, end_date, use_cache)
    news_df = news_loader.fetch_data(start_date, end_date, use_cache)
    
    return stock_df, news_df


if __name__ == '__main__':
    # Test the data loaders
    print("="*60)
    print("🧪 TESTING DATA LOADERS")
    print("="*60)
    
    ticker = 'AAPL'
    start = '2024-01-01'
    end = '2024-12-31'
    
    print(f"\n📊 Testing with {ticker} from {start} to {end}\n")
    
    # Test stock data
    stock_loader = StockDataLoader(ticker)
    stock_df = stock_loader.fetch_data(start, end)
    
    if not stock_df.empty:
        print(f"\n📈 Stock Data Preview:")
        print(stock_df.head())
        print(f"\nShape: {stock_df.shape}")
    
    # Test news data
    news_loader = NewsDataLoader(ticker)
    news_df = news_loader.fetch_data(start, end)
    
    if not news_df.empty:
        print(f"\n📰 News Data Preview:")
        print(news_df.head())
        print(f"\nShape: {news_df.shape}")
    
    print("\n" + "="*60)
    print("✅ DATA LOADER TEST COMPLETE")
    print("="*60)
