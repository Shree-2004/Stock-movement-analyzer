"""
Data Preprocessing Module - Clean and prepare stock and news data.

This module provides functions to:
1. Calculate daily returns and create target labels
2. Compute technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
3. Clean and process news headlines
4. Align data by date
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class StockPreprocessor:
    """
    Preprocessor for stock market data.
    """
    
    def __init__(self, df):
        """
        Initialize stock preprocessor.
        
        Args:
            df (pd.DataFrame): Stock data with OHLCV columns
        """
        self.df = df.copy()
        self.df.index = pd.to_datetime(self.df.index)
    
    def calculate_returns(self):
        """
        Calculate daily returns.
        
        Returns:
            pd.DataFrame: DataFrame with 'Returns' column added
        """
        self.df['Returns'] = self.df['Close'].pct_change()
        print("✅ Calculated daily returns")
        return self.df
    
    def create_target_label(self):
        """
        Create binary target label: 1 if next-day return > 0, else 0.
        
        Returns:
            pd.DataFrame: DataFrame with 'Target' column added
        """
        # Shift returns by -1 to get next day's return
        self.df['Next_Day_Return'] = self.df['Returns'].shift(-1)
        
        # Create binary label
        self.df['Target'] = (self.df['Next_Day_Return'] > 0).astype(int)
        
        print("✅ Created target labels (1=Up, 0=Down)")
        print(f"   - Up days: {self.df['Target'].sum()}")
        print(f"   - Down days: {(self.df['Target'] == 0).sum()}")
        
        return self.df
    
    def add_technical_indicators(self, sma_periods=[10, 20, 50], 
                                 ema_periods=[12, 26],
                                 rsi_period=14,
                                 macd_fast=12, macd_slow=26, macd_signal=9,
                                 bb_period=20, bb_std=2):
        """
        Add technical indicators to the DataFrame.
        
        Args:
            sma_periods (list): Periods for Simple Moving Averages
            ema_periods (list): Periods for Exponential Moving Averages
            rsi_period (int): Period for RSI
            macd_fast (int): Fast period for MACD
            macd_slow (int): Slow period for MACD
            macd_signal (int): Signal period for MACD
            bb_period (int): Period for Bollinger Bands
            bb_std (int): Standard deviations for Bollinger Bands
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators added
        """
        close = self.df['Close']
        
        # Simple Moving Averages
        for period in sma_periods:
            sma = SMAIndicator(close=close, window=period)
            self.df[f'SMA_{period}'] = sma.sma_indicator()
        
        # Exponential Moving Averages
        for period in ema_periods:
            ema = EMAIndicator(close=close, window=period)
            self.df[f'EMA_{period}'] = ema.ema_indicator()
        
        # RSI
        rsi = RSIIndicator(close=close, window=rsi_period)
        self.df['RSI'] = rsi.rsi()
        
        # MACD
        macd = MACD(close=close, window_fast=macd_fast, 
                   window_slow=macd_slow, window_sign=macd_signal)
        self.df['MACD'] = macd.macd()
        self.df['MACD_Signal'] = macd.macd_signal()
        self.df['MACD_Diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(close=close, window=bb_period, window_dev=bb_std)
        self.df['BB_High'] = bb.bollinger_hband()
        self.df['BB_Mid'] = bb.bollinger_mavg()
        self.df['BB_Low'] = bb.bollinger_lband()
        self.df['BB_Width'] = bb.bollinger_wband()
        
        # Additional features (with safe division)
        # Avoid division by zero: use np.where to check if SMA_20 > 0
        self.df['Price_to_SMA_20'] = np.where(
            self.df['SMA_20'] > 0,
            close / self.df['SMA_20'],
            1.0  # Default to 1.0 if SMA_20 is 0
        )
        self.df['Volume_Change'] = self.df['Volume'].pct_change()
        
        print("✅ Added technical indicators:")
        print(f"   - SMAs: {sma_periods}")
        print(f"   - EMAs: {ema_periods}")
        print(f"   - RSI ({rsi_period} period)")
        print(f"   - MACD ({macd_fast}/{macd_slow}/{macd_signal})")
        print(f"   - Bollinger Bands ({bb_period} period)")
        
        return self.df
    
    def remove_nan(self):
        """
        Remove rows with NaN values (typically from indicator calculation).
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        initial_rows = len(self.df)
        self.df = self.df.dropna()
        removed_rows = initial_rows - len(self.df)
        
        print(f"✅ Removed {removed_rows} rows with NaN values")
        print(f"   Remaining rows: {len(self.df)}")
        
        return self.df
    
    def get_processed_data(self):
        """
        Get the processed DataFrame.
        
        Returns:
            pd.DataFrame: Processed stock data
        """
        return self.df


class NewsPreprocessor:
    """
    Preprocessor for news data.
    """
    
    def __init__(self, df):
        """
        Initialize news preprocessor.
        
        Args:
            df (pd.DataFrame): News data with 'date' and 'headline' columns
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Clean a single text string.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def clean_headlines(self):
        """
        Clean all headlines.
        
        Returns:
            pd.DataFrame: DataFrame with 'cleaned_headline' column added
        """
        self.df['cleaned_headline'] = self.df['headline'].apply(self.clean_text)
        print("✅ Cleaned news headlines")
        return self.df
    
    def aggregate_daily(self):
        """
        Aggregate news by date (count headlines per day).
        
        Returns:
            pd.DataFrame: Daily aggregated news data
        """
        daily_counts = self.df.groupby('date').size().reset_index(name='news_count')
        print(f"✅ Aggregated news by date: {len(daily_counts)} unique dates")
        return daily_counts
    
    def get_processed_data(self):
        """
        Get the processed DataFrame.
        
        Returns:
            pd.DataFrame: Processed news data
        """
        return self.df


def align_stock_and_news(stock_df, news_df):
    """
    Align stock and news data by date.
    
    Args:
        stock_df (pd.DataFrame): Processed stock data with DateTimeIndex
        news_df (pd.DataFrame): Processed news data with 'date' column
        
    Returns:
        pd.DataFrame: Combined DataFrame with both stock and news data
    """
    # Ensure stock_df has date as column
    stock_df = stock_df.copy()
    stock_df['date'] = stock_df.index
    
    # Merge on date
    combined_df = stock_df.merge(news_df, on='date', how='left')
    
    # Fill missing news counts with 0
    if 'news_count' in combined_df.columns:
        combined_df['news_count'] = combined_df['news_count'].fillna(0)
    
    # Set date back as index
    combined_df = combined_df.set_index('date')
    
    print(f"✅ Aligned stock and news data: {len(combined_df)} rows")
    
    return combined_df


def preprocess_all_data(stock_df, news_df):
    """
    Complete preprocessing pipeline for stock and news data.
    
    Args:
        stock_df (pd.DataFrame): Raw stock data
        news_df (pd.DataFrame): Raw news data
        
    Returns:
        pd.DataFrame: Fully preprocessed and combined data
    """
    print("\n" + "="*60)
    print("🔧 STARTING DATA PREPROCESSING")
    print("="*60 + "\n")
    
    # Process stock data
    print("📊 Processing stock data...")
    stock_processor = StockPreprocessor(stock_df)
    stock_processor.calculate_returns()
    stock_processor.create_target_label()
    stock_processor.add_technical_indicators()
    stock_processor.remove_nan()
    processed_stock = stock_processor.get_processed_data()
    
    print("\n📰 Processing news data...")
    # Process news data
    news_processor = NewsPreprocessor(news_df)
    news_processor.clean_headlines()
    daily_news = news_processor.aggregate_daily()
    
    print("\n🔗 Aligning stock and news data...")
    # Combine data
    combined_df = align_stock_and_news(processed_stock, daily_news)
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nFinal dataset shape: {combined_df.shape}")
    print(f"Features: {combined_df.shape[1]}")
    print(f"Samples: {combined_df.shape[0]}")
    
    return combined_df


if __name__ == '__main__':
    # Test preprocessing with sample data
    from src.data_loader import load_stock_and_news
    
    print("="*60)
    print("🧪 TESTING PREPROCESSING MODULE")
    print("="*60)
    
    # Load sample data
    ticker = 'AAPL'
    start = '2024-01-01'
    end = '2024-12-31'
    
    stock_df, news_df = load_stock_and_news(ticker, start, end)
    
    if not stock_df.empty and not news_df.empty:
        # Preprocess
        processed_df = preprocess_all_data(stock_df, news_df)
        
        print("\n📋 Processed Data Preview:")
        print(processed_df.head())
        
        print("\n📊 Column Names:")
        print(processed_df.columns.tolist())
        
        print("\n📈 Target Distribution:")
        print(processed_df['Target'].value_counts())
    else:
        print("❌ Could not load data for testing")
