"""
Feature Engineering Module - Combine technical and sentiment features.

This module provides functions to:
1. Combine technical indicators with sentiment features
2. Create final feature matrix for ML models
3. Handle missing data and feature scaling
4. Ensure no data leakage
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_stock_and_news
from src.preprocessing import preprocess_all_data
from src.sentiment import FinBERTSentimentAnalyzer, add_sentiment_to_news, aggregate_daily_sentiment


def create_feature_matrix(stock_df, news_df, include_sentiment=True):
    """
    Create complete feature matrix combining technical and sentiment features.
    
    Args:
        stock_df (pd.DataFrame): Raw stock data
        news_df (pd.DataFrame): Raw news data
        include_sentiment (bool): Whether to include sentiment features
        
    Returns:
        tuple: (features_df, target_series, feature_names)
    """
    print("\n" + "="*60)
    print("🔧 CREATING FEATURE MATRIX")
    print("="*60 + "\n")
    
    # Step 1: Preprocess stock data
    from src.preprocessing import StockPreprocessor
    
    stock_processor = StockPreprocessor(stock_df)
    stock_processor.calculate_returns()
    stock_processor.create_target_label()
    stock_processor.add_technical_indicators()
    processed_stock = stock_processor.get_processed_data()
    
    # Step 2: Add sentiment features if requested
    if include_sentiment and not news_df.empty:
        print("📰 Adding sentiment features...")
        
        # Analyze sentiment
        analyzer = FinBERTSentimentAnalyzer()
        news_with_sentiment = add_sentiment_to_news(news_df, analyzer)
        
        # Aggregate daily
        daily_sentiment = aggregate_daily_sentiment(news_with_sentiment)
        
        # Merge with stock data
        # Fix timezone mismatch: convert both to timezone-naive
        processed_stock['date'] = pd.to_datetime(processed_stock.index).tz_localize(None)
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date']).dt.tz_localize(None)
        
        combined_df = processed_stock.merge(daily_sentiment, on='date', how='left')
        combined_df = combined_df.set_index('date')
        
        # Forward-fill sentiment for days without news
        sentiment_cols = ['sentiment_mean', 'sentiment_std', 'sentiment_momentum',
                         'positive_count', 'negative_count']
        for col in sentiment_cols:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].fillna(method='ffill').fillna(0)
        
        # Fill news_count with 0
        if 'news_count' in combined_df.columns:
            combined_df['news_count'] = combined_df['news_count'].fillna(0)
        
        print(f"✅ Added {len(sentiment_cols)} sentiment features")
    else:
        combined_df = processed_stock
        print("ℹ️  Skipping sentiment features")
    
    # Step 3: Select features for ML
    # Exclude target, returns, and OHLCV raw data
    exclude_cols = ['Target', 'Next_Day_Return', 'Returns', 
                   'Open', 'High', 'Low', 'Close', 'Volume']
    
    feature_cols = [col for col in combined_df.columns if col not in exclude_cols]
    
    # Clean data: handle infinity and NaN values
    # Replace infinity with NaN
    combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
    
    # Remove any remaining NaN rows
    combined_df = combined_df.dropna()
    
    features_df = combined_df[feature_cols]
    target_series = combined_df['Target']
    
    print(f"\n✅ Feature matrix created:")
    print(f"   - Features: {len(feature_cols)}")
    print(f"   - Samples: {len(features_df)}")
    print(f"   - Target distribution: Up={target_series.sum()}, Down={(target_series==0).sum()}")
    
    return features_df, target_series, feature_cols


def prepare_train_test_split(features_df, target_series, test_size=0.3):
    """
    Create time-series train/test split (NO random shuffle!).
    
    Args:
        features_df (pd.DataFrame): Feature matrix
        target_series (pd.Series): Target labels
        test_size (float): Proportion of data for testing
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print("\n" + "="*60)
    print("✂️  CREATING TRAIN/TEST SPLIT")
    print("="*60 + "\n")
    
    # Calculate split point
    split_idx = int(len(features_df) * (1 - test_size))
    
    # Split chronologically (NO shuffle!)
    X_train = features_df.iloc[:split_idx]
    X_test = features_df.iloc[split_idx:]
    y_train = target_series.iloc[:split_idx]
    y_test = target_series.iloc[split_idx:]
    
    print(f"📊 Split details:")
    print(f"   - Train period: {X_train.index[0]} to {X_train.index[-1]}")
    print(f"   - Test period: {X_test.index[0]} to {X_test.index[-1]}")
    print(f"   - Train samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print(f"✅ Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_feature_importance_names(feature_cols):
    """
    Get readable names for features.
    
    Args:
        feature_cols (list): List of feature column names
        
    Returns:
        dict: Mapping of feature names to readable descriptions
    """
    descriptions = {
        'SMA_10': '10-day Simple Moving Average',
        'SMA_20': '20-day Simple Moving Average',
        'SMA_50': '50-day Simple Moving Average',
        'EMA_12': '12-day Exponential Moving Average',
        'EMA_26': '26-day Exponential Moving Average',
        'RSI': 'Relative Strength Index',
        'MACD': 'MACD Line',
        'MACD_Signal': 'MACD Signal Line',
        'MACD_Diff': 'MACD Histogram',
        'BB_High': 'Bollinger Band Upper',
        'BB_Mid': 'Bollinger Band Middle',
        'BB_Low': 'Bollinger Band Lower',
        'BB_Width': 'Bollinger Band Width',
        'Price_to_SMA_20': 'Price / 20-day SMA Ratio',
        'Volume_Change': 'Volume Change %',
        'sentiment_mean': 'Average Daily Sentiment',
        'sentiment_std': 'Sentiment Volatility',
        'sentiment_momentum': 'Sentiment Change',
        'news_count': 'Number of News Articles',
        'positive_count': 'Positive News Count',
        'negative_count': 'Negative News Count'
    }
    
    return {col: descriptions.get(col, col) for col in feature_cols}


if __name__ == '__main__':
    # Test feature engineering
    print("="*60)
    print("🧪 TESTING FEATURE ENGINEERING MODULE")
    print("="*60)
    
    ticker = 'AAPL'
    start = '2024-01-01'
    end = '2024-12-31'
    
    # Load data
    stock_df, news_df = load_stock_and_news(ticker, start, end)
    
    if not stock_df.empty:
        # Test WITHOUT sentiment
        print("\n" + "="*60)
        print("🔬 Test 1: Features WITHOUT Sentiment")
        print("="*60)
        
        features_no_sent, target_no_sent, cols_no_sent = create_feature_matrix(
            stock_df, news_df, include_sentiment=False
        )
        
        X_train, X_test, y_train, y_test, scaler = prepare_train_test_split(
            features_no_sent, target_no_sent
        )
        
        print(f"\n📋 Features (no sentiment): {cols_no_sent}")
        
        # Test WITH sentiment
        print("\n\n" + "="*60)
        print("🔬 Test 2: Features WITH Sentiment")
        print("="*60)
        
        features_with_sent, target_with_sent, cols_with_sent = create_feature_matrix(
            stock_df, news_df, include_sentiment=True
        )
        
        X_train2, X_test2, y_train2, y_test2, scaler2 = prepare_train_test_split(
            features_with_sent, target_with_sent
        )
        
        print(f"\n📋 Features (with sentiment): {cols_with_sent}")
        
        print("\n" + "="*60)
        print("✅ FEATURE ENGINEERING TEST COMPLETE")
        print("="*60)
    else:
        print("❌ Could not load data for testing")
