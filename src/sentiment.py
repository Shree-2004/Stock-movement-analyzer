"""
Sentiment Analysis Module - Analyze financial news sentiment using FinBERT.

This module provides functions to:
1. Load the FinBERT model from HuggingFace
2. Analyze sentiment of news headlines
3. Aggregate daily sentiment scores
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from tqdm import tqdm


class FinBERTSentimentAnalyzer:
    """
    Sentiment analyzer using FinBERT model.
    
    FinBERT is a BERT model pre-trained on financial text (earnings reports, news).
    It understands financial jargon better than generic sentiment models.
    """
    
    def __init__(self, model_name='ProsusAI/finbert'):
        """
        Initialize FinBERT sentiment analyzer.
        
        Args:
            model_name (str): HuggingFace model name
        """
        print(f"📥 Loading FinBERT model: {model_name}")
        print("   (First run will download ~500MB model)")
        
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Create pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
            
            device_name = "GPU" if self.device == 0 else "CPU"
            print(f"✅ FinBERT loaded successfully (using {device_name})")
            
        except Exception as e:
            print(f"❌ Error loading FinBERT: {e}")
            print("   Using fallback simple sentiment analyzer")
            self.sentiment_pipeline = None
    
    def analyze_single(self, text):
        """
        Analyze sentiment of a single text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: {'label': str, 'score': float, 'sentiment_value': float}
        """
        if not text or pd.isna(text):
            return {'label': 'neutral', 'score': 0.0, 'sentiment_value': 0.0}
        
        if self.sentiment_pipeline is None:
            return self._fallback_sentiment(text)
        
        try:
            # Truncate if too long
            text = text[:512]
            
            # Get sentiment
            result = self.sentiment_pipeline(text)[0]
            
            # Convert to numerical value: positive=1, neutral=0, negative=-1
            label = result['label'].lower()
            score = result['score']
            
            if 'positive' in label:
                sentiment_value = score
            elif 'negative' in label:
                sentiment_value = -score
            else:  # neutral
                sentiment_value = 0.0
            
            return {
                'label': label,
                'score': score,
                'sentiment_value': sentiment_value
            }
            
        except Exception as e:
            print(f"⚠️  Error analyzing text: {e}")
            return {'label': 'neutral', 'score': 0.0, 'sentiment_value': 0.0}
    
    def analyze_batch(self, texts, batch_size=16):
        """
        Analyze sentiment of multiple texts in batches.
        
        Args:
            texts (list): List of texts to analyze
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of sentiment dictionaries
        """
        if self.sentiment_pipeline is None:
            return [self._fallback_sentiment(text) for text in texts]
        
        results = []
        
        print(f"🔍 Analyzing {len(texts)} headlines...")
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            
            # Truncate texts
            batch = [text[:512] if text and not pd.isna(text) else "" for text in batch]
            
            try:
                batch_results = self.sentiment_pipeline(batch)
                
                for result in batch_results:
                    label = result['label'].lower()
                    score = result['score']
                    
                    if 'positive' in label:
                        sentiment_value = score
                    elif 'negative' in label:
                        sentiment_value = -score
                    else:
                        sentiment_value = 0.0
                    
                    results.append({
                        'label': label,
                        'score': score,
                        'sentiment_value': sentiment_value
                    })
                    
            except Exception as e:
                print(f"⚠️  Error in batch: {e}")
                # Add neutral sentiments for failed batch
                results.extend([{'label': 'neutral', 'score': 0.0, 'sentiment_value': 0.0}] * len(batch))
        
        return results
    
    def _fallback_sentiment(self, text):
        """
        Simple fallback sentiment analyzer (keyword-based).
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment dictionary
        """
        if not text or pd.isna(text):
            return {'label': 'neutral', 'score': 0.0, 'sentiment_value': 0.0}
        
        text = text.lower()
        
        # Simple keyword lists
        positive_words = ['gain', 'profit', 'rise', 'surge', 'beat', 'strong', 'growth', 
                         'upgrade', 'bullish', 'high', 'success', 'win', 'positive']
        negative_words = ['loss', 'fall', 'drop', 'decline', 'miss', 'weak', 'concern',
                         'downgrade', 'bearish', 'low', 'fail', 'negative', 'risk']
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count > neg_count:
            sentiment_value = min(pos_count / 5, 1.0)
            return {'label': 'positive', 'score': sentiment_value, 'sentiment_value': sentiment_value}
        elif neg_count > pos_count:
            sentiment_value = -min(neg_count / 5, 1.0)
            return {'label': 'negative', 'score': abs(sentiment_value), 'sentiment_value': sentiment_value}
        else:
            return {'label': 'neutral', 'score': 0.0, 'sentiment_value': 0.0}


def add_sentiment_to_news(news_df, analyzer=None):
    """
    Add sentiment analysis to news DataFrame.
    
    Args:
        news_df (pd.DataFrame): News data with 'headline' or 'cleaned_headline' column
        analyzer (FinBERTSentimentAnalyzer): Sentiment analyzer instance
        
    Returns:
        pd.DataFrame: News data with sentiment columns added
    """
    if analyzer is None:
        analyzer = FinBERTSentimentAnalyzer()
    
    # Use cleaned headlines if available, otherwise use raw headlines
    text_column = 'cleaned_headline' if 'cleaned_headline' in news_df.columns else 'headline'
    
    # Analyze sentiments
    sentiments = analyzer.analyze_batch(news_df[text_column].tolist())
    
    # Add to DataFrame
    news_df['sentiment_label'] = [s['label'] for s in sentiments]
    news_df['sentiment_score'] = [s['score'] for s in sentiments]
    news_df['sentiment_value'] = [s['sentiment_value'] for s in sentiments]
    
    print(f"✅ Added sentiment analysis to {len(news_df)} headlines")
    
    return news_df


def aggregate_daily_sentiment(news_df):
    """
    Aggregate sentiment scores by date.
    
    Args:
        news_df (pd.DataFrame): News data with 'date' and sentiment columns
        
    Returns:
        pd.DataFrame: Daily aggregated sentiment features
    """
    # Group by date
    daily_sentiment = news_df.groupby('date').agg({
        'sentiment_value': ['mean', 'std', 'count'],
        'sentiment_label': lambda x: (x == 'positive').sum()
    }).reset_index()
    
    # Flatten column names
    daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std', 
                               'news_count', 'positive_count']
    
    # Fill NaN std with 0 (happens when only 1 news per day)
    daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0)
    
    # Calculate negative count
    daily_sentiment['negative_count'] = news_df.groupby('date').apply(
        lambda x: (x['sentiment_label'] == 'negative').sum()
    ).values
    
    # Calculate sentiment momentum (change from previous day)
    daily_sentiment['sentiment_momentum'] = daily_sentiment['sentiment_mean'].diff()
    daily_sentiment['sentiment_momentum'] = daily_sentiment['sentiment_momentum'].fillna(0)
    
    print(f"✅ Aggregated daily sentiment: {len(daily_sentiment)} days")
    print(f"   Average sentiment: {daily_sentiment['sentiment_mean'].mean():.3f}")
    print(f"   Positive days: {(daily_sentiment['sentiment_mean'] > 0).sum()}")
    print(f"   Negative days: {(daily_sentiment['sentiment_mean'] < 0).sum()}")
    
    return daily_sentiment


if __name__ == '__main__':
    # Test sentiment analysis
    from src.data_loader import load_stock_and_news
    
    print("="*60)
    print("🧪 TESTING SENTIMENT ANALYSIS MODULE")
    print("="*60 + "\n")
    
    # Test with sample headlines
    print("📝 Testing with sample headlines:\n")
    
    sample_headlines = [
        "Apple reports record quarterly earnings, stock surges",
        "Apple faces regulatory challenges in Europe",
        "Analysts upgrade Apple stock to buy rating",
        "Apple stock falls on weak iPhone sales",
        "Apple announces new product lineup"
    ]
    
    analyzer = FinBERTSentimentAnalyzer()
    
    for headline in sample_headlines:
        result = analyzer.analyze_single(headline)
        print(f"Headline: {headline}")
        print(f"  → {result['label'].upper()} (score: {result['sentiment_value']:.3f})\n")
    
    # Test with real news data
    print("\n" + "="*60)
    print("📰 Testing with news data")
    print("="*60 + "\n")
    
    ticker = 'AAPL'
    start = '2024-01-01'
    end = '2024-12-31'
    
    _, news_df = load_stock_and_news(ticker, start, end)
    
    if not news_df.empty:
        # Add sentiment
        news_with_sentiment = add_sentiment_to_news(news_df, analyzer)
        
        print("\n📊 News with Sentiment Preview:")
        print(news_with_sentiment[['date', 'headline', 'sentiment_label', 'sentiment_value']].head(10))
        
        # Aggregate daily
        daily_sentiment = aggregate_daily_sentiment(news_with_sentiment)
        
        print("\n📈 Daily Sentiment Preview:")
        print(daily_sentiment.head())
    else:
        print("❌ Could not load news data for testing")
