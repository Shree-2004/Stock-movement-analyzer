"""
Streamlit Dashboard - Interactive web app for stock prediction analysis.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from src.data_loader import load_stock_and_news
from src.model import compare_models_with_without_sentiment
from src.features import create_feature_matrix


# Page config
st.set_page_config(
    page_title="Stock Movement Analyzer",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 News Sentiment + Stock Movement Analyzer")
st.markdown("*Predict stock movements using ML and news sentiment analysis*")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Stock ticker dropdown with Indian stocks on top
indian_stocks = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "WIPRO.NS", "MARUTI.NS", "TITAN.NS"
]

us_stocks = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "JPM", "V", "WMT"
]

# Combine with separator
stock_options = ["--- Indian Stocks (NSE) ---"] + indian_stocks + \
                ["--- US Stocks ---"] + us_stocks

ticker = st.sidebar.selectbox(
    "Stock Ticker",
    options=stock_options,
    index=1,  # Default to first Indian stock (RELIANCE.NS)
    help="Select a stock ticker. Indian stocks use .NS suffix (NSE) or .BO (BSE)"
)

# Skip separator options
if ticker.startswith("---"):
    ticker = indian_stocks[0]  # Default to RELIANCE if separator selected

ticker = ticker.upper()

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime(2024, 1, 1)
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now()
    )

test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 30) / 100

run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")

# Main content
if run_analysis:
    with st.spinner(f"Loading data for {ticker}..."):
        stock_df, news_df = load_stock_and_news(
            ticker,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    
    if stock_df.empty:
        st.error(f"❌ Could not load data for {ticker}")
    else:
        # Display data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Stock Data Points", len(stock_df))
        with col2:
            st.metric("📰 News Articles", len(news_df))
        with col3:
            latest_price = stock_df['Close'].iloc[-1]
            # Detect currency based on ticker
            if ticker.endswith('.NS') or ticker.endswith('.BO'):
                currency_symbol = "₹"
                currency_name = "INR"
            else:
                currency_symbol = "$"
                currency_name = "USD"
            st.metric("💰 Latest Price", f"{currency_symbol}{latest_price:.2f}")
        
        # Stock price chart
        st.subheader("📈 Stock Price History")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=stock_df.index,
            open=stock_df['Open'],
            high=stock_df['High'],
            low=stock_df['Low'],
            close=stock_df['Close'],
            name='Price',
            hovertext=[f"Date: {date.strftime('%Y-%m-%d')}<br>"
                      f"Open: {currency_symbol}{row['Open']:.2f}<br>"
                      f"High: {currency_symbol}{row['High']:.2f}<br>"
                      f"Low: {currency_symbol}{row['Low']:.2f}<br>"
                      f"Close: {currency_symbol}{row['Close']:.2f}<br>"
                      f"Volume: {row['Volume']:,.0f}"
                      for date, row in stock_df.iterrows()],
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title=f"{ticker} Stock Price",
            yaxis_title=f"Price ({currency_symbol})",
            xaxis_title="Date",
            height=400,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent News Section
        if not news_df.empty:
            st.markdown("---")
            st.subheader("📰 Recent News Headlines with Sentiment")
            
            # Analyze sentiment for news
            from src.sentiment import FinBERTSentimentAnalyzer, add_sentiment_to_news
            
            with st.spinner("Analyzing news sentiment..."):
                analyzer = FinBERTSentimentAnalyzer()
                news_with_sentiment = add_sentiment_to_news(news_df.copy(), analyzer)
            
            # Get last 10 news articles
            recent_news = news_with_sentiment.tail(10).sort_values('date', ascending=False)
            
            if len(recent_news) > 0:
                for idx, row in recent_news.iterrows():
                    date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
                    headline = row['headline']
                    sentiment = row.get('sentiment', 'neutral')
                    sentiment_score = row.get('sentiment_score', 0.0)
                    
                    # Color code based on sentiment
                    if sentiment == 'positive':
                        badge_color = "🟢"
                        sentiment_text = f"<span style='color: green; font-weight: bold;'>Positive ({sentiment_score:.2f})</span>"
                    elif sentiment == 'negative':
                        badge_color = "🔴"
                        sentiment_text = f"<span style='color: red; font-weight: bold;'>Negative ({sentiment_score:.2f})</span>"
                    else:
                        badge_color = "🟡"
                        sentiment_text = f"<span style='color: orange; font-weight: bold;'>Neutral ({sentiment_score:.2f})</span>"
                    
                    # Display in an expander
                    with st.expander(f"{badge_color} {date_str} - {headline[:70]}..."):
                        st.write(f"**Full Headline:** {headline}")
                        st.markdown(f"**Sentiment:** {sentiment_text}", unsafe_allow_html=True)
            else:
                st.info("No recent news available for this period")
        
        # Run ML models
        st.markdown("---")
        st.subheader("🤖 Machine Learning Analysis")
        
        with st.spinner("Training models... This may take a few minutes..."):
            results = compare_models_with_without_sentiment(stock_df, news_df, test_size)
        
        # Display comparison
        st.markdown("### 📊 Model Performance Comparison")
        
        comparison_df = results['comparison_df']
        st.dataframe(comparison_df, use_container_width=True)
        
        # Best model
        best_model = results['best_model']
        best_acc = results['with_sentiment'][best_model]['results']['accuracy']
        
        st.success(f"🏆 Best Model: **{best_model.upper()}** with **{best_acc:.2%}** accuracy")
        
        # Next-Day Prediction Section
        st.markdown("---")
        st.markdown("### 🔮 Next-Day Prediction")
        
        # Get the best model's prediction for the last day
        best_model_obj = results['with_sentiment'][best_model]['model']
        
        # Get latest features
        from src.features import create_feature_matrix
        features_df, _, _ = create_feature_matrix(stock_df, news_df, include_sentiment=True)
        
        if len(features_df) > 0:
            # Get last row features
            last_features = features_df.iloc[[-1]]
            
            # Make prediction
            prediction = best_model_obj.predict(last_features)[0]
            prediction_proba = best_model_obj.predict_proba(last_features)[0]
            
            # Display prediction
            col1, col2, col3 = st.columns([2, 2, 3])
            
            with col1:
                if prediction == 1:
                    st.markdown("### 📈 **UP**")
                    st.markdown(f"<div style='background-color: #28a745; padding: 20px; border-radius: 10px; text-align: center;'>"
                               f"<h1 style='color: white; margin: 0;'>⬆️ UP</h1>"
                               f"<p style='color: white; margin: 0;'>Predicted Direction</p></div>", 
                               unsafe_allow_html=True)
                else:
                    st.markdown("### 📉 **DOWN**")
                    st.markdown(f"<div style='background-color: #dc3545; padding: 20px; border-radius: 10px; text-align: center;'>"
                               f"<h1 style='color: white; margin: 0;'>⬇️ DOWN</h1>"
                               f"<p style='color: white; margin: 0;'>Predicted Direction</p></div>", 
                               unsafe_allow_html=True)
            
            with col2:
                confidence = max(prediction_proba) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
                st.metric("Model Used", best_model.upper())
            
            with col3:
                st.markdown("**Prediction Details:**")
                st.write(f"- **Up Probability:** {prediction_proba[1]*100:.1f}%")
                st.write(f"- **Down Probability:** {prediction_proba[0]*100:.1f}%")
                st.write(f"- **Latest Price:** {currency_symbol}{stock_df['Close'].iloc[-1]:.2f}")
                
                if not news_df.empty:
                    try:
                        # Convert both dates to compatible datetime types (timezone-naive)
                        last_date = pd.Timestamp(stock_df.index[-1])
                        # Remove timezone if present
                        if hasattr(last_date, 'tz') and last_date.tz is not None:
                            last_date = last_date.tz_localize(None)
                        cutoff_date = last_date - pd.Timedelta(days=7)
                        
                        # Ensure news_df['date'] is datetime64 and timezone-naive
                        news_dates = pd.to_datetime(news_df['date'])
                        if hasattr(news_dates.dt, 'tz') and news_dates.dt.tz is not None:
                            news_dates = news_dates.dt.tz_localize(None)
                        
                        recent_news_count = len(news_df[news_dates >= cutoff_date])
                        st.write(f"- **News (Last 7 days):** {recent_news_count} articles")
                    except Exception as e:
                        # Silently skip if there's any datetime comparison issue
                        pass
            
            st.info(f"💡 **Note:** This prediction is based on the **{best_model.upper()}** model trained on historical data. "
                   f"The model achieved **{best_acc:.1%}** accuracy on test data. Past performance does not guarantee future results.")
        
        st.markdown("---")
        
        # Detailed results for best model
        st.markdown(f"### 🔍 Detailed Results: {best_model.upper()}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Without Sentiment")
            results_no_sent = results['without_sentiment'][best_model]['results']
            st.metric("Accuracy", f"{results_no_sent['accuracy']:.2%}")
            st.metric("Precision", f"{results_no_sent['precision']:.2%}")
            st.metric("Recall", f"{results_no_sent['recall']:.2%}")
            st.metric("F1-Score", f"{results_no_sent['f1']:.2%}")
        
        with col2:
            st.markdown("#### With Sentiment")
            results_with_sent = results['with_sentiment'][best_model]['results']
            st.metric("Accuracy", f"{results_with_sent['accuracy']:.2%}")
            st.metric("Precision", f"{results_with_sent['precision']:.2%}")
            st.metric("Recall", f"{results_with_sent['recall']:.2%}")
            st.metric("F1-Score", f"{results_with_sent['f1']:.2%}")
        
        # Confusion matrices
        st.markdown("### 📋 Confusion Matrices")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Without Sentiment")
            cm_no = results_no_sent['confusion_matrix']
            fig_cm_no = px.imshow(
                cm_no,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Down', 'Up'],
                y=['Down', 'Up'],
                text_auto=True,
                color_continuous_scale='Blues'
            )
            fig_cm_no.update_layout(height=300)
            st.plotly_chart(fig_cm_no, use_container_width=True)
        
        with col2:
            st.markdown("#### With Sentiment")
            cm_with = results_with_sent['confusion_matrix']
            fig_cm_with = px.imshow(
                cm_with,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Down', 'Up'],
                y=['Down', 'Up'],
                text_auto=True,
                color_continuous_scale='Greens'
            )
            fig_cm_with.update_layout(height=300)
            st.plotly_chart(fig_cm_with, use_container_width=True)
        
        # Feature importance
        st.markdown("### 🎯 Feature Importance")
        
        model_obj = results['with_sentiment'][best_model]['model']
        features_df, _, feature_names = create_feature_matrix(stock_df, news_df, include_sentiment=True)
        importance_df = model_obj.get_feature_importance(feature_names)
        
        fig_importance = px.bar(
            importance_df.head(15),
            x='importance',
            y='feature',
            orientation='h',
            title=f"Top 15 Features - {best_model.upper()}",
            labels={'importance': 'Importance Score', 'feature': 'Feature'}
        )
        fig_importance.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Key insights
        st.markdown("### 💡 Key Insights")
        
        for model_type in ['logistic', 'random_forest', 'xgboost']:
            acc_without = results['without_sentiment'][model_type]['results']['accuracy']
            acc_with = results['with_sentiment'][model_type]['results']['accuracy']
            improvement = acc_with - acc_without
            
            if improvement > 0:
                st.success(f"✅ **{model_type.upper()}**: Sentiment improved accuracy by **{improvement*100:.2f}%**")
            elif improvement < 0:
                st.warning(f"⚠️ **{model_type.upper()}**: Sentiment decreased accuracy by **{abs(improvement)*100:.2f}%**")
            else:
                st.info(f"➖ **{model_type.upper()}**: No change with sentiment")

else:
    # Welcome screen
    st.info("👈 Configure settings in the sidebar and click **Run Analysis** to start!")
    
    st.markdown("""
    ## 🎯 What This App Does
    
    This application predicts **next-day stock price direction (Up/Down)** using:
    
    1. **📊 Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands
    2. **📰 News Sentiment**: FinBERT-based analysis of financial news
    3. **🤖 Machine Learning**: Logistic Regression, Random Forest, XGBoost
    
    ## 🔬 The Key Question
    
    **Does news sentiment improve stock movement prediction accuracy?**
    
    ## 📚 How It Works
    
    1. Fetches stock price data from Yahoo Finance
    2. Collects financial news headlines
    3. Analyzes sentiment using FinBERT (financial BERT model)
    4. Computes technical indicators
    5. Trains ML models with and without sentiment features
    6. Compares performance to measure sentiment impact
    
    ## 🚀 Get Started
    
    1. Select a stock ticker from the dropdown (Indian or US stocks)
    2. Select date range
    3. Click **Run Analysis**
    4. Wait for results (may take 2-3 minutes)
    
    **Popular Indian Stocks**: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS  
    **Popular US Stocks**: AAPL, MSFT, GOOGL, TSLA
    
    ---
    
    *Built with Python, Streamlit, FinBERT, and scikit-learn*
    """)

# Footer
st.markdown("---")
st.markdown("*📈 News Sentiment + Stock Movement Analyzer | Built for ML Portfolio*")
