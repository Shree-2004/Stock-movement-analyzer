# 📈 News Sentiment + Stock Movement Analyzer

> **Predict stock price movements using machine learning, technical indicators, and news sentiment analysis**

## 🎯 Project Goal

This project builds an end-to-end system that predicts **next-day stock price direction (Up/Down)** by combining:
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands
- **News Sentiment**: FinBERT-based sentiment analysis of financial news
- **Machine Learning**: Logistic Regression, Random Forest, XGBoost

The key question: **Does news sentiment improve prediction accuracy?**

---

## 🚀 Quick Start

### 1. Setup Environment

```powershell
# Navigate to project directory
cd "C:\Users\Shree Londhe\Desktop\Stock Movement Analyzer\news_stock_analyzer"

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Verify installation
python -c "import pandas, yfinance, transformers; print('✅ All dependencies installed')"
```

### 2. Configure API Keys (Optional)

```powershell
# Copy template
copy .env.template .env

# Edit .env and add your API keys (optional - sample data works without keys)
# Get free keys from:
# - NewsAPI: https://newsapi.org
# - Finnhub: https://finnhub.io
```

### 3. Run the Project

#### Option A: Command Line (Full Pipeline)

```powershell
# Set Python path
$env:PYTHONPATH="C:\Users\Shree Londhe\Desktop\Stock Movement Analyzer\news_stock_analyzer"

# Run with default settings (AAPL, 2022-2024)
python main.py

# Run with custom parameters
python main.py --ticker MSFT --start 2023-01-01 --end 2024-12-31 --test-size 0.3
```

#### Option B: Streamlit Dashboard (Interactive)

```powershell
# Launch dashboard
streamlit run app.py

# Then open browser to http://localhost:8501
# Enter ticker, select dates, click "Run Analysis"
```

#### Option C: Test Individual Modules

```powershell
$env:PYTHONPATH="C:\Users\Shree Londhe\Desktop\Stock Movement Analyzer\news_stock_analyzer"

# Test data loader
python -m src.data_loader

# Test preprocessing
python -m src.preprocessing

# Test sentiment analysis
python -m src.sentiment

# Test feature engineering
python -m src.features

# Test ML models
python -m src.model
```

---

## 📁 Project Structure

```
news_stock_analyzer/
├── src/                    # Source code modules
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration settings
│   ├── utils.py           # Utility functions
│   ├── data_loader.py     # Stock & news data collection (coming next)
│   ├── preprocessing.py   # Data cleaning & technical indicators
│   ├── sentiment.py       # FinBERT sentiment analysis
│   ├── features.py        # Feature engineering
│   ├── model.py           # ML model training & evaluation
│   └── evaluation.py      # Performance metrics & visualization
├── data/                   # Data storage
│   └── cache/             # Cached API responses
├── models/                 # Trained model files
├── notebooks/              # Jupyter notebooks for exploration
├── app.py                  # Streamlit dashboard (coming soon)
├── requirements.txt        # Python dependencies
├── .env.template          # API key template
└── README.md              # This file
```

---

## 🧠 Methodology

### Step 1: Problem Definition
- **Task**: Binary classification (Up/Down)
- **Why direction vs. price?** Stock prices are non-stationary; direction is more stable and actionable
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### Step 2: Data Collection
- **Stock Data**: OHLCV from Yahoo Finance (free, no API key)
- **News Data**: Company-specific headlines from NewsAPI or Finnhub

### Step 3: Feature Engineering
- **Technical Indicators**: Capture price momentum, trends, volatility
- **Sentiment Features**: Aggregate daily news sentiment
- **Target Label**: 1 if next-day return > 0, else 0

### Step 4: Model Training
- **Time-series split**: Train on past, test on future (no data leakage!)
- **Models**: Logistic Regression (baseline), Random Forest, XGBoost
- **Comparison**: With vs. without sentiment features

### Step 5: Evaluation
- Compare model performance
- Analyze when sentiment helps (earnings seasons? major events?)
- Visualize results in Streamlit dashboard

---

## 📊 Expected Results

- **Baseline (Technical only)**: 52-58% accuracy
- **With Sentiment**: 55-62% accuracy (3-5% improvement)
- **Best Model**: Random Forest or XGBoost
- **Key Insight**: Sentiment helps most during high-volatility events

---

## 🔑 API Keys

| Service | Purpose | Free Tier | Sign Up |
|---------|---------|-----------|---------|
| **yfinance** | Stock prices | Unlimited | No key needed |
| **NewsAPI** | News headlines | 100 requests/day | [newsapi.org](https://newsapi.org) |
| **Finnhub** | News headlines | 60 calls/minute | [finnhub.io](https://finnhub.io) |
| **HuggingFace** | FinBERT model | Unlimited (local) | No key needed |

---

## 🛠️ Technologies Used

- **Data**: pandas, numpy, yfinance
- **NLP**: transformers (FinBERT), torch, nltk
- **ML**: scikit-learn, xgboost
- **Technical Indicators**: ta (Technical Analysis library)
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard**: Streamlit

---

## 📚 What We Built

✅ **STEP 1: Project Setup** (COMPLETED)
- Created project structure
- Set up virtual environment
- Installed dependencies
- Created configuration files

✅ **STEP 2: Data Collection** (COMPLETED)
- Implemented stock data loader (yfinance)
- Implemented news data fetcher (NewsAPI/Finnhub)
- Added caching and error handling
- Tested data retrieval

✅ **STEP 3: Data Preprocessing** (COMPLETED)
- Calculated technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Cleaned news headlines
- Aligned data by date
- Created target labels

✅ **STEP 4: Sentiment Analysis** (COMPLETED)
- Loaded FinBERT model
- Analyzed news sentiment
- Aggregated daily scores
- Tested sentiment pipeline

✅ **STEP 5: Feature Engineering** (COMPLETED)
- Combined technical + sentiment features
- Implemented time-series split
- Added feature scaling
- Ensured no data leakage

✅ **STEP 6: Model Training** (COMPLETED)
- Trained Logistic Regression
- Trained Random Forest
- Trained XGBoost
- Compared with/without sentiment

✅ **STEP 7: Evaluation** (COMPLETED)
- Generated confusion matrices
- Calculated all metrics
- Created visualizations
- Analyzed sentiment impact

✅ **STEP 8: Dashboard** (COMPLETED)
- Built Streamlit app
- Added interactive charts
- Implemented model comparison
- Created feature importance viz

✅ **STEP 9: Documentation** (COMPLETED)
- Comprehensive README
- Usage examples
- Interview prep notes
- Complete walkthrough

---

## 🎓 Interview Talking Points

### Why this project?
- Combines **NLP + Finance + ML** - three hot areas
- Real-world application with measurable impact
- Demonstrates end-to-end ML pipeline

### Key Challenges
- **Data alignment**: Matching news dates with trading days
- **Avoiding data leakage**: Only use past data to predict future
- **Sentiment accuracy**: Financial news requires domain-specific models (FinBERT)
- **Class imbalance**: Markets tend to go up over time

### Limitations
- Only considers news sentiment (not fundamentals, macroeconomics)
- Historical patterns may not hold in future
- Transaction costs not included
- Assumes instant execution

### Future Improvements
- Multi-stock portfolio approach
- Social media sentiment (Twitter, Reddit)
- Deep learning (LSTM, Transformers)
- Real-time prediction system
- Risk management strategies

---

## 📝 License

This project is for educational purposes.

---

## 👨‍💻 Author

Built as a comprehensive ML/NLP portfolio project.

---

## 🙏 Acknowledgments

- **FinBERT**: ProsusAI for the financial sentiment model
- **yfinance**: For free stock market data
- **NewsAPI/Finnhub**: For news data APIs

---

**Ready to predict the market? Let's go! 🚀**
