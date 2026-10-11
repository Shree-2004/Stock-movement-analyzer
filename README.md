# 📈 News Sentiment + Stock Movement Analyzer

> **Predict stock price movements using machine learning, technical indicators, and news sentiment analysis**

> ### ⚠️ Status: the sentiment comparison has **not** been run on real news
>
> When no NewsAPI/Finnhub key is configured, the news pipeline falls back to
> **generated placeholder headlines** (`src/data_loader.py`,
> `_get_sample_news_data`): 8 fixed templates cycling every 3 days, identical
> for every ticker. Every cached run in this repo used that fallback.
>
> So the project's central question — *does news sentiment improve prediction
> accuracy?* — **is still open.** The price, indicator, split and model code is
> real and runs; the sentiment input is a stub. See
> [What is and isn't measured](#-what-is-and-isnt-measured).

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
git clone https://github.com/Shree-2004/stock-movement-analyzer.git
cd stock-movement-analyzer

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)

```powershell
# Copy template
copy .env.template .env

# Without a key the news pipeline returns PLACEHOLDER headlines, not real news.
# A real sentiment result requires a key (see the Status note at the top).
# Get free keys from:
# - NewsAPI: https://newsapi.org
# - Finnhub: https://finnhub.io
```

### 3. Run the Project

#### Option A: Command Line (Full Pipeline)

```powershell
# Set Python path
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
stock-movement-analyzer/
├── src/                    # Source code modules
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration settings
│   ├── utils.py           # Utility functions
│   ├── data_loader.py     # Stock (yfinance) + news; falls back to placeholders
│   ├── preprocessing.py   # Data cleaning & technical indicators
│   ├── sentiment.py       # FinBERT sentiment analysis
│   ├── features.py        # Feature engineering
│   ├── model.py           # ML model training & evaluation
├── data/                   # Data storage
│   └── cache/             # Cached API responses
├── models/                 # Trained model files
├── app.py                  # Streamlit dashboard
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

## 📊 What is and isn't measured

**Real and working.** OHLCV from Yahoo Finance; the five technical-indicator
families; a chronological train/test split with no shuffle; `StandardScaler`
fit on train only and `transform`-ed on test. There is no scaling or ordering
leak — that part was done correctly.

**Not established: anything about news sentiment.**

An earlier local run on AAPL 2024 reported 57.33% accuracy without sentiment
and 60.00% with, on a 75-day test set. Two separate problems make that
unusable as a result:

**1. The sentiment input was synthetic.** 8 templated headlines on a 24-day
loop, carrying no information about the stock by construction. The run also
ranked `Sentiment_mean` 2nd by feature importance — which is evidence of a
model fitting a periodic artifact, not evidence that sentiment has value.

**2. The test set is too small to separate those numbers from chance.** Every
reported figure is an integer count out of 75:

| | without sentiment | with sentiment | difference |
|---|---|---|---|
| Logistic Regression | 41/75 | 42/75 | +1 prediction |
| Random Forest | 42/75 | 44/75 | +2 predictions |
| XGBoost | 43/75 | 45/75 | +2 predictions |

The headline "+2.67% improvement" is **two extra correct predictions**. A 95%
confidence interval on 60% runs from **48.9% to 71.1%** — it includes chance
(z = 1.73 against a 50% null).

**What answering the question would require:** a real dated news corpus with
publication timestamps (so no article can postdate the day it is used to
predict), and a test set large enough to detect an effect of a few percentage
points — which, at these effect sizes, means thousands of observations rather
than 75.

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

⚠️ **STEP 4: Sentiment Analysis** (CODE COMPLETE, NOT VALIDATED — ran on placeholder headlines, not real news)
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

⚠️ **STEP 7: Evaluation** (RUN, BUT NOT CONCLUSIVE — see [What is and isn't measured](#-what-is-and-isnt-measured))
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
- **The sentiment result is not validated** — cached runs used placeholder
  headlines, and the 75-day test set cannot separate a few percentage points
  from chance. Treat the sentiment half as unfinished.
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
