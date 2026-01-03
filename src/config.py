"""
Configuration settings for the News Stock Analyzer project.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')

# Default Stock Settings
DEFAULT_TICKER = os.getenv('DEFAULT_STOCK_TICKER', 'AAPL')
START_DATE = os.getenv('START_DATE', '2022-01-01')
END_DATE = os.getenv('END_DATE', '2024-12-31')

# Technical Indicator Parameters
SMA_PERIODS = [10, 20, 50]
EMA_PERIODS = [12, 26]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2

# Model Parameters
TEST_SIZE = 0.3
RANDOM_STATE = 42
N_SPLITS = 5  # For time series cross-validation

# File Paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
CACHE_DIR = os.path.join(DATA_DIR, 'cache')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
