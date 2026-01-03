"""
Utility functions for the News Stock Analyzer project.
"""

import os
import json
import pickle
from datetime import datetime, timedelta
import pandas as pd


def validate_date(date_str):
    """
    Validate date string format (YYYY-MM-DD).
    
    Args:
        date_str (str): Date string to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def get_trading_days(start_date, end_date):
    """
    Get list of trading days between start and end dates.
    
    Args:
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        
    Returns:
        list: List of trading day dates
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # B = business days
    return date_range.tolist()


def save_to_cache(data, filename, cache_dir='data/cache'):
    """
    Save data to cache file.
    
    Args:
        data: Data to save (DataFrame, dict, etc.)
        filename (str): Cache filename
        cache_dir (str): Cache directory path
    """
    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, filename)
    
    if isinstance(data, pd.DataFrame):
        data.to_pickle(filepath)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    print(f"✅ Data cached to {filepath}")


def load_from_cache(filename, cache_dir='data/cache'):
    """
    Load data from cache file.
    
    Args:
        filename (str): Cache filename
        cache_dir (str): Cache directory path
        
    Returns:
        Data from cache or None if not found
    """
    filepath = os.path.join(cache_dir, filename)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        # Try loading as pickle first
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Loaded data from cache: {filepath}")
        return data
    except:
        # Try loading as pandas DataFrame
        try:
            data = pd.read_pickle(filepath)
            print(f"✅ Loaded DataFrame from cache: {filepath}")
            return data
        except:
            return None


def check_data_quality(df, required_columns):
    """
    Check data quality and report issues.
    
    Args:
        df (pd.DataFrame): DataFrame to check
        required_columns (list): List of required column names
        
    Returns:
        dict: Quality report
    """
    report = {
        'missing_columns': [],
        'null_counts': {},
        'duplicate_rows': 0,
        'total_rows': len(df)
    }
    
    # Check for missing columns
    for col in required_columns:
        if col not in df.columns:
            report['missing_columns'].append(col)
    
    # Check for null values
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            report['null_counts'][col] = null_count
    
    # Check for duplicates
    report['duplicate_rows'] = df.duplicated().sum()
    
    return report


def print_data_quality_report(report):
    """
    Print data quality report in a readable format.
    
    Args:
        report (dict): Quality report from check_data_quality()
    """
    print("\n" + "="*50)
    print("📊 DATA QUALITY REPORT")
    print("="*50)
    
    print(f"\n📈 Total Rows: {report['total_rows']}")
    
    if report['missing_columns']:
        print(f"\n⚠️  Missing Columns: {', '.join(report['missing_columns'])}")
    else:
        print("\n✅ All required columns present")
    
    if report['null_counts']:
        print("\n⚠️  Null Values Found:")
        for col, count in report['null_counts'].items():
            print(f"   - {col}: {count} nulls")
    else:
        print("\n✅ No null values")
    
    if report['duplicate_rows'] > 0:
        print(f"\n⚠️  Duplicate Rows: {report['duplicate_rows']}")
    else:
        print("\n✅ No duplicate rows")
    
    print("="*50 + "\n")


def format_percentage(value):
    """
    Format decimal as percentage string.
    
    Args:
        value (float): Decimal value (e.g., 0.75)
        
    Returns:
        str: Formatted percentage (e.g., "75.00%")
    """
    return f"{value * 100:.2f}%"
