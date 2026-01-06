"""
Simple Momentum Strategy - Main Execution
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from simple_momentum import SimpleMomentumStrategy


def load_data(tickers, start_date, end_date):
    """Download price data from Yahoo Finance."""
    print(f"\nDownloading data for {len(tickers)} stocks from Yahoo Finance...")
    print(f"Period: {start_date} to {end_date}")
    
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        prices = data['Close']
    
    # Drop any stocks with all NaN
    prices = prices.dropna(axis=1, how='all')
    
    print(f"✓ Downloaded {len(prices)} days for {len(prices.columns)} stocks")
    
    return prices


def main():
    """Run simple momentum backtest."""
    
    # Universe: 20 liquid large-caps
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'TSLA', 'NVDA', 'JPM', 'V', 'WMT',
        'JNJ', 'PG', 'MA', 'UNH', 'HD',
        'DIS', 'BAC', 'ADBE', 'NFLX', 'CRM'
    ]
    
    # Time period: 3 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    
    # Load data
    prices = load_data(
        tickers,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    # Run strategy
    strategy = SimpleMomentumStrategy(prices, lookback_days=252)
    
    results = strategy.run_backtest(
        n_long=10,
        n_short=10,
        rebalance_days=21,  # Monthly rebalance
        train_test_split=0.7
    )
    
    print("\n" + "="*60)
    print("BACKTEST COMPLETE!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = main()
