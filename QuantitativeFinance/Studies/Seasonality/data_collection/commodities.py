# src/data_collection/commodities.py

import yfinance as yf
from datetime import datetime

def load_commodities_data(tickers, start='2000-01-01', end=None):
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data
