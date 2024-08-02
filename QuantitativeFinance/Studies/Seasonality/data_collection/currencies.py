# src/data_collection/currencies.py

import yfinance as yf
from datetime import datetime

def load_currency_data(tickers, start='2000-01-01', end=None):
    """
    Load historical data for given currency pairs using yfinance.
    """
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data

def load_americas_currency_data(start='2000-01-01', end=None):
    """
    Load historical data for currency pairs in the Americas sector.
    """
    tickers = ['USDCAD=X', 'USDMXN=X', 'CADMXN=X']
    return load_currency_data(tickers, start, end)

def load_european_currency_data(start='2000-01-01', end=None):
    """
    Load historical data for currency pairs in the European sector.
    """
    tickers = ['EURUSD=X']
    return load_currency_data(tickers, start, end)

def load_asian_pacific_currency_data(start='2000-01-01', end=None):
    """
    Load historical data for currency pairs in the Asian Pacific sector.
    """
    tickers = ['USDJPY=X', 'USDKRW=X', 'JPYKRW=X']
    return load_currency_data(tickers, start, end)

def process_currency_data(df):
    """
    Process the currency data to calculate returns and other metrics.
    """
    df['Return'] = df.pct_change() * 100
    return df

# # Example usage
# if __name__ == "__main__":
#     start_date = '2000-01-01'
#     end_date = datetime.now().strftime('%Y-%m-%d')

#     # Load data for each sector
#     americas_data = load_americas_currency_data(start=start_date, end=end_date)
#     european_data = load_european_currency_data(start=start_date, end=end_date)
#     asian_pacific_data = load_asian_pacific_currency_data(start=start_date, end=end_date)

#     # Process the data
#     americas_data = process_currency_data(americas_data)
#     european_data = process_currency_data(european_data)
#     asian_pacific_data = process_currency_data(asian_pacific_data)

#     # Print the first few rows of each dataset
#     print("Americas Sector Data:\n", americas_data.head())
#     print("\nEuropean Sector Data:\n", european_data.head())
#     print("\nAsian Pacific Sector Data:\n", asian_pacific_data.head())
