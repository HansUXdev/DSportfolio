# src/data_collection/load_data.py

import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime

def load_price_data(tickers, start='2000-01-01', end=None):
    """
    Load historical data for given tickers using yfinance.
    
    Parameters:
    tickers (str or list): A single ticker symbol or a list of ticker symbols.
    start (str): The start date for fetching data in 'YYYY-MM-DD' format.
    end (str): The end date for fetching data in 'YYYY-MM-DD' format. Defaults to today's date if None.
    
    Returns:
    pd.DataFrame: DataFrame containing historical price data.
    """
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)
    
    if data.empty:
        print(f"No data found for {tickers}")
        return pd.DataFrame()
    
    # Ensure all necessary columns are included for single ticker
    if isinstance(tickers, str):
        required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        if not all(col in data.columns for col in required_columns):
            print(f"Missing required columns in data for {tickers}. Available columns: {data.columns}")
            return pd.DataFrame()
    else:
        # For multiple tickers, ensure 'Adj Close' is present
        if 'Adj Close' not in data.columns.levels[0]:
            print(f"Missing 'Adj Close' column in data for {tickers}. Available columns: {data.columns.levels[0]}")
            return pd.DataFrame()

    return data

# # Example usage
# tickers = ['AAPL', 'MSFT']
# start_date = '2020-01-01'
# end_date = '2023-01-01'
# df = load_price_data(tickers, start=start_date, end=end_date)
# print(df.head())

def load_fed_data(series, start_date='2000-01-01'):
    return pdr.get_data_fred(series, start=start_date)


############################################################################################################
# Fetch Financial Data Function with Technical Indicators
############################################################################################################
def fetch_financial_data(ticker='SPY', start_year=1993, end_year=None, interval='1d', export_csv=False, csv_file=None,  calculate_indicators=False,):
    """
    Fetches data for a specified ticker from Yahoo Finance from the given start year to the current year or specified end year at specified intervals.
    
    Parameters:
        ticker (str): The ticker symbol for the asset. Defaults to 'SPY'.
        start_year (int): The year from which to start fetching the data. Defaults to 1993.
        end_year (int): The last year for which to fetch the data. Defaults to the current year if None.
        interval (str): The data interval ('1d' for daily, '1wk' for weekly, '1mo' for monthly, '1h' for hourly).
        export_csv (bool): Whether to export the data to a CSV file. Defaults to False.
        csv_file (str): The path of the CSV file to export the data to. Automatically determined if None.
    """
    # If end_year is not specified, use the current year
    # Hourly data can only be fetched for the last 730 days
    if end_year is None:
        end_year = pd.Timestamp.today().year
        
    # If csv_file is not specified, automatically generate a file name based on the ticker and interval
    if csv_file is None:
        csv_file = f'{ticker}_{interval}_data_{start_year}_to_{end_year}.csv'
    
    # Adjust end_date to ensure data is fetched through the end of the end_year
    end_date = f"{end_year}-12-31"
    
    # Download the data before attempting to access it
    data = yf.download(ticker, start=f'{start_year}-01-01', end=end_date, interval=interval)
    
    # Ensure 'data' has been successfully downloaded and is not empty
    if not data.empty and calculate_indicators:
        data = bollinger_bands(data)
        data = macd(data)
        data = rsi(data)
        data = woodie_pivots(data)
        data = obv(data)
        data = atr(data)
        data = stochastic_oscillator(data)

        # Non-stationary data processing
        data = calculate_price_differences(data, 'Close')  # Calculate price differences
        data = calculate_log_returns(data, 'Close')  # Calculate log returns for the 'Close' column
        data = calculate_volume_changes(data, 'Volume')  # Calculate volume changes


    else:
        print("Data download failed or returned an empty DataFrame.")
    
    if export_csv and not data.empty:
        data.to_csv(csv_file)
        print(f'Data exported to {csv_file}')
    
    return data 
############################################################################################################
# Example usage
############################################################################################################
# spy_data_daily = fetch_financial_data(ticker='SPY', interval='1d', export_csv=True, calculate_indicators=True)  # Daily data for SPY
# aapl_data_weekly = fetch_financial_data(ticker='AAPL', interval='1wk', start_year=2000, export_csv=True)  # Weekly data for AAPL starting from 2000
