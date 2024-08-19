import yfinance as yf
import pandas as pd
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


def fetch_fed_data(series, start_date='2008-01-01'):
    """
    Fetch macroeconomic data from FRED.
    """
    return pdr.get_data_fred(series, start=start_date)

def merge_macro_data(financial_df, macro_df):
    """
    Merge financial data with macroeconomic data.
    """
    macro_df = macro_df.resample('M').ffill()  # Ensure macro data is also monthly
    merged_df = financial_df.merge(macro_df, left_index=True, right_index=True, how='inner')
    print("Merged Data:\n", merged_df.head())
    return merged_df


############################################################################################################
# Fundementals
############################################################################################################
def get_fundamentals_data(ticker):
    data = yf.Ticker(ticker)
    fundamentals = {
        'balance_sheet': data.balance_sheet,
        'cashflow': data.cashflow,
        'earnings': data.earnings,
        'financials': data.financials
    }
    return fundamentals


# Financial Data Fetching

def fetch_fundamentals(ticker):
    """
    Fetches comprehensive fundamental data for a given ticker, including balance sheet and cash flow,
    and returns it as a DataFrame.

    Args:
    - ticker (str): The ticker symbol of the stock.

    Returns:
    - DataFrame: DataFrame with merged market, technical, and fundamental data.
    """
    try:
        # Define start date and end date based on current date and one year ago
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        ticker_obj = yf.Ticker(ticker)
        
        # Fetch Beta from ticker's info
        beta_value = ticker_obj.info.get('beta', 0)
        
        balance_sheet = ticker_obj.balance_sheet
        cashflow = ticker_obj.cashflow

        balance_sheet_transposed = balance_sheet.T
        cashflow_transposed = cashflow.T

        fundamentals = pd.concat([balance_sheet_transposed, cashflow_transposed], axis=1)
        fundamentals.index.names = ['Date']
        
        # Insert Beta as the first column
        fundamentals.insert(0, 'Beta', beta_value)

        fundamentals.fillna(method='backfill', inplace=True)
        fundamentals.fillna(method='ffill', inplace=True)
        fundamentals.fillna(0, inplace=True)

        # Example of calculating growth rate of free cash flows (replace with your actual data)
        free_cash_flows = pd.Series([100, 120, 140, 160, 180])
        growth_rate = free_cash_flows.pct_change().mean()
        print("Free Cash Flow Growth Rate:", growth_rate)

        return fundamentals

    except Exception as e:
        print(f"Failed to fetch or process fundamental data for {ticker}: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of failure

def fetch_financial_data(ticker, start_year=1993, end_year=None, interval='1d', export_csv=False, csv_file=None, calculate_indicators=True):
    # Define the expected types for each argument
    expected_types = {
        'ticker': str,
        'start_year': int,
        'end_year': (int, type(None)),  # int or None
        'interval': str,
        'export_csv': bool,
        'csv_file': (str, type(None)),  # str or None
        'calculate_indicators': bool
    }

    # Gather the current arguments in a dictionary
    arguments = {
        'ticker': ticker,
        'start_year': start_year,
        'end_year': end_year,
        'interval': interval,
        'export_csv': export_csv,
        'csv_file': csv_file,
        'calculate_indicators': calculate_indicators
    }

    # Check argument types
    check_argument_types(arguments, expected_types)

    if end_year is None:
        end_year = pd.Timestamp.today().year

    if csv_file is None:
        csv_file = f'{ticker}_{interval}_data_{start_year}_to_{end_year}.csv'

    end_date = f"{end_year}-12-31"
    print(f"DataTypes: ticker {type(ticker)}  start {type(start_year)}  end {type(end_date)}  export {type(export_csv)}   csv {type(csv_file)}    calc {type(calculate_indicators)}            ")
    data = yf.download(ticker, start=f'{start_year}-01-01', end=end_date, interval=interval)
    print(f"data: {type(data)}")
    print(f"Downloaded data type: {type(data)}")

    if not data.empty and calculate_indicators:
        data = bollinger_bands(data, 20, 2)
        print(f"After bollinger_bands: {type(data)}")
        data = macd(data)
        print(f"After macd: {type(data)}")
        data = rsi(data)
        print(f"After rsi: {type(data)}")
        data = woodie_pivots(data)
        print(f"After woodie_pivots: {type(data)}")
        data = atr(data)
        print(f"After atr: {type(data)}")
        data = stochastic_oscillator(data)
        print(f"After stochastic_oscillator: {type(data)}")

        data = calculate_price_differences(data, 'Close')
        print(f"After calculate_price_differences: {type(data)}")
        data = calculate_log_returns(data, 'Close')
        print(f"After calculate_log_returns: {type(data)}")
        data = calculate_volume_changes(data, 'Volume')
        print(f"After calculate_volume_changes: {type(data)}")
    else:
        print("Data download failed or returned an empty DataFrame.")

    if export_csv and not data.empty:
        data.to_csv(csv_file)
        print(f'Data exported to {csv_file}')

    return data
