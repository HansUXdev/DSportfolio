#!/usr/bin/env python
# coding: utf-8

# In[284]:


get_ipython().system('pip install pandas numpy matplotlib yfinance backtrader datetime timedelta asyncio')
from IPython.core.display import clear_output
clear_output()


# In[285]:


import concurrent.futures

import yfinance as yf
import backtrader as bt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose


# In[286]:


# Set option to display all columns
pd.set_option('display.max_columns', None)

# Optionally, set the max rows displayed in the output as well
pd.set_option('display.max_rows', 100)


# # Introduction
# This notebook is designed to showcase a comprehensive analysis of a diverse portfolio, integrating Index Funds, Leveraged ETFs, monthly dividend REITs and ETFs, and quarterly dividend stocks. The analysis includes portfolio beta evaluation, backtesting a Dollar-Cost Averaging (DCA) strategy that optimizes timing and risk management with technical and quantative analysis, and hedged with options pricing.
# 

# In[293]:


# Define Portfolio Assets
index_funds = ['SPY', 'QQQ', 'DAX']
leveraged_etfs = ['TQQQ', 'UMDD', 'UDOW', 'SOXL', 'NVDL', 'TSLL']
monthly_dividend_reits_etfs = ['O', 'AGNC', 'CSHI', 'JEPI', 'NUSI']
quarterly_dividend_stocks = [
    'SPYD', 'MSFT', 'INTC', 'F', 'CSCO', 'BAC', 'PFE', 'BX', 'MO', 
    'DOW', 'WMT', 'T', 'KMB', 'SWK', 'IBM', 'PEP', 'KO', 'JNJ'
]
hedging = ['VIX', 'UVXY', 'SPXS' ]


# ### Beta in Finance
# 
# In finance, Beta (β) is a measure of a stock's volatility in relation to the market. It indicates how sensitive the price of a particular stock is to movements in the overall market. Specifically, beta quantifies the systematic risk of a stock, which is the risk that cannot be diversified away because it is inherent to the entire market.
# 
# A beta of 1 implies that the stock's price tends to move with the market, while a beta greater than 1 suggests that the stock is more volatile than the market. Conversely, a beta less than 1 indicates that the stock is less volatile than the market.
# 
# ### Functions Requiring Beta
# 
# Several functions in finance require beta as an input parameter. Here's how beta is utilized in each of them:
# 
# 1. **`get_cost_of_equity(risk_free_rate, beta, market_return)`**:
#    - Calculates the cost of equity using the Capital Asset Pricing Model (CAPM), where beta is a key component in assessing the risk of the stock relative to the market.
# 
# 2. **`calculate_expected_return(risk_free_rate, beta, market_return, market_risk_premium)`**:
#    - Utilizes beta within the Capital Asset Pricing Model (CAPM) to compute the expected return of an asset, where beta reflects the systematic risk associated with the asset.
# 
# 3. **`three_stage_dividend_discount_model(symbol, discount_rate)`**:
#    - Involves beta indirectly through `calculate_intrinsic_value`, where beta influences the discount rate used in the valuation.
# 
# 4. **`residual_income_model(net_income, equity, required_return)`**:
#    - Although not directly utilizing beta, the required return on equity (which is part of the model) can be derived from CAPM, where beta plays a pivotal role.
# 
# These functions rely on beta to estimate the cost of equity, expected returns, or to indirectly influence the discount rate used in valuation models, as beta serves as a measure of systematic risk inherent in the stock.
# 
# ### Beta Hedging
# Beta hedging is a risk management strategy used by investors to mitigate the impact of market fluctuations on their portfolios. It involves adjusting the beta of the portfolio to align with different market conditions. In this hypothetical scenario, maintaining a portfolio with a beta of 2-3 during bullish cycles implies a higher sensitivity to market movements, potentially leading to greater gains when the market is performing well. Conversely, adjusting the beta to -2 during bearish cycles aims to reduce losses or even profit from market downturns due to the inverse correlation. Transitioning to a beta of 0 during flat cycles indicates a focus on income generation rather than market movements, allowing for stable returns despite stagnant market conditions. By strategically adjusting the beta of the portfolio according to market cycles, investors aim to optimize risk-adjusted returns and better navigate various market environments.
# 

# In[294]:


def calculate_beta(stock_symbol, market_symbol='^GSPC', start='2020-01-01', end='2023-01-01'):
    """
    Calculate beta for a given stock symbol relative to a market index.

    Args:
        stock_symbol (str): Stock symbol of the company for which beta is to be calculated.
        market_symbol (str): Symbol of the market index. Default is '^GSPC' (S&P 500).
        start (str): Start date for fetching historical data in 'YYYY-MM-DD' format. Default is '2020-01-01'.
        end (str): End date for fetching historical data in 'YYYY-MM-DD' format. Default is '2023-01-01'.

    Returns:
        dict: A dictionary containing calculated betas using different methods or error messages.

    Notes:
        - Beta is calculated using different methods: Linear Regression, Covariance Method, Variance Ratio, and scipy's linregress.
        - If an error occurs during data download or calculation, it returns a dictionary with an 'error' key and corresponding error message.
    """
    try:
        stock_data = yf.download(stock_symbol, start=start, end=end)['Adj Close']
        market_data = yf.download(market_symbol, start=start, end=end)['Adj Close']
    except Exception as e:
        return {'error': f'Failed to download data: {str(e)}'}
    
    returns = pd.DataFrame({
        'stock_returns': stock_data.pct_change(),
        'market_returns': market_data.pct_change()
    }).dropna()
    
    betas = {}
    
    # Method 1: Linear Regression
    try:
        model = LinearRegression().fit(returns[['market_returns']], returns['stock_returns'])
        betas['linear_regression'] = model.coef_[0]
    except Exception as e:
        betas['linear_regression_error'] = str(e)
    
    # Method 2: Covariance Method
    try:
        covariance = returns.cov().iloc[0, 1]
        market_var = returns['market_returns'].var()
        betas['covariance_method'] = covariance / market_var
    except Exception as e:
        betas['covariance_method_error'] = str(e)
    
    # Method 3: Variance Ratio
    try:
        stock_var = returns['stock_returns'].var()
        betas['variance_ratio'] = stock_var / market_var
    except Exception as e:
        betas['variance_ratio_error'] = str(e)
    
    # Method 4: Using scipy linregress for an alternative linear regression method
    try:
        slope, _, _, _, _ = linregress(returns['market_returns'], returns['stock_returns'])
        betas['linregress'] = slope
    except Exception as e:
        betas['linregress_error'] = str(e)
    
    return betas


# In[295]:


# Combine all asset lists into a single dictionary for easier iteration
assets = {
    'Index Funds': index_funds,
    'LETFS': leveraged_etfs,
    'Monthly Dividend REITs/ETFs': monthly_dividend_reits_etfs,
    'Quarterly Dividend Stocks': quarterly_dividend_stocks,
    'Hedging ETFS': hedging
}

# Initialize a dictionary to store beta values for each asset category
beta_values = {category: {} for category in assets.keys()}

# Iterate through each category and asset to calculate beta values
for category, asset_list in assets.items():
    print(f"\nCalculating beta for {category}:")
    for asset in asset_list:
        try:
            beta = calculate_beta(asset)
            print(f"  {asset}:")
            for method, value in beta.items():
                print(f"    {method}: {value}")
        except Exception as e:
            print(f"  Error calculating beta for {asset}: {str(e)}")
        beta_values[category][asset] = beta

clear_output()


# In[ ]:


beta_values['Index Funds']


# In[ ]:


beta_values['LETFS']


# In[ ]:


beta_values['Monthly Dividend REITs/ETFs']


# In[ ]:


beta_values['Quarterly Dividend Stocks']


# In[ ]:


beta_values['Hedging ETFS']


# In[267]:


############################################################################################################
# Define Technical Indicators Functions
############################################################################################################
def bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    data['Bollinger_High'] = rolling_mean + (rolling_std * num_std)
    data['Bollinger_Low'] = rolling_mean - (rolling_std * num_std)
    return data

def macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

def rsi(data, periods=14, ema=True):
    close_delta = data['Close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    if ema:
        ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    else:
        ma_up = up.rolling(window=periods).mean()
        ma_down = down.rolling(window=periods).mean()
    rsi = ma_up / ma_down
    data['RSI'] = 100 - (100 / (1 + rsi))
    return data

def woodie_pivots(data):
    high = data['High']
    low = data['Low']
    close = data['Close']
    pivot = (high + low + 2 * close) / 4
    data['Pivot'] = pivot
    data['R1'] = 2 * pivot - low
    data['S1'] = 2 * pivot - high
    data['R2'] = pivot + (high - low)
    data['S2'] = pivot - (high - low)
    data['R3'] = high + 2 * (pivot - low)
    data['S3'] = low - 2 * (high - pivot)
    data['R4'] = pivot + 3 * (high - low)
    data['S4'] = pivot - 3 * (high - low)
    return data

def obv(data):
    data['OBV'] = np.where(data['Close'] > data['Close'].shift(1), data['Volume'],
                           np.where(data['Close'] < data['Close'].shift(1), -data['Volume'], 0)).cumsum()
    return data

def atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    data['ATR'] = true_range.rolling(window=window).mean()
    return data

def stochastic_oscillator(data, window=14):
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    data['%K'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    data['%D'] = data['%K'].rolling(window=3).mean()
    return data

############################################################################################################
# Process other non-stationary data
############################################################################################################
# Function to detrend time series data using a linear regression model
def detrend_data(data, column):
    # Linear regression model requires reshaped index as a feature
    X = np.arange(len(data)).reshape(-1, 1)
    y = data[column].values  # Original values to detrend
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict the trend
    trend = model.predict(X)
    
    # Detrend by subtracting the trend from the original data
    detrended = y - trend
    data[f'{column}_detrended'] = detrended
    
    # Return the detrended data and the trend for further analysis
    return data, trend

def seasonal_decomposition(data, column, period):
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(data[column], model='multiplicative', period=period)
    
    # Add components to DataFrame
    data['trend_component'] = decomposition.trend
    data['seasonal_component'] = decomposition.seasonal
    data['residual_component'] = decomposition.resid
    
    # Seasonally adjust the data
    data[column + '_seasonally_adjusted'] = data[column] / data['seasonal_component']
    
    return data

# Function to calculate price differences
def calculate_price_differences(data, column):
    data[f'{column}_diff'] = data[column].diff()
    return data

# Function to calculate log returns
def calculate_log_returns(data, column):
    data[f'{column}_log_return'] = np.log(data[column] / data[column].shift(1))
    return data

# Function to calculate volume changes
def calculate_volume_changes(data, volume_column):
    data[f'{volume_column}_changes'] = data[volume_column].diff()
    return data



# In[268]:


def fetch_and_merge_fundamentals(ticker, market_technical_data):
    """
    Fetches comprehensive fundamental data for a given ticker, including balance sheet and cash flow,
    and merges it with existing market and technical data.

    Args:
    - ticker (str): The ticker symbol of the stock.
    - market_technical_data (DataFrame): DataFrame with existing market and technical data.

    Returns:
    - DataFrame: Enhanced DataFrame with merged market, technical, and fundamental data.
    """
    try:
        ticker_obj = yf.Ticker(ticker)

        # Fetch balance sheet and cash flow data
        balance_sheet = ticker_obj.balance_sheet
        cashflow = ticker_obj.cashflow

        # Transform the data; ensure that the index is date and transpose the DataFrame
        balance_sheet_transposed = balance_sheet.T
        cashflow_transposed = cashflow.T

        # Combine balance sheet and cash flow data
        fundamentals = pd.concat([balance_sheet_transposed, cashflow_transposed], axis=1)

        # Make sure there's a 'Date' column for merging
        fundamentals.index.names = ['Date']

        # Merge with market and technical data
        combined_data = market_technical_data.merge(fundamentals, left_index=True, right_index=True, how='outer')

        # Handle missing values: backfill, forward fill, or fill with zeros
        combined_data.fillna(method='backfill', inplace=True)
        combined_data.fillna(method='ffill', inplace=True)
        combined_data.fillna(0, inplace=True)

        return combined_data

    except Exception as e:
        print(f"Failed to fetch or process fundamental data for {ticker}: {e}")
        return market_technical_data  # Return original data in case of failure


# In[269]:


############################################################################################################
# Fetch Financial Data Function with Technical Indicators
# TODO: Add concurency?
############################################################################################################
def fetch_financial_data(ticker='SPY', start_year=1993, end_year=None, interval='1d', calculate_indicators=False, include_fundamentals=False, export_csv=False, csv_file=None,):
    """
    Fetches data for a specified ticker from Yahoo Finance from the given start year to the current year or specified end year at specified intervals.
    
    Parameters:
        ticker (str): The ticker symbol for the asset. Defaults to 'SPY'.
        start_year (int): The year from which to start fetching the data. Defaults to 1993.
        end_year (int): The last year for which to fetch the data. Defaults to the current year if None.
        interval (str): The data interval ('1d' for daily, '1wk' for weekly, '1mo' for monthly, '1h' for hourly).
        export_csv (bool): Whether to export the data to a CSV file. Defaults to False.
        csv_file (str): The path of the CSV file to export the data to. Automatically determined if None.
        calculate_indicators (bool): Flag to calculate technical indicators.
        include_fundamentals (bool): Flag to include fundamental data.

    Returns:
        DataFrame: DataFrame containing the requested financial data.
    """
    # Adjust for hourly data to limit to the last 730 days
    if interval == '1h':
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    else:
        start_date = f'{start_year}-01-01'
    
    if end_year is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = f"{end_year}-12-31"
    
    if csv_file is None:
        csv_file = f'{ticker}_{interval}_data_{start_date}_to_{end_date}.csv'
    
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    
    if not data.empty:
        if calculate_indicators:
            # Here you would call your indicator functions on the `data` DataFrame
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
            # Handling NaN values by forward filling then dropping rows with NaN values
            data.ffill(inplace=True)
            data.dropna(inplace=True)
       # Fetch and include fundamental data
        if include_fundamentals:
            ticker_obj = yf.Ticker(ticker)
            try:
                # Fetch fundamental data
                dividends = ticker_obj.dividends.last('1Y').sum()  # Sum of dividends over the last year
                splits = len(ticker_obj.splits.last('1Y'))  # Count of splits over the last year
                # Attempt to summarize cashflow, financials, and balance_sheet
                latest_cashflow = ticker_obj.cashflow.iloc[:, 0]  # Latest cash flow data
                latest_financials = ticker_obj.financials.iloc[:, 0]  # Latest financial data
                latest_balance_sheet = ticker_obj.balance_sheet.iloc[:, 0]  # Latest balance sheet data
                
                # Create summary metrics (example: total cash flow, net income, total assets)
                total_cashflow = latest_cashflow.get('Total Cash From Operating Activities')
                net_income = latest_financials.get('Net Income')
                total_assets = latest_balance_sheet.get('Total Assets')
                
                # Append these as new columns to 'data'; handle missing values as needed
                data['Dividends_Sum_Last_Year'] = dividends
                data['Splits_Count_Last_Year'] = splits
                data['Total_Cashflow'] = total_cashflow
                data['Net_Income'] = net_income
                data['Total_Assets'] = total_assets
                
            except Exception as e:
                print(f"Failed to fetch fundamental data for {ticker}: {e}")
                # Optionally, handle missing fundamental data (e.g., fill with NaN or zeros)

            # Get split, dividend, and balance sheet data
            try:
                # Ensure the 'data' DataFrame has a 'Date' index for proper merging
                data.index.name = 'Date'
                
                # Call the fetch_and_merge_fundamentals function with our ticker and current 'data' DataFrame
                data = fetch_and_merge_fundamentals(ticker, data)
            except Exception as e:
                print(f"Failed to merge fundamental data for {ticker}: {e}")
                # Export CSV if requested
                if export_csv:
                    data.to_csv(csv_file)
                    print(f'Data exported to {csv_file}')

            else:
                print("Data download failed or returned an empty DataFrame.")
    return data


# In[278]:





# In[280]:


# Example usage
index_funds = ['SPY', 'QQQ', 'DAX']  # List of your funds
funds_data = fetch_data_for_all_funds(index_funds)

# Now, extract the data for each fund into its own variable
SPY_data = funds_data.get('SPY')
QQQ_data = funds_data.get('QQQ')
DAX_data = funds_data.get('DAX')
DAX_data
# # Initialize an empty dictionary to store data for each fund
# funds_data = {}

# # Iterate over the index funds to fetch data for each and store it in the dictionary
# for fund in index_funds:
#     print(f"Fetching data for {fund}...")
#     # Note: Adjust the parameters according to your fetch_financial_data function's definition
#     # Here it's assumed that fetch_financial_data only requires the ticker symbol as a parameter
#     funds_data[fund] = fetch_financial_data(ticker=fund, calculate_indicators=True)

# # Now, extract the data for each fund into its own variable
# SPY_data = funds_data['SPY']
# QQQ_data = funds_data['QQQ']
# DAX_data = funds_data.get('DAX')  # Using .get() for 'DAX' in case it's not available/fetched correctly


# In[ ]:





# In[281]:


# Initialize an empty dictionary to store data for each ETF
etfs_data = {}

# Iterate over the leveraged ETFs to fetch data for each and store it in the dictionary
for etf in leveraged_etfs:
    print(f"Fetching data for {etf}...")
    # Adjust parameters as per your fetch_financial_data function's requirements
    etfs_data[etf] = fetch_financial_data(ticker=etf, calculate_indicators=True)
# Now, let's assume you want to access the data specifically for TQQQ, UDOW, and SOXL
etfs_data['NVDL']
# TQQQ_data = etfs_data['TQQQ']
# UMDD_data = etfs_data['UMDD']
# UDOW_data = etfs_data['UDOW']
# SOXL_data = etfs_data['SOXL']
# NVDL_data = etfs_data['NVDL']
# TSLL_data = etfs_data['TSLL']
# BITX_data = etfs_data['BITX']


# In[276]:


monthly_dividend_data = {}

# Iterate over the monthly dividend REITs and ETFs to fetch data for each and store it in the dictionary
for asset in monthly_dividend_reits_etfs:
    print(f"Fetching data for {asset}...")
    # Adjust parameters as per your fetch_financial_data function's requirements
    monthly_dividend_data[asset] = fetch_financial_data(ticker=asset, calculate_indicators=True, include_fundamentals=True, export_csv=True)

# Access the data specifically for each asset
# monthly_dividend_data
# monthly_dividend_data['O']
# Access the market data for 'O'

# Clear the output before printing the data
clear_output()

# # o_market_data = monthly_dividend_data['O']
monthly_dividend_data['O']
# fetch_financial_data(ticker='O', calculate_indicators=True, include_fundamentals=True, export_csv=True)


# In[277]:


# Initialize an empty dictionary to store data for each stock
quarterly_dividend_data = {}

# Iterate over the quarterly dividend stocks to fetch data for each and store it in the dictionary
for stock in quarterly_dividend_stocks:
    print(f"Fetching data for {stock}...")
    quarterly_dividend_data[stock] = fetch_financial_data(ticker=stock, calculate_indicators=True, include_fundamentals=True)
# Now, you can access the data specifically for each stock, for example:
# MSFT_data = quarterly_dividend_data['MSFT']
# quarterly_dividend_data
# MSFT_data
clear_output()


# In[283]:


quarterly_dividend_data['MSFT']


# 

# # Valuation Models for Each Asset Class:

# In[226]:


def calculate_cashflow_growth_rate(free_cash_flows):
    return free_cash_flows.pct_change().mean()

def project_future_free_cash_flows(last_cash_flow, growth_rate, years):
    return [last_cash_flow * (1 + growth_rate) ** i for i in range(1, years + 1)]


def calculate_terminal_value(last_cash_flow, growth_rate, required_rate, years):
    return last_cash_flow * (1 + growth_rate) / (required_rate - growth_rate) / (1 + required_rate) ** years

def calculate_fair_value(discounted_cash_flows, terminal_value, outstanding_shares):
    total_present_value = sum(discounted_cash_flows) + terminal_value
    return total_present_value / outstanding_shares

def get_cost_of_equity(risk_free_rate, beta, market_return):
    return risk_free_rate + beta * (market_return - risk_free_rate)

def get_cost_of_debt(interest_rate, tax_rate):
    return interest_rate * (1 - tax_rate)

def get_proportions(market_value_equity, market_value_debt):
    total_value = market_value_equity + market_value_debt
    return market_value_equity / total_value, market_value_debt / total_value

def calculate_wacc(cost_of_equity, cost_of_debt, equity_proportion, debt_proportion, tax_rate):
    wacc = (cost_of_equity * equity_proportion) + ((1 - tax_rate) * cost_of_debt * debt_proportion)
    return wacc

def calculate_intrinsic_value(dividend_data, discount_rate):
    intrinsic_value = 0
    for year, dividend in enumerate(dividend_data, start=1):
        if year <= 5:
            growth_rate = 0.05
        elif 5 < year <= 10:
            growth_rate = 0.03
        else:
            growth_rate = 0.01
        intrinsic_value += dividend / ((1 + discount_rate) ** year)
    return intrinsic_value

def calculate_cost_of_equity(beta, risk_free_rate, market_return):
    """
    Calculate the cost of equity using the CAPM formula.
    
    :param beta: Beta of the stock
    :param risk_free_rate: Risk-free rate
    :param market_return: Expected market return
    :return: Cost of equity
    """
    return risk_free_rate + beta * (market_return - risk_free_rate)


# In[227]:


def dcf_valuation(cash_flows, discount_rate):
    """
    Calculate the present value of cash flows using the discounted cash flow (DCF) method.
    
    Args:
    - cash_flows (list): List of projected cash flows.
    - discount_rate (float): Discount rate (required rate of return).
    
    Returns:
    - float: Present value of the cash flows.
    """
    dcf_value = sum(cf / (1 + discount_rate)**n for n, cf in enumerate(cash_flows, start=1))
    return dcf_value

def calculate_expected_return(risk_free_rate, beta, market_return, market_risk_premium):
    """
    Calculate the expected return of an asset using the Capital Asset Pricing Model (CAPM).
    
    Args:
    - risk_free_rate (float): Risk-free rate (e.g., yield on Treasury bills).
    - beta (float): Beta coefficient of the asset.
    - market_return (float): Expected return of the market portfolio.
    - market_risk_premium (float): Market risk premium.
    
    Returns:
    - float: Expected return of the asset.
    """
    expected_return = risk_free_rate + beta * market_risk_premium
    return expected_return

def three_stage_dividend_discount_model(symbol, discount_rate):
    dividend_data = fetch_dividend_data(symbol)
    intrinsic_value = calculate_intrinsic_value(dividend_data, discount_rate)
    return intrinsic_value


def residual_income_model(net_income, equity, required_return):
    """
    Calculate the value of equity using the Residual Income Model.
    
    Args:
    - net_income (float): Net income of the company.
    - equity (float): Book value of equity.
    - required_return (float): Required rate of return on equity.
    
    Returns:
    - float: Estimated value of equity using the Residual Income Model.
    """
    # Calculate the present value of expected future residual income
    residual_income = net_income - (required_return * equity)
    
    # Value of equity is the book value of equity plus the present value of expected future residual income
    equity_value = equity + residual_income
    
    return equity_value



# In[ ]:





# In[ ]:





# # Backtesting for DCA Strategy

# In[ ]:





# # Options Pricing Analysis

# In[ ]:




