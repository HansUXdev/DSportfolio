from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression


import yfinance as yf
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

import yoptions as yo
import yfinance as yf
import numpy as np
from pandas_datareader import data as pdr
from datetime import datetime

from data_forecasting import (arima_forecast, garch_forecast, get_fundamental_ratios, accuracy_score, optimize_model_hyperparameter)
############################################################################################################
# Utility Functions
############################################################################################################
def resample_to_quarterly(df):
    return df.resample('Q').ffill()

def resample_to_monthly(df):
    return df.resample('M').ffill()

def resample_to_weekly(df):
    return df.resample('W').ffill()

def resample_to_daily(df):
    return df.resample('D').ffill()

def resample_to_hourly(df):
    return df.resample('H').ffill()




def calculate_returns(df):
    """Calculate the daily returns."""
    df['Return'] = df['Adj Close'].pct_change() * 100
    return df
def display_all_monthly_statistics(df):
    """Display all monthly statistics for a DataFrame."""
    df_monthly = resample_to_monthly(df)
    df_monthly['Monthly Return'] = df_monthly['Adj Close'].pct_change() * 100
    display_seasonality_stats(df_monthly)

def display_seasonality_stats(df):
    """Display seasonality statistics."""
    stats = df.groupby(df.index.month)['Monthly Return'].agg(['mean', 'std', 'max', 'min'])
    stats.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for month in stats.index:
        mean_return = stats.loc[month, 'mean']
        print(f"{month}: Mean = {mean_return:.2f}")



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

def load_data(ticker='SPY', start='2008-01-01', end=None):
    """
    Load historical data for a given ticker using yfinance.
    """
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end)
    return df

def half_kelly_criterion(mean_return, std_return):
    win_prob = (mean_return / std_return) ** 2 / ((mean_return / std_return) ** 2 + 1)
    loss_prob = 1 - win_prob
    odds = mean_return / std_return
    kelly_fraction = (win_prob - loss_prob) / odds
    half_kelly_fraction = kelly_fraction / 2
    return half_kelly_fraction

def calculate_half_kelly_fractions(seasonality_stats):
    seasonality_stats['half_kelly_fraction'] = seasonality_stats.apply(
        lambda row: half_kelly_criterion(row['mean'], row['std']), axis=1
    )
    return seasonality_stats
def position_size_half_kelly(signals, seasonality_stats, iv_series):
    signals['half_kelly_fraction'] = signals.index.month.map(seasonality_stats['half_kelly_fraction'])
    signals['position_size'] = signals['half_kelly_fraction'].fillna(0) * signals['Buy']
    return signals
def backtest_strategy_with_half_kelly(df, signals):
    df['Strategy Return'] = df['Monthly Return'] * signals['position_size'].shift(1)
    df['Cumulative Return'] = (1 + df['Monthly Return']/100).cumprod() - 1
    df['Cumulative Strategy Return'] = (1 + df['Strategy Return']/100).cumprod() - 1
    return df


def create_summary_csv(tickers, start_date, end_date, filename='summary.csv'):
    summary_data = []
    
    for ticker in tickers:
        df = load_price_data(ticker, start=start_date, end=end_date)
        
        if isinstance(df, pd.Series):
            df = df.to_frame(name='Adj Close')
        
        if 'Adj Close' not in df.columns:
            print(f"Column 'Adj Close' not found in the data for {ticker}. Available columns: {df.columns}")
            continue
        
        df = calculate_returns(df)
        seasonality_table = create_seasonality_table(df)
        
        for month, stats in seasonality_table.iterrows():
            mean_return = stats['mean']
            std_dev = stats['std']
            count = stats['count']
            positive_prob = stats['positive_prob']
            kelly_size = apply_kelly_method(mean_return, std_dev, positive_prob)
            
            summary_data.append({
                'Ticker': ticker,
                'Month': month,
                'Mean Return': mean_return,
                'Standard Deviation': std_dev,
                'Count': count,
                'Positive Probability': positive_prob,
                'Kelly Size': kelly_size
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(filename, index=False)
    print(f"Summary CSV created: {filename}")
def apply_kelly_method(mean_return, std_dev, win_prob):
    b = mean_return / std_dev  # Assuming b is the edge ratio
    kelly_fraction = win_prob - ((1 - win_prob) / b)
    return kelly_fraction

# Main Analysis Function
def analyze_ticker(ticker, start_date, end_date):
    df = load_price_data(ticker, start=start_date, end=end_date)
    
    if isinstance(df, pd.Series):
        df = df.to_frame(name='Adj Close')
    
    if 'Adj Close' not in df.columns:
        print(f"Column 'Adj Close' not found in the data for {ticker}. Available columns: {df.columns}")
        return
    
    if 'Close' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns:
        print(f"Columns 'Close', 'High', and 'Low' are required. Available columns: {df.columns}")
        return
    
    df['Return'] = df['Adj Close'].pct_change() * 100
    df = ichimoku_cloud(df)
    df = add_technical_indicators(df)

    # ARIMA and GARCH Forecasts
    arima_forecast(df)
    garch_forecast(df)

    # Fundamental Ratios
    pe_ratio, pb_ratio, debt_to_equity = get_fundamental_ratios(ticker)
    print(f"P/E Ratio: {pe_ratio}, P/B Ratio: {pb_ratio}, Debt to Equity: {debt_to_equity}")

    # Machine Learning with Hyperparameter Optimization
    df['Target'] = (df['Return'] > 0).astype(int)
    features = ['Adj Close', 'Return']
    X = df[features].shift(1).dropna()
    y = df['Target'].shift(1).dropna()
    X, y = X.align(y, join='inner')
    best_model = optimize_model_hyperparameters(X, y)
    y_pred = best_model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Optimized Model Accuracy: {accuracy:.2f}")

    # Backtest Strategy
    backtest_strategy(df)

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

############################################################################################################
# Technicals which should replace the need for ta-lib
############################################################################################################

def add_technical_indicators(df):
    try:
        df = bollinger_bands(df)
        df = macd(df)
        df = rsi(df)
        df = woodie_pivots(df)
        # df = obv(df)
        df = atr(df)
        df = stochastic_oscillator(df)
    except KeyError as e:
        print(f"Missing column for technical indicator calculation: {e}")
    return df

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

def ichimoku_cloud(df):
    if 'High' not in df.columns or 'Low' not in df.columns:
        print("Data does not contain 'High' or 'Low' columns necessary for Ichimoku Cloud.")
        return df

    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2

    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['kijun_sen'] = (high_26 + low_26) / 2

    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)

    df['chikou_span'] = df['Adj Close'].shift(-26)

    return df
def get_daily_woodies_pivots_with_bollinger(ticker, start_date, end_date, window=20, no_of_stds=2, strike_increment=1):
    # Fetch historical daily data
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    
    # Calculate Bollinger Bands
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    data['Bollinger High'] = rolling_mean + (rolling_std * no_of_stds)
    data['Bollinger Low'] = rolling_mean - (rolling_std * no_of_stds)
    

    # Calculate Woodie's pivot points
    data['Pivot Point'] = (data['High'] + data['Low'] + 2 * data['Close']) / 4
    data['R1'] = 2 * data['Pivot Point'] - data['Low']
    data['S1'] = 2 * data['Pivot Point'] - data['High']
    data['R2'] = data['Pivot Point'] + (data['High'] - data['Low'])     
    data['S2'] = data['Pivot Point'] - (data['High'] - data['Low'])
    data['R3'] = data['Pivot Point'] + 2 * (data['High'] - data['Low'])
    data['S3'] = data['Pivot Point'] - 2 * (data['High'] - data['Low']) 
    data['R4'] = data['Pivot Point'] + 3 * (data['High'] - data['Low'])

    # Round Bollinger Bands and pivot points to the nearest strike
    rounding_columns = ['Bollinger High', 'Bollinger Low', 'Pivot Point', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3', 'R4']
    for column in rounding_columns:
        data[column + '_Strike'] = np.round(data[column] / strike_increment) * strike_increment

    # Spreads between the pivot points
    data['Bearcall_Spread_width'] = data['R1_Strike'] - data['Pivot Point_Strike']
    data['Bullput_Spread_width'] = data['Pivot Point_Strike'] - data['S1_Strike']
    
    data['Bearcall_Spread_width_2'] = data['R2_Strike'] - data['Pivot Point_Strike']
    data['Bullput_Spread_width_2'] = data['Pivot Point_Strike'] - data['S2_Strike']
    
    data['Bearcall_Spread_width_3'] = data['R3_Strike'] - data['Pivot Point_Strike']
    data['Bullput_Spread_width_3'] = data['Pivot Point_Strike'] - data['S3_Strike']

    data['BB_Spread'] = data['Bollinger High_Strike'] - data['Bollinger Low_Strike']

    # Percentage change for the spread
    # data['Bearcall_Spread_Percentage'] = data['Bearcall_Spread_width'] / data['Close']
    # data['Bullput_Spread_Percentage'] = data['Bullput_SBullput_Spread_widthpread_'] / data['Close']
    
    return data

############################################################################################################
# Valuation Models
############################################################################################################

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


############################################################################################################
# ???
############################################################################################################
def prepare_data(data, target_columns, shift_target_by=-1):
    """
    Prepares features and target by selecting the correct columns and shifting the target.

    Parameters:
        data (pd.DataFrame): The input DataFrame with all the data.
        target_columns (list): The list of columns to be used as target.
        shift_target_by (int): The number of rows to shift the target by for prediction.

    Returns:
        tuple: Tuple containing the features and target DataFrames.
    """
    # Check if target columns exist
    missing_targets = [col for col in target_columns if col not in data.columns]
    if missing_targets:
        raise ValueError(f"The following target columns are missing from the data: {missing_targets}")
    
    # Assume all other columns are features
    feature_columns = [col for col in data.columns if col not in target_columns]
    
    # Handle missing values
    data = data.dropna(subset=feature_columns + target_columns)

    # Select features and target
    features = data[feature_columns]
    target = data[target_columns].shift(shift_target_by)

    # Drop the rows with NaN values in the target
    valid_indices = target.dropna().index
    features = features.loc[valid_indices]
    target = target.dropna()

    return features, target

def split_data(features, target, test_size=0.2, random_state=42):
    """
    Splits features and target into training and testing sets.

    Parameters:
        features (pd.DataFrame): Features DataFrame.
        target (pd.DataFrame): Target DataFrame.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: Tuple containing the split data.
    """
    return train_test_split(features, target, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Trains a RandomForestRegressor model on the provided data.

    Parameters:
        X_train (pd.DataFrame): The training features.
        y_train (pd.DataFrame): The training target.
        n_estimators (int): The number of trees in the forest.
        random_state (int): Random state for reproducibility.

    Returns:
        RandomForestRegressor: The trained model.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    # Flatten y_train to a 1D array if it's not already
    y_train = y_train.values.flatten() if isinstance(y_train, pd.DataFrame) else y_train
    model.fit(X_train, y_train)
    return model

def evaluate_feature_importances(model, feature_columns):
    """
    Evaluates and prints the feature importances of the model.

    Parameters:
        model (RandomForestRegressor): The trained model.
        feature_columns (list): The list of columns that were used as features.

    Returns:
        None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")
    for f in range(len(feature_columns)):
        print(f"{f + 1}. feature {feature_columns[indices[f]]} ({importances[indices[f]]})")


def check_data_errors(data):
    errors = []

    # Check for missing values
    if data.isnull().values.any():
        errors.append("Issue: Data contains missing values.")
    
    # Check for duplicate dates
    if data.index.duplicated().any():
        errors.append("Issue: Data contains duplicate dates.")
    
    # Outliers in price data
    z_scores = np.abs((data['Close'] - data['Close'].mean()) / data['Close'].std())
    if z_scores[z_scores > 3].any():
        errors.append("Issue: Data contains potential outliers in 'Close' prices.")
    
    # Volume checks
    if (data['Volume'] == 0).any():
        errors.append("Issue: Data contains days with zero volume.")
    if ((data['Volume'].diff() / data['Volume']).abs() > 5).any():
        errors.append("Issue: Data contains unexpected spikes in volume.")
    
    # Continuity of dates, excluding weekends and public holidays
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=data.index.min(), end=data.index.max())
    business_days = pd.date_range(start=data.index.min(), end=data.index.max(), freq='B')
    business_days = business_days[~business_days.isin(holidays)]  # Exclude holidays
    
    missing_dates = business_days.difference(data.index).tolist()
    if missing_dates:
        formatted_dates = ', '.join([d.strftime('%Y-%m-%d') for d in missing_dates])
        errors.append(f"Issue: Data might be missing trading days: {formatted_dates}")

    return errors

############################################################################################################
# Currency Data Functions
############################################################################################################
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

############################################################################################################
# Options Utilities
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
def black_scholes(S, K, T, r, sigma, option_type):
    # S: spot price of the asset
    # K: strike price
    # T: time to maturity
    # r: risk-free rate
    # sigma: volatility of the asset
    # option_type: 'call' or 'put'
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    else:
        price = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    return price

def binomial_tree(S, K, T, r, sigma, N, option_type):
    # Parameters as described in the Black-Scholes function
    # N: number of binomial steps
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    C = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        C[i, N] = max(0, S * d**i * u**(N - i) - K if option_type == "call" else K - S * d**i * u**(N - i))
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            C[i, j] = (p * C[i, j + 1] + (1 - p) * C[i + 1, j + 1]) * np.exp(-r * dt)
    return C[0, 0]

def monte_carlo_simulation(S, K, T, r, sigma, n_simulations, option_type):
    dt = T / 365
    results = []
    for _ in range(n_simulations):
        path = S * np.cumprod(np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(size=365)))
        if option_type == "call":
            results.append(max(0, path[-1] - K))
        else:
            results.append(max(0, K - path[-1]))
    mc_price = np.mean(results) * np.exp(-r * T)
    return mc_price


# # Section 4: Finite Difference Methods for American Options
# print("## Finite Difference Methods for American Options")
def finite_difference_american_option(S, K, T, r, sigma, option_type):
    # Grid parameters
    N = 1000  # time steps
    M = 200  # price steps
    dt = T / N
    dS = 2 * S / M
    grid = np.zeros((M+1, N+1))
    S_values = np.linspace(0, 2 * S, M+1)
    
    # Set up the final conditions
    if option_type == "call":
        grid[:, -1] = np.maximum(S_values - K, 0)
    else:
        grid[:, -1] = np.maximum(K - S_values, 0)
    
    # Coefficients for the matrix
    a = 0.5 * dt * (sigma**2 * np.arange(M+1)**2 - r * np.arange(M+1))
    b = -dt * (sigma**2 * np.arange(M+1)**2 + r)
    c = 0.5 * dt * (sigma**2 * np.arange(M+1)**2 + r * np.arange(M+1))
    
    # Solving the equation backwards in time
    for j in reversed(range(N)):
        rhs = grid[:, j+1]
        # Set up the matrix
        mat = np.zeros((3, M+1))
        mat[0, 1:] = -a[1:]
        mat[1, :] = 1 - b
        mat[2, :-1] = -c[:-1]
        grid[1:-1, j] = solve_banded((1, 1), mat[:, 1:-1], rhs[1:-1])
        # Apply early exercise condition
        if option_type == "call":
            grid[:, j] = np.maximum(grid[:, j], S_values - K)
        else:
            grid[:, j] = np.maximum(grid[:, j], K - S_values)

    return grid[M//2, 0]
def heston_model(S, K, T, r, kappa, theta, xi, rho, v0, n_simulations, option_type):
    dt = T / 365
    prices = np.zeros(n_simulations)
    v = np.maximum(v0 + np.zeros(n_simulations), 0)
    
    for t in range(1, 365):
        dw1 = np.random.normal(size=n_simulations)
        dw2 = rho * dw1 + np.sqrt(1 - rho**2) * np.random.normal(size=n_simulations)
        S += S * (r * dt + np.sqrt(v) * np.sqrt(dt) * dw1)
        v += kappa * (theta - v) * dt + xi * np.sqrt(v) * np.sqrt(dt) * dw2
        v = np.maximum(v, 0)
    
    if option_type == "call":
        prices = np.exp(-r * T) * np.maximum(S - K, 0)
    else:
        prices = np.exp(-r * T) * np.maximum(K - S, 0)
    
    return np.mean(prices)
def merton_jump_diffusion(S, K, T, r, sigma, lambda_, mu_j, sigma_j, option_type):
    """
    Merton's Jump Diffusion model for option pricing.
    lambda_: Jump frequency per year
    mu_j: Mean jump size
    sigma_j: Jump size volatility
    """
    def integrand(k):
        # Merton's characteristic function part for jump
        jump_part = np.exp(-lambda_ * T + k * np.log(1 + mu_j) + 0.5 * k**2 * sigma_j**2 * T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2 - k * mu_j) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            return np.exp(-r * T) * jump_part * (S * si.norm.cdf(d1) - K * si.norm.cdf(d2))
        else:
            return np.exp(-r * T) * jump_part * (K * si.norm.cdf(-d2) - S * si.norm.cdf(-d1))

    # Numerical integration for the jump diffusion part
    price, _ = quad(integrand, 0, np.inf)
    return price

def barrier_option(S, K, H, T, r, sigma, option_type, barrier_type):
    """
    Analytical price for European barrier options.
    H: Barrier level
    barrier_type: 'up-and-out' or 'down-and-out'
    """
    # Coefficients for barrier options
    mu = (r - 0.5 * sigma**2) / (sigma**2)
    lambda_ = np.sqrt(mu**2 + 2 * r / sigma**2)
    x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    
    if barrier_type == "up-and-out":
        if option_type == "call":
            if H > K:
                price = black_scholes(S, K, T, r, sigma, option_type) \
                        - black_scholes(S, H, T, r, sigma, option_type) \
                        - (H - K) * np.exp(-r * T) * si.norm.cdf(x2) \
                        + (H - K) * np.exp(-r * T) * si.norm.cdf(y2)
            else:
                price = 0  # Option knocked out
        else:
            raise ValueError("Barrier put options are not typically used with up-and-out barriers.")
    elif barrier_type == "down-and-out":
        if option_type == "call":
            price = black_scholes(S, K, T, r, sigma, option_type) \
                    - (H - K) * np.exp(-r * T) * si.norm.cdf(-y2)
        else:
            price = black_scholes(S, K, T, r, sigma, option_type) \
                    - black_scholes(S, H, T, r, sigma, option_type) \
                    - (H - K) * np.exp(-r * T) * si.norm.cdf(-y1)

    return price

def calculate_implied_volatility(options_df):
    options_df['mid'] = (options_df['bid'] + options_df['ask']) / 2
    atm_options = options_df.loc[(options_df['strike'] == options_df['strike'].median()) & (options_df['option_type'] == 'call')]
    atm_options['implied_volatility'] = atm_options['impliedVolatility']
    return atm_options['implied_volatility']

############################################################################################################
# Fetch Options
############################################################################################################
# def get_options_data(ticker, start, end):
#     data = yf.Ticker(ticker)
#     options_dates = data.options
#     options_data = []
    
#     for date in options_dates:
#         if pd.to_datetime(date) > pd.to_datetime(end):
#             break
#         option_chain = data.option_chain(date)
#         calls = option_chain.calls
#         puts = option_chain.puts
#         calls['option_type'] = 'call'
#         puts['option_type'] = 'put'
#         options_data.append(calls)
#         options_data.append(puts)
    
#     options_df = pd.concat(options_data)
#     options_df['date'] = pd.to_datetime(options_df['lastTradeDate'])
#     options_df.set_index('date', inplace=True)
#     options_df = options_df[start:end]
#     return options_df

def get_options_data(ticker, start, end):
    data = yf.Ticker(ticker)
    options_dates = data.options
    options_data = []
    for date in options_dates:
        if date >= start and date <= end:
            opt = data.option_chain(date)
            calls = opt.calls
            puts = opt.puts
            calls['type'], puts['type'] = 'call', 'put'
            options_data.extend([calls, puts])
    
    # Check if options_data is not empty
    if not options_data:
        raise ValueError("No options data found for the given date range.")
    
    options_df = pd.concat(options_data)
    options_df['date'] = pd.to_datetime(options_df['lastTradeDate'])
    options_df.set_index('date', inplace=True)
    return options_df

# Function to fetch option Greeks for given strikes
def get_option_greeks_for_strikes(ticker, strikes):
    # Get the options data for the given ticker
    stock = yf.Ticker(ticker)
    options_data = {}

    # Iterate through expiration dates and strikes
    for expiration in stock.options:
        opt = stock.option_chain(expiration)
        all_options = pd.concat([opt.calls, opt.puts])
        selected_options = all_options[all_options['strike'].isin(strikes)]
        
        # Fetch the Greeks for the selected options
        for index, row in selected_options.iterrows():
            greeks = (row['delta'], row['gamma'], row['theta'], row['vega'], row['rho'])
            strike = row['strike']
            options_data[strike] = greeks

    return options_data

# Function to get options chain with greeks for SPY
def get_spy_options():
    # Assume you already have calculated or have a specific pivot value
    pivot_point = 450  # Example pivot point

    # Get the options chain for SPY
    chain = yo.get_chain_greeks(stock_ticker='SPY', dividend_yield=0.018, option_type='c')
    
    # Find the option closest to the pivot point
    closest_option = chain.iloc[(chain['Strike'] - pivot_point).abs().argsort()[:1]]
    
    return closest_option

############################################################################################################
# Fetch Financial Data 
############################################################################################################


def load_fed_data(series, start_date='2000-01-01'):
    return pdr.get_data_fred(series, start=start_date)

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
# src/data_collection/resample_data.py
# # Example usage
# tickers = ['AAPL', 'MSFT']
# start_date = '2020-01-01'
# end_date = '2023-01-01'
# df = load_price_data(tickers, start=start_date, end=end_date)
# print(df.head())

# # Apply the function to fetch and merge fundamental data for MSFT
# fundamental_data = fetch_fundamentals('MSFT')

# # Displaying the first few rows of the fundamental dataset
# fundamental_data
# # Define the ticker symbol
# Strikes = get_daily_woodies_pivots_with_bollinger('AAPL', start, end) 
# print(Strikes.tail(10))
# # Example usage
# spy_data = fetch_and_process_data("SPY")  # Assuming this function returns data with DateTimeIndex
# errors = check_data_errors(spy_data)
# if errors:
#     for error in errors:
#         print(error)
# else:
#     print("No issues detected in the data.")

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

# # Define the ticker symbol
# Strikes = get_daily_woodies_pivots_with_bollinger('AAPL', start, end) 
# print(Strikes.tail(10))

# # Apply the function to fetch and merge fundamental data for MSFT
# fundamental_data = fetch_fundamentals('MSFT')

# # Displaying the first few rows of the fundamental dataset
# fundamental_data


# Valuation Models

# # Example usage
# tickers = ['AAPL', 'MSFT']
# start_date = '2020-01-01'
# end_date = '2023-01-01'
# df = load_price_data(tickers, start=start_date, end=end_date)
# print(df.head())
# spy_data_daily = fetch_financial_data(ticker='SPY', interval='1d', export_csv=True, calculate_indicators=True)  # Daily data for SPY
# aapl_data_weekly = fetch_financial_data(ticker='AAPL', interval='1wk', start_year=2000, export_csv=True)  # Weekly data for AAPL starting from 2000