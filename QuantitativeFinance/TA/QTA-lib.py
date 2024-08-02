# %% [markdown]
# # Import these functions

# %%
!pip install yahoo_fin
!pip install yoptions

# %%
import concurrent.futures

# Fundemental & Technical Data
import yfinance as yf 

# Options Data
import yoptions as yo
from yahoo_fin import options

# General Helper Libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta



# %%
# Set option to display all columns
pd.set_option('display.max_columns', None)

# Optionally, set the max rows displayed in the output as well
pd.set_option('display.max_rows', 100)

# %% [markdown]
# ## Technical Features (Technical Indicators known to be reliable for ML and forecasting)

# %%
def calculate_ema(data, period):
    alpha = 2 / (period + 1)
    return data.ewm(alpha=alpha, adjust=False).mean()

def calculate_dema(data, period):
    ema1 = calculate_ema(data, period)
    ema2 = calculate_ema(ema1, period)
    return 2 * ema1 - ema2


# Define a function to compute the moving averages according to the type.
def moving_average(df, ma_type, period):
    if ma_type == "DEMA":
        return calculate_dema(df, period)
    elif ma_type == "EMA":
        return calculate_ema(df, period)
    elif ma_type == "SMA":
        return df.rolling(window=period).mean()


# %%
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

# %%
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

# %%
def bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    data['Bollinger_High'] = rolling_mean + (rolling_std * num_std)
    data['Bollinger_Low'] = rolling_mean - (rolling_std * num_std)
    return data

# Calculate the predictive rolling mean and standard deviation
def predict_bollinger_bands(series, period, num_std, shift_periods):
    rolling_mean = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()

    # Predict the future values by shifting the mean and std
    predicted_mean = rolling_mean.shift(-shift_periods)
    predicted_std = rolling_std.shift(-shift_periods)

    # Calculate the upper and lower predictive bands
    predicted_upper_band = predicted_mean + (predicted_std * num_std)
    predicted_lower_band = predicted_mean - (predicted_std * num_std)

    return predicted_upper_band, predicted_lower_band

# Calculate the predictive bands for 3 periods ahead

# Calculate the rolling mean and standard deviation for the Bollinger Bands
def calculate_bollinger_bands(series, period, num_std):
    rolling_mean = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

# Calculate predictive Bollinger Bands by shifting the bands forward
def calculate_predictive_bands(series, period, num_std, shift_periods):
    upper_band, middle_band, lower_band = calculate_bollinger_bands(series, period, num_std)
    # Shift the calculated bands forward by the shift_periods
    predictive_upper_band = upper_band.shift(periods=shift_periods)
    predictive_lower_band = lower_band.shift(periods=shift_periods)
    # For the predicted SMA, simply use the shifted rolling mean
    predictive_middle_band = middle_band.shift(periods=shift_periods)
    return predictive_upper_band, predictive_middle_band, predictive_lower_band



# %%
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


# %% [markdown]
# ## Data Preparation Helpers

# %%

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

# Apply the function to fetch and merge fundamental data for MSFT
fundamental_data = fetch_fundamentals('MSFT')

# Displaying the first few rows of the fundamental dataset
fundamental_data


# %%
def fetch_technical_data(ticker='SPY', start_year=1993, end_year=None, interval='1d', calculate_indicators=False, include_fundamentals=False, export_csv=False, csv_file=None,):
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
    return data

MSFT = fetch_technical_data('MSFT', interval="1d", calculate_indicators=True, include_fundamentals=True)

MSFT.head()

# %% [markdown]
# # Data Preprocessing

# %%
"""
calculate_returns(data, periods): Calculates returns over specified periods (e.g., 1, 2, 3 months) for given stock prices.
calculate_volatility(data, periods): Computes volatility over specified periods for given stock prices.
calculate_moving_averages(data, periods): Calculates moving averages over specified periods for given stock prices.
calculate_MA_gaps(data, periods): Computes moving average gaps over specified periods to capture trends.
apply_log_transform(data, columns): Applies logarithmic transformation to specified columns to normalize data.
add_lag_features(data, columns, lags): Adds lagged features for specified columns and lag periods to capture time-series dependencies.
"""

# %% [markdown]
# # Advanced Feature Engineering

# %%
"""
encode_cyclical_features(data, column, max_val): Encodes cyclical features like days of the week or months in a year using sine and cosine transformations to preserve their cyclical nature.
calculate_technical_indicators(data, indicators): Calculates a range of technical indicators (e.g., RSI, MACD, Bollinger Bands) specified by the user for given stock prices.
apply_PCA_reduction(data, n_components): Applies PCA (Principal Component Analysis) to reduce the dimensionality of the feature space while retaining n_components principal components.
"""


# %% [markdown]
# # Trading Strategy Simulation and Evaluation

# %%
"""
simulate_trading_strategy(predictions, transaction_costs): Simulates a trading strategy based on model predictions and calculates net profit after accounting for transaction costs.
calculate_max_drawdown(returns): Calculates the maximum drawdown from a series of returns, useful for risk assessment.
backtest_strategy(time_series, strategy_function): Conducts a backtest of a trading strategy function on a historical time series dataset to evaluate performance over time.
"""



# %% [markdown]
# # Model Training and Evaluation
# 
# 
# 

# %%
"""
train_test_split_time_series(data, test_ratio): Splits time-series data into training and testing sets with a specified ratio, preserving the temporal order.
cross_validate_time_series(model, data, cv_splits): Performs time-series cross-validation with a given model, data, and number of splits.
evaluate_model_performance(model, X_test, y_test, metrics): Evaluates the model performance using specified metrics (e.g., RMSE, MAE, Pearson correlation).
hyperparameter_optimization(model, param_grid, X_train, y_train): Conducts hyperparameter optimization for a given model and parameter grid.
"""



# %% [markdown]
# # Prediction and Ranking

# %%
"""
predict_and_rank(model, data, features): Generates predictions using a trained model and ranks the predictions for trading strategies.
adjust_predictions_based_on_median(data, securities_to_adjust): Dynamically adjusts predictions based on the median target value for specific securities.
calculate_spread_return_sharpe(data, portfolio_size, weight_ratio): Calculates the Sharpe ratio based on spread return for a given portfolio size and weight ratio.
"""

# %% [markdown]
# # Data Visualization

# %%
"""
plot_stock_prices(data, title): Plots stock prices or returns over time.
plot_feature_importances(model, feature_names): Plots the feature importances for a trained model.
plot_prediction_vs_actual(data, predictions, actual, title): Plots predicted vs. actual values for visual comparison.
"""


# %% [markdown]
# # Data Integrity and Cleaning

# %%
"""
detect_outliers(data, method): Detects outliers in the dataset using a specified method (e.g., IQR, Z-score).
impute_missing_values(data, imputation_strategy): Imputes missing values using a specified strategy (e.g., mean, median, k-NN).
"""


# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# # 

# %% [markdown]
# # 


