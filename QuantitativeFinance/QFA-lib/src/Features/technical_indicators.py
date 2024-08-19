############################################################################################################
# Fundementals
############################################################################################################

# from .Features.utils import check_argument_types


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


# Super Smoother Filter function
def super_smoother(series, period):
    # Constants for the filter calculation
    a1 = np.exp(-1.414 * np.pi / period)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    # Preallocate the output series with the same initial value as input
    filtered_series = np.full_like(series, series.iloc[0])

    # Apply the filter to the series
    for i in range(2, len(series)):
        filtered_series[i] = c1 * series.iloc[i] + c2 * filtered_series[i - 1] + c3 * filtered_series[i - 2]

    return pd.Series(filtered_series, index=series.index)

# Applying the Super Smoother Filter to the closing prices
# filtered_close_prices = super_smoother(close_prices, period=10)

def bollinger_bands(data, period, num_std):
    expected_types = {
        'data': pd.DataFrame,
        'window': int,
        'no_of_std': (int, float)
    }
    arguments = {
        'data': data,
        'window': period,
        'no_of_std': num_std
    }
    check_argument_types(arguments, expected_types)
    rolling_mean = data.rolling(window=period).mean()
    rolling_std = data.rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band


# Define the function to shift the series for prediction
def predict_moving_average(data, period):
    return data.rolling(window=period).mean().shift(-period)




# Calculate predictive Bollinger Bands by shifting the bands forward
def calculate_predictive_bands(data, period, num_std, shift_periods):
    upper_band, middle_band, lower_band = bollinger_bands(data, period, num_std)
    # Shift the calculated bands forward by the shift_periods
    predictive_upper_band = upper_band.shift(periods=shift_periods)
    predictive_lower_band = lower_band.shift(periods=shift_periods)
    # For the predicted SMA, simply use the shifted rolling mean
    predictive_middle_band = middle_band.shift(periods=shift_periods)
    return predictive_upper_band, predictive_middle_band, predictive_lower_band


def macd(data, short_window=12, long_window=26, signal_window=9):
    expected_types = {
        'data': pd.DataFrame,
        'short_window': int,
        'long_window': int,
        'signal_window': int
    }
    arguments = {
        'data': data,
        'short_window': short_window,
        'long_window': long_window,
        'signal_window': signal_window
    }
    check_argument_types(arguments, expected_types)

    if 'Close' in data.columns:
        short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
        long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        data['MACD'] = macd
        data['Signal Line'] = signal
        return data
    else:
        raise ValueError("Column 'Close' not found in the data")


def rsi(data, periods=14, ema=True):
    expected_types = {
        'data': pd.DataFrame,
        'periods': int,
        'ema': bool
    }
    arguments = {
        'data': data,
        'periods': periods,
        'ema': ema
    }
    check_argument_types(arguments, expected_types)

    close_delta = data['Close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    if ema:
        ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    else:
        ma_up = up.rolling(window=periods).mean()
        ma_down = down.rolling(window=periods).mean()

    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
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