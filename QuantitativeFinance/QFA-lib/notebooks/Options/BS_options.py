# 
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm

####################################
# Fetch and Process Data
####################################

def fetch_and_process_data(_asset):
    # Fetch data for the specified asset
    hist = yf.download(_asset, start='2022-01-01') 

    # Indicator calculations as defined earlier
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
            ma_up = up.rolling(window=periods, adjust=False).mean()
            ma_down = down.rolling(window=periods, adjust=False).mean()
        
        rsi = ma_up / ma_down
        data['RSI'] = 100 - (100 / (1 + rsi))
        return data
        
    def obv(data):
        """Calculate On-Balance Volume."""
        obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        data['OBV'] = obv
        return data

    def atr(data, window=14):
        """Calculate Average True Range (ATR)."""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        data['ATR'] = true_range.rolling(window=window).mean()
        return data

    def woodie_pivots(data):
        # Calculate Woodie's pivot points
        data['Pivot'] = (data['High'] + data['Low'] + 2 * data['Close']) / 4
        data['R1'] = 2 * data['Pivot'] - data['Low']
        data['S1'] = 2 * data['Pivot'] - data['High']
        data['R2'] = data['Pivot'] + (data['High'] - data['Low'])
        data['S2'] = data['Pivot'] - (data['High'] - data['Low'])
        data['R3'] = data['High'] + 2 * (data['Pivot'] - data['Low'])
        data['S3'] = data['Low'] - 2 * (data['High'] - data['Pivot'])
        data['R4'] = data['Pivot'] + 3 * (data['High'] - data['Low'])
        data['S4'] = data['Pivot'] - 3 * (data['High'] - data['Low'])
        return data

    # Apply each indicator function to the data
    hist = bollinger_bands(hist)
    hist = macd(hist)
    hist = rsi(hist)
    hist = woodie_pivots(hist)
    hist = obv(hist)
    hist = atr(hist)
    # Repeat for other indicators as necessary...

    # Note: No explicit parallel processing applied here due to sequential dependency of calculations on data.

    # Ensure all NaN values created by indicators are handled appropriately
    hist.dropna(inplace=True)

    return hist

# Example usage
spy_data = fetch_and_process_data("SPY")

####################################
# Credit Spread Pricing
####################################

import pandas as pd
import numpy as np

def round_to_nearest_50_cents(value):
    """Round the value to the nearest 50 cents."""
    return np.round(value * 2, 0) / 2

def generate_credit_spreads(data):
    # Initialize a list to hold the spreads for each day
    spreads = []
    
    # Iterate through each row in the DataFrame
    for index, row in data.iterrows():
        # Initialize a dictionary for the current day's spreads
        day_spreads = {
            'Date': index,
            'R1-R2 Put Spread': None,
            'R1-R3 Put Spread': None,
            'R1-R4 Put Spread': None,
            'S1-S2 Call Spread': None,
            'S1-S3 Call Spread': None,
            'S1-S4 Call Spread': None,
        }
        
        # Calculate the put credit spreads for each pair of resistances
        for i in range(2, 5):
            sell_strike = round_to_nearest_50_cents(row['R1'])
            buy_strike = round_to_nearest_50_cents(row[f'R{i}'])
            day_spreads[f'R1-R{i} Put Spread'] = (sell_strike, buy_strike)
        
        # Calculate the call credit spreads for each pair of supports
        for i in range(2, 5):
            sell_strike = round_to_nearest_50_cents(row['S1'])
            buy_strike = round_to_nearest_50_cents(row[f'S{i}'])
            day_spreads[f'S1-S{i} Call Spread'] = (sell_strike, buy_strike)
        
        # Add the current day's spreads to the list
        spreads.append(day_spreads)
    
    # Convert the list of spreads into a DataFrame for easier viewing
    spreads_df = pd.DataFrame(spreads)
    spreads_df.set_index('Date', inplace=True)
    return spreads_df

# Assuming spy_data is already defined and contains the necessary columns
# Generate the credit spreads
credit_spreads = generate_credit_spreads(spy_data)



####################################
# Option Spread Probability Estimation Using Black-Scholes Model
####################################

def calculate_option_probabilities(spy_data, credit_spreads, volatility, risk_free_rate, time_to_expiration):
    """
    Calculate the probabilities of credit spreads expiring worthless using the Black-Scholes model.
    
    Parameters:
    - spy_data: DataFrame containing 'Close' prices of the stock.
    - credit_spreads: DataFrame containing the strike prices for the spreads.
    - volatility: Annualized volatility of the stock.
    - risk_free_rate: Annual risk-free interest rate.
    - time_to_expiration: Time to expiration of the options in years.
    
    Returns:
    - DataFrame with probabilities of each spread expiring worthless.
    """
    def calculate_cdf(strike, current_price, volatility, time_to_expiration, risk_free_rate):
        """Calculate the cumulative distribution function for Black-Scholes."""
        d1 = (np.log(current_price / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiration) / (volatility * np.sqrt(time_to_expiration))
        return norm.cdf(d1)
    
    probabilities = []
    
    for index, row in spy_data.iterrows():
        current_price = row['Close']
        
        # Adjusted to directly use 'R' and 'S' values from `credit_spreads`
        for i in range(1, 5):
            if f'R{i}' in credit_spreads.columns and f'S{i}' in credit_spreads.columns:
                # For put spreads, using R values
                sell_strike_put = row[f'R{i}']
                buy_strike_put = row[f'R{i}'] - 1  # Example adjustment, customize as needed
                prob_put = calculate_cdf(sell_strike_put, current_price, volatility, time_to_expiration, risk_free_rate)
                
                # For call spreads, using S values
                sell_strike_call = row[f'S{i}']
                buy_strike_call = row[f'S{i}'] + 1  # Example adjustment, customize as needed
                prob_call = 1 - calculate_cdf(buy_strike_call, current_price, volatility, time_to_expiration, risk_free_rate)
                
                probabilities.append({
                    'Date': index,
                    f'R{i} Put Spread Probability': prob_put,
                    f'S{i} Call Spread Probability': prob_call
                })
    
    probabilities_df = pd.DataFrame(probabilities).set_index('Date')
    return probabilities_df

# Assuming you have calculated or have the values for the following variables:
volatility = 0.2  # Example volatility
risk_free_rate = 0.01  # Example risk-free rate
time_to_expiration = 1/52  # 1 week to expiration

# Calculate probabilities
probabilities_df = calculate_option_probabilities(spy_data, spy_data, volatility, risk_free_rate, time_to_expiration)
print(probabilities_df)



