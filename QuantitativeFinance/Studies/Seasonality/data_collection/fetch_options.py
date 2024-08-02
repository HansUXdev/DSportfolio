# src/data_collection/fetch_options.py

import yfinance as yf
import pandas as pd

def get_options_data(ticker, start, end):
    data = yf.Ticker(ticker)
    options_dates = data.options
    options_data = []
    
    for date in options_dates:
        if pd.to_datetime(date) > pd.to_datetime(end):
            break
        option_chain = data.option_chain(date)
        calls = option_chain.calls
        puts = option_chain.puts
        calls['option_type'] = 'call'
        puts['option_type'] = 'put'
        options_data.append(calls)
        options_data.append(puts)
    
    options_df = pd.concat(options_data)
    options_df['date'] = pd.to_datetime(options_df['lastTradeDate'])
    options_df.set_index('date', inplace=True)
    options_df = options_df[start:end]
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

# # Define the ticker symbol
# Strikes = get_daily_woodies_pivots_with_bollinger('AAPL', start, end) 
# print(Strikes.tail(10))


import yoptions as yo

# Function to get options chain with greeks for SPY
def get_spy_options():
    # Assume you already have calculated or have a specific pivot value
    pivot_point = 450  # Example pivot point

    # Get the options chain for SPY
    chain = yo.get_chain_greeks(stock_ticker='SPY', dividend_yield=0.018, option_type='c')
    
    # Find the option closest to the pivot point
    closest_option = chain.iloc[(chain['Strike'] - pivot_point).abs().argsort()[:1]]
    
    return closest_option
