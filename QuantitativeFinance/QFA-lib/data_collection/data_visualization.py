# src/visualization/plot_data.py

import matplotlib.pyplot as plt

def plot_technical_indicators(df):
    plt.figure(figsize=(14, 10))

    # Plot Close Price and Bollinger Bands
    plt.subplot(3, 1, 1)
    plt.plot(df['Adj Close'], label='Close Price')
    plt.plot(df['Upper Band'], label='Upper Bollinger Band', linestyle='--', color='r')
    plt.plot(df['Lower Band'], label='Lower Bollinger Band', linestyle='--', color='b')
    plt.fill_between(df.index, df['Upper Band'], df['Lower Band'], alpha=0.1)
    plt.title('Monthly Close Price and Bollinger Bands')
    plt.legend()

    # Plot RSI
    plt.subplot(3, 1, 2)
    plt.plot(df['RSI'], label='RSI', color='purple')
    plt.axhline(70, linestyle='--', alpha=0.5, color='r')
    plt.axhline(30, linestyle='--', alpha=0.5, color='g')
    plt.title('Monthly Relative Strength Index (RSI)')
    plt.legend()

    # Plot MACD
    plt.subplot(3, 1, 3)
    plt.plot(df['MACD'], label='MACD', color='g')
    plt.plot(df['Signal Line'], label='Signal Line', color='orange')
    plt.fill_between(df.index, df['MACD'] - df['Signal Line'], 0, alpha=0.2, color='grey')
    plt.title('Monthly MACD and Signal Line')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_monthly_technical_indicators(df):
    """
    Plot RSI, Bollinger Bands, and MACD for the given dataframe.
    """
    plt.figure(figsize=(14, 10))

    # Plot Close Price and Bollinger Bands
    plt.subplot(4, 1, 1)
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['Upper Band'], label='Upper Bollinger Band', linestyle='--', color='r')
    plt.plot(df['Lower Band'], label='Lower Bollinger Band', linestyle='--', color='b')
    plt.fill_between(df.index, df['Upper Band'], df['Lower Band'], alpha=0.1)
    plt.title('Monthly Close Price and Bollinger Bands')
    plt.legend()

    # Plot RSI
    plt.subplot(4, 1, 2)
    plt.plot(df['RSI'], label='RSI', color='purple')
    plt.axhline(70, linestyle='--', alpha=0.5, color='r')
    plt.axhline(30, linestyle='--', alpha=0.5, color='g')
    plt.title('Monthly Relative Strength Index (RSI)')
    plt.legend()

    # Plot MACD
    plt.subplot(4, 1, 3)
    plt.plot(df['MACD'], label='MACD', color='g')
    plt.plot(df['Signal Line'], label='Signal Line', color='orange')
    plt.fill_between(df.index, df['MACD'] - df['Signal Line'], 0, alpha=0.2, color='grey')
    plt.title('Monthly MACD and Signal Line')
    plt.legend()

    plt.tight_layout()
    plt.show()

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

def plot_with_macro_data(merged_df):
    """
    Plot financial indicators with macroeconomic data.
    """
    fig, axs = plt.subplots(4, 1, figsize=(14, 12))

    # Plot Close Price and Bollinger Bands
    axs[0].plot(merged_df['Close'], label='Close Price')
    axs[0].plot(merged_df['Upper Band'], label='Upper Bollinger Band', linestyle='--', color='r')
    axs[0].plot(merged_df['Lower Band'], label='Lower Bollinger Band', linestyle='--', color='b')
    axs[0].fill_between(merged_df.index, merged_df['Upper Band'], merged_df['Lower Band'], alpha=0.1)
    axs[0].set_title('Monthly Close Price and Bollinger Bands')
    axs[0].legend()

    # Plot RSI
    axs[1].plot(merged_df['RSI'], label='RSI', color='purple')
    axs[1].axhline(70, linestyle='--', alpha=0.5, color='r')
    axs[1].axhline(30, linestyle='--', alpha=0.5, color='g')
    axs[1].set_title('Monthly Relative Strength Index (RSI)')
    axs[1].legend()

    # Plot MACD
    axs[2].plot(merged_df['MACD'], label='MACD', color='g')
    axs[2].plot(merged_df['Signal Line'], label='Signal Line', color='orange')
    axs[2].fill_between(merged_df.index, merged_df['MACD'] - merged_df['Signal Line'], 0, alpha=0.2, color='grey')
    axs[2].set_title('Monthly MACD and Signal Line')
    axs[2].legend()

    # Plot Fed Funds Rate
    axs[3].plot(merged_df['FEDFUNDS'], label='Federal Funds Rate', color='blue')
    axs[3].set_title('Federal Funds Rate')
    axs[3].legend()

    plt.tight_layout()
    plt.show()

def test_data_quality(data):
    """
    Tests the data for quality and consistency.
    
    Parameters:
        data (pd.DataFrame): The DataFrame containing financial data.
    
    Returns:
        dict: A dictionary containing the test results.
    """
    results = {}

    # Test for NaN values
    nan_count = data.isna().sum().sum()
    results['total_nan_count'] = nan_count

    # Test for detrending
    detrended_columns = [col for col in data.columns if 'detrended' in col]
    results['detrended_columns'] = detrended_columns
    if detrended_columns:
        results['detrending_test'] = "Detrended columns found."
    else:
        results['detrending_test'] = "No detrended columns found."

    # Check for log returns and price differences
    log_return_columns = [col for col in data.columns if 'log_return' in col]
    price_diff_columns = [col for col in data.columns if '_diff' in col]
    results['log_return_columns'] = log_return_columns
    results['price_diff_columns'] = price_diff_columns

    # Check for volume changes
    volume_change_columns = [col for col in data.columns if 'volume_changes' in col]
    results['volume_change_columns'] = volume_change_columns

    # Statistical checks for detrended columns
    for col in detrended_columns:
        mean = data[col].mean()
        std_dev = data[col].std()
        results[f'{col}_mean'] = mean
        results[f'{col}_std_dev'] = std_dev

    return results

# Example usage
# Assuming 'spy_data_daily' is a DataFrame obtained from the fetch_financial_data function and includes some of the processed columns
# spy_data_daily = pd.DataFrame()  # Placeholder for the actual spy_data_daily DataFrame
# test_results = test_data_quality(spy_data_daily)
# print(test_results)


def plot_ghost_candles(future_predictions, start_date):
    future_dates = pd.date_range(start=start_date, periods=len(future_predictions), freq='B')
    
    fig = go.Figure(data=[go.Candlestick(x=future_dates,
                                         open=future_predictions[:, 0],
                                         high=future_predictions[:, 1],
                                         low=future_predictions[:, 2],
                                         close=future_predictions[:, 3],
                                         name='Forecast')])
    fig.update_layout(title='Forecasted OHLC Prices for the Next Days',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)
    fig.show()
def forecast_future(stock_data, model, scaler, seq_length, num_features, future_days=10):
    # Directly use the DataFrame without slicing it before the function call
    # Ensure stock_data is in the expected DataFrame format with columns
    
    # Extract the last sequence from the DataFrame
    last_sequence_df = stock_data.iloc[-seq_length:]  # Ensure this is a DataFrame slice
    last_sequence = scaler.transform(last_sequence_df[features])
    last_sequence = np.expand_dims(last_sequence, axis=0)
    
    future_predictions = []
    
    for _ in range(future_days):
        prediction = model.predict(last_sequence)
        future_predictions.append(prediction[0])
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[:, -1, :] = prediction

    future_predictions = np.array(future_predictions)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    return future_predictions

def forecast_and_plot_complete(ticker, features, start_date, end_date, seq_length, future_days=10):
    stock_data = download_stock_data(ticker, start_date, end_date)
#     scaler, scaled_data = scale_data(stock_data[features].values)
    scaler, scaled_data = scale_data(stock_data[features].values, features)
    
    X, y = create_sequences(scaled_data, seq_length)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    num_features = len(features)
    X_train = X_train.reshape((X_train.shape[0], seq_length, num_features))
    X_test = X_test.reshape((X_test.shape[0], seq_length, num_features))
    
    model = build_and_train_model(X_train, y_train, seq_length, num_features, epochs=50, batch_size=32)
    
    # Generate predictions for the test set
    predicted_values_test = model.predict(X_test)
    mae, mse, rmse = calculate_metrics(scaler.inverse_transform(y_test), scaler.inverse_transform(predicted_values_test))
    print(f"Test Set - MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
    
    # Forecast future prices directly from the entire stock_data DataFrame
    future_predictions = forecast_future(stock_data, model, scaler, seq_length, num_features, future_days)
    plot_ghost_candles(future_predictions, '2024-03-11')


### Plotting Spreads
def plot_spreads(data):
    """
    Plots the calculated bear call spreads, bull put spreads, and Bollinger Bands spread.

    Args:
    data (pandas.DataFrame): DataFrame containing the calculated spreads.
    """
    # Plot settings
    plt.figure(figsize=(14, 7))
    plt.title(f"Spreads for {ticker}")

    # Plot each of the spreads
    plt.plot(data.index, data['Bearcall_Spread_'], label='Bear Call Spread (R1 - Pivot)')
    plt.plot(data.index, data['Bullput_Spread_'], label='Bull Put Spread (Pivot - S1)')
    plt.plot(data.index, data['Bearcall_Spread_2'], label='Bear Call Spread 2 (R2 - Pivot)')
    plt.plot(data.index, data['Bullput_Spread_2'], label='Bull Put Spread 2 (Pivot - S2)')
    plt.plot(data.index, data['Bearcall_Spread_ 3'], label='Bear Call Spread 3 (R3 - Pivot)')
    plt.plot(data.index, data['Bullput_Spread_3'], label='Bull Put Spread 3 (Pivot - S3)')
    plt.plot(data.index, data['BB_Spread'], label='Bollinger Bands Spread')

    # Labels and legend
    plt.xlabel('Date')
    plt.ylabel('Spread Value')
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.show()


def display_all_monthly_statistics(seasonality_table, seasonality_volume_table):
    """
    Display the statistics for each month in the seasonality table.
    """
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for month in months:
        monthly_data = seasonality_table.loc[month]
        monthly_volume = seasonality_volume_table.loc[month]
        mean_return = monthly_data.mean()
        std_return = monthly_data.std()
        highest_return_year = monthly_data.idxmax()
        highest_return_value = monthly_data.max()
        lowest_return_year = monthly_data.idxmin()
        lowest_return_value = monthly_data.min()
        mean_volume = monthly_volume.mean()
        highest_volume_year = monthly_volume.idxmax()
        highest_volume_value = monthly_volume.max()
        lowest_volume_year = monthly_volume.idxmin()
        lowest_volume_value = monthly_volume.min()

        print(f"Mean {month} Return: {mean_return:.2f}%")
        print(f"Standard Deviation of {month} Returns: {std_return:.2f}%")
        print(f"Highest {month} Return: {highest_return_value:.2f}% in {highest_return_year}")
        print(f"Lowest {month} Return: {lowest_return_value:.2f}% in {lowest_return_year}")
        print(f"Mean {month} Volume: {mean_volume:.2f}")
        print(f"Highest {month} Volume: {highest_volume_value:.2f} in {highest_volume_year}")
        print(f"Lowest {month} Volume: {lowest_volume_value:.2f} in {lowest_volume_year}")
        print()




# src/visualization/plot_strategy.py

import matplotlib.pyplot as plt

def plot_cumulative_returns_with_half_kelly(original, strategy, title):
    plt.figure(figsize=(14, 7))
    plt.plot(original.index, original['Cumulative Return'], label='Original Return')
    plt.plot(strategy.index, strategy['Cumulative Strategy Return'], label='Strategy Return with Half-Kelly Position Sizing')
    plt.legend()
    plt.title(title)
    plt.show()
