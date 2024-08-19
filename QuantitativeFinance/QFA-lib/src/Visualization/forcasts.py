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
























