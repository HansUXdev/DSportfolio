# ./analysis/forecasting.py

def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data.sort_index().asfreq('B', method='ffill')
    stock_data.index = pd.to_datetime(stock_data.index)
    return stock_data
def forecast_future(df, model, steps=10):
    last_sequence = df[-steps:].values.reshape(1, -1)
    forecast = model.predict(last_sequence)
    return forecast

def scale_data(data, feature_names):
    scaler = MinMaxScaler()
    # Convert the data to a DataFrame to ensure feature names are used
    data_df = pd.DataFrame(data, columns=feature_names)
    scaled_data = scaler.fit_transform(data_df)
    return scaler, scaled_data


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

def build_and_train_model(X_train, y_train, seq_length, num_features, epochs, batch_size):
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, num_features)),
        LSTM(50, activation='relu'),
        Dense(num_features)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def plot_forecasts(stock_data, predicted_values, y_test, seq_length, split_idx, scaler):
    dates = stock_data.index[split_idx + seq_length:]
    predicted_prices = scaler.inverse_transform(predicted_values)
    actual_prices = scaler.inverse_transform(y_test)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    feature_names = ['Open', 'High', 'Low', 'Close']

    for i, feature in enumerate(feature_names):
        fig.add_trace(go.Scatter(x=dates, y=actual_prices[:, i], mode='lines', name=f'Actual {feature}'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=predicted_prices[:, i], mode='lines', name=f'Predicted {feature}', line=dict(dash='dash')), row=i+1, col=1)

    fig.update_layout(height=800, width=1000, title_text="Stock Price Forecasting")
    fig.update_yaxes(title_text="<b>Price</b>")
    fig.update_xaxes(title_text="<b>Date</b>", row=4, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()

def forecast_and_plot(stock_data, features, seq_length):
#     scaler, scaled_data = scale_data(stock_data[features].values)
    scaler, scaled_data = scale_data(stock_data[features].values, features)
    X, y = create_sequences(scaled_data, seq_length)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], len(features)))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], len(features)))
    
    model = build_and_train_model(X_train, y_train, seq_length, len(features), epochs=50, batch_size=32)
    predicted_values = model.predict(X_test)
    mae, mse, rmse = calculate_metrics(scaler.inverse_transform(y_test), scaler.inverse_transform(predicted_values))
    
    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
    plot_forecasts(stock_data, predicted_values, y_test, seq_length, split_idx, scaler)
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