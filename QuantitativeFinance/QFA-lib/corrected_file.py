# %%
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime

from scipy.stats import linregress
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="The 'unit' keyword in TimedeltaIndex construction is deprecated and will be removed in a future version. Use pd.to_timedelta instead.")
def load_data(ticker='SPY', start='2008-01-01', end=None):
    """
    Load historical data for a given ticker using yfinance.
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end)
    return df
def visualize_seasonality_table(seasonality_table):
    Visualize the seasonality table using a heatmap.
    plt.figure(figsize=(10, 8))
    sns.heatmap(seasonality_table, annot=True, fmt=".2f", cmap='RdYlGn', center=0)
    plt.title('Seasonality of Index Fund Returns')
    plt.show()
def extract_monthly_data(seasonality_table, month):
    Extract data for a specific month.
    return seasonality_table.loc[month]
def visualize_monthly_data(monthly_data, month):
    Visualize the extracted data for the selected month.
    plt.figure(figsize=(10, 6))
    plt.bar(monthly_data.index, monthly_data.values, color='dodgerblue')
    plt.xlabel('Year')
    plt.ylabel(f'Average {month} Return (%)')
    plt.title(f'Average {month} Returns Over the Years')
    plt.xticks(rotation=45)
def perform_quantitative_analysis(monthly_data, month):
    Perform quantitative analysis on the extracted monthly data.
    mean_return = monthly_data.mean()
    std_return = monthly_data.std()
    highest_return_year = monthly_data.idxmax()
    highest_return_value = monthly_data.max()
    lowest_return_year = monthly_data.idxmin()
    lowest_return_value = monthly_data.min()
    print(f"Mean {month} Return: {mean_return:.2f}%")
    print(f"Standard Deviation of {month} Returns: {std_return:.2f}%")
    print(f"Highest {month} Return: {highest_return_value:.2f}% in {highest_return_year}")
    print(f"Lowest {month} Return: {lowest_return_value:.2f}% in {lowest_return_year}")
    # Determine Overall Trend
    years = monthly_data.index.astype(int)
    returns = monthly_data.values
    if len(set(returns)) > 1:  # Check for variance in data
        slope, intercept, r_value, p_value, std_err = linregress(years, returns)
        plt.bar(monthly_data.index, monthly_data.values, color='dodgerblue', label=f'{month} Return')
        plt.plot(monthly_data.index, intercept + slope*years, 'r', label=f'Trend Line (slope={slope:.2f})')
        plt.title(f'Average {month} Returns Over the Years with Trend Line')
        plt.legend()
        print(f"Trend Line Slope: {slope:.2f}% per year")
        print(f"R-squared: {r_value**2:.2f}")
    else:
        print(f"No variance in {month} data to compute trend line.")
def display_all_monthly_statistics(seasonality_table, seasonality_volume_table):
    Display the statistics for each month in the seasonality table.
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month in months:
        monthly_data = seasonality_table.loc[month]
        monthly_volume = seasonality_volume_table.loc[month]
        mean_volume = monthly_volume.mean()
        highest_volume_year = monthly_volume.idxmax()
        highest_volume_value = monthly_volume.max()
        lowest_volume_year = monthly_volume.idxmin()
        lowest_volume_value = monthly_volume.min()
        print(f"Mean {month} Volume: {mean_volume:.2f}")
        print(f"Highest {month} Volume: {highest_volume_value:.2f} in {highest_volume_year}")
        print(f"Lowest {month} Volume: {lowest_volume_value:.2f} in {lowest_volume_year}")
        print()
def resample_monthly(df):
    Resample the dataframe to monthly frequency.
    df_monthly = df.resample('M').ffill()
    print("Resampled Data:\n", df_monthly.head())
    return df_monthly
def calculate_technical_indicators(df):
    Calculate RSI, Bollinger Bands, and MACD for the given dataframe.
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    # Calculate Bollinger Bands
    df['20 Day MA'] = df['Close'].rolling(window=20).mean()
    df['20 Day STD'] = df['Close'].rolling(window=20).std()
    df['Upper Band'] = df['20 Day MA'] + (df['20 Day STD'] * 2)
    df['Lower Band'] = df['20 Day MA'] - (df['20 Day STD'] * 2)
    # Calculate MACD
    df['12 EMA'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['26 EMA'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['12 EMA'] - df['26 EMA']
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    print("Technical Indicators:\n", df[['RSI', 'Upper Band', 'Lower Band', 'MACD', 'Signal Line']].head())
def plot_monthly_technical_indicators(df):
    Plot RSI, Bollinger Bands, and MACD for the given dataframe.
    plt.figure(figsize=(14, 10))
    # Plot Close Price and Bollinger Bands
    plt.subplot(4, 1, 1)
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['Upper Band'], label='Upper Bollinger Band', linestyle='--', color='r')
    plt.plot(df['Lower Band'], label='Lower Bollinger Band', linestyle='--', color='b')
    plt.fill_between(df.index, df['Upper Band'], df['Lower Band'], alpha=0.1)
    plt.title('Monthly Close Price and Bollinger Bands')
    # Plot RSI
    plt.subplot(4, 1, 2)
    plt.plot(df['RSI'], label='RSI', color='purple')
    plt.axhline(70, linestyle='--', alpha=0.5, color='r')
    plt.axhline(30, linestyle='--', alpha=0.5, color='g')
    plt.title('Monthly Relative Strength Index (RSI)')
    # Plot MACD
    plt.subplot(4, 1, 3)
    plt.plot(df['MACD'], label='MACD', color='g')
    plt.plot(df['Signal Line'], label='Signal Line', color='orange')
    plt.fill_between(df.index, df['MACD'] - df['Signal Line'], 0, alpha=0.2, color='grey')
    plt.title('Monthly MACD and Signal Line')
    plt.tight_layout()
def fetch_fed_data(series, start_date='2008-01-01'):
    Fetch macroeconomic data from FRED.
    return pdr.get_data_fred(series, start=start_date)
def merge_macro_data(financial_df, macro_df):
    Merge financial data with macroeconomic data.
    macro_df = macro_df.resample('M').ffill()  # Ensure macro data is also monthly
    merged_df = financial_df.merge(macro_df, left_index=True, right_index=True, how='inner')
    print("Merged Data:\n", merged_df.head())
    return merged_df
def plot_with_macro_data(merged_df):
    Plot financial indicators with macroeconomic data.
    fig, axs = plt.subplots(4, 1, figsize=(14, 12))
    axs[0].plot(merged_df['Close'], label='Close Price')
    axs[0].plot(merged_df['Upper Band'], label='Upper Bollinger Band', linestyle='--', color='r')
    axs[0].plot(merged_df['Lower Band'], label='Lower Bollinger Band', linestyle='--', color='b')
    axs[0].fill_between(merged_df.index, merged_df['Upper Band'], merged_df['Lower Band'], alpha=0.1)
    axs[0].set_title('Monthly Close Price and Bollinger Bands')
    axs[0].legend()
    axs[1].plot(merged_df['RSI'], label='RSI', color='purple')
    axs[1].axhline(70, linestyle='--', alpha=0.5, color='r')
    axs[1].axhline(30, linestyle='--', alpha=0.5, color='g')
    axs[1].set_title('Monthly Relative Strength Index (RSI)')
    axs[1].legend()
    axs[2].plot(merged_df['MACD'], label='MACD', color='g')
    axs[2].plot(merged_df['Signal Line'], label='Signal Line', color='orange')
    axs[2].fill_between(merged_df.index, merged_df['MACD'] - merged_df['Signal Line'], 0, alpha=0.2, color='grey')
    axs[2].set_title('Monthly MACD and Signal Line')
    axs[2].legend()
    # Plot Fed Funds Rate
    axs[3].plot(merged_df['FEDFUNDS'], label='Federal Funds Rate', color='blue')
    axs[3].set_title('Federal Funds Rate')
    axs[3].legend()
# Define the half-Kelly criterion function
def half_kelly_criterion(mean_return, std_return):
    win_prob = (mean_return / std_return) ** 2 / ((mean_return / std_return) ** 2 + 1)
    loss_prob = 1 - win_prob
    odds = mean_return / std_return
    kelly_fraction = (win_prob - loss_prob) / odds
    half_kelly_fraction = kelly_fraction / 2
    return half_kelly_fraction
# Define portfolio return and variance calculations
def calculate_portfolio_return(weights, returns):
    return np.dot(weights, returns)
def calculate_portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))
# Objective function to maximize portfolio return
def objective_function(weights):
    portfolio_return = calculate_portfolio_return(weights, returns)
    portfolio_variance = calculate_portfolio_variance(weights, cov_matrix)
    portfolio_std_dev = np.sqrt(portfolio_variance)
    return -portfolio_return  # Negative for minimization
def visualize_seasonality_table(seasonality_table, title):
    """Visualize the seasonality table as a heatmap."""
    sns.heatmap(seasonality_table, annot=True, cmap='RdYlGn', center=0)
    plt.title(title)
def apply_kelly_method(mean_return, std_dev, win_prob):
    """Calculate the Kelly criterion for position sizing."""
    b = mean_return / std_dev  # Assuming b is the edge ratio
    kelly_fraction = win_prob - ((1 - win_prob) / b)
    return kelly_fraction
def display_all_monthly_statistics_with_kelly(df):
    """Display all monthly statistics for a DataFrame with Kelly position size."""
    df_monthly = resample_to_monthly(df)
    df_monthly['Monthly Return'] = df_monthly['Adj Close'].pct_change() * 100
    stats = df_monthly.groupby(df_monthly.index.month)['Monthly Return'].agg(['mean', 'std', 'count'])
    stats['positive_prob'] = df_monthly.groupby(df_monthly.index.month)['Monthly Return'].apply(lambda x: (x > 0).mean())
    stats['kelly_size'] = stats.apply(lambda row: apply_kelly_method(row['mean'], row['std'], row['positive_prob']), axis=1)
    stats.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month, row in stats.iterrows():
        print(f"{month}: Mean = {row['mean']:.2f}, Std Dev = {row['std']:.2f}, Count = {row['count']}, Positive Prob = {row['positive_prob']:.2f}, Kelly Size = {row['kelly_size']:.2f}")
    return stats
def machine_learning_analysis(df):
    """Perform machine learning analysis using RandomForest and return the model and accuracy."""
    df['Target'] = (df['Return'] > 0).astype(int)  # Binary classification: 1 if return is positive, 0 otherwise
    features = ['Adj Close', 'Return']  # Example features; you can add more technical indicators
    X = df[features].shift(1)  # Shift features to avoid look-ahead bias
    y = df['Target'].shift(1)
    # Drop rows with NaN values to ensure consistent lengths
    X, y = X.dropna(), y.dropna()
    X, y = X.align(y, join='inner', axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    return model, accuracy
def create_summary_csv(tickers, start_date, end_date, filename='summary.csv'):
    """Create a CSV file with mean, std, count, positive_prob, and Kelly size for all assets."""
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
def analyze_ticker(ticker, start_date, end_date):
        return
    visualize_seasonality_table(seasonality_table, f'Seasonality of {ticker} Returns')
    display_all_monthly_statistics_with_kelly(df)
    # Machine Learning Analysis
    model, accuracy = machine_learning_analysis(df)
def load_price_data(ticker, start, end):
    """Load historical price data from Yahoo Finance."""
    return yf.download(ticker, start=start, end=end)
def calculate_returns(df):
    """Calculate the daily returns."""
    df['Return'] = df['Adj Close'].pct_change() * 100
def resample_to_monthly(df):
    """Resample daily data to monthly data."""
    return df.resample('M').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Adj Close': 'last',
        'Volume': 'sum'
def seasonality_analysis(df):
    """Perform seasonality analysis."""
    return df.groupby(df.index.month)['Monthly Return'].agg(['mean', 'std', 'count', 'positive_prob'])
def create_seasonality_table(df):
    """Create a seasonality table for returns."""
    df = df.dropna(subset=['Return'])
    seasonality_table = df_monthly.groupby(df_monthly.index.month)['Monthly Return'].agg(['mean', 'std', 'count'])
    seasonality_table['positive_prob'] = df_monthly.groupby(df_monthly.index.month)['Monthly Return'].apply(lambda x: (x > 0).mean())
    return seasonality_table
def calculate_returns(df, period='M'):
    Calculate percentage returns based on the specified period.
    df['Return'] = df['Close'].resample(period).ffill().pct_change() * 100
    df['Volume'] = df['Volume'].resample(period).sum()  # Sum volume for each period
def create_seasonality_table(df, column='Return'):
    Create a pivot table for seasonality analysis.
    seasonality_table = df.pivot_table(values=column, index=df.index.year, columns=df.index.month, aggfunc=np.mean)
    # Ensure all months are represented
    seasonality_table = seasonality_table.reindex(columns=range(1, 13))
    seasonality_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    seasonality_table = seasonality_table.transpose()
def plot_technical_indicators(df):
    plt.subplot(3, 1, 1)
    plt.title('Close Price and Bollinger Bands')
    plt.subplot(3, 1, 2)
    plt.title('Relative Strength Index (RSI)')
    plt.subplot(3, 1, 3)
    plt.title('MACD and Signal Line')
def visualize_seasonality_table(seasonality_table, ticker):
    plt.title(f'Seasonality of {ticker} Returns')
def display_all_monthly_statistics(seasonality_table, seasonality_volume_table, ticker):
        print(f"{ticker} - {month} Statistics")