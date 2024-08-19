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
