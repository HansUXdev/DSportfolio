# ./analysis/seasonality_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

def seasonal_decomposition(df, column, period=12):
    decomposition = seasonal_decompose(df[column], model='additive', period=period)
    return decomposition

def seasonality_analysis(df):
    seasonality = df.groupby(df.index.month)['Monthly Return'].agg(['mean', 'std', 'count'])
    seasonality['positive_prob'] = df.groupby(df.index.month)['Monthly Return'].apply(lambda x: (x > 0).mean())
    seasonality.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return seasonality

def display_seasonality_stats(df):
    """Display seasonality statistics."""
    stats = df.groupby(df.index.month)['Monthly Return'].agg(['mean', 'std', 'max', 'min'])
    stats.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for month in stats.index:
        mean_return = stats.loc[month, 'mean']
        print(f"{month}: Mean = {mean_return:.2f}")

def plot_seasonality(df, title):
    """Plot seasonality of the given DataFrame."""
    monthly_means = df.groupby(df.index.month)['Monthly Return'].mean()
    monthly_means.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_means, marker='o')
    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel('Mean Monthly Return (%)')
    plt.grid(True)
    plt.show()


def calculate_returns(df):
    """Calculate the daily returns."""
    df['Return'] = df['Adj Close'].pct_change() * 100
    return df

def create_seasonality_table(df):
    """Create a seasonality table for returns."""
    df = df.dropna(subset=['Return'])
    df_monthly = resample_to_monthly(df)
    df_monthly['Monthly Return'] = df_monthly['Adj Close'].pct_change() * 100
    return seasonality_analysis(df_monthly)

def visualize_seasonality_table(seasonality_table, title):
    """Visualize the seasonality table as a heatmap."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(seasonality_table, annot=True, cmap='RdYlGn', center=0)
    plt.title(title)
    plt.show()

def display_all_monthly_statistics(df):
    """Display all monthly statistics for a DataFrame."""
    df_monthly = resample_to_monthly(df)
    df_monthly['Monthly Return'] = df_monthly['Adj Close'].pct_change() * 100
    display_seasonality_stats(df_monthly)