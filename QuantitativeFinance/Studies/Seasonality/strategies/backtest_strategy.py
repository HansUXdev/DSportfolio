# strategies/backtest_strategy.py

def backtest_strategy_with_half_kelly(df, signals):
    df['Strategy Return'] = df['Monthly Return'] * signals['position_size'].shift(1)
    df['Cumulative Return'] = (1 + df['Monthly Return']/100).cumprod() - 1
    df['Cumulative Strategy Return'] = (1 + df['Strategy Return']/100).cumprod() - 1
    return df
