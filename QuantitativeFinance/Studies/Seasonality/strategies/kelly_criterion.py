def half_kelly_criterion(mean_return, std_return):
    win_prob = (mean_return / std_return) ** 2 / ((mean_return / std_return) ** 2 + 1)
    loss_prob = 1 - win_prob
    odds = mean_return / std_return
    kelly_fraction = (win_prob - loss_prob) / odds
    half_kelly_fraction = kelly_fraction / 2
    return half_kelly_fraction

def calculate_half_kelly_fractions(seasonality_stats):
    seasonality_stats['half_kelly_fraction'] = seasonality_stats.apply(
        lambda row: half_kelly_criterion(row['mean'], row['std']), axis=1
    )
    return seasonality_stats
def position_size_half_kelly(signals, seasonality_stats, iv_series):
    signals['half_kelly_fraction'] = signals.index.month.map(seasonality_stats['half_kelly_fraction'])
    signals['position_size'] = signals['half_kelly_fraction'].fillna(0) * signals['Buy']
    return signals
