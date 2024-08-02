# src/visualization/plot_strategy.py

import matplotlib.pyplot as plt

def plot_cumulative_returns_with_half_kelly(original, strategy, title):
    plt.figure(figsize=(14, 7))
    plt.plot(original.index, original['Cumulative Return'], label='Original Return')
    plt.plot(strategy.index, strategy['Cumulative Strategy Return'], label='Strategy Return with Half-Kelly Position Sizing')
    plt.legend()
    plt.title(title)
    plt.show()
