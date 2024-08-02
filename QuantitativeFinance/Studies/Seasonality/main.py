# main.py

from src.data_collection.load_data import load_price_data, load_fed_data
from src.data_collection.resample_data import resample_to_monthly
from src.data_collection.fetch_options import get_options_data
from src.analysis.calculate_returns import calculate_monthly_returns
from src.analysis.technical_indicators import calculate_technical_indicators
from src.analysis.seasonality_analysis import seasonality_analysis, seasonality_analysis_iv
from src.analysis.implied_volatility import calculate_implied_volatility
from src.strategies.kelly_criterion import calculate_half_kelly_fractions
from src.strategies.position_sizing import position_size_half_kelly
from src.strategies.backtest_strategy import backtest_strategy_with_half_kelly
from src.visualization.plot_data import plot_technical_indicators
from src.visualization.plot_strategy import plot_cumulative_returns_with_half_kelly

# def main():
#     # Data collection
#     tickers = ['SPY', 'SPI-U', 'SPX-S']
#     price_data = load_price_data(tickers)
#     fed_funds = load_fed_data('FEDFUNDS')
#     spy_options = get_options_data('SPY', '2020-01-01', '2023-01-01')

#     # Data processing
#     price_data_monthly = resample_to_monthly(price_data)
#     fed_funds_monthly = resample_to_monthly(fed_funds)
    
#     spy_data = calculate_monthly_returns(price_data_monthly[['SPY']])
#     spy_data = calculate_technical_indicators(spy_data)
    
#     spy_iv = calculate_implied_volatility(spy_options)
    
#     seasonality_stats = seasonality_analysis(spy_data)
#     seasonality_iv_stats = seasonality_analysis_iv(spy_iv)
    
#     seasonality_stats = calculate_half_kelly_fractions(seasonality_stats)
#     spy_signals = position_size_half_kelly(spy_data, seasonality_stats, spy_iv)
    
#     spy_backtest = backtest_strategy_with_half_kelly(spy_data, spy_signals)
    
#     # Visualization
#     plot_technical_indicators(spy_data)
#     plot_cumulative_returns_with_half_kelly(spy_backtest, spy_backtest, 'SPY Returns: Original vs. Strategy with Half-Kelly Position Sizing')

# if __name__ == "__main__":
#     main()
