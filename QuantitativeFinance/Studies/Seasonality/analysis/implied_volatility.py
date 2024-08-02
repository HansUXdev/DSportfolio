# src/analysis/implied_volatility.py

def calculate_implied_volatility(options_df):
    options_df['mid'] = (options_df['bid'] + options_df['ask']) / 2
    atm_options = options_df.loc[(options_df['strike'] == options_df['strike'].median()) & (options_df['option_type'] == 'call')]
    atm_options['implied_volatility'] = atm_options['impliedVolatility']
    return atm_options['implied_volatility']
