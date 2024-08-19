############################################################################################################
def check_argument_types(arguments, expected_types):
    for arg_name, arg_value in arguments.items():
        if not isinstance(arg_value, expected_types[arg_name]):
            raise TypeError(f"Argument '{arg_name}' should be of type {expected_types[arg_name].__name__}, "
                            f"but got {type(arg_value).__name__}")
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
