# src/data_collection/resample_data.py

def resample_to_monthly(df):
    return df.resample('M').ffill()
