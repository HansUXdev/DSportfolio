import matplotlib.pyplot as plt  
from statsmodels.tsa.seasonal import seasonal_decompose  
from statsmodels.tsa.stattools import adfuller  
  
class TimeSeriesAnalysis:  
    def decompose_series(self, df, column, model='additive', freq=None):  
        """Decompose the time series into trend, seasonal, and residual components."""  
        decomposition = seasonal_decompose(df[column], model=model, period=freq)  
        return decomposition  
  
    def check_stationarity(self, df, column):  
        """Perform the Augmented Dickey-Fuller test to check for stationarity."""  
        result = adfuller(df[column])  
        return result  
  
    def plot_decomposition(self, decomposition):  
        """Plot the decomposed components of the time series."""  
        decomposition.plot()  
        plt.show()  