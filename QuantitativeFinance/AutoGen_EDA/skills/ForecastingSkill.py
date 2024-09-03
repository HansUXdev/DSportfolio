from statsmodels.tsa.arima.model import ARIMA  
  
class ForecastingSkill:  
    def arima_forecast(self, df, column, order=(1, 1, 1), steps=10):  
        """Perform ARIMA forecasting."""  
        model = ARIMA(df[column], order=order)  
        model_fit = model.fit()  
        forecast = model_fit.forecast(steps=steps)  
        return forecast  
  
    def plot_forecast(self, df, column, forecast):  
        """Plot the original series and the forecast."""  
        plt.plot(df[column], label='Original')  
        plt.plot(forecast, label='Forecast', color='red')  
        plt.legend()  
        plt.show()  