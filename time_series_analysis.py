# time_series_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from math import sqrt

def load_data(file_path):
    """
    Load time series data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded time series data.
    """
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    return data

def plot_time_series(data, title='Time Series Data'):
    """
    Plot the time series data.

    Parameters:
    data (pd.DataFrame): Time series data.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Time Series')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def decompose_time_series(data, model='additive', period=None):
    """
    Decompose the time series into trend, seasonal, and residual components.

    Parameters:
    data (pd.DataFrame): Time series data.
    model (str): Model type ('additive' or 'multiplicative').
    period (int): Period of the seasonal component.

    Returns:
    statsmodels.tsa.seasonal.DecomposeResult: Decomposition result.
    """
    decomposition = seasonal_decompose(data, model=model, period=period)
    decomposition.plot()
    plt.show()
    return decomposition

def test_stationarity(data):
    """
    Test the stationarity of the time series using the Augmented Dickey-Fuller test.

    Parameters:
    data (pd.DataFrame): Time series data.

    Returns:
    tuple: ADF test results.
    """
    result = adfuller(data)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    return result

def plot_acf_pacf(data, lags=None):
    """
    Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).

    Parameters:
    data (pd.DataFrame): Time series data.
    lags (int): Number of lags to include in the plot.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(data, lags=lags, ax=ax1)
    plot_pacf(data, lags=lags, ax=ax2)
    plt.show()

def fit_arima_model(data, order=(1, 1, 1)):
    """
    Fit an ARIMA model to the time series data.

    Parameters:
    data (pd.DataFrame): Time series data.
    order (tuple): ARIMA order (p, d, q).

    Returns:
    statsmodels.tsa.arima.model.ARIMAResults: Fitted ARIMA model.
    """
    model = ARIMA(data, order=order)
    fitted_model = model.fit()
    print(fitted_model.summary())
    return fitted_model

def forecast(fitted_model, steps=10):
    """
    Forecast future values using the fitted ARIMA model.

    Parameters:
    fitted_model (statsmodels.tsa.arima.model.ARIMAResults): Fitted ARIMA model.
    steps (int): Number of steps to forecast.

    Returns:
    pd.DataFrame: Forecasted values.
    """
    forecast = fitted_model.forecast(steps=steps)
    return forecast

def evaluate_model(fitted_model, data):
    """
    Evaluate the ARIMA model using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

    Parameters:
    fitted_model (statsmodels.tsa.arima.model.ARIMAResults): Fitted ARIMA model.
    data (pd.DataFrame): Time series data.

    Returns:
    tuple: MSE and RMSE values.
    """
    predictions = fitted_model.predict()
    mse = mean_squared_error(data, predictions)
    rmse = sqrt(mse)
    print('MSE:', mse)
    print('RMSE:', rmse)
    return mse, rmse

def main():
    # Example usage
    file_path = 'path_to_your_data.csv'
    data = load_data(file_path)

    plot_time_series(data)

    decomposition = decompose_time_series(data, model='additive', period=12)

    stationarity_result = test_stationarity(data)

    plot_acf_pacf(data, lags=20)

    fitted_model = fit_arima_model(data, order=(1, 1, 1))

    forecast_values = forecast(fitted_model, steps=10)
    print('Forecasted Values:', forecast_values)

    mse, rmse = evaluate_model(fitted_model, data)

if __name__ == '__main__':
    main()
