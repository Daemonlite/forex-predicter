import arrow
import bleach
import requests

import os
import json
from sklearn.linear_model import LinearRegression
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ForexPredictor:
    def __init__(self, apikey):
        self.apikey = apikey
        self.history_file = 'currency_history.json'

    """
   
    symbol: The name of the equity of your choice. For example: symbol=IBM

    Intraday: Refers to trading activity that occurs within the same trading day. Intraday OHLCV bars represent the price and volume data
    for a given financial instrument 
    (such as a stock or currency pair) over intervals within a single trading day (e.g., 1-minute intervals, 5-minute intervals).

    Outputsize parameter: This parameter is used to specify the number of data points (bars) that the API should return in the response. In this statement,
    it's mentioned that when the outputsize parameter is not set, the default behavior is to return the most recent 100 intraday OHLCV bars.

    """

    """
    This method will return the most recent 100 intraday Open, High, Low, Close, and Volume (OHLCV) 
    bars by default when the outputsize parameter is not set
    The outputsize parameter is used to specify the number of data points to be returned.

    """
    def most_recent_comp_OHLCV(self,symbol,interval):
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}min&apikey={self.apikey}"
        response = requests.get(url)
        data = response.json()
        
        # Check if the response contains data
        if 'Time Series' in data:
            # Extract OHLCV data from the response
            time_series_data = data['Time Series']
            # Sort keys (timestamps) to get the most recent data first
            sorted_timestamps = sorted(time_series_data.keys(), reverse=True)
            # Initialize a list to store OHLCV data
            ohlcv_data = []
            # Iterate over the timestamps and extract OHLCV values
            for timestamp in sorted_timestamps[:100]:  # Limit to most recent 100 data points
                ohlcv_values = time_series_data[timestamp]
                ohlcv_data.append({
                    'timestamp': timestamp,
                    'open': ohlcv_values['1. open'],
                    'high': ohlcv_values['2. high'],
                    'low': ohlcv_values['3. low'],
                    'close': ohlcv_values['4. close'],
                    'volume': ohlcv_values['5. volume']
                })
            return ohlcv_data
        else:
            # If there's no data, return None or handle the case accordingly
            return None
        

    def currency_market(self, fsymbol, tsymbol):
        url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={fsymbol}&to_symbol={tsymbol}&apikey={self.apikey}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Save data to history file
            self.save_to_history(data)
            return data
        else:
            print("Error fetching data:", response.status_code)
            return None

    def save_to_history(self, data):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as file:
                history = json.load(file)
        else:
            history = []

        history.append(data)

        with open(self.history_file, 'w') as file:
            json.dump(history, file)

    def check_history(self):
        if os.path.exists(self.history_file):
            return True
        else:
            return False

    def predictor(self):
        logger.warning("prediction start")
        # Load historical data from the history file
        with open(self.history_file, 'r') as file:
            history_data = json.load(file)
            

        # Extract features (X) and targets (y) from historical data
        # Extract features (X) and targets (y) from historical data
        X = []
        y_open = []
        y_high = []
        y_low = []
        y_close = []

        time_series = history_data["Time Series FX (Daily)"]

        for date, data in time_series.items():
            # Extract features and targets from each data point
            open_price = float(data["1. open"])
            high_price = float(data["2. high"])
            low_price = float(data["3. low"])
            close_price = float(data["4. close"])
            
            X.append([open_price, high_price, low_price, close_price])
            y_open.append(open_price)
            y_high.append(high_price)
            y_low.append(low_price)
            y_close.append(close_price)


        # Convert lists to numpy arrays
        X = np.array(X)
        y_open = np.array(y_open)
        y_high = np.array(y_high)
        y_low = np.array(y_low)
        y_close = np.array(y_close)

        # Fit a linear regression model for each target
        open_model = LinearRegression().fit(X, y_open)
        high_model = LinearRegression().fit(X, y_high)
        low_model = LinearRegression().fit(X, y_low)
        close_model = LinearRegression().fit(X, y_close)

        # Predict the next prices based on the last historical data point
        last_data_point = history_data[-1]
        last_features = np.array([[float(last_data_point['open']), float(last_data_point['high']), float(last_data_point['low']), float(last_data_point['close'])]])

        next_open = open_model.predict(last_features)[0]
        next_high = high_model.predict(last_features)[0]
        next_low = low_model.predict(last_features)[0]
        next_close = close_model.predict(last_features)[0]

        # Save predictions to a JSON file
        predictions = {
            'next_open': next_open,
            'next_high': next_high,
            'next_low': next_low,
            'next_close': next_close
        }

        with open('predictions.json', 'w') as file:
            json.dump(predictions, file)

        logger.warning("predictions made successfully")

        return "predictions made successfully"

        

    

        
    