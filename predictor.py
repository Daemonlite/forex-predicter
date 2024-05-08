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
        self.history_file = "currency_history.json"
        self.comp_file = "company_stock_history.json"

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

    def most_recent_comp_OHLCV_to_json(self, symbol, interval):
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}min&apikey={self.apikey}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            self.save_to_history(data, type="comp")
            logger.warning(f"Predicted OHLCV data saved to {self.comp_file}")
        else:
            logger.warning("No data found.")

    def predict_comp(self):
        logger.warning("prediction start")
        with open(self.comp_file, "r") as file:
            history_data = json.load(file)["OHLCV Data"]

        # Extract features (X) and targets (y) from historical data
        X = []
        y_open = []
        y_high = []
        y_low = []
        y_close = []

        for data_point in history_data:
            # Extract features and targets from each data point
            features = [
                float(data_point["open"]),
                float(data_point["high"]),
                float(data_point["low"]),
                float(data_point["close"]),
            ]
            target_open = float(data_point["open"])
            target_high = float(data_point["high"])
            target_low = float(data_point["low"])
            target_close = float(data_point["close"])

            X.append(features)
            y_open.append(target_open)
            y_high.append(target_high)
            y_low.append(target_low)
            y_close.append(target_close)

        # Convert lists to numpy arrays
        X = np.array(X)
        y_open = np.array(y_open)
        y_high = np.array(y_high)
        y_low = np.array(y_low)
        y_close = np.array(y_close)

        # Fit linear regression models
        open_model = LinearRegression().fit(X, y_open)
        high_model = LinearRegression().fit(X, y_high)
        low_model = LinearRegression().fit(X, y_low)
        close_model = LinearRegression().fit(X, y_close)

        # Predict the next OHLCV values using the trained models
        last_data_point = list(history_data)[0]
        last_features = np.array(
            [
                float(last_data_point["open"]),
                float(last_data_point["high"]),
                float(last_data_point["low"]),
                float(last_data_point["close"]),
            ]
        ).reshape(1, -1)

        next_open = open_model.predict(last_features)[0]
        next_high = high_model.predict(last_features)[0]
        next_low = low_model.predict(last_features)[0]
        next_close = close_model.predict(last_features)[0]

        # Construct the predicted OHLCV data
        predicted_data = {
            "1. open": str(next_open),
            "2. high": str(next_high),
            "3. low": str(next_low),
            "4. close": str(next_close),
            "5. volume": "0",  # Volume is not predicted, so setting it to 0
        }

        with open("comp_predictions.json", "w") as file:
            json.dump(predicted_data, file)
            logger.warning("predictions made successfully")

    def currency_market(self, fsymbol, tsymbol):
        url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={fsymbol}&to_symbol={tsymbol}&apikey={self.apikey}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Save data to history file
            self.save_to_history(data, type="currency")
            return data
        else:
            print("Error fetching data:", response.status_code)
            return None

    def save_to_history(self, data, type):
        if type == "currency":
            if os.path.exists(self.history_file):
                with open(self.history_file, "r") as file:
                    history = json.load(file)
            else:
                history = []

            history.append(data)

            with open(self.history_file, "w") as file:
                json.dump(history, file)

        if type == "comp":
            if os.path.exists(self.comp_file):
                with open(self.comp_file, "r") as file:
                    history = json.load(file)
            else:
                history = []

                history.append(data)

                with open(self.comp_file, "w") as file:
                    json.dump(history, file)

    def check_history(self):
        if os.path.exists(self.history_file):
            return True
        else:
            return False

    def predictor(self):
        logger.warning("prediction start")
        # Load historical data from the history file
        with open(self.history_file, "r") as file:
            history_data = json.load(file)
            print(history_data)

        # Extract features (X) and targets (y) from historical data
        X = []
        y_open = []
        y_high = []
        y_low = []
        y_close = []

        for data_point in history_data:
            
            time_series = data_point["Time Series FX (Daily)"]
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
        last_data_point = history_data[-1]["Time Series FX (Daily)"]
        last_features = np.array(
            [
                [
                    float(last_data_point[date]["1. open"]),
                    float(last_data_point[date]["2. high"]),
                    float(last_data_point[date]["3. low"]),
                    float(last_data_point[date]["4. close"]),
                ]
                for date in last_data_point
            ]
        )

        next_open = open_model.predict(last_features)[0]
        next_high = high_model.predict(last_features)[0]
        next_low = low_model.predict(last_features)[0]
        next_close = close_model.predict(last_features)[0]

        # Save predictions to a JSON file
        predictions = {
            "next_open": next_open,
            "next_high": next_high,
            "next_low": next_low,
            "next_close": next_close,
        }

        with open("predictions.json", "w") as file:
            json.dump(predictions, file)

        logger.warning("predictions made successfully")

        return "predictions made successfully"
