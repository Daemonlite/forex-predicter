from predictor import ForexPredictor
from decouple import config
import logging

logger = logging.getLogger(__name__)

apikey = config("ALPHA_VINTAGE_KEY")

import time


def main():
    # select what to predict
    predictor = ForexPredictor(apikey)

    pred = input(
        "What do you want to predict on the market? \n 1. Company Stock\n 2. Currency Forex \n : "
    )

    if pred == "1":
        symbol = input("enter company symbol egs.IBM, : ")
        interval = input("input time interval egs.1,2,3 all are in minutes : ")
        predictor.most_recent_comp_OHLCV_to_json(symbol=symbol, interval=interval)

        print("Waiting for 10 seconds before making predictions...")
        time.sleep(10)

        predictor.predict_comp()

    elif pred == "2":

        # First, set your forex currencies to predict
        from_symbol = input(
            "Enter the currency you want to predict (e.g., USD, GBP, EUR): "
        )
        to_symbol = input("Enter your local currency: ")

        # Create an instance of ForexPredictor with the provided API key

        # Fetch historical data for the specified currency pair
        predictor.currency_market(fsymbol=from_symbol, tsymbol=to_symbol)

        # Wait for ten seconds
        print("Waiting for 10 seconds before making predictions...")
        time.sleep(10)

        # Predict the later outcome
        predictor.predictor()


if __name__ == "__main__":
    main()
