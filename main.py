from predictor import ForexPredictor
from decouple import config
import logging

logger = logging.getLogger(__name__)

apikey = config('ALPHA_VINTAGE_KEY')

import time

def main():
    # First, set your forex currencies to predict
    from_symbol = input('Enter the currency you want to predict (e.g., USD, GBP, EUR): ')
    to_symbol = input('Enter your local currency: ')
    
    # Create an instance of ForexPredictor with the provided API key
    predictor = ForexPredictor(apikey)
    
    # Fetch historical data for the specified currency pair
    predictor.currency_market(fsymbol=from_symbol, tsymbol=to_symbol)
    
    # Wait for ten seconds
    print("Waiting for 10 seconds before making predictions...")
    time.sleep(10)
    
    # Predict the later outcome
    predictor.predictor()




if __name__ == "__main__":
    main()
