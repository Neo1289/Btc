import pandas as pd
import os
import logging
import requests
from datetime import datetime
import schedule
import time
from pathlib import Path
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sys
import signal

# ===========================
# Configuration and Constants
# ===========================

# Environment Variables or Defaults
API_URL = os.getenv("COINGECKO_API_URL", "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart")
VS_CURRENCY = os.getenv("VS_CURRENCY", "usd")
DAYS = os.getenv("DAYS", "365")
INTERVAL = os.getenv("INTERVAL", "daily")
LOG_FILE = os.getenv("LOG_FILE", "daily_query.log")
CSV_FILE_NAME = os.getenv("CSV_FILE_NAME", "daily_bitcoin_price.csv")
SCHEDULE_TIME = os.getenv("SCHEDULE_TIME", "13:50")  # Format: "HH:MM"

# Retry Configuration
RETRIES = int(os.getenv("RETRIES", "5"))
BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", "1"))
STATUS_FORCELIST = [429, 500, 502, 503, 504]

# ===========================
# Logging Configuration
# ===========================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ===========================
# Function Definitions
# ===========================

def get_bitcoin_daily_transactions():
    """
    Fetches Bitcoin market data from CoinGecko API, processes it, and saves to a CSV file.
    Implements retries with exponential backoff for robustness.
    """
    try:
        logging.info("Starting data fetch from CoinGecko API.")

        # Prepare API parameters
        params = {
            'vs_currency': VS_CURRENCY,
            'days': DAYS,
            'interval': INTERVAL
        }

        # Setup session with retry strategy
        session = requests.Session()
        retry_strategy = Retry(
            total=RETRIES,
            backoff_factor=BACKOFF_FACTOR,
            status_forcelist=STATUS_FORCELIST,
            allowed_methods=["GET"],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # Make the API request
        response = session.get(API_URL, params=params, timeout=10)

        # Check response status
        if response.status_code == 200:
            data = response.json()

            prices = data.get('prices', [])
            if not prices:
                logging.warning("No price data found in the API response.")
                return

            # Process dates and closing prices
            dates = [
                datetime.utcfromtimestamp(price[0] / 1000).strftime('%Y-%m-%d')
                for price in prices
            ]
            closing_prices = [price[1] for price in prices]

            df_price = pd.DataFrame({
                'date': dates,
                'Closing Price (USD)': closing_prices
            })

            # Remove duplicate dates, keeping the latest entry
            df_price = df_price.drop_duplicates(subset=['date'], keep='last')

            ma_list = [50, 100, 150, 200]

            # Calculate the moving averages
            for ma in ma_list:
                df_price[f'{ma}_MA'] = df_price['Closing Price (USD)'].rolling(window=ma).mean()

            # Calculate the 200-day EMA
            df_price['200_EMA'] = df_price['Closing Price (USD)'].ewm(span=200, adjust=False).mean()

            # Calculate the RSI
            df_price['Price_Change'] = df_price['Closing Price (USD)'].diff()
            df_price['Gain'] = df_price['Price_Change'].apply(lambda x: x if x > 0 else 0)
            df_price['Loss'] = df_price['Price_Change'].apply(lambda x: -x if x < 0 else 0)
            window_length = 14
            df_price['Avg_Gain'] = df_price['Gain'].rolling(window=window_length).mean()
            df_price['Avg_Loss'] = df_price['Loss'].rolling(window=window_length).mean()
            df_price['RS'] = df_price['Avg_Gain'] / df_price['Avg_Loss'].replace(0, np.nan)
            df_price['RSI'] = 100 - (100 / (1 + df_price['RS']))

            # Calculate the MACD
            df_price['12_EMA'] = df_price['Closing Price (USD)'].ewm(span=12, adjust=False).mean()
            df_price['26_EMA'] = df_price['Closing Price (USD)'].ewm(span=26, adjust=False).mean()
            df_price['MACD_Line'] = df_price['12_EMA'] - df_price['26_EMA']
            df_price['Signal_Line'] = df_price['MACD_Line'].ewm(span=9, adjust=False).mean()
            df_price['MACD_Histogram'] = df_price['MACD_Line'] - df_price['Signal_Line']

            # Define CSV file path
            script_dir = Path(__file__).resolve().parent
            csv_file_price = script_dir / CSV_FILE_NAME

            # Ensure 'date' column is datetime
            df_price['date'] = pd.to_datetime(df_price['date'])

            # Write to CSV (overwrite)
            df_price.to_csv(csv_file_price, mode='w', header=True, index=False)

            logging.info(f"Successfully wrote {len(df_price)} rows to {csv_file_price}.")

        else:
            logging.error(f"Failed to fetch data from CoinGecko API. Status code: {response.status_code}. Response: {response.text}")

    except requests.exceptions.RequestException as req_err:
        logging.error(f"RequestException occurred: {req_err}", exc_info=True)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

def start_scheduler():
    """
    Initializes and starts the scheduler to run the data fetching function at the scheduled time.
    """
    logging.info("Scheduler started. Waiting for the scheduled time...")
    schedule.every().day.at(SCHEDULE_TIME).do(get_bitcoin_daily_transactions)

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)  # Reduced sleep time for better responsiveness
    except KeyboardInterrupt:
        logging.info("Scheduler stopped manually via KeyboardInterrupt.")
    except Exception as e:
        logging.error(f"Scheduler encountered an error: {e}", exc_info=True)

def graceful_shutdown(signum, frame):
    """
    Handles graceful shutdown on receiving termination signals.
    """
    logging.info(f"Received termination signal ({signum}). Shutting down gracefully...")
    sys.exit(0)

# ===========================
# Main Execution
# ===========================

if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, graceful_shutdown)   # Handle Ctrl+C
    signal.signal(signal.SIGTERM, graceful_shutdown)  # Handle termination signals

    # Perform an initial run immediately (optional)
    get_bitcoin_daily_transactions()

    # Start the scheduler
    start_scheduler()
