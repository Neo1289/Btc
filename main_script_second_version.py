import pandas as pd
import os
import logging
import requests
from datetime import datetime
import schedule
import time
from pathlib import Path
import numpy as np

# Configure logging at the top
logging.basicConfig(filename="daily_query.log", level=logging.INFO)

def get_bitcoin_daily_transactions():
    try:

        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': '365',
            'interval': 'daily'
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()

            prices = data['prices']
            dates = [datetime.utcfromtimestamp(price[0] / 1000).strftime('%Y-%m-%d') for price in prices]
            closing_prices = [price[1] for price in prices]

            df_price = pd.DataFrame({
                'date': dates,
                'Closing Price (USD)': closing_prices
            })

            # Remove duplicate dates, keeping the latest entry
            df_price = df_price.drop_duplicates(subset=['date'], keep='last')

            ma_list = [50, 100, 150, 200]

            # Calculate the moving averages
            for i in ma_list:
                df_price[f'{i}_MA'] = df_price['Closing Price (USD)'].rolling(window=i).mean()

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

            script_dir = Path(__file__).resolve().parent
            csv_file_price = script_dir / "daily_bitcoin_price.csv"

            # Ensure dates are datetime objects
            df_price['date'] = pd.to_datetime(df_price['date'])

            # Write to CSV (overwrite or append as needed)
            df_price.to_csv(csv_file_price, mode='w', header=True, index=False)

            logging.info(f"Successfully wrote {len(df_price)} rows to {csv_file_price} on {pd.Timestamp.now()}")

        else:
            logging.error(f"Failed to fetch data from CoinGecko API. Status code: {response.status_code}")

    except Exception as e:
        logging.error(f"Error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    logging.info("Scheduler started. Waiting for the scheduled time...")
    try:
        schedule.every().day.at("13:50").do(get_bitcoin_daily_transactions)
        while True:
            schedule.run_pending()
            time.sleep(25) 
    except KeyboardInterrupt:
        logging.info("Scheduler stopped manually.")
    except Exception as e:
        logging.error(f"Scheduler stopped due to an error: {e}", exc_info=True)
