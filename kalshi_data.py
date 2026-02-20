import time, requests, csv, os
import pandas as pd
from dotenv import load_dotenv
from pprint import pprint

import kalshi_python
from kalshi_python.models.get_market_response import GetMarketResponse
from kalshi_python.rest import ApiException
from kalshi_python_sync import Configuration, KalshiClient

#===================== Config ==============================
load_dotenv()
API_KEY_ID = os.getenv("KALSHI_KEY_ID")
PRIVATE_KEY_PATH = "kalshi.pem"

def configure_client(path: str):
    config = Configuration(
        host="https://api.elections.kalshi.com/trade-api/v2"
    )

    with open(path, "r") as f:
        config.private_key_pem = f.read()

    config.api_key_id = API_KEY_ID

    client = KalshiClient(config)

    balance = client.get_balance()
    print(f"Balance: ${balance.balance / 100:.2f}")

    return client, balance

#===========================================================

#===================== Get Market ==========================
def get_market(client, ticker):
    try:
        market = client.get_market(ticker)
        print("The response of MarketsApi->get_market:\n")
        pprint(market)
    except Exception as e:
        print(f"Error fetching market: {e}")
        return None
#===========================================================


if __name__ == "__main__":
    client, balance = configure_client(PRIVATE_KEY_PATH)
    get_market(client, "Kenya Open Winner?")
