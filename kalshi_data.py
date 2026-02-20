import time
import requests
import csv
import pandas as pd
from kalshi_python_sync import Configuration, KalshiClient
import os
from dotenv import load_dotenv

#===================== Config ====================
load_dotenv()
API_KEY_ID = os.getenv("KALSHI_KEY_ID")

if not API_KEY_ID:
    raise RuntimeError(
        "KALSHI_KEY_ID is not set. "
        "Set it in your shell or in a .env file before running."
    )

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

if __name__ == "__main__":
    client, balance = configure_client(PRIVATE_KEY_PATH)
