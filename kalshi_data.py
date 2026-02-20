import time
import requests
import os
import csv
import pandas as pd
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.exceptions import InvalidSignature
from kalshi_python_sync import Configuration, KalshiClient

# ==================== Constants ====================

BASE = "https://api.elections.kalshi.com"
API_KEY_ID = os.getenv("KALSHI_KEY_ID")
PRIVATE_KEY_PATH = "Main Key.txt"

# ====================================================

#===================== Init Stuff ====================
def sign_pss_text(private_key: rsa.RSAPrivateKey, text: str) -> str:
    message = text.encode('utf-8')
    try:
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')
    except InvalidSignature as e:
        raise ValueError("RSA sign PSS failed") from e

def load_private_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def sign_request(private_key, timestamp_ms: str, method: str, path: str) -> str:
    path_no_query = path.split("?")[0]
    msg = f"{timestamp_ms}{method.upper()}{path_no_query}".encode("utf-8")
    sig = private_key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")

def authed_get(path: str, params=None):
    pk = load_private_key(PRIVATE_KEY_PATH)
    ts = str(int(time.time() * 1000))
    sig = sign_request(pk, ts, "GET", path)
    headers = {
        "KALSHI-ACCESS-KEY": API_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": sig,
    }
    return requests.get(BASE + path, headers=headers, params=params)

#===========================================================

#======================= Config ============================

# Configure the client
def configure_client(path: str):
    config = Configuration(
        host="https://api.elections.kalshi.com/trade-api/v2"
    )
    # For authenticated requests
    # Read private key from file
    with open(path, "r") as f:
        private_key = f.read()

    config.api_key_id = API_KEY_ID
    config.private_key_pem = private_key

    # Initialize the client
    client = KalshiClient(config)

    # Make API calls
    balance = client.get_balance()
    print(f"Balance: ${balance.balance / 100:.2f}")
    return client, balance

if __name__ == "__main__":
    client, balance = configure_client()
