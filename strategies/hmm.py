"""
Hidden Markov Model for prediction market price series.

Strategies 1 (quant) and 2 (news-informed) both feed into this model.
Hidden states represent market regimes:
  0 — stable     (price drifting slowly, low spread, normal volume)
  1 — drifting   (trending move, widening spread)
  2 — resolving  (price approaching 0 or 100, sharp volume spike)

Usage:
    from strategies.hmm import MarketHMM
    model = MarketHMM()
    model.fit()                    # trains on data/candlesticks/*.csv
    result = model.predict(df)     # df from fetch_live_candlesticks()
    # result = {"fair_value": 65.3, "confidence": 0.71, "regime": "drifting"}
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CANDLES_DIR = Path(__file__).parent.parent / "data" / "candlesticks"
REGIMES = {0: "stable", 1: "drifting", 2: "resolving"}


class MarketHMM:
    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._model = None
        self._fitted = False

    # ------------------------------------------------------------------ #
    #  Feature engineering                                                 #
    # ------------------------------------------------------------------ #

    def _build_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Build 4-feature matrix from a candlestick DataFrame.
        Returns None if the DataFrame lacks required columns or has too few rows.

        Features:
          price_mid         — (yes_bid_close + yes_ask_close) / 2
          spread_pct        — (yes_ask_close - yes_bid_close) / 100
          volume_delta      — volume.diff() (normalised to [-1, 1])
          time_remaining    — linear 0→1 across the sequence
        """
        required = {"yes_bid_close", "yes_ask_close", "volume"}
        if not required.issubset(df.columns):
            return None

        df = df.dropna(subset=list(required)).copy()
        if len(df) < self.n_components:
            return None

        price_mid = (df["yes_bid_close"] + df["yes_ask_close"]) / 2.0
        spread_pct = (df["yes_ask_close"] - df["yes_bid_close"]) / 100.0
        vol_delta = df["volume"].diff().fillna(0)
        # normalise vol_delta to [-1, 1]
        vd_max = vol_delta.abs().max()
        vol_delta_norm = vol_delta / vd_max if vd_max > 0 else vol_delta
        n = len(df)
        time_remaining = np.linspace(0.0, 1.0, n)

        X = np.column_stack([
            price_mid.values,
            spread_pct.values,
            vol_delta_norm.values,
            time_remaining,
        ])
        return X.astype(float)

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, candles_dir: Path = CANDLES_DIR) -> None:
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning("hmmlearn not installed — HMM strategy disabled")
            return

        candles_dir = Path(candles_dir)
        if not candles_dir.exists():
            logger.warning(f"Candlesticks directory not found: {candles_dir}")
            return

        csv_files = list(candles_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(
                f"No candlestick CSVs found in {candles_dir}. "
                "Run `python -m data.history` to collect training data. "
                "HMM will use midpoint fallback until trained."
            )
            return

        sequences, lengths = [], []
        for path in csv_files:
            try:
                df = pd.read_csv(path)
                X = self._build_features(df)
                if X is not None and len(X) >= self.n_components:
                    sequences.append(X)
                    lengths.append(len(X))
            except Exception as e:
                logger.debug(f"Skipping {path.name}: {e}")

        if not sequences:
            logger.warning("No valid sequences to train on after filtering.")
            return

        X_all = np.vstack(sequences)
        model = GaussianHMM(
            n_components=self.n_components,
            covariance_type="diag",
            n_iter=100,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_all, lengths)

        self._model = model
        self._fitted = True
        logger.info(f"HMM trained on {len(sequences)} sequences ({len(X_all)} timesteps)")

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    def predict(self, ticker_df: pd.DataFrame) -> dict:
        """
        Returns dict: {fair_value (0–100), confidence (0–1), regime (str)}.
        Falls back to midpoint estimate when model is not fitted.
        """
        fallback_value = self._midpoint_fallback(ticker_df)

        if not self._fitted or self._model is None:
            return {"fair_value": fallback_value, "confidence": 0.33, "regime": "stable"}

        X = self._build_features(ticker_df)
        if X is None or len(X) < 2:
            return {"fair_value": fallback_value, "confidence": 0.33, "regime": "stable"}

        try:
            state_seq = self._model.predict(X)
            # Posterior probabilities for last observation
            log_post = self._model.predict_proba(X)
            last_probs = log_post[-1]
            last_state = int(state_seq[-1])
            confidence = float(last_probs[last_state])
            regime = REGIMES.get(last_state, "stable")

            # Fair value: smoothed price_mid from last few candles, informed by regime
            price_mids = X[:, 0]   # first feature column
            if regime == "resolving":
                # Weight recent candles more heavily when resolving
                weights = np.exp(np.linspace(-2, 0, len(price_mids)))
                fair_value = float(np.average(price_mids, weights=weights))
            else:
                fair_value = float(np.mean(price_mids[-5:]))

            fair_value = max(1.0, min(99.0, fair_value))
            return {"fair_value": fair_value, "confidence": confidence, "regime": regime}

        except Exception as e:
            logger.debug(f"HMM predict error: {e}")
            return {"fair_value": fallback_value, "confidence": 0.33, "regime": "stable"}

    @staticmethod
    def _midpoint_fallback(df: pd.DataFrame) -> float:
        if df.empty:
            return 50.0
        for bid_col, ask_col in [
            ("yes_bid_close", "yes_ask_close"),
            ("yes_bid_open", "yes_ask_open"),
        ]:
            if bid_col in df.columns and ask_col in df.columns:
                last = df.iloc[-1]
                bid = last.get(bid_col)
                ask = last.get(ask_col)
                if pd.notna(bid) and pd.notna(ask):
                    return float((bid + ask) / 2)
        return 50.0

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({"model": self._model, "fitted": self._fitted}, path)
        logger.info(f"HMM saved → {path}")

    def load(self, path: str) -> None:
        import joblib
        data = joblib.load(path)
        self._model = data["model"]
        self._fitted = data["fitted"]
        logger.info(f"HMM loaded ← {path}")
