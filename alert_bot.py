"""
Sweep-Alarm: erkennt Sweep-Reversal-Setups auf geschlossenen 5m-Kerzen
und schickt dir eine Telegram-Nachricht mit Coin, Richtung, Entry, SL, TP,
Positionsgroesse und Risiko. Er handelt NICHT selbst.

Aufruf:
    python alert_bot.py          normaler Lauf (alle paar Minuten per GitHub Actions)
    python alert_bot.py --test   sendet eine Testnachricht und prueft den Phemex-Zugriff
    python alert_bot.py --loop 31   prueft 31 Minuten lang jede Minute

Zugangsdaten kommen aus Umgebungsvariablen (GitHub Secrets):
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""

import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# EINSTELLUNGEN (hier anpassen)
# ----------------------------------------------------------------------
COINS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT",
    "ADA/USDT:USDT", "LTC/USDT:USDT", "DOT/USDT:USDT", "NEAR/USDT:USDT",
    "SUI/USDT:USDT", "TRX/USDT:USDT", "UNI/USDT:USDT", "ATOM/USDT:USDT",
]
CAPITAL = 400.0          # dein Kapital in USDT
RISK_PCT = 0.5           # Risiko je Trade in % (erste 30 Trades: 0.5)
FEE_PCT = 0.06           # Gebuehr je Seite in %

SWING_LOOKBACK = 15      # letzte Hoch/Tief aus den letzten 15 Kerzen (5m)
SWEEP_MIN_PCT = 0.05     # Sweep mindestens 0.05 % ueber/unter dem Swing
WICK_MIN = 0.6           # Lunte mindestens 60 % der Kerzenspanne
SL_BUFFER_PCT = 0.05     # SL-Puffer hinter dem Sweep-Extrem
CONFIRM_WINDOW = 3       # Schluss ueber/unter EMA20 darf bis zu 3 Kerzen nach dem Sweep kommen
MIN_RR = 1.0             # Ziel muss mindestens 1R entfernt sein
MIN_SCORE = 3            # Qualitaet mindestens 3 von 5, sonst keine Nachricht
MAX_FEE_SHARE = 30.0     # Gebuehren duerfen hoechstens 30 % des Risikos sein
CHASE_R = 0.25           # nicht mehr einsteigen, wenn der Kurs mehr als 0.25R weggelaufen ist
MAX_AGE_MIN = 12         # Signal nur melden, wenn die Bestaetigungskerze nicht aelter ist

STATE_FILE = "state.json"
TF_MIN = 5


# ----------------------------------------------------------------------
# DATEN
# ----------------------------------------------------------------------
def fetch_5m(exchange, symbol, limit=500):
    rows = exchange.fetch_ohlcv(symbol, "5m", limit=limit)
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    # unfertige (laufende) Kerze entfernen
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    df = df[df["timestamp"] + pd.Timedelta(minutes=TF_MIN) <= now].reset_index(drop=True)
    return df


def add_indicators(df):
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["swing_high"] = df["high"].shift(1).rolling(SWING_LOOKBACK).max()
    df["swing_low"] = df["low"].shift(1).rolling(SWING_LOOKBACK).min()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
    df["sweep_depth_low"] = (df["swing_low"] - df["low"]) / df["swing_low"] * 100
    df["sweep_depth_high"] = (df["high"] - df["swing_high"]) / df["swing_high"] * 100
    df["long_sweep"] = (df["low"] < df["swing_low"]) & (df["sweep_depth_low"] >= SWEEP_MIN_PCT) \
        & (df["lower_wick"] >= WICK_MIN)
    df["short_sweep"] = (df["high"] > df["swing_high"]) & (df["sweep_depth_high"] >= SWEEP_MIN_PCT) \
        & (df["upper_wick"] >= WICK_MIN)
    return df


def add_htf_trend(df):
    """15m-EMA50 und deren Steigung, nur mit abgeschlossenen 15m-Kerzen."""
    df = df.copy()
    htf = (df.set_index("timestamp").resample("15min")
           .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna())
    htf["ema50"] = htf["close"].ewm(span=50, adjust=False).mean()
    htf["slope"] = htf["ema50"] - htf["ema50"].shift(3)
    known = htf[["slope"]].shift(1)
    df["htf_slope"] = known.reindex(df["timestamp"], method="ffill")["slope"].values
    return df


# ----------------------------------------------------------------------
# SETUP-ERKENNUNG
# ----------------------------------------------------------------------
def compute_tp(entry, side, ema50, ema200, swing_level, sl):
    risk = abs(entry - sl)
    if risk == 0:
        return None
    for level in (ema50, ema200, swing_level):
        if level is None or np.isnan(level):
            continue
        if side == "long" and level > entry and (level - entry) >= MIN_RR * risk:
            return float(level)
        if side == "short" and level < entry and (entry - level) >= MIN_RR * risk:
            return float(level)
    return None


def find_setups(df, symbol):
    """Sucht Setups, die in den letzten 3 geschlossenen Kerzen bestaetigt wurden."""
    out = []
    L = len(df) - 1
    if L < 220:
        return out
    last_close = float(df.loc[L, "close"])
    for
