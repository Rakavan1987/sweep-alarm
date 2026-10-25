"""
Sweep-Alarm: erkennt Sweep-Reversal-Setups auf geschlossenen 5m-Kerzen
und schickt dir eine Telegram-Nachricht mit Coin, Richtung, Entry, SL, TP,
Positionsgroesse und Risiko. Er handelt NICHT selbst.

Aufruf:
    python alert_bot.py          normaler Lauf (alle paar Minuten per GitHub Actions)
    python alert_bot.py --test   sendet eine Testnachricht und prueft den Phemex-Zugriff

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
    for side in ("long", "short"):
        flag = "long_sweep" if side == "long" else "short_sweep"
        for s in range(max(SWING_LOOKBACK + 1, L - 2 - CONFIRM_WINDOW), L + 1):
            if not df.loc[s, flag]:
                continue
            extreme = float(df.loc[s, "low"] if side == "long" else df.loc[s, "high"])
            confirm = None
            for c in range(s, min(s + CONFIRM_WINDOW, L) + 1):
                if c > s:  # neues Extrem nach dem Sweep macht das Setup ungueltig
                    if side == "long" and df.loc[c, "low"] < extreme:
                        break
                    if side == "short" and df.loc[c, "high"] > extreme:
                        break
                ok = df.loc[c, "close"] > df.loc[c, "ema20"] if side == "long" \
                    else df.loc[c, "close"] < df.loc[c, "ema20"]
                slope = df.loc[c, "htf_slope"]
                trend_ok = np.isnan(slope) or (slope >= 0 if side == "long" else slope <= 0)
                if ok and trend_ok:
                    confirm = c
                    break
            if confirm is None or confirm < L - 2:
                continue

            c = confirm
            entry = float(df.loc[c, "close"])
            sl = extreme * (1 - SL_BUFFER_PCT / 100) if side == "long" else extreme * (1 + SL_BUFFER_PCT / 100)
            swing_level = df.loc[s, "swing_high"] if side == "long" else df.loc[s, "swing_low"]
            tp = compute_tp(entry, side, df.loc[c, "ema50"], df.loc[c, "ema200"], swing_level, sl)
            if tp is None:
                continue
            risk = abs(entry - sl)
            # Kurs darf seit der Bestaetigung nicht zu weit weggelaufen sein
            moved = (last_close - entry) if side == "long" else (entry - last_close)
            wrong_side = (last_close <= sl) if side == "long" else (last_close >= sl)
            if wrong_side or moved > CHASE_R * risk:
                continue

            sl_pct = risk / entry * 100
            risk_usd = CAPITAL * RISK_PCT / 100
            size = risk_usd / (sl_pct / 100)
            fee_usd = size * FEE_PCT / 100 * 2
            fee_share = fee_usd / risk_usd * 100
            rr = abs(tp - entry) / risk
            wick = float(df.loc[s, "lower_wick" if side == "long" else "upper_wick"])
            depth = float(df.loc[s, "sweep_depth_low" if side == "long" else "sweep_depth_high"])
            slope = df.loc[c, "htf_slope"]
            strict_trend = (not np.isnan(slope)) and (slope > 0 if side == "long" else slope < 0)
            score = int(wick >= 0.7) + int(depth >= 0.10) + int(strict_trend) \
                + int(rr >= 1.5) + int(fee_share <= 15)
            if score < MIN_SCORE or fee_share > MAX_FEE_SHARE:
                continue

            max_chase = entry + CHASE_R * risk if side == "long" else entry - CHASE_R * risk
            out.append({
                "symbol": symbol, "side": side, "confirm_ts": df.loc[c, "timestamp"],
                "entry": entry, "sl": sl, "tp": tp, "max_chase": max_chase,
                "sl_pct": sl_pct, "rr": rr, "size": size, "risk_usd": risk_usd,
                "fee_usd": fee_usd, "fee_share": fee_share, "score": score,
                "wick": wick, "depth": depth, "trend": "mit dem 15m-Trend" if strict_trend else "15m-Trend flach",
                "last_close": last_close,
            })
            break  # pro Richtung nur das erste gueltige Setup
    return out


# ----------------------------------------------------------------------
# NACHRICHT
# ----------------------------------------------------------------------
def price_fmt(x):
    if x >= 1000:
        return f"{x:,.2f}".replace(",", "'")
    if x >= 10:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.3f}"
    return f"{x:.5f}"


def build_message(s, btc_note):
    coin = s["symbol"].split("/")[0]
    arrow = "🟢 LONG" if s["side"] == "long" else "🔴 SHORT"
    lev_min = max(1, math.ceil(s["size"] / CAPITAL))
    lev_max = max(1, math.floor(50 / s["sl_pct"]))
    lev_txt = f"{lev_min}x bis {lev_max}x" if lev_min <= lev_max else "kein sinnvoller Hebel (SL zu eng)"
    age = (datetime.now(timezone.utc).replace(tzinfo=None) - s["confirm_ts"].to_pydatetime()).total_seconds() / 60 - TF_MIN
    stars = "★" * s["score"] + "☆" * (5 - s["score"])
    tp_gain = s["risk_usd"] * s["rr"]
    lines = [
        f"{arrow} {coin}USDT   Qualitaet {stars} ({s['score']}/5)",
        f"Kerze geschlossen vor ca. {max(0, round(age))} Min",
        "",
        f"Entry:  {price_fmt(s['entry'])}",
        f"SL:     {price_fmt(s['sl'])}  ({s['sl_pct']:.2f} %)",
        f"TP:     {price_fmt(s['tp'])}  (CRV 1:{s['rr']:.1f})",
        f"Nicht mehr einsteigen {'ueber' if s['side']=='long' else 'unter'} {price_fmt(s['max_chase'])}",
        "",
        f"Positionsgroesse: {s['size']:.0f} USDT",
        f"Risiko: {s['risk_usd']:.2f} USDT ({RISK_PCT} %)  |  Gewinn bei TP: ca. +{tp_gain:.2f} USDT",
        f"Hebel: {lev_txt}  |  Gebuehren: ca. {s['fee_usd']:.2f} USDT",
        "",
        f"Sweep-Lunte {s['wick']*100:.0f} %, Sweep-Tiefe {s['depth']:.2f} %, {s['trend']}",
    ]
    if btc_note:
        lines.append(btc_note)
    lines += ["", "Pruefe den Chart und die Checkliste. Bei Unsicherheit: Screenshot (5m + 15m) an Claude."]
    return "\n".join(lines)


def btc_context(exchange):
    try:
        df = add_htf_trend(add_indicators(fetch_5m(exchange, "BTC/USDT:USDT")))
        slope = df["htf_slope"].iloc[-1]
        if np.isnan(slope):
            return ""
        return "BTC 15m-Trend: " + ("steigend" if slope > 0 else "fallend" if slope < 0 else "flach")
    except Exception:
        return ""


def send_telegram(text):
    import requests
    token = os.environ["TELEGRAM_TOKEN"]
    chat = os.environ["TELEGRAM_CHAT_ID"]
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text}, timeout=20)
    r.raise_for_status()


# ----------------------------------------------------------------------
# STATE (damit dasselbe Signal nicht zweimal gemeldet wird)
# ----------------------------------------------------------------------
def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    cutoff = (datetime.now(timezone.utc) - pd.Timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S")
    state = {k: v for k, v in state.items() if v >= cutoff}
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=1, sort_keys=True)


# ----------------------------------------------------------------------
# HAUPTPROGRAMM
# ----------------------------------------------------------------------
def make_exchange():
    import ccxt
    return ccxt.phemex({"enableRateLimit": True})


def run(exchange=None, send=send_telegram):
    exchange = exchange or make_exchange()
    state = load_state()
    btc_note = None
    errors, sent = 0, 0
    for symbol in COINS:
        try:
            df = add_htf_trend(add_indicators(fetch_5m(exchange, symbol)))
        except Exception as e:
            errors += 1
            print(f"{symbol}: Fehler beim Laden: {e}")
            continue
        for s in find_setups(df, symbol):
            key = f"{s['symbol']}|{s['side']}|{s['confirm_ts']}"
            age_min = (datetime.now(timezone.utc).replace(tzinfo=None)
                       - s["confirm_ts"].to_pydatetime()).total_seconds() / 60 - TF_MIN
            if key in state or age_min > MAX_AGE_MIN:
                continue
            if btc_note is None:
                btc_note = btc_context(exchange)
            send(build_message(s, btc_note))
            state[key] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            sent += 1
            print(f"{symbol}: Signal {s['side']} gesendet")
    save_state(state)
    print(f"Fertig. Gesendet: {sent}, Ladefehler: {errors}/{len(COINS)}")
    return sent, errors


def loop(minutes, pause_s=60):
    """Prueft eine Weile lang jede Minute. So kommt das Signal kurz nach dem
    Kerzenschluss, auch wenn GitHub den Start des Laufs verzoegert."""
    exchange = make_exchange()
    end = time.time() + minutes * 60
    n = 0
    while True:
        n += 1
        print(f"--- Durchgang {n} ({datetime.now(timezone.utc):%H:%M:%S} UTC) ---")
        try:
            run(exchange)
        except Exception as e:  # ein Fehler darf die Schleife nicht beenden
            print(f"Fehler im Durchgang: {e}")
        if time.time() + pause_s >= end:
            break
        time.sleep(pause_s)


def test():
    ex = make_exchange()
    df = fetch_5m(ex, "BTC/USDT:USDT")
    last = df.iloc[-1]
    msg = (f"✅ Sweep-Alarm verbunden.\nBTC letzte geschlossene 5m-Kerze: {last['timestamp']} UTC, "
           f"Close {price_fmt(float(last['close']))}.\nBeobachtete Coins: "
           + ", ".join(c.split('/')[0] for c in COINS))
    send_telegram(msg)
    print(msg)


if __name__ == "__main__":
    if "--test" in sys.argv:
        test()
    elif "--once" in sys.argv:
        sent, errors = run()
        if errors == len(COINS):
            sys.exit(1)  # alles fehlgeschlagen -> GitHub zeigt rotes X
    elif "--loop" in sys.argv:
        loop(float(sys.argv[sys.argv.index("--loop") + 1]))
    else:
        loop(28)  # Standard: 28 Minuten lang jede Minute pruefen
