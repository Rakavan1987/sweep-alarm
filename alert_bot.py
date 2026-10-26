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
MAX_AGE_MIN = 4          # Signal nur melden, wenn die Bestaetigungskerze nicht aelter ist

SWING15 = 10             # 15m-Sweep: letztes Hoch/Tief aus den letzten 10 Kerzen (15m)
SWEEP15_MIN_PCT = 0.10   # 15m-Sweep mindestens 0.10 % ueber/unter dem Swing

SHADOW_WICKS = (0.3, 0.4, 0.5)   # Schatten-Varianten mit kuerzerer Lunte (nur Auswertung, keine Nachricht)
SHADOW_MAX_AGE_MIN = 12          # Schatten-Trade nur aufnehmen, wenn die Bestaetigung nicht aelter ist
SHADOW_KEEP_DAYS = 14
SUMMARY_EVERY_H = 22             # Schatten-Bilanz hoechstens alle 22 Stunden per Telegram

STATE_FILE = "state.json"
TF_MIN = 5

# Zaehler: warum wurde ein Sweep NICHT gemeldet (jeder Fall nur einmal gezaehlt)
STATS = set()


_quiet = False  # waehrend der Schatten-Suche keine Statistik schreiben


def note(symbol, side, ts, reason):
    if _quiet:
        return
    STATS.add((symbol, side, str(ts), reason))


def print_stats():
    from collections import Counter
    c = Counter(r for (_, _, _, r) in STATS)
    print("=== Filter-Statistik dieses Laufs ===")
    if not c:
        print("keine Sweep-Kandidaten gesehen")
    for reason, n in sorted(c.items(), key=lambda x: -x[1]):
        print(f"{n:4d}  {reason}")
    # Einzelfaelle (UTC), damit man einen bestimmten Sweep nachschlagen kann
    rows = sorted(STATS, key=lambda x: (x[2], x[0]))
    print("--- Einzelfaelle (Zeit in UTC) ---")
    for sym, side, ts, reason in rows[:120]:
        print(f"{ts[5:16]}  {sym.split('/')[0]:<6} {side:<5} {reason}")
    STATS.clear()


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
    df["raw_long"] = (df["low"] < df["swing_low"]) & (df["sweep_depth_low"] >= SWEEP_MIN_PCT)
    df["raw_short"] = (df["high"] > df["swing_high"]) & (df["sweep_depth_high"] >= SWEEP_MIN_PCT)
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
            sts = df.loc[s, "timestamp"]
            if not df.loc[s, flag]:
                if df.loc[s, "raw_long" if side == "long" else "raw_short"]:
                    note(symbol, side, sts, "Sweep ohne Lunte (Lunte < 60 %)")
                continue
            note(symbol, side, sts, "Sweep mit Lunte gefunden")
            extreme = float(df.loc[s, "low"] if side == "long" else df.loc[s, "high"])
            confirm = None
            trend_blocked = False
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
                if ok and not trend_ok:
                    trend_blocked = True
                if ok and trend_ok:
                    confirm = c
                    break
            if confirm is None:
                note(symbol, side, sts, "15m-Trend blockt" if trend_blocked
                     else "kein Schluss ueber/unter EMA20 in 3 Kerzen")
                continue
            if confirm < L - 2:
                note(symbol, side, sts, "Bestaetigung schon zu alt")
                continue

            c = confirm
            entry = float(df.loc[c, "close"])
            sl = extreme * (1 - SL_BUFFER_PCT / 100) if side == "long" else extreme * (1 + SL_BUFFER_PCT / 100)
            swing_level = df.loc[s, "swing_high"] if side == "long" else df.loc[s, "swing_low"]
            tp = compute_tp(entry, side, df.loc[c, "ema50"], df.loc[c, "ema200"], swing_level, sl)
            if tp is None:
                note(symbol, side, sts, "kein Ziel mit CRV >= 1")
                continue
            risk = abs(entry - sl)
            # Kurs darf seit der Bestaetigung nicht zu weit weggelaufen sein
            moved = (last_close - entry) if side == "long" else (entry - last_close)
            wrong_side = (last_close <= sl) if side == "long" else (last_close >= sl)
            if wrong_side or moved > CHASE_R * risk:
                note(symbol, side, sts, "Kurs schon zu weit weggelaufen")
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
            if fee_share > MAX_FEE_SHARE:
                note(symbol, side, sts, "Gebuehren zu hoch (SL zu eng)")
                continue
            if score < MIN_SCORE:
                note(symbol, side, sts, "Qualitaet unter 3 von 5")
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


def add_15m_sweeps(df):
    """15m-Kerzen aus den 5m-Kerzen bauen (nur vollstaendige) und Sweeps markieren."""
    d = df.set_index("timestamp")
    h = d.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    h["n"] = d["close"].resample("15min").count()
    h = h[h["n"] == 3].copy()
    h["swing_low"] = h["low"].shift(1).rolling(SWING15).min()
    h["swing_high"] = h["high"].shift(1).rolling(SWING15).max()
    rng = (h["high"] - h["low"]).replace(0, np.nan)
    h["lower_wick"] = (h[["open", "close"]].min(axis=1) - h["low"]) / rng
    h["upper_wick"] = (h["high"] - h[["open", "close"]].max(axis=1)) / rng
    h["depth_low"] = (h["swing_low"] - h["low"]) / h["swing_low"] * 100
    h["depth_high"] = (h["high"] - h["swing_high"]) / h["swing_high"] * 100
    h["long_sweep"] = (h["low"] < h["swing_low"]) & (h["depth_low"] >= SWEEP15_MIN_PCT) \
        & (h["lower_wick"] >= WICK_MIN)
    h["short_sweep"] = (h["high"] > h["swing_high"]) & (h["depth_high"] >= SWEEP15_MIN_PCT) \
        & (h["upper_wick"] >= WICK_MIN)
    return h


def find_setups_15m(df, symbol):
    """EXPERIMENTELL: Sweep auf der 15m-Kerze, Bestaetigung auf 5m (Schluss ueber/unter EMA20).
    Kein 15m-Trendfilter. Wird in der Nachricht als 'neu, ungetestet' markiert."""
    out = []
    L = len(df) - 1
    if L < 220:
        return out
    h = add_15m_sweeps(df)
    if len(h) < SWING15 + 2:
        return out
    idx = {ts: i for i, ts in enumerate(df["timestamp"])}
    last_close = float(df.loc[L, "close"])
    for side in ("long", "short"):
        flag = "long_sweep" if side == "long" else "short_sweep"
        for bs in h.index[-4:]:
            row = h.loc[bs]
            if not bool(row[flag]):
                continue
            end = idx.get(bs + pd.Timedelta(minutes=10))  # letzte 5m-Kerze dieser 15m-Kerze
            if end is None:
                continue
            extreme = float(row["low"] if side == "long" else row["high"])
            note(symbol, side, bs, "15m-Sweep mit Lunte gefunden")
            confirm = None
            for c in range(end, min(end + CONFIRM_WINDOW, L) + 1):
                if c > end:
                    if side == "long" and df.loc[c, "low"] < extreme:
                        break
                    if side == "short" and df.loc[c, "high"] > extreme:
                        break
                ok = df.loc[c, "close"] > df.loc[c, "ema20"] if side == "long" \
                    else df.loc[c, "close"] < df.loc[c, "ema20"]
                if ok:
                    confirm = c
                    break
            if confirm is None:
                note(symbol, side, bs, "15m: kein 5m-Schluss ueber/unter EMA20")
                continue
            if confirm < L - 2:
                note(symbol, side, bs, "15m: Bestaetigung schon zu alt")
                continue
            c = confirm
            entry = float(df.loc[c, "close"])
            sl = extreme * (1 - SL_BUFFER_PCT / 100) if side == "long" else extreme * (1 + SL_BUFFER_PCT / 100)
            swing_level = row["swing_high"] if side == "long" else row["swing_low"]
            tp = compute_tp(entry, side, df.loc[c, "ema50"], df.loc[c, "ema200"], swing_level, sl)
            if tp is None:
                note(symbol, side, bs, "15m: kein Ziel mit CRV >= 1")
                continue
            risk = abs(entry - sl)
            moved = (last_close - entry) if side == "long" else (entry - last_close)
            wrong_side = (last_close <= sl) if side == "long" else (last_close >= sl)
            if wrong_side or moved > CHASE_R * risk:
                note(symbol, side, bs, "15m: Kurs schon zu weit weggelaufen")
                continue
            sl_pct = risk / entry * 100
            risk_usd = CAPITAL * RISK_PCT / 100
            size = risk_usd / (sl_pct / 100)
            fee_usd = size * FEE_PCT / 100 * 2
            fee_share = fee_usd / risk_usd * 100
            rr = abs(tp - entry) / risk
            wick = float(row["lower_wick" if side == "long" else "upper_wick"])
            depth = float(row["depth_low" if side == "long" else "depth_high"])
            slope = df.loc[c, "htf_slope"]
            strict_trend = (not np.isnan(slope)) and (slope > 0 if side == "long" else slope < 0)
            score = int(wick >= 0.7) + int(depth >= 0.20) + int(strict_trend) \
                + int(rr >= 1.5) + int(fee_share <= 15)
            if fee_share > MAX_FEE_SHARE:
                note(symbol, side, bs, "15m: Gebuehren zu hoch (SL zu eng)")
                continue
            if score < MIN_SCORE:
                note(symbol, side, bs, "15m: Qualitaet unter 3 von 5")
                continue
            max_chase = entry + CHASE_R * risk if side == "long" else entry - CHASE_R * risk
            out.append({
                "symbol": symbol, "side": side, "confirm_ts": df.loc[c, "timestamp"], "kind": "15m",
                "entry": entry, "sl": sl, "tp": tp, "max_chase": max_chase,
                "sl_pct": sl_pct, "rr": rr, "size": size, "risk_usd": risk_usd,
                "fee_usd": fee_usd, "fee_share": fee_share, "score": score,
                "wick": wick, "depth": depth,
                "trend": "mit dem 15m-Trend" if strict_trend else "gegen/neutral zum 15m-Trend",
                "last_close": last_close,
            })
            break
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
        f"{'🧪 15m-SWEEP (neu, ungetestet)' + chr(10) if s.get('kind') == '15m' else ''}"
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
    now = datetime.now(timezone.utc)
    cutoff = (now - pd.Timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S")
    cutoff_sh = (now - pd.Timedelta(days=SHADOW_KEEP_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    out = {k: v for k, v in state.items()
           if not k.startswith("__") and isinstance(v, str) and v >= cutoff}
    if isinstance(state.get("__summary__"), str):
        out["__summary__"] = state["__summary__"]
    shadow = state.get("__shadow__") or {}
    out["__shadow__"] = {k: r for k, r in shadow.items() if r.get("ts", "") >= cutoff_sh}
    with open(STATE_FILE, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)


# ----------------------------------------------------------------------
# SCHATTEN-TRACKER: Varianten still mitlaufen lassen und selbst auswerten
# ----------------------------------------------------------------------
def shadow_update(shadow, raw, df, symbol, live_setups):
    """Neue Kandidaten aller Varianten merken und offene anhand der Kerzen auswerten."""
    global _quiet, WICK_MIN
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cands = [("15m" if x.get("kind") == "15m" else "w60", x) for x in live_setups]
    _quiet = True
    try:
        for w in SHADOW_WICKS:
            old = WICK_MIN
            WICK_MIN = w
            try:
                dfw = add_htf_trend(add_indicators(raw))
            finally:
                WICK_MIN = old
            for x in find_setups(dfw, symbol):
                cands.append((f"w{int(round(w * 100))}", x))
    finally:
        _quiet = False
    for variant, x in cands:
        age = (now - x["confirm_ts"].to_pydatetime()).total_seconds() / 60 - TF_MIN
        sid = f"{variant}|{symbol}|{x['side']}|{x['confirm_ts']}"
        if age > SHADOW_MAX_AGE_MIN or sid in shadow:
            continue
        shadow[sid] = {"v": variant, "sym": symbol, "side": x["side"], "ts": str(x["confirm_ts"]),
                       "entry": float(x["entry"]), "sl": float(x["sl"]), "tp": float(x["tp"]),
                       "rr": float(x["rr"]), "fee_r": float(x["fee_share"]) / 100.0,
                       "res": None, "r": 0.0}
    # offene Trades dieses Coins auswerten (bei gleichzeitigem Treffer gilt SL)
    for sid, r in shadow.items():
        if r["res"] or r["sym"] != symbol:
            continue
        after = df[df["timestamp"] > pd.Timestamp(r["ts"])]
        for _, c in after.iterrows():
            if r["side"] == "long":
                hit_sl, hit_tp = c["low"] <= r["sl"], c["high"] >= r["tp"]
            else:
                hit_sl, hit_tp = c["high"] >= r["sl"], c["low"] <= r["tp"]
            if hit_sl:
                r["res"], r["r"] = "SL", -1.0 - r["fee_r"]
                break
            if hit_tp:
                r["res"], r["r"] = "TP", r["rr"] - r["fee_r"]
                break
        if not r["res"]:
            if (now - pd.Timestamp(r["ts"]).to_pydatetime()).total_seconds() > 24 * 3600:
                r["res"], r["r"] = "EXP", 0.0


def shadow_table(shadow):
    labels = [("w30", "Lunte>=30%"), ("w40", "Lunte>=40%"), ("w50", "Lunte>=50%"),
              ("w60", "Lunte>=60% (live)"), ("15m", "15m-Sweep (live)")]
    lines = []
    for key, label in labels:
        items = [r for r in shadow.values() if r["v"] == key]
        tp = sum(r["res"] == "TP" for r in items)
        sl = sum(r["res"] == "SL" for r in items)
        offen = sum(r["res"] is None for r in items)
        done = tp + sl
        wr = f"{tp / done * 100:.0f}%" if done else "-"
        total = sum(r["r"] for r in items if r["res"] in ("TP", "SL"))
        lines.append(f"{label:<19} Trades {len(items):3d} | TP {tp:3d} SL {sl:3d} offen {offen:2d} | "
                     f"Treffer {wr:>4} | netto {total:+.1f} R")
    return "\n".join(lines)


def print_shadow(shadow):
    print("=== Schatten-Bilanz (nur Auswertung, nicht gehandelt; R = Vielfaches des Risikos, nach Gebuehren) ===")
    print(shadow_table(shadow))


def maybe_send_summary(send=send_telegram):
    st = load_state()
    shadow = st.get("__shadow__") or {}
    if not any(r.get("res") in ("TP", "SL") for r in shadow.values()):
        return
    now = datetime.now(timezone.utc)
    last = st.get("__summary__")
    if isinstance(last, str) and last >= (now - pd.Timedelta(hours=SUMMARY_EVERY_H)).strftime("%Y-%m-%d %H:%M:%S"):
        return
    try:
        send("📊 Schatten-Bilanz (nicht gehandelte Varianten, netto nach Gebuehren)\n" + shadow_table(shadow)
             + "\n\nR = Vielfaches des Risikos. Wenig Trades = noch nicht aussagekraeftig.")
        st["__summary__"] = now.strftime("%Y-%m-%d %H:%M:%S")
        save_state(st)
    except Exception as e:
        print(f"Bilanz-Nachricht fehlgeschlagen: {e}")


# ----------------------------------------------------------------------
# HAUPTPROGRAMM
# ----------------------------------------------------------------------
def make_exchange():
    import ccxt
    return ccxt.phemex({"enableRateLimit": True})


def run(exchange=None, send=send_telegram):
    exchange = exchange or make_exchange()
    state = load_state()
    shadow = state.get("__shadow__") or {}
    btc_note = None
    errors, sent = 0, 0
    for symbol in COINS:
        try:
            raw = fetch_5m(exchange, symbol)
            df = add_htf_trend(add_indicators(raw))
        except Exception as e:
            errors += 1
            print(f"{symbol}: Fehler beim Laden: {e}")
            continue
        setups = find_setups(df, symbol)
        try:  # die 15m-Suche darf die normale Suche nie stoeren
            seen = {(x["side"], x["confirm_ts"]) for x in setups}
            setups += [x for x in find_setups_15m(df, symbol) if (x["side"], x["confirm_ts"]) not in seen]
        except Exception as e:
            print(f"{symbol}: Fehler bei der 15m-Suche: {e}")
        try:  # der Schatten-Tracker darf den Alarm nie stoeren
            shadow_update(shadow, raw, df, symbol, setups)
        except Exception as e:
            print(f"{symbol}: Fehler im Schatten-Tracker: {e}")
        for s in setups:
            key = f"{s['symbol']}|{s['side']}|{s['confirm_ts']}" + ("|15m" if s.get("kind") == "15m" else "")
            age_min = (datetime.now(timezone.utc).replace(tzinfo=None)
                       - s["confirm_ts"].to_pydatetime()).total_seconds() / 60 - TF_MIN
            if key in state:
                continue
            if age_min > MAX_AGE_MIN:
                note(symbol, s["side"], s["confirm_ts"], "Signal schon zu alt")
                continue
            note(symbol, s["side"], s["confirm_ts"], "SIGNAL gesendet")
            if btc_note is None:
                btc_note = btc_context(exchange)
            send(build_message(s, btc_note))
            state[key] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            sent += 1
            print(f"{symbol}: Signal {s['side']} gesendet")
    state["__shadow__"] = shadow
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
    print_stats()
    try:
        print_shadow(load_state().get("__shadow__") or {})
        maybe_send_summary()
    except Exception as e:
        print(f"Bilanz-Fehler: {e}")


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
        print_stats()
        if errors == len(COINS):
            sys.exit(1)  # alles fehlgeschlagen -> GitHub zeigt rotes X
    elif "--loop" in sys.argv:
        loop(float(sys.argv[sys.argv.index("--loop") + 1]))
    else:
        loop(28)  # Standard: 28 Minuten lang jede Minute pruefen
