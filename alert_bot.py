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
    # neu dazu (was es auf Phemex nicht gibt, wird beim Start automatisch uebersprungen)
    "XLM/USDT:USDT", "BCH/USDT:USDT", "ETC/USDT:USDT", "AAVE/USDT:USDT", "HBAR/USDT:USDT",
    "FIL/USDT:USDT", "ARB/USDT:USDT", "INJ/USDT:USDT",
    "WLD/USDT:USDT", "ONDO/USDT:USDT", "ENA/USDT:USDT",
]
CAPITAL = 400.0          # dein Kapital in USDT
RISK_PCT = 0.5           # Risiko je Trade in % (erste 30 Trades: 0.5)
FEE_PCT = 0.06           # Gebuehr je Seite in %

SWING_LOOKBACK = 15      # letzte Hoch/Tief aus den letzten 15 Kerzen (5m)
SWEEP_MIN_PCT = 0.05     # Sweep mindestens 0.05 % ueber/unter dem Swing
WICK_MIN = 0.4           # Lunte mindestens 40 % -> Setup wird gemeldet
A_WICK = 0.6             # ab 60 % Lunte = A-Setup (volles Risiko), 40-60 % = B-Setup (halbes Risiko)
B_RISK_FACTOR = 0.5      # B-Setups: nur halbes Risiko
SL_BUFFER_PCT = 0.05     # SL-Puffer hinter dem Sweep-Extrem
CONFIRM_WINDOW = 3       # Schluss ueber/unter EMA20 darf bis zu 3 Kerzen nach dem Sweep kommen
MIN_RR = 1.0             # Ziel muss mindestens 1R entfernt sein
TREND_FILTER = True       # 15m-Trendfilter an (Schatten-Variante schaltet ihn aus)
MIN_SCORE = 3            # Qualitaet mindestens 3 von 5, sonst keine Nachricht
MAX_FEE_SHARE = 30.0     # Gebuehren duerfen hoechstens 30 % des Risikos sein
CHASE_R = 0.25           # nicht mehr einsteigen, wenn der Kurs mehr als 0.25R weggelaufen ist
MAX_AGE_MIN = 4          # Signal nur melden, wenn die Bestaetigungskerze nicht aelter ist

SWING15 = 10             # 15m-Sweep: letztes Hoch/Tief aus den letzten 10 Kerzen (15m)
SWEEP15_MIN_PCT = 0.10   # 15m-Sweep mindestens 0.10 % ueber/unter dem Swing

# Schatten-Varianten (nur Auswertung, keine Nachricht): jede lockert genau EINE Regel
SHADOW_VARIANTS = [
    ("w30", {"WICK_MIN": 0.3}),
    ("w40", {"WICK_MIN": 0.4}),
    ("w50", {"WICK_MIN": 0.5}),
    ("notrend", {"TREND_FILTER": False}),
    ("conf5", {"CONFIRM_WINDOW": 5}),
    ("score2", {"MIN_SCORE": 2}),
]
WAVE_MIN = 4                     # ab 4 Alarmen in dieselbe Richtung gleichzeitig: Marktwelle = EINE Wette
WAVE_WINDOW_MIN = 20             # so lange gilt eine Welle als aktiv
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


PRIORITY = [  # je Fall zaehlt nur das aussagekraeftigste Ergebnis
    "SIGNAL gesendet", "Signal schon zu alt", "Kurs schon zu weit weggelaufen",
    "kein Ziel mit CRV >= 1", "Gebuehren zu hoch (SL zu eng)", "Qualitaet unter 3 von 5",
    "15m-Trend blockt", "kein Schluss", "Bestaetigung schon zu alt", "wartet auf Bestaetigung",
    "Sweep mit Lunte gefunden", "Sweep ohne Lunte",
]


def _clean(reason):
    kind = "15m" if (reason.startswith("15m-Sweep") or reason.startswith("15m:")) else "5m"
    r = reason[4:].strip() if reason.startswith("15m:") else reason
    if r.startswith("15m-Sweep mit Lunte"):
        r = "Sweep mit Lunte gefunden"
    if r.startswith("kein Schluss") or r.startswith("kein 5m-Schluss"):
        r = "kein Schluss"
    if r.startswith("Sweep ohne Lunte"):
        r = "Sweep ohne Lunte"
    return kind, r


def final_cases():
    cases = {}
    for sym, side, ts, reason in STATS:
        kind, r = _clean(reason)
        cases.setdefault((kind, sym, side, ts), set()).add(r)
    out = {}
    for key, rs in cases.items():
        out[key] = next((p for p in PRIORITY if p in rs), sorted(rs)[0])
    return out


def print_stats():
    from collections import Counter
    fin = final_cases()
    sent = [k for k, r in fin.items() if r == "SIGNAL gesendet"]
    fin = {k: ("alle Filter bestanden" if r == "Sweep mit Lunte gefunden" else r) for k, r in fin.items()
           if r not in ("SIGNAL gesendet", "Signal schon zu alt")}
    print("=== Filter-Statistik dieses Laufs (je Fall nur das Endergebnis) ===")
    print(f"Signale gesendet: {len(sent)}")
    if not fin:
        print("keine Sweep-Kandidaten gesehen")
    for kind, title in (("5m", "5m-Sweeps"), ("15m", "15m-Sweeps")):
        c = Counter(r for (k, _, _, _), r in fin.items() if k == kind)
        if not c:
            continue
        print(f"-- {title}: {sum(n for r, n in c.items() if r != 'Sweep ohne Lunte')} mit Lunte, "
              f"{c.get('Sweep ohne Lunte', 0)} ohne Lunte --")
        for reason, n in sorted(c.items(), key=lambda x: -x[1]):
            if reason != "Sweep ohne Lunte":
                print(f"{n:4d}  {reason}")
    # Einzelfaelle (UTC) der Sweeps MIT Lunte, die juengsten zuletzt
    rows = sorted(((ts, sym, side, kind, r) for (kind, sym, side, ts), r in fin.items()
                   if r != "Sweep ohne Lunte"))
    print("--- Einzelfaelle mit Lunte (Zeit in UTC, juengste 300) ---")
    for ts, sym, side, kind, r in rows[-300:]:
        print(f"{ts[5:16]}  {sym.split('/')[0]:<6} {side:<5} {'15m ' if kind == '15m' else '5m  '}{r}")
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
    for n in (30, 60):  # weiter entfernte Ziele, falls das naechste Ziel zu nah ist
        df[f"swing_high{n}"] = df["high"].shift(1).rolling(n).max()
        df[f"swing_low{n}"] = df["low"].shift(1).rolling(n).min()
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
def compute_tp(entry, side, ema50, ema200, swing_level, sl, far_levels=()):
    """Erstes Ziel in der Reihenfolge EMA50, EMA200, naechstes Hoch/Tief mit mindestens 1R.
    Reicht keins davon, werden weiter entfernte Hochs/Tiefs (30, 60 Kerzen) geprueft."""
    risk = abs(entry - sl)
    if risk == 0:
        return None
    for level in (ema50, ema200, swing_level, *far_levels):
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
                    note(symbol, side, sts, f"Sweep ohne Lunte (Lunte < {WICK_MIN*100:.0f} %)")
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
                trend_ok = (not TREND_FILTER) or np.isnan(slope) or (slope >= 0 if side == "long" else slope <= 0)
                if ok and not trend_ok:
                    trend_blocked = True
                if ok and trend_ok:
                    confirm = c
                    break
            if confirm is None:
                if trend_blocked:
                    note(symbol, side, sts, "15m-Trend blockt")
                elif s + CONFIRM_WINDOW > L:
                    note(symbol, side, sts, "wartet auf Bestaetigung")
                else:
                    note(symbol, side, sts, f"kein Schluss ueber/unter EMA20 in {CONFIRM_WINDOW} Kerzen")
                continue
            if confirm < L - 2:
                note(symbol, side, sts, "Bestaetigung schon zu alt")
                continue

            c = confirm
            entry = float(df.loc[c, "close"])
            sl = extreme * (1 - SL_BUFFER_PCT / 100) if side == "long" else extreme * (1 + SL_BUFFER_PCT / 100)
            swing_level = df.loc[s, "swing_high"] if side == "long" else df.loc[s, "swing_low"]
            far = [df.loc[s, f"swing_{'high' if side == 'long' else 'low'}{n}"] for n in (30, 60)]
            tp = compute_tp(entry, side, df.loc[c, "ema50"], df.loc[c, "ema200"], swing_level, sl, far)
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
            grade = "A" if wick >= A_WICK else "B"
            if grade == "B":
                risk_usd *= B_RISK_FACTOR
                size *= B_RISK_FACTOR
                fee_usd *= B_RISK_FACTOR
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
                "fee_usd": fee_usd, "fee_share": fee_share, "score": score, "grade": grade,
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
                if end + CONFIRM_WINDOW > L:
                    note(symbol, side, bs, "15m: wartet auf Bestaetigung")
                else:
                    note(symbol, side, bs, "15m: kein 5m-Schluss ueber/unter EMA20")
                continue
            if confirm < L - 2:
                note(symbol, side, bs, "15m: Bestaetigung schon zu alt")
                continue
            c = confirm
            entry = float(df.loc[c, "close"])
            sl = extreme * (1 - SL_BUFFER_PCT / 100) if side == "long" else extreme * (1 + SL_BUFFER_PCT / 100)
            swing_level = row["swing_high"] if side == "long" else row["swing_low"]
            far = [df.loc[end, f"swing_{'high' if side == 'long' else 'low'}{n}"] for n in (30, 60)]
            tp = compute_tp(entry, side, df.loc[c, "ema50"], df.loc[c, "ema200"], swing_level, sl, far)
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
            grade = "A" if wick >= A_WICK else "B"
            if grade == "B":
                risk_usd *= B_RISK_FACTOR
                size *= B_RISK_FACTOR
                fee_usd *= B_RISK_FACTOR
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
                "fee_usd": fee_usd, "fee_share": fee_share, "score": score, "grade": grade,
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


def build_message(s, btc_note, wave_note=None):
    coin = s["symbol"].split("/")[0]
    arrow = "🟢 LONG" if s["side"] == "long" else "🔴 SHORT"
    lev_min = max(1, math.ceil(s["size"] / CAPITAL))
    lev_max = max(1, math.floor(50 / s["sl_pct"]))
    lev_txt = f"{lev_min}x bis {lev_max}x" if lev_min <= lev_max else "kein sinnvoller Hebel (SL zu eng)"
    age = (datetime.now(timezone.utc).replace(tzinfo=None) - s["confirm_ts"].to_pydatetime()).total_seconds() / 60 - TF_MIN
    stars = "★" * s["score"] + "☆" * (5 - s["score"])
    tp_gain = s["risk_usd"] * s["rr"]
    lines = [
        f"💰 ORDER: {s['size']:.0f} USDT (nicht mehr!)",
        (wave_note + chr(10)) if wave_note else "",
        f"{'🧪 15m-SWEEP (neu, ungetestet)' + chr(10) if s.get('kind') == '15m' else ''}"
        f"{'⚠️ B-SETUP: Lunte nur ' + format(s['wick']*100, '.0f') + ' % - nur HALBES Risiko, Vorsicht' + chr(10) if s.get('grade') == 'B' else ''}"
        f"{arrow} {coin}USDT   Qualitaet {stars} ({s['score']}/5)",
        f"Kerze geschlossen vor ca. {max(0, round(age))} Min",
        "",
        f"Entry:  {price_fmt(s['entry'])}",
        f"SL:     {price_fmt(s['sl'])}  ({s['sl_pct']:.2f} %)",
        f"TP:     {price_fmt(s['tp'])}  (CRV 1:{s['rr']:.1f})",
        f"Nicht mehr einsteigen {'ueber' if s['side']=='long' else 'unter'} {price_fmt(s['max_chase'])}",
        "",
        f"Positionsgroesse: {s['size']:.0f} USDT",
        f"Risiko: {s['risk_usd']:.2f} USDT ({s['risk_usd']/CAPITAL*100:.2f} %)  |  Gewinn bei TP: ca. +{tp_gain:.2f} USDT",
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
    if isinstance(state.get("__wave__"), dict):
        out["__wave__"] = state["__wave__"]
    if isinstance(state.get("__recent__"), list):
        out["__recent__"] = state["__recent__"]
    with open(STATE_FILE, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)


# ----------------------------------------------------------------------
# SCHATTEN-TRACKER: Varianten still mitlaufen lassen und selbst auswerten
# ----------------------------------------------------------------------
_BTC = None   # BTC-5m-Daten mit 15m-Trend (fuer die Schatten-Variante "mit/gegen BTC")


def btc_aligned(side, ts):
    """True = Setup laeuft mit der BTC-15m-Richtung, False = dagegen, None = unbekannt/flach."""
    if _BTC is None:
        return None
    try:
        slope = _BTC[_BTC["timestamp"] <= pd.Timestamp(ts)]["htf_slope"].iloc[-1]
        if np.isnan(slope) or slope == 0:
            return None
        return bool(slope > 0) if side == "long" else bool(slope < 0)
    except Exception:
        return None


def shadow_update(shadow, raw, df, symbol, live_setups):
    """Neue Kandidaten aller Varianten merken und offene anhand der Kerzen auswerten."""
    global _quiet
    g = globals()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cands = [("15m" if x.get("kind") == "15m" else "w60", x) for x in live_setups
             if x.get("kind") == "15m" or x.get("grade", "A") == "A"]
    for lab, x in list(cands):   # dieselben Live-Setups, getrennt nach BTC-Richtung
        al = btc_aligned(x["side"], x["confirm_ts"])
        if al is True:
            cands.append(("btc", x))
        elif al is False:
            cands.append(("vsbtc", x))
    _quiet = True
    try:
        for name, override in SHADOW_VARIANTS:
            old = {k: g[k] for k in override}
            g.update(override)
            try:
                dfw = add_htf_trend(add_indicators(raw))
                found = find_setups(dfw, symbol)
            finally:
                g.update(old)
            for x in found:
                cands.append((name, x))
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
    labels = [("w60", "nur A (Lunte>=60%)"), ("w40", "live A+B (>=40%)"), ("w30", "Lunte>=30%"),
              ("w50", "Lunte>=50%"), ("notrend", "ohne 15m-Trend"), ("conf5", "Bestaet. 5 Kerzen"),
              ("score2", "Qualitaet ab 2/5"), ("15m", "15m-Sweep (live)"),
              ("btc", "live MIT BTC"), ("vsbtc", "live GEGEN BTC")]
    lines = []
    for key, label in labels:
        items = [r for r in shadow.values() if r["v"] == key]
        tp = sum(r["res"] == "TP" for r in items)
        sl = sum(r["res"] == "SL" for r in items)
        offen = sum(r["res"] is None for r in items)
        done = tp + sl
        wr = f"{tp / done * 100:.0f}%" if done else "-"
        total = sum(r["r"] for r in items if r["res"] in ("TP", "SL"))
        lines.append(f"{label:<18} Trades {len(items):3d} | TP {tp:3d} SL {sl:3d} offen {offen:2d} | "
                     f"Treffer {wr:>4} | netto {total:+.1f} R")
    return "\n".join(lines) + shadow_hours(shadow)


def shadow_hours(shadow):
    """Live-Setups (5m A+B und 15m) nach Tageszeit in Schweizer Zeit: wann kommen gute Setups?"""
    try:
        from zoneinfo import ZoneInfo
        ch = ZoneInfo("Europe/Zurich")
    except Exception:
        return ""
    buckets = [("00-06 Nacht", 0, 6), ("06-09 Morgen", 6, 9), ("09-17 Arbeit", 9, 17),
               ("17-23 Abend", 17, 23), ("23-24 spaet", 23, 24)]
    rows = {b[0]: [0, 0, 0, 0.0] for b in buckets}   # Trades, TP, SL, R
    for r in shadow.values():
        if r.get("v") not in ("w40", "15m"):
            continue
        try:
            h = pd.Timestamp(r["ts"]).tz_localize("UTC").tz_convert(ch).hour
        except Exception:
            continue
        for name, a, b in buckets:
            if a <= h < b:
                row = rows[name]
                row[0] += 1
                if r["res"] == "TP":
                    row[1] += 1
                    row[3] += r["r"]
                elif r["res"] == "SL":
                    row[2] += 1
                    row[3] += r["r"]
                break
    lines = ["", "Nach Tageszeit (CH-Zeit, Live-Setups 5m A+B und 15m):"]
    for name, _, _ in buckets:
        t, tp, sl, rr = rows[name]
        lines.append(f"{name:<13} Trades {t:3d} | TP {tp:3d} SL {sl:3d} | netto {rr:+.1f} R")
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


_missing_printed = False


def usable_coins(exchange):
    """Nur Coins verwenden, die es auf Phemex wirklich gibt (der Rest wird uebersprungen)."""
    global _missing_printed
    try:
        markets = exchange.load_markets()
    except Exception:
        return list(COINS)
    ok = [c for c in COINS if c in markets]
    missing = [c for c in COINS if c not in markets]
    if missing and not _missing_printed:
        print("Nicht auf Phemex gefunden, uebersprungen:", ", ".join(c.split("/")[0] for c in missing))
        _missing_printed = True
    return ok or list(COINS)


_TS_FMT = "%Y-%m-%d %H:%M:%S"


def _minutes_since(ts_str, now):
    try:
        return (now - datetime.strptime(ts_str, _TS_FMT)).total_seconds() / 60
    except Exception:
        return 9999


def dispatch(queue, state, send, get_btc_note):
    """Schickt die neuen Alarme. Kommen viele in dieselbe Richtung gleichzeitig (Marktwelle),
    geht nur der beste als Alarm raus, der Rest wird als EINE Wette gebuendelt."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    nowstr = now.strftime(_TS_FMT)
    sent = 0
    recent = [r for r in (state.get("__recent__") or []) if _minutes_since(r[1], now) <= WAVE_WINDOW_MIN]
    wave = state.get("__wave__") if isinstance(state.get("__wave__"), dict) else None
    for side in ("long", "short"):
        grp = [(x, k) for x, k in queue if x["side"] == side]
        if not grp:
            continue
        label = "LONG" if side == "long" else "SHORT"
        coins = ", ".join(x["symbol"].split("/")[0] for x, _ in grp)
        active = bool(wave and wave.get("side") == side and _minutes_since(wave.get("start", ""), now) <= WAVE_WINDOW_MIN)
        n_recent = sum(1 for r in recent if r[0] == side)
        if active:
            for x, k in grp:
                note(x["symbol"], x["side"], x["confirm_ts"], "in Marktwelle gebuendelt")
                state[k] = nowstr
            send(f"🌊 Marktwelle {label} geht weiter: {len(grp)} weitere Alarme ({coins}). "
                 f"Gleiche Wette wie vorher, nicht zusaetzlich handeln.")
            sent += 1
        elif len(grp) + n_recent >= WAVE_MIN:
            best = max(grp, key=lambda sk: (sk[0]["score"], 1 if sk[0].get("grade") == "A" else 0, sk[0]["wick"]))
            total = len(grp) + n_recent
            wave_note = (f"🌊 MARKTWELLE: {total} Alarme {label} in kurzer Zeit (neu: {coins}). "
                         f"Das ist EINE Wette, nicht {total}. Hier der beste, handle hoechstens EINEN.")
            for x, k in grp:
                note(x["symbol"], x["side"], x["confirm_ts"],
                     "SIGNAL gesendet (Marktwelle)" if (x, k) == best else "in Marktwelle gebuendelt")
                state[k] = nowstr
            send(build_message(best[0], get_btc_note(), wave_note))
            wave = {"side": side, "start": nowstr}
            state["__wave__"] = wave
            sent += 1
        else:
            for x, k in grp:
                note(x["symbol"], x["side"], x["confirm_ts"], "SIGNAL gesendet")
                send(build_message(x, get_btc_note()))
                state[k] = nowstr
                recent.append([side, nowstr])
                sent += 1
                print(f"{x['symbol']}: Signal {x['side']} gesendet")
    state["__recent__"] = recent
    return sent


def run(exchange=None, send=send_telegram):
    exchange = exchange or make_exchange()
    state = load_state()
    shadow = state.get("__shadow__") or {}
    btc_note = None
    queue = []
    errors, sent = 0, 0
    global _BTC
    try:
        _BTC = add_htf_trend(add_indicators(fetch_5m(exchange, "BTC/USDT:USDT")))
    except Exception as e:
        _BTC = None
        print(f"BTC-Daten fuer die Schatten-Variante nicht geladen: {e}")
    coins = usable_coins(exchange)
    for symbol in coins:
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
            queue.append((s, key))
    if queue:
        def get_btc_note():
            nonlocal btc_note
            if btc_note is None:
                btc_note = btc_context(exchange)
            return btc_note
        sent += dispatch(queue, state, send, get_btc_note)
    state["__shadow__"] = shadow
    save_state(state)
    print(f"Fertig. Gesendet: {sent}, Ladefehler: {errors}/{len(coins)}")
    return sent, errors


def persist_state():
    """Im GitHub-Lauf den Stand zwischendurch committen, damit er bei einem Abbruch
    (z. B. Run abbrechen und neu starten) nicht verloren geht. Fehler sind egal."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    import subprocess
    try:
        cmds = [["git", "config", "user.name", "sweep-alarm"],
                ["git", "config", "user.email", "sweep-alarm@users.noreply.github.com"],
                ["git", "add", STATE_FILE]]
        for c in cmds:
            subprocess.run(c, check=True, capture_output=True, timeout=60)
        if subprocess.run(["git", "diff", "--cached", "--quiet", "--", STATE_FILE],
                          capture_output=True, timeout=60).returncode == 0:
            return  # nichts geaendert
        subprocess.run(["git", "commit", "-m", "state"], check=True, capture_output=True, timeout=60)
        branch = os.environ.get("GITHUB_REF_NAME") or "main"
        # Hat jemand zwischendurch etwas ins Repo hochgeladen (z. B. neue alert_bot.py),
        # wuerde ein einfacher push abgelehnt. Darum erst obendrauf setzen (state.json gewinnt).
        subprocess.run(["git", "pull", "--rebase", "--autostash", "-X", "theirs", "origin", branch],
                       check=True, capture_output=True, timeout=120)
        subprocess.run(["git", "push", "origin", f"HEAD:{branch}"], check=True, capture_output=True, timeout=120)
        print("Stand gespeichert (git push).")
    except Exception as e:
        err = getattr(e, "stderr", b"") or b""
        print(f"Stand speichern fehlgeschlagen: {e} {err.decode(errors='ignore')[:300]}")


PERSIST_EVERY_MIN = 20


def loop(minutes, pause_s=60):
    """Prueft eine Weile lang jede Minute. So kommt das Signal kurz nach dem
    Kerzenschluss, auch wenn GitHub den Start des Laufs verzoegert."""
    exchange = make_exchange()
    end = time.time() + minutes * 60
    n = 0
    last_persist = time.time()
    while True:
        n += 1
        print(f"--- Durchgang {n} ({datetime.now(timezone.utc):%H:%M:%S} UTC) ---")
        try:
            run(exchange)
        except Exception as e:  # ein Fehler darf die Schleife nicht beenden
            print(f"Fehler im Durchgang: {e}")
        if time.time() - last_persist >= PERSIST_EVERY_MIN * 60:
            persist_state()
            last_persist = time.time()
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
        if errors >= 10:
            sys.exit(1)  # alles fehlgeschlagen -> GitHub zeigt rotes X
    elif "--loop" in sys.argv:
        loop(float(sys.argv[sys.argv.index("--loop") + 1]))
    else:
        loop(28)  # Standard: 28 Minuten lang jede Minute pruefen
