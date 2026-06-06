#!/usr/bin/env python3
"""
30M FVG Zone Scanner + Signal Bot
- Detects new 30M FVGs automatically
- Scores each zone for probability
- Alerts when approaching zone
- Posts entry signal at 50% level
- Tracks TP1, TP2, WIN/LOSS automatically
- Posts to Telegram and/or Discord
"""

import os, time, logging, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ── CONFIG ────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")
DISCORD_WEBHOOK    = os.getenv("DISCORD_WEBHOOK",    "")
TWELVE_DATA_KEY    = os.getenv("TWELVE_DATA_KEY",    "08b6188ba9c34307b9091c7570124b1f")
SCAN_INTERVAL      = int(os.getenv("SCAN_INTERVAL",  "60"))   # scan every 60 seconds
APPROACH_DIST      = float(os.getenv("APPROACH_DIST","20"))   # alert when 20pts away
MIN_SCORE          = int(os.getenv("MIN_SCORE",      "4"))    # minimum zone score
MAX_DAILY          = int(os.getenv("MAX_DAILY",      "8"))    # max signals per day

SYMBOLS = ["XAU/USD", "XAG/USD", "BTC/USD"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── STATE ─────────────────────────────────────────────────────────
active_zones   = {}   # symbol -> list of active FVG zones
open_trade     = None # currently open trade
daily_signals  = 0
last_day       = None
approached     = {}   # track which zones we already sent approach alert for

# ── DATA ──────────────────────────────────────────────────────────
def fetch(symbol: str, interval: str, size: int = 100) -> pd.DataFrame:
    url = (f"https://api.twelvedata.com/time_series"
           f"?symbol={symbol}&interval={interval}&outputsize={size}&apikey={TWELVE_DATA_KEY}")
    r = requests.get(url, timeout=15)
    data = r.json()
    if "values" not in data:
        raise ValueError(f"API error: {data.get('message', data)}")
    df = pd.DataFrame(data["values"])
    for col in ["open","high","low","close"]:
        df[col] = df[col].astype(float)
    return df.iloc[::-1].reset_index(drop=True)

def live_price(symbol: str) -> float:
    url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVE_DATA_KEY}"
    return float(requests.get(url, timeout=10).json()["price"])

# ── SESSION ───────────────────────────────────────────────────────
def in_session() -> bool:
    h = datetime.now(timezone.utc).hour
    return (7 <= h < 16) or (12 <= h < 21)

# ── FVG DETECTION ─────────────────────────────────────────────────
def detect_fvgs(df: pd.DataFrame) -> list:
    fvgs = []
    for i in range(2, len(df)):
        # Bearish FVG: candle[i-2].low > candle[i].high — gap down
        if df.low.iloc[i-2] > df.high.iloc[i]:
            fvgs.append({
                "type":    "BEARISH",
                "top":     round(df.low.iloc[i-2], 2),
                "bot":     round(df.high.iloc[i], 2),
                "mid":     round((df.low.iloc[i-2] + df.high.iloc[i]) / 2, 2),
                "bar_idx": i,
                "candle":  df.iloc[i-1],  # the middle candle (impulse)
            })
        # Bullish FVG: candle[i-2].high < candle[i].low — gap up
        if df.high.iloc[i-2] < df.low.iloc[i]:
            fvgs.append({
                "type":    "BULLISH",
                "top":     round(df.low.iloc[i], 2),
                "bot":     round(df.high.iloc[i-2], 2),
                "mid":     round((df.low.iloc[i] + df.high.iloc[i-2]) / 2, 2),
                "bar_idx": i,
                "candle":  df.iloc[i-1],
            })
    return fvgs

# ── ZONE SCORING ──────────────────────────────────────────────────
def score_zone(fvg: dict, df_30m: pd.DataFrame, df_daily: pd.DataFrame, price: float) -> dict:
    score   = 0
    reasons = []

    # 1. Strong impulse candle created the FVG
    candle   = fvg["candle"]
    body     = abs(candle.close - candle.open)
    rng      = max(candle.high - candle.low, 0.01)
    avg_body = abs(df_30m.close - df_30m.open).mean()
    if body > avg_body * 1.5:
        score += 2
        reasons.append("Strong impulse")

    # 2. Aligns with previous day high/low
    if len(df_daily) >= 2:
        p_high = float(df_daily.high.iloc[-2])
        p_low  = float(df_daily.low.iloc[-2])
        buf    = 15
        if abs(fvg["mid"] - p_high) <= buf:
            score += 2
            reasons.append("Prev day high")
        if abs(fvg["mid"] - p_low) <= buf:
            score += 2
            reasons.append("Prev day low")

    # 3. Fibonacci confluence
    swing_h = df_30m.high.tail(20).max()
    swing_l = df_30m.low.tail(20).min()
    fib_rng = swing_h - swing_l
    fibs    = [0.382, 0.5, 0.618, 0.705, 0.786]
    for f in fibs:
        level = swing_h - fib_rng * f
        if abs(fvg["mid"] - level) <= 10:
            score += 2
            reasons.append(f"Fib {f}")
            break

    # 4. First touch — zone never been tested
    recent_prices = df_30m.close.tail(20)
    if fvg["type"] == "BEARISH":
        tested = any(p >= fvg["bot"] for p in recent_prices.iloc[:-3])
    else:
        tested = any(p <= fvg["top"] for p in recent_prices.iloc[:-3])
    if not tested:
        score += 2
        reasons.append("First touch")

    # 5. Zone size reasonable (not too big)
    zone_size = fvg["top"] - fvg["bot"]
    if zone_size < 30:
        score += 1
        reasons.append("Tight zone")

    return {**fvg, "score": score, "reasons": reasons}

# ── MESSAGING ─────────────────────────────────────────────────────
def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True}
    requests.post(url, data=data, timeout=15).raise_for_status()

def send_discord(content: str = None, embed: dict = None):
    if not DISCORD_WEBHOOK:
        return
    payload = {}
    if content:
        payload["content"] = content
    if embed:
        payload["embeds"] = [embed]
    requests.post(DISCORD_WEBHOOK, json=payload, timeout=15).raise_for_status()

def notify(text: str, embed: dict = None):
    try: send_telegram(text)
    except Exception as e: log.error(f"Telegram error: {e}")
    try: send_discord(content=text if not embed else None, embed=embed)
    except Exception as e: log.error(f"Discord error: {e}")

def zone_alert(symbol: str, zone: dict):
    sym   = symbol.replace("/", "")
    stars = "🔥" if zone["score"] >= 7 else "⭐" if zone["score"] >= 5 else "👀"
    typ   = zone["type"]
    emoji = "🔴" if typ == "BEARISH" else "🟢"

    text = (
        f"{stars} *NEW 30M FVG ZONE — {sym}*\n"
        f"\n"
        f"{emoji} *Type:* {typ}\n"
        f"📊 *Zone:* ${zone['bot']} — ${zone['top']}\n"
        f"🎯 *50% Level:* ${zone['mid']}\n"
        f"⭐ *Score:* {zone['score']}/9\n"
        f"✅ *Confluences:* {', '.join(zone['reasons']) if zone['reasons'] else 'Base FVG'}\n"
        f"\n"
        f"_Watch for price to return to this zone_\n"
        f"_Entry alert will fire at 50% level_"
    )
    notify(text)
    log.info(f"Zone alert: {sym} {typ} score={zone['score']}")

def approach_alert(symbol: str, zone: dict, price: float, dist: float):
    sym   = symbol.replace("/", "")
    typ   = zone["type"]
    emoji = "🔴" if typ == "BEARISH" else "🟢"

    text = (
        f"⚠️ *APPROACHING ZONE — {sym}*\n"
        f"\n"
        f"{emoji} *{typ} FVG*\n"
        f"📍 *Zone:* ${zone['bot']} — ${zone['top']}\n"
        f"💰 *Current Price:* ${round(price, 2)}\n"
        f"📏 *Distance:* {round(dist, 1)} pts away\n"
        f"\n"
        f"_Get ready — entry alert coming soon_"
    )
    notify(text)
    log.info(f"Approach alert: {sym} {typ} dist={dist}")

def entry_signal(symbol: str, zone: dict, price: float):
    sym       = symbol.replace("/", "")
    typ       = zone["type"]
    direction = "SELL" if typ == "BEARISH" else "BUY"
    emoji     = "🔴" if direction == "SELL" else "🟢"
    now       = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")

    sl  = round(zone["top"] + 5 if direction == "SELL" else zone["bot"] - 5, 2)
    risk= abs(price - sl)
    tp1 = round(price - risk       if direction == "SELL" else price + risk,       2)
    tp2 = round(price - risk * 2.0 if direction == "SELL" else price + risk * 2.0, 2)

    text = (
        f"{emoji} *{sym} — {direction}* {emoji}\n"
        f"\n"
        f"📍 *Setup:* 30M FVG {typ}\n"
        f"⚡ *Type:* MARKET ORDER\n\n"
        f"💰 *Entry:* `${round(price, 2)}`\n"
        f"🛑 *Stop Loss:* `${sl}`\n"
        f"✅ *TP1:* `${tp1}`\n"
        f"✅ *TP2:* `${tp2}`\n"
        f"⚖️ *R:R:* 1:2\n"
        f"\n"
        f"⚠️ _Not financial advice. Risk max 1-2%._\n"
        f"_{now}_"
    )
    notify(text)
    log.info(f"Entry signal: {sym} {direction} entry={price} sl={sl} tp1={tp1} tp2={tp2}")

    return {
        "symbol":    symbol,
        "direction": direction,
        "entry":     round(price, 2),
        "sl":        sl,
        "tp1":       tp1,
        "tp2":       tp2,
        "tp1_hit":   False,
        "tp2_hit":   False,
    }

def tp_hit_msg(trade: dict, tp_level: str):
    sym = trade["symbol"].replace("/", "")
    tp  = trade["tp1"] if tp_level == "TP1" else trade["tp2"]
    pts = round(abs(tp - trade["entry"]), 2)

    if tp_level == "TP1":
        text = (
            f"*TRADE UPDATE — {sym}*\n"
            f"\n"
            f"TP1 SMASHED 🚀\n\n"
            f"✅ Move Stop Loss to entry — trade is now *risk free*\n"
            f"📊 Let it run to TP2\n"
            f""
        )
    else:
        text = (
            f"*TRADE UPDATE — {sym}*\n"
            f"\n"
            f"TP2 HIT 🔥 +{pts} pts\n\n"
            f"💰 Consider closing or trailing stop\n"
            f""
        )
    notify(text)

def result_msg(trade: dict, result: str, exit_price: float):
    sym  = trade["symbol"].replace("/", "")
    pts  = round(abs(exit_price - trade["entry"]), 2)
    win  = result == "WIN"
    text = (
        f"{'✅ WIN' if win else '❌ LOSS'} — *{sym} {trade['direction']}*\n"
        f"\n"
        f"*Entry:* ${trade['entry']}\n"
        f"*Exit:* ${exit_price}\n"
        f"*Result:* {'+'if win else '-'}{pts} pts\n"
        f"\n"
        f"_Not financial advice_"
    )
    notify(text)

# ── TRADE MONITOR ─────────────────────────────────────────────────
def monitor_open_trade():
    global open_trade
    if not open_trade:
        return
    try:
        price = live_price(open_trade["symbol"])
        d     = open_trade["direction"]

        if d == "BUY":
            if not open_trade["tp1_hit"] and price >= open_trade["tp1"]:
                open_trade["tp1_hit"] = True
                tp_hit_msg(open_trade, "TP1")
            if not open_trade["tp2_hit"] and price >= open_trade["tp2"]:
                open_trade["tp2_hit"] = True
                tp_hit_msg(open_trade, "TP2")
                result_msg(open_trade, "WIN", open_trade["tp2"])
                open_trade = None
                return
            if price <= open_trade["sl"]:
                r = "WIN" if open_trade["tp1_hit"] else "LOSS"
                e = open_trade["entry"] if open_trade["tp1_hit"] else open_trade["sl"]
                result_msg(open_trade, r, e)
                open_trade = None
        else:
            if not open_trade["tp1_hit"] and price <= open_trade["tp1"]:
                open_trade["tp1_hit"] = True
                tp_hit_msg(open_trade, "TP1")
            if not open_trade["tp2_hit"] and price <= open_trade["tp2"]:
                open_trade["tp2_hit"] = True
                tp_hit_msg(open_trade, "TP2")
                result_msg(open_trade, "WIN", open_trade["tp2"])
                open_trade = None
                return
            if price >= open_trade["sl"]:
                r = "WIN" if open_trade["tp1_hit"] else "LOSS"
                e = open_trade["entry"] if open_trade["tp1_hit"] else open_trade["sl"]
                result_msg(open_trade, r, e)
                open_trade = None

    except Exception as e:
        log.error(f"Monitor error: {e}")

# ── MAIN SCAN ─────────────────────────────────────────────────────
def scan(symbol: str):
    global open_trade, daily_signals, approached

    try:
        price   = live_price(symbol)
        df_30m  = fetch(symbol, "30min", 50)
        df_daily= fetch(symbol, "1day",  5)
        time.sleep(1)

        # Detect all FVGs from last 10 bars
        fvgs = detect_fvgs(df_30m.tail(15))

        # Score and filter
        zones = []
        for fvg in fvgs[-5:]:  # only last 5 FVGs
            scored = score_zone(fvg, df_30m, df_daily, price)
            if scored["score"] >= MIN_SCORE:
                zones.append(scored)

        # Check for new zones (not already tracked)
        existing = active_zones.get(symbol, [])
        for zone in zones:
            zone_id = f"{zone['type']}_{zone['top']}_{zone['bot']}"
            if not any(f"{z['type']}_{z['top']}_{z['bot']}" == zone_id for z in existing):
                existing.append(zone)
                zone_alert(symbol, zone)

        active_zones[symbol] = existing

        # Check each zone
        for zone in list(active_zones.get(symbol, [])):
            zone_id  = f"{zone['type']}_{zone['top']}_{zone['bot']}"
            typ      = zone["type"]
            mid      = zone["mid"]

            if typ == "BEARISH":
                dist = price - zone["bot"]  # how far below zone top price is
                approaching = 0 < dist <= APPROACH_DIST
                at_50 = zone["bot"] <= price <= zone["top"] and price >= mid
            else:
                dist = zone["top"] - price
                approaching = 0 < dist <= APPROACH_DIST
                at_50 = zone["bot"] <= price <= zone["top"] and price <= mid

            # Approach alert
            if approaching and zone_id not in approached:
                approach_alert(symbol, zone, price, dist)
                approached[zone_id] = True

            # Entry signal — at or past 50% WITH rejection candle confirmed
            if at_50 and not open_trade and daily_signals < MAX_DAILY and in_session():
                # Fetch 1M candles to check for rejection
                try:
                    df_1m  = fetch(symbol, "1min", 10)
                    last_c = df_1m.iloc[-1]
                    prev_c = df_1m.iloc[-2]
                    body   = abs(last_c.close - last_c.open)
                    rng    = max(last_c.high - last_c.low, 0.0001)

                    # Bearish rejection for sell zone
                    bear_eng  = last_c.close < last_c.open and prev_c.close > prev_c.open and last_c.close < prev_c.open
                    shooting  = last_c.close < last_c.open and (last_c.high - last_c.open) >= body * 1.5
                    bear_pin  = last_c.close < last_c.open and (last_c.high - last_c.open) > rng * 0.5
                    bear_rej  = bear_eng or shooting or bear_pin

                    # Bullish rejection for buy zone
                    bull_eng  = last_c.close > last_c.open and prev_c.close < prev_c.open and last_c.close > prev_c.open
                    hammer    = last_c.close > last_c.open and (last_c.open - last_c.low) >= body * 1.5
                    bull_pin  = last_c.close > last_c.open and (last_c.open - last_c.low) > rng * 0.5
                    bull_rej  = bull_eng or hammer or bull_pin

                    confirmed = (zone["type"] == "BEARISH" and bear_rej) or (zone["type"] == "BULLISH" and bull_rej)

                    if confirmed:
                        trade = entry_signal(symbol, zone, price)
                        open_trade    = trade
                        daily_signals += 1
                        active_zones[symbol].remove(zone)
                        if zone_id in approached:
                            del approached[zone_id]
                        break
                    else:
                        log.info(f"At 50% but no rejection candle yet — waiting")
                except Exception as e:
                    log.error(f"Rejection check error: {e}")

            # Invalidate zone if price closes through it
            if typ == "BEARISH" and price > zone["top"] + 5:
                active_zones[symbol].remove(zone)
                log.info(f"Zone invalidated: {symbol} {typ}")
            elif typ == "BULLISH" and price < zone["bot"] - 5:
                active_zones[symbol].remove(zone)
                log.info(f"Zone invalidated: {symbol} {typ}")

    except Exception as e:
        log.error(f"Scan error for {symbol}: {e}")

# ── MAIN ─────────────────────────────────────────────────────────
def main():
    global daily_signals, last_day

    log.info("=" * 50)
    log.info("  30M FVG Zone Scanner")
    log.info(f"  Symbols: {', '.join(SYMBOLS)}")
    log.info(f"  Scan: every {SCAN_INTERVAL}s")
    log.info(f"  Min score: {MIN_SCORE}")
    log.info(f"  Approach alert: {APPROACH_DIST}pts")
    log.info("=" * 50)

    notify(
        "🤖 *FVG Zone Scanner Online*\n"
        f"Scanning: {', '.join(s.replace('/', '') for s in SYMBOLS)}\n"
        f"Timeframe: 30M FVGs\n"
        f"Min score: {MIN_SCORE}/9\n"
        f"Approach alert: {APPROACH_DIST}pts from zone\n"
        f"Entry: at 50% of FVG\n"
        f"Sessions: London + New York"
    )

    while True:
        now = datetime.now(timezone.utc)

        # Reset daily
        if last_day != now.date():
            daily_signals = 0
            last_day      = now.date()
            approached    = {}
            log.info("Daily reset")

        # Monitor open trade
        monitor_open_trade()

        # Scan all symbols
        for symbol in SYMBOLS:
            scan(symbol)
            time.sleep(2)

        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()
