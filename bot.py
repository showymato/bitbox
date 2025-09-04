import os
import json
import logging
import asyncio
import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ------------------- CONFIG -------------------
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
LIMIT = 500

SESSION_FILTER = True
SESSION_START_UTC = 7   # 07:00 UTC (London open)
SESSION_END_UTC   = 20  # 20:00 UTC (NY close)

STATE_FILE = "config.json"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ALLOWED_CHAT_IDS = [int(cid) for cid in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if cid]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScalperBot")

# ------------------- HELPERS -------------------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"open_trade": None}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def utc_hour():
    return datetime.now(timezone.utc).hour

# ------------------- STRATEGY -------------------
class Scalper:
    def __init__(self):
        self.exchange = ccxt.binance({"enableRateLimit": True})
        self.state = load_state()

    async def fetch_ohlcv(self):
        ohlcv = await self.exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
        df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df

    def indicators(self, df):
        df["ema9"] = df["c"].ewm(span=9).mean()
        df["ema21"] = df["c"].ewm(span=21).mean()
        df["ema50"] = df["c"].ewm(span=50).mean()
        df["ema200"] = df["c"].ewm(span=200).mean()
        df["rsi"] = self.rsi(df["c"], 14)
        df["atr"] = self.atr(df, 14)
        df["vwap"] = (df["c"] * df["v"]).cumsum() / df["v"].cumsum()
        df["vol_avg"] = df["v"].rolling(20).mean()
        return df

    def rsi(self, series, period=14):
        delta = series.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(period).mean()
        avg_loss = pd.Series(loss).rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def atr(self, df, period=14):
        hl = df["h"] - df["l"]
        hc = abs(df["h"] - df["c"].shift())
        lc = abs(df["l"] - df["c"].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def entry_signal(self, df):
        row = df.iloc[-1]
        if SESSION_FILTER and not (SESSION_START_UTC <= utc_hour() <= SESSION_END_UTC):
            return None

        # Long
        if row["ema9"] > row["ema21"] and row["ema21"] > row["ema50"]:
            if row["c"] > row["vwap"] and 40 < row["rsi"] < 65 and row["v"] > row["vol_avg"]:
                return {"side": "BUY", "price": row["c"], "atr": row["atr"]}
        # Short
        if row["ema9"] < row["ema21"] and row["ema21"] < row["ema50"]:
            if row["c"] < row["vwap"] and 35 < row["rsi"] < 60 and row["v"] > row["vol_avg"]:
                return {"side": "SELL", "price": row["c"], "atr": row["atr"]}
        return None

    def exit_signal(self, df):
        trade = self.state.get("open_trade")
        if not trade:
            return None

        row = df.iloc[-1]
        price = row["c"]
        atr = trade["atr"]

        if trade["side"] == "BUY":
            # SL
            if price <= trade["entry"] - atr:
                return {"exit": "SL hit", "price": price}
            # TP
            if price >= trade["entry"] + 2*atr:
                return {"exit": "TP hit", "price": price}
            # RSI extreme
            if row["rsi"] > 75:
                return {"exit": "RSI extreme", "price": price}
            # Trailing stop
            if price - trade["entry"] >= atr:
                ts = trade.get("trailing_stop", trade["entry"])
                new_ts = max(ts, price - atr)
                trade["trailing_stop"] = new_ts
                save_state(self.state)
                if price <= new_ts:
                    return {"exit": "Trailing Stop", "price": price}

        if trade["side"] == "SELL":
            # SL
            if price >= trade["entry"] + atr:
                return {"exit": "SL hit", "price": price}
            # TP
            if price <= trade["entry"] - 2*atr:
                return {"exit": "TP hit", "price": price}
            # RSI extreme
            if row["rsi"] < 25:
                return {"exit": "RSI extreme", "price": price}
            # Trailing stop
            if trade["entry"] - price >= atr:
                ts = trade.get("trailing_stop", trade["entry"])
                new_ts = min(ts, price + atr)
                trade["trailing_stop"] = new_ts
                save_state(self.state)
                if price >= new_ts:
                    return {"exit": "Trailing Stop", "price": price}

        return None

# ------------------- TELEGRAM HANDLERS -------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id not in ALLOWED_CHAT_IDS:
        return
    await update.message.reply_text("🤖 Scalper Bot running with full entry/exit logic!")

async def check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id not in ALLOWED_CHAT_IDS:
        return
    bot = Scalper()
    df = await bot.fetch_ohlcv()
    df = bot.indicators(df)

    # Check exit first
    exit_sig = bot.exit_signal(df)
    if exit_sig:
        bot.state["open_trade"] = None
        save_state(bot.state)
        msg = f"❌ EXIT {exit_sig['exit']} @ {exit_sig['price']:.2f}"
        await update.message.reply_text(msg)
        return

    # Otherwise check entry
    if not bot.state.get("open_trade"):
        entry_sig = bot.entry_signal(df)
        if entry_sig:
            bot.state["open_trade"] = {
                "side": entry_sig["side"],
                "entry": entry_sig["price"],
                "atr": entry_sig["atr"]
            }
            save_state(bot.state)
            msg = f"✅ ENTRY {entry_sig['side']} @ {entry_sig['price']:.2f} | ATR={entry_sig['atr']:.2f}"
        else:
            msg = "⚠️ No new signal."
    else:
        trade = bot.state["open_trade"]
        msg = f"📈 Holding {trade['side']} from {trade['entry']:.2f} | ATR={trade['atr']:.2f}"

    await update.message.reply_text(msg)

# ------------------- MAIN LOOP -------------------
# async def main():
#     app = Application.builder().token(TELEGRAM_TOKEN).build()
#     app.add_handler(CommandHandler("start", start))
#     app.add_handler(CommandHandler("check", check))
#     await app.run_polling()

# if __name__ == "__main__":
#     asyncio.run(main())
# ------------------- MAIN LOOP -------------------
async def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("check", check))
    await app.run_polling()  # keep polling forever

if __name__ == "__main__":
    import asyncio
    asyncio.get_event_loop().run_until_complete(main())  # remove asyncio.run()

