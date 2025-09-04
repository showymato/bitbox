import os
import json
import logging
import asyncio
import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypesimport os
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

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ScalperBot")

# ------------------- HELPERS -------------------
def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading state: {e}")
    return {"open_trade": None}

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving state: {e}")

def utc_hour():
    return datetime.now(timezone.utc).hour

# ------------------- STRATEGY -------------------
class Scalper:
    def __init__(self):
        self.exchange = None
        self.state = load_state()
    
    async def initialize_exchange(self):
        if not self.exchange:
            self.exchange = ccxt.binance({
                "enableRateLimit": True,
                "sandbox": False,
                "timeout": 30000,
            })
    
    async def close_exchange(self):
        if self.exchange:
            try:
                await self.exchange.close()
            except Exception as e:
                logger.warning(f"Error closing exchange: {e}")
            finally:
                self.exchange = None

    async def fetch_ohlcv(self):
        try:
            await self.initialize_exchange()
            ohlcv = await self.exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
            df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV: {e}")
            raise

    def indicators(self, df):
        if len(df) < 200:
            raise ValueError("Not enough historical data for indicators")
        
        df = df.copy()
        df["ema9"] = df["c"].ewm(span=9, adjust=False).mean()
        df["ema21"] = df["c"].ewm(span=21, adjust=False).mean()
        df["ema50"] = df["c"].ewm(span=50, adjust=False).mean()
        df["ema200"] = df["c"].ewm(span=200, adjust=False).mean()
        df["rsi"] = self.rsi(df["c"], 14)
        df["atr"] = self.atr(df, 14)
        df["vwap"] = (df["c"] * df["v"]).cumsum() / df["v"].cumsum()
        df["vol_avg"] = df["v"].rolling(20).mean()
        return df

    def rsi(self, series, period=14):
        delta = series.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        gain_series = pd.Series(gain, index=series.index)
        loss_series = pd.Series(loss, index=series.index)
        avg_gain = gain_series.rolling(window=period, min_periods=period).mean()
        avg_loss = loss_series.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def atr(self, df, period=14):
        high = df["h"]
        low = df["l"]
        close = df["c"]
        prev_close = close.shift(1)
        
        hl = high - low
        hc = abs(high - prev_close)
        lc = abs(low - prev_close)
        
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return atr

    def entry_signal(self, df):
        if len(df) < 200:
            return None
            
        row = df.iloc[-1]
        
        required_fields = ["ema9", "ema21", "ema50", "vwap", "rsi", "vol_avg", "atr"]
        if any(pd.isna(row[field]) for field in required_fields):
            return None
        
        if SESSION_FILTER and not (SESSION_START_UTC <= utc_hour() <= SESSION_END_UTC):
            return None

        # Long signal
        if (row["ema9"] > row["ema21"] and 
            row["ema21"] > row["ema50"] and
            row["c"] > row["vwap"] and 
            40 < row["rsi"] < 65 and 
            row["v"] > row["vol_avg"]):
            return {"side": "BUY", "price": float(row["c"]), "atr": float(row["atr"])}

        # Short signal
        if (row["ema9"] < row["ema21"] and 
            row["ema21"] < row["ema50"] and
            row["c"] < row["vwap"] and 
            35 < row["rsi"] < 60 and 
            row["v"] > row["vol_avg"]):
            return {"side": "SELL", "price": float(row["c"]), "atr": float(row["atr"])}

        return None

    def exit_signal(self, df):
        trade = self.state.get("open_trade")
        if not trade:
            return None

        if len(df) < 1:
            return None
            
        row = df.iloc[-1]
        price = float(row["c"])
        atr = trade["atr"]
        entry_price = trade["entry"]

        if trade["side"] == "BUY":
            if price <= entry_price - atr:
                return {"exit": "SL hit", "price": price}
            if price >= entry_price + 2*atr:
                return {"exit": "TP hit", "price": price}
            if not pd.isna(row["rsi"]) and row["rsi"] > 75:
                return {"exit": "RSI extreme", "price": price}
            
            if price - entry_price >= atr:
                current_ts = trade.get("trailing_stop", entry_price)
                new_ts = max(current_ts, price - atr)
                if new_ts > current_ts:
                    trade["trailing_stop"] = new_ts
                    save_state(self.state)
                if price <= new_ts:
                    return {"exit": "Trailing Stop", "price": price}

        elif trade["side"] == "SELL":
            if price >= entry_price + atr:
                return {"exit": "SL hit", "price": price}
            if price <= entry_price - 2*atr:
                return {"exit": "TP hit", "price": price}
            if not pd.isna(row["rsi"]) and row["rsi"] < 25:
                return {"exit": "RSI extreme", "price": price}
            
            if entry_price - price >= atr:
                current_ts = trade.get("trailing_stop", entry_price)
                new_ts = min(current_ts, price + atr)
                if new_ts < current_ts:
                    trade["trailing_stop"] = new_ts
                    save_state(self.state)
                if price >= new_ts:
                    return {"exit": "Trailing Stop", "price": price}

        return None

# ------------------- TELEGRAM HANDLERS -------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if ALLOWED_CHAT_IDS and update.effective_chat.id not in ALLOWED_CHAT_IDS:
        await update.message.reply_text("❌ Unauthorized access")
        return
    
    await update.message.reply_text(
        "🤖 **Scalper Bot Active!**\n\n"
        "Commands:\n"
        "/check - Check for signals\n"
        "/status - Current position\n"
        "/config - Bot configuration\n\n"
        f"Trading: {SYMBOL} on {TIMEFRAME} timeframe"
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if ALLOWED_CHAT_IDS and update.effective_chat.id not in ALLOWED_CHAT_IDS:
        return
    
    bot = Scalper()
    trade = bot.state.get("open_trade")
    
    if trade:
        msg = (f"📊 **Current Position**\n"
               f"Side: {trade['side']}\n"
               f"Entry: ${trade['entry']:.2f}\n"
               f"ATR: ${trade['atr']:.2f}")
        if "trailing_stop" in trade:
            msg += f"\nTrailing Stop: ${trade['trailing_stop']:.2f}"
    else:
        msg = "💤 No open position"
    
    await update.message.reply_text(msg)

async def config_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if ALLOWED_CHAT_IDS and update.effective_chat.id not in ALLOWED_CHAT_IDS:
        return
    
    session_status = "ON" if SESSION_FILTER else "OFF"
    current_hour = utc_hour()
    in_session = SESSION_START_UTC <= current_hour <= SESSION_END_UTC
    
    msg = (f"⚙️ **Bot Configuration**\n"
           f"Symbol: {SYMBOL}\n"
           f"Timeframe: {TIMEFRAME}\n"
           f"Session Filter: {session_status}\n"
           f"Trading Hours: {SESSION_START_UTC}:00 - {SESSION_END_UTC}:00 UTC\n"
           f"Current Hour: {current_hour}:00 UTC\n"
           f"In Session: {'✅' if in_session else '❌'}")
    
    await update.message.reply_text(msg)

async def check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if ALLOWED_CHAT_IDS and update.effective_chat.id not in ALLOWED_CHAT_IDS:
        return

    bot = None
    try:
        bot = Scalper()
        df = await bot.fetch_ohlcv()
        df = bot.indicators(df)
        
        exit_sig = bot.exit_signal(df)
        if exit_sig:
            bot.state["open_trade"] = None
            save_state(bot.state)
            msg = f"❌ **EXIT SIGNAL**\n{exit_sig['exit']} @ ${exit_sig['price']:.2f}"
            await update.message.reply_text(msg)
            return

        if not bot.state.get("open_trade"):
            entry_sig = bot.entry_signal(df)
            if entry_sig:
                bot.state["open_trade"] = {
                    "side": entry_sig["side"],
                    "entry": entry_sig["price"],
                    "atr": entry_sig["atr"],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                save_state(bot.state)
                
                entry_price = entry_sig["price"]
                atr = entry_sig["atr"]
                
                if entry_sig["side"] == "BUY":
                    sl = entry_price - atr
                    tp = entry_price + 2*atr
                else:
                    sl = entry_price + atr
                    tp = entry_price - 2*atr
                
                msg = (f"✅ **ENTRY SIGNAL**\n"
                       f"Side: {entry_sig['side']}\n"
                       f"Price: ${entry_price:.2f}\n"
                       f"Stop Loss: ${sl:.2f}\n"
                       f"Take Profit: ${tp:.2f}\n"
                       f"ATR: ${atr:.2f}")
            else:
                current_price = df.iloc[-1]["c"]
                rsi = df.iloc[-1]["rsi"]
                msg = (f"⚠️ **No Signal**\n"
                       f"Price: ${current_price:.2f}\n"
                       f"RSI: {rsi:.1f}")
        else:
            trade = bot.state["open_trade"]
            current_price = df.iloc[-1]["c"]
            entry_price = trade["entry"]
            
            if trade["side"] == "BUY":
                pnl = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl = ((entry_price - current_price) / entry_price) * 100
            
            pnl_emoji = "🟢" if pnl > 0 else "🔴"
            
            msg = (f"📈 **Holding Position**\n"
                   f"Side: {trade['side']}\n"
                   f"Entry: ${entry_price:.2f}\n"
                   f"Current: ${current_price:.2f}\n"
                   f"PnL: {pnl_emoji} {pnl:+.2f}%")
            
            if "trailing_stop" in trade:
                msg += f"\nTrailing Stop: ${trade['trailing_stop']:.2f}"

        await update.message.reply_text(msg)
        
    except Exception as e:
        logger.error(f"Error in check command: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")
    finally:
        if bot:
            await bot.close_exchange()

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")

# ------------------- MAIN FUNCTION - FIXED FOR RENDER -------------------
def main():
    """
    Entry point for the bot - designed to work on Render and other platforms
    """
    # Validate environment variables
    if not TELEGRAM_TOKEN:
        logger.error("❌ TELEGRAM_TOKEN environment variable not set!")
        return
    
    if not ALLOWED_CHAT_IDS:
        logger.warning("⚠️ TELEGRAM_CHAT_IDS not set - bot will accept all users!")
    
    logger.info("🚀 Starting Scalper Bot...")
    
    # Create application
    try:
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Add handlers
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("check", check))
        app.add_handler(CommandHandler("status", status))
        app.add_handler(CommandHandler("config", config_info))
        app.add_error_handler(error_handler)
        
        logger.info("✅ Bot handlers registered successfully")
        logger.info("🔄 Starting polling...")
        
        # Run the bot - this is the key fix!
        app.run_polling(
            drop_pending_updates=True,
            close_loop=False  # This prevents the event loop error
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()  # No asyncio.run() - this is the key fix!


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

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ScalperBot")

# ------------------- HELPERS -------------------
def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading state: {e}")
    return {"open_trade": None}

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving state: {e}")

def utc_hour():
    return datetime.now(timezone.utc).hour

# ------------------- STRATEGY -------------------
class Scalper:
    def __init__(self):
        self.exchange = None
        self.state = load_state()
    
    async def initialize_exchange(self):
        if not self.exchange:
            self.exchange = ccxt.binance({
                "enableRateLimit": True,
                "sandbox": False,  # Set to True for testing
                "timeout": 30000,
            })
    
    async def close_exchange(self):
        if self.exchange:
            await self.exchange.close()
            self.exchange = None

    async def fetch_ohlcv(self):
        try:
            await self.initialize_exchange()
            ohlcv = await self.exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
            df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV: {e}")
            raise

    def indicators(self, df):
        # Ensure we have enough data
        if len(df) < 200:
            raise ValueError("Not enough historical data for indicators")
        
        df = df.copy()
        df["ema9"] = df["c"].ewm(span=9, adjust=False).mean()
        df["ema21"] = df["c"].ewm(span=21, adjust=False).mean()
        df["ema50"] = df["c"].ewm(span=50, adjust=False).mean()
        df["ema200"] = df["c"].ewm(span=200, adjust=False).mean()
        df["rsi"] = self.rsi(df["c"], 14)
        df["atr"] = self.atr(df, 14)
        df["vwap"] = (df["c"] * df["v"]).cumsum() / df["v"].cumsum()
        df["vol_avg"] = df["v"].rolling(20).mean()
        return df

    def rsi(self, series, period=14):
        delta = series.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        gain_series = pd.Series(gain, index=series.index)
        loss_series = pd.Series(loss, index=series.index)
        avg_gain = gain_series.rolling(window=period, min_periods=period).mean()
        avg_loss = loss_series.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def atr(self, df, period=14):
        high = df["h"]
        low = df["l"]
        close = df["c"]
        prev_close = close.shift(1)
        
        hl = high - low
        hc = abs(high - prev_close)
        lc = abs(low - prev_close)
        
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return atr

    def entry_signal(self, df):
        if len(df) < 200:
            return None
            
        row = df.iloc[-1]
        
        # Check for NaN values
        required_fields = ["ema9", "ema21", "ema50", "vwap", "rsi", "vol_avg", "atr"]
        if any(pd.isna(row[field]) for field in required_fields):
            return None
        
        if SESSION_FILTER and not (SESSION_START_UTC <= utc_hour() <= SESSION_END_UTC):
            return None

        # Long signal
        if (row["ema9"] > row["ema21"] and 
            row["ema21"] > row["ema50"] and
            row["c"] > row["vwap"] and 
            40 < row["rsi"] < 65 and 
            row["v"] > row["vol_avg"]):
            return {"side": "BUY", "price": float(row["c"]), "atr": float(row["atr"])}

        # Short signal
        if (row["ema9"] < row["ema21"] and 
            row["ema21"] < row["ema50"] and
            row["c"] < row["vwap"] and 
            35 < row["rsi"] < 60 and 
            row["v"] > row["vol_avg"]):
            return {"side": "SELL", "price": float(row["c"]), "atr": float(row["atr"])}

        return None

    def exit_signal(self, df):
        trade = self.state.get("open_trade")
        if not trade:
            return None

        if len(df) < 1:
            return None
            
        row = df.iloc[-1]
        price = float(row["c"])
        atr = trade["atr"]
        entry_price = trade["entry"]

        if trade["side"] == "BUY":
            # Stop Loss
            if price <= entry_price - atr:
                return {"exit": "SL hit", "price": price}
            
            # Take Profit
            if price >= entry_price + 2*atr:
                return {"exit": "TP hit", "price": price}
            
            # RSI extreme exit
            if not pd.isna(row["rsi"]) and row["rsi"] > 75:
                return {"exit": "RSI extreme", "price": price}
            
            # Trailing stop logic
            if price - entry_price >= atr:
                current_ts = trade.get("trailing_stop", entry_price)
                new_ts = max(current_ts, price - atr)
                if new_ts > current_ts:
                    trade["trailing_stop"] = new_ts
                    save_state(self.state)
                if price <= new_ts:
                    return {"exit": "Trailing Stop", "price": price}

        elif trade["side"] == "SELL":
            # Stop Loss
            if price >= entry_price + atr:
                return {"exit": "SL hit", "price": price}
            
            # Take Profit
            if price <= entry_price - 2*atr:
                return {"exit": "TP hit", "price": price}
            
            # RSI extreme exit
            if not pd.isna(row["rsi"]) and row["rsi"] < 25:
                return {"exit": "RSI extreme", "price": price}
            
            # Trailing stop logic
            if entry_price - price >= atr:
                current_ts = trade.get("trailing_stop", entry_price)
                new_ts = min(current_ts, price + atr)
                if new_ts < current_ts:
                    trade["trailing_stop"] = new_ts
                    save_state(self.state)
                if price >= new_ts:
                    return {"exit": "Trailing Stop", "price": price}

        return None

# ------------------- TELEGRAM HANDLERS -------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if ALLOWED_CHAT_IDS and update.effective_chat.id not in ALLOWED_CHAT_IDS:
        await update.message.reply_text("❌ Unauthorized access")
        return
    
    await update.message.reply_text(
        "🤖 **Scalper Bot Active!**\n\n"
        "Commands:\n"
        "/check - Check for signals\n"
        "/status - Current position\n"
        "/config - Bot configuration\n\n"
        f"Trading: {SYMBOL} on {TIMEFRAME} timeframe"
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if ALLOWED_CHAT_IDS and update.effective_chat.id not in ALLOWED_CHAT_IDS:
        return
    
    bot = Scalper()
    trade = bot.state.get("open_trade")
    
    if trade:
        msg = (f"📊 **Current Position**\n"
               f"Side: {trade['side']}\n"
               f"Entry: ${trade['entry']:.2f}\n"
               f"ATR: ${trade['atr']:.2f}")
        if "trailing_stop" in trade:
            msg += f"\nTrailing Stop: ${trade['trailing_stop']:.2f}"
    else:
        msg = "💤 No open position"
    
    await update.message.reply_text(msg)

async def config_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if ALLOWED_CHAT_IDS and update.effective_chat.id not in ALLOWED_CHAT_IDS:
        return
    
    session_status = "ON" if SESSION_FILTER else "OFF"
    current_hour = utc_hour()
    in_session = SESSION_START_UTC <= current_hour <= SESSION_END_UTC
    
    msg = (f"⚙️ **Bot Configuration**\n"
           f"Symbol: {SYMBOL}\n"
           f"Timeframe: {TIMEFRAME}\n"
           f"Session Filter: {session_status}\n"
           f"Trading Hours: {SESSION_START_UTC}:00 - {SESSION_END_UTC}:00 UTC\n"
           f"Current Hour: {current_hour}:00 UTC\n"
           f"In Session: {'✅' if in_session else '❌'}")
    
    await update.message.reply_text(msg)

async def check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if ALLOWED_CHAT_IDS and update.effective_chat.id not in ALLOWED_CHAT_IDS:
        return

    bot = None
    try:
        bot = Scalper()
        df = await bot.fetch_ohlcv()
        df = bot.indicators(df)
        
        # Check exit first
        exit_sig = bot.exit_signal(df)
        if exit_sig:
            bot.state["open_trade"] = None
            save_state(bot.state)
            pnl = ""
            msg = f"❌ **EXIT SIGNAL**\n{exit_sig['exit']} @ ${exit_sig['price']:.2f}{pnl}"
            await update.message.reply_text(msg)
            return

        # Check entry if no open trade
        if not bot.state.get("open_trade"):
            entry_sig = bot.entry_signal(df)
            if entry_sig:
                bot.state["open_trade"] = {
                    "side": entry_sig["side"],
                    "entry": entry_sig["price"],
                    "atr": entry_sig["atr"],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                save_state(bot.state)
                
                # Calculate targets
                entry_price = entry_sig["price"]
                atr = entry_sig["atr"]
                
                if entry_sig["side"] == "BUY":
                    sl = entry_price - atr
                    tp = entry_price + 2*atr
                else:
                    sl = entry_price + atr
                    tp = entry_price - 2*atr
                
                msg = (f"✅ **ENTRY SIGNAL**\n"
                       f"Side: {entry_sig['side']}\n"
                       f"Price: ${entry_price:.2f}\n"
                       f"Stop Loss: ${sl:.2f}\n"
                       f"Take Profit: ${tp:.2f}\n"
                       f"ATR: ${atr:.2f}")
            else:
                current_price = df.iloc[-1]["c"]
                rsi = df.iloc[-1]["rsi"]
                msg = (f"⚠️ **No Signal**\n"
                       f"Price: ${current_price:.2f}\n"
                       f"RSI: {rsi:.1f}")
        else:
            trade = bot.state["open_trade"]
            current_price = df.iloc[-1]["c"]
            entry_price = trade["entry"]
            
            if trade["side"] == "BUY":
                pnl = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl = ((entry_price - current_price) / entry_price) * 100
            
            pnl_emoji = "🟢" if pnl > 0 else "🔴"
            
            msg = (f"📈 **Holding Position**\n"
                   f"Side: {trade['side']}\n"
                   f"Entry: ${entry_price:.2f}\n"
                   f"Current: ${current_price:.2f}\n"
                   f"PnL: {pnl_emoji} {pnl:+.2f}%")
            
            if "trailing_stop" in trade:
                msg += f"\nTrailing Stop: ${trade['trailing_stop']:.2f}"

        await update.message.reply_text(msg)
        
    except Exception as e:
        logger.error(f"Error in check command: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")
    finally:
        if bot:
            await bot.close_exchange()

# ------------------- ERROR HANDLER -------------------
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")

# ------------------- MAIN LOOP -------------------
async def main():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN environment variable not set!")
        return
    
    if not ALLOWED_CHAT_IDS:
        logger.warning("TELEGRAM_CHAT_IDS not set - bot will accept all users!")
    
    logger.info("Starting Scalper Bot...")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("check", check))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("config", config_info))
    
    # Add error handler
    app.add_error_handler(error_handler)
    
    # Start polling
    logger.info("Bot is running... Press Ctrl+C to stop")
    await app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
