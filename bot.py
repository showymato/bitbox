"""
Production Crypto Trading Bot
Modular, async-safe architecture for 5m BTC scalping
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ==================== CONFIGURATION ====================
@dataclass
class BotConfig:
    """Centralized configuration"""
    # Trading
    symbol: str = "BTC/USDT"
    timeframe: str = "5m"
    limit: int = 500
    
    # Strategy
    session_filter: bool = True
    session_start_utc: int = 7
    session_end_utc: int = 20
    
    # Risk Management
    max_position_size: float = 100.0  # USDT
    max_daily_trades: int = 10
    
    # Files
    state_file: str = "bot_state.json"
    log_file: str = "bot.log"
    
    # External
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    allowed_chat_ids: list = None
    
    def __post_init__(self):
        if self.allowed_chat_ids is None:
            chat_ids = os.getenv("TELEGRAM_CHAT_IDS", "")
            self.allowed_chat_ids = [int(cid.strip()) for cid in chat_ids.split(",") if cid.strip()]

# Global config instance
config = BotConfig()

# ==================== LOGGING SETUP ====================
def setup_logging():
    """Enhanced logging with file output"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("ScalperBot")

logger = setup_logging()

# ==================== STATE MANAGEMENT ====================
class StateManager:
    """Handles bot state persistence"""
    
    def __init__(self, filename: str):
        self.filename = filename
        self._state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load state from file"""
        try:
            if os.path.exists(self.filename):
                with open(self.filename, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading state: {e}")
        
        return {
            "open_trade": None,
            "daily_trades": 0,
            "last_trade_date": None,
            "total_trades": 0,
            "total_pnl": 0.0
        }
    
    def save_state(self):
        """Save state to file"""
        try:
            with open(self.filename, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def get(self, key: str, default=None):
        """Get state value"""
        return self._state.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set state value and save"""
        self._state[key] = value
        self.save_state()
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple state values"""
        self._state.update(updates)
        self.save_state()

# Global state manager
state_manager = StateManager(config.state_file)

# ==================== TRADING BOT ====================
class ScalperBot:
    """Main trading bot class"""
    
    def __init__(self):
        self.exchange = None
        
    def initialize_exchange(self):
        """Initialize exchange connection"""
        if not self.exchange:
            self.exchange = ccxt.binance({
                "enableRateLimit": True,
                "sandbox": False,
                "timeout": 30000,
            })
    
    def close_exchange(self):
        """Close exchange connection"""
        if self.exchange:
            try:
                self.exchange.close()
            except Exception as e:
                logger.warning(f"Error closing exchange: {e}")
            finally:
                self.exchange = None
    
    def fetch_ohlcv(self):
        """Fetch OHLCV data"""
        try:
            self.initialize_exchange()
            ohlcv = self.exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
            
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            if len(df) < 200:
                raise ValueError(f"Insufficient data: {len(df)} candles (need 200+)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            raise
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators"""
        df = df.copy()
        
        # EMAs
        df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        gain_series = pd.Series(gain, index=df.index)
        loss_series = pd.Series(loss, index=df.index)
        
        avg_gain = gain_series.rolling(window=14, min_periods=14).mean()
        avg_loss = loss_series.rolling(window=14, min_periods=14).mean()
        
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # ATR
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=14, min_periods=14).mean()
        
        # VWAP
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        
        # Volume average
        df["volume_avg"] = df["volume"].rolling(20).mean()
        
        return df
    
    def is_session_active(self):
        """Check if we're in active trading session"""
        if not SESSION_FILTER:
            return True
        
        current_hour = datetime.now(timezone.utc).hour
        return SESSION_START_UTC <= current_hour <= SESSION_END_UTC
    
    def validate_indicators(self, row):
        """Validate that row has all required indicators"""
        required_fields = ["ema9", "ema21", "ema50", "vwap", "rsi", "volume_avg", "atr"]
        return not any(pd.isna(row[field]) for field in required_fields)
    
    def check_entry_signal(self, df):
        """Check for entry signals"""
        if len(df) < 200 or not self.is_session_active():
            return None
        
        row = df.iloc[-1]
        
        if not self.validate_indicators(row):
            return None
        
        # Check daily trade limit
        today = datetime.now(timezone.utc).date().isoformat()
        if bot_state.get("last_trade_date") != today:
            bot_state["daily_trades"] = 0
            bot_state["last_trade_date"] = today
            save_state(bot_state)
        
        if bot_state.get("daily_trades", 0) >= MAX_DAILY_TRADES:
            return None
        
        # Long signal
        if (row["ema9"] > row["ema21"] and 
            row["ema21"] > row["ema50"] and
            row["close"] > row["vwap"] and 
            40 <= row["rsi"] <= 65 and 
            row["volume"] > row["volume_avg"]):
            
            return {
                "side": "BUY",
                "price": float(row["close"]),
                "atr": float(row["atr"]),
                "timestamp": row["timestamp"].isoformat(),
                "rsi": float(row["rsi"])
            }
        
        # Short signal
        if (row["ema9"] < row["ema21"] and 
            row["ema21"] < row["ema50"] and
            row["close"] < row["vwap"] and 
            35 <= row["rsi"] <= 60 and 
            row["volume"] > row["volume_avg"]):
            
            return {
                "side": "SELL",
                "price": float(row["close"]),
                "atr": float(row["atr"]),
                "timestamp": row["timestamp"].isoformat(),
                "rsi": float(row["rsi"])
            }
        
        return None
    
    def check_exit_signal(self, df):
        """Check for exit signals"""
        trade = bot_state.get("open_trade")
        if not trade or len(df) < 1:
            return None
        
        row = df.iloc[-1]
        current_price = float(row["close"])
        entry_price = trade["entry_price"]
        atr = trade["atr"]
        side = trade["side"]
        
        # Stop Loss and Take Profit
        if side == "BUY":
            stop_loss = entry_price - atr
            take_profit = entry_price + (2 * atr)
            
            if current_price <= stop_loss:
                return {"reason": "Stop Loss", "price": current_price}
            
            if current_price >= take_profit:
                return {"reason": "Take Profit", "price": current_price}
            
            # RSI extreme exit
            if not pd.isna(row["rsi"]) and row["rsi"] > 75:
                return {"reason": "RSI Overbought", "price": current_price}
            
            # Trailing stop logic
            profit = current_price - entry_price
            if profit >= atr:
                current_trailing = trade.get("trailing_stop", entry_price)
                new_trailing = max(current_trailing, current_price - atr)
                
                if new_trailing > current_trailing:
                    trade["trailing_stop"] = new_trailing
                    bot_state["open_trade"] = trade
                    save_state(bot_state)
                
                if current_price <= new_trailing:
                    return {"reason": "Trailing Stop", "price": current_price}
        
        elif side == "SELL":
            stop_loss = entry_price + atr
            take_profit = entry_price - (2 * atr)
            
            if current_price >= stop_loss:
                return {"reason": "Stop Loss", "price": current_price}
            
            if current_price <= take_profit:
                return {"reason": "Take Profit", "price": current_price}
            
            # RSI extreme exit
            if not pd.isna(row["rsi"]) and row["rsi"] < 25:
                return {"reason": "RSI Oversold", "price": current_price}
            
            # Trailing stop logic
            profit = entry_price - current_price
            if profit >= atr:
                current_trailing = trade.get("trailing_stop", entry_price)
                new_trailing = min(current_trailing, current_price + atr)
                
                if new_trailing < current_trailing:
                    trade["trailing_stop"] = new_trailing
                    bot_state["open_trade"] = trade
                    save_state(bot_state)
                
                if current_price >= new_trailing:
                    return {"reason": "Trailing Stop", "price": current_price}
        
        return None
    
    def process_signals(self):
        """Main signal processing logic"""
        try:
            # Fetch data and calculate indicators
            df = self.fetch_ohlcv()
            df = self.calculate_indicators(df)
            
            current_price = float(df.iloc[-1]["close"])
            current_rsi = float(df.iloc[-1]["rsi"]) if not pd.isna(df.iloc[-1]["rsi"]) else None
            
            # Check for exit signal first
            exit_signal = self.check_exit_signal(df)
            if exit_signal:
                trade = bot_state["open_trade"]
                entry_price = trade["entry_price"]
                exit_price = exit_signal["price"]
                side = trade["side"]
                
                # Calculate PnL
                if side == "BUY":
                    pnl = exit_price - entry_price
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                else:
                    pnl = entry_price - exit_price
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                
                # Update state
                bot_state["open_trade"] = None
                bot_state["total_trades"] = bot_state.get("total_trades", 0) + 1
                bot_state["total_pnl"] = bot_state.get("total_pnl", 0.0) + pnl
                bot_state["daily_trades"] = bot_state.get("daily_trades", 0) + 1
                save_state(bot_state)
                
                logger.info(f"EXIT: {side} @ {exit_price:.2f} | PnL: {pnl:+.2f} ({pnl_pct:+.2f}%)")
                
                return {
                    "action": "EXIT",
                    "reason": exit_signal["reason"],
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "side": side,
                    "current_price": current_price,
                    "rsi": current_rsi
                }
            
            # Check for entry signal if no open trade
            if not bot_state.get("open_trade"):
                entry_signal = self.check_entry_signal(df)
                if entry_signal:
                    trade_data = {
                        "side": entry_signal["side"],
                        "entry_price": entry_signal["price"],
                        "atr": entry_signal["atr"],
                        "entry_time": entry_signal["timestamp"],
                        "entry_rsi": entry_signal["rsi"]
                    }
                    
                    bot_state["open_trade"] = trade_data
                    save_state(bot_state)
                    
                    # Calculate targets
                    entry_price = entry_signal["price"]
                    atr = entry_signal["atr"]
                    
                    if entry_signal["side"] == "BUY":
                        stop_loss = entry_price - atr
                        take_profit = entry_price + (2 * atr)
                    else:
                        stop_loss = entry_price + atr
                        take_profit = entry_price - (2 * atr)
                    
                    logger.info(f"ENTRY: {entry_signal['side']} @ {entry_price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
                    
                    return {
                        "action": "ENTRY",
                        "side": entry_signal["side"],
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "atr": atr,
                        "current_price": current_price,
                        "rsi": current_rsi
                    }
            
            # No signal - return current status
            open_trade = bot_state.get("open_trade")
            if open_trade:
                entry_price = open_trade["entry_price"]
                side = open_trade["side"]
                
                if side == "BUY":
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                
                return {
                    "action": "HOLD",
                    "open_trade": open_trade,
                    "current_price": current_price,
                    "unrealized_pnl_pct": pnl_pct,
                    "rsi": current_rsi
                }
            
            return {
                "action": "WAIT",
                "current_price": current_price,
                "rsi": current_rsi
            }
            
        except Exception as e:
            logger.error(f"Error processing signals: {e}")
            raise
        finally:
            self.close_exchange()

# Global bot instance
bot = ScalperBot() - current_price
            if profit >= atr:
                current_trailing = trade.get("trailing_stop", entry_price)
                new_trailing = min(current_trailing, current_price + atr)
                
                if new_trailing < current_trailing:
                    trade["trailing_stop"] = new_trailing
                    state_manager.set("open_trade", trade)
                
                if current_price >= new_trailing:
                    return {"reason": "Trailing Stop", "price": current_price}
        
        return None

# ==================== MAIN BOT CLASS ====================
class ScalperBot:
    """Main bot orchestrator"""
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.signal_generator = SignalGenerator()
    
    async def process_signals(self) -> Dict[str, Any]:
        """Main signal processing logic"""
        try:
            # Fetch data
            df = await self.data_fetcher.fetch_ohlcv()
            df = self.signal_generator.indicator_engine.calculate_all(df)
            
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "current_price": float(df.iloc[-1]["close"]),
                "rsi": float(df.iloc[-1]["rsi"]) if not pd.isna(df.iloc[-1]["rsi"]) else None
            }
            
            # Check for exit signal first
            exit_signal = self.signal_generator.generate_exit_signal(df)
            if exit_signal:
                trade = state_manager.get("open_trade")
                entry_price = trade["entry_price"]
                exit_price = exit_signal["price"]
                side = trade["side"]
                
                # Calculate PnL
                if side == "BUY":
                    pnl = exit_price - entry_price
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                else:
                    pnl = entry_price - exit_price
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                
                # Update state
                total_trades = state_manager.get("total_trades", 0) + 1
                total_pnl = state_manager.get("total_pnl", 0.0) + pnl
                daily_trades = state_manager.get("daily_trades", 0) + 1
                
                state_manager.update({
                    "open_trade": None,
                    "total_trades": total_trades,
                    "total_pnl": total_pnl,
                    "daily_trades": daily_trades
                })
                
                result.update({
                    "action": "EXIT",
                    "reason": exit_signal["reason"],
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "side": side
                })
                
                logger.info(f"EXIT: {side} @ {exit_price:.2f} | PnL: {pnl:+.2f} ({pnl_pct:+.2f}%)")
                return result
            
            # Check for entry signal if no open trade
            open_trade = state_manager.get("open_trade")
            if not open_trade:
                entry_signal = self.signal_generator.generate_entry_signal(df)
                if entry_signal:
                    trade_data = {
                        "side": entry_signal["side"],
                        "entry_price": entry_signal["price"],
                        "atr": entry_signal["atr"],
                        "entry_time": entry_signal["timestamp"],
                        "entry_rsi": entry_signal["rsi"]
                    }
                    
                    state_manager.set("open_trade", trade_data)
                    
                    # Calculate targets
                    entry_price = entry_signal["price"]
                    atr = entry_signal["atr"]
                    
                    if entry_signal["side"] == "BUY":
                        stop_loss = entry_price - atr
                        take_profit = entry_price + (2 * atr)
                    else:
                        stop_loss = entry_price + atr
                        take_profit = entry_price - (2 * atr)
                    
                    result.update({
                        "action": "ENTRY",
                        "side": entry_signal["side"],
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "atr": atr
                    })
                    
                    logger.info(f"ENTRY: {entry_signal['side']} @ {entry_price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
                    return result
            
            # No signal - return status
            result["action"] = "HOLD" if open_trade else "WAIT"
            if open_trade:
                entry_price = open_trade["entry_price"]
                current_price = result["current_price"]
                side = open_trade["side"]
                
                if side == "BUY":
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                
                result.update({
                    "open_trade": open_trade,
                    "unrealized_pnl_pct": pnl_pct
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing signals: {e}")
            raise
        finally:
            await self.data_fetcher.close()

# Global bot instance
bot = ScalperBot()

# ==================== TELEGRAM HANDLERS ====================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    if config.allowed_chat_ids and update.effective_chat.id not in config.allowed_chat_ids:
        await update.message.reply_text("❌ Unauthorized access")
        return
    
    welcome_msg = (
        "🤖 **Scalper Bot v2.0 Active!**\n\n"
        "📊 **Commands:**\n"
        "/check - Check signals and position\n"
        "/status - Trading statistics\n"
        "/config - Bot configuration\n\n"
        f"📈 Trading: {config.symbol} ({config.timeframe})\n"
        f"⏰ Session: {config.session_start_utc}:00-{config.session_end_utc}:00 UTC"
    )
    
    await update.message.reply_text(welcome_msg, parse_mode="Markdown")

async def check_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /check command"""
    if config.allowed_chat_ids and update.effective_chat.id not in config.allowed_chat_ids:
        return
    
    try:
        result = await bot.process_signals()
        
        if result["action"] == "ENTRY":
            msg = (
                f"✅ **ENTRY SIGNAL**\n"
                f"Side: {result['side']}\n"
                f"Price: ${result['entry_price']:.2f}\n"
                f"Stop Loss: ${result['stop_loss']:.2f}\n"
                f"Take Profit: ${result['take_profit']:.2f}\n"
                f"ATR: ${result['atr']:.2f}"
            )
        
        elif result["action"] == "EXIT":
            pnl_emoji = "🟢" if result['pnl'] > 0 else "🔴"
            msg = (
                f"❌ **EXIT SIGNAL**\n"
                f"Reason: {result['reason']}\n"
                f"Exit Price: ${result['exit_price']:.2f}\n"
                f"PnL: {pnl_emoji} ${result['pnl']:+.2f} ({result['pnl_pct']:+.2f}%)"
            )
        
        elif result["action"] == "HOLD":
            trade = result["open_trade"]
            pnl_emoji = "🟢" if result['unrealized_pnl_pct'] > 0 else "🔴"
            msg = (
                f"📊 **HOLDING POSITION**\n"
                f"Side: {trade['side']}\n"
                f"Entry: ${trade['entry_price']:.2f}\n"
                f"Current: ${result['current_price']:.2f}\n"
                f"PnL: {pnl_emoji} {result['unrealized_pnl_pct']:+.2f}%"
            )
            
            if "trailing_stop" in trade:
                msg += f"\nTrailing: ${trade['trailing_stop']:.2f}"
        
        else:  # WAIT
            msg = (
                f"⏳ **WAITING FOR SIGNAL**\n"
                f"Price: ${result['current_price']:.2f}\n"
                f"RSI: {result['rsi']:.1f}" if result['rsi'] else "RSI: N/A"
            )
        
        await update.message.reply_text(msg, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Error in check command: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    if config.allowed_chat_ids and update.effective_chat.id not in config.allowed_chat_ids:
        return
    
    total_trades = state_manager.get("total_trades", 0)
    total_pnl = state_manager.get("total_pnl", 0.0)
    daily_trades = state_manager.get("daily_trades", 0)
    open_trade = state_manager.get("open_trade")
    
    win_rate = "N/A"
    avg_pnl = "N/A"
    
    if total_trades > 0:
        avg_pnl = f"${total_pnl / total_trades:.2f}"
    
    msg = (
        f"📈 **TRADING STATISTICS**\n"
        f"Total Trades: {total_trades}\n"
        f"Total PnL: ${total_pnl:+.2f}\n"
        f"Avg PnL/Trade: {avg_pnl}\n"
        f"Daily Trades: {daily_trades}/{config.max_daily_trades}\n"
        f"Position: {'Open' if open_trade else 'Closed'}"
    )
    
    await update.message.reply_text(msg, parse_mode="Markdown")

async def config_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /config command"""
    if config.allowed_chat_ids and update.effective_chat.id not in config.allowed_chat_ids:
        return
    
    current_hour = datetime.now(timezone.utc).hour
    in_session = config.session_start_utc <= current_hour <= config.session_end_utc
    
    msg = (
        f"⚙️ **BOT CONFIGURATION**\n"
        f"Symbol: {config.symbol}\n"
        f"Timeframe: {config.timeframe}\n"
        f"Session Filter: {'ON' if config.session_filter else 'OFF'}\n"
        f"Trading Hours: {config.session_start_utc}:00-{config.session_end_utc}:00 UTC\n"
        f"Current Time: {current_hour}:00 UTC\n"
        f"In Session: {'✅' if in_session else '❌'}\n"
        f"Max Daily Trades: {config.max_daily_trades}\n"
        f"Max Position: ${config.max_position_size}"
    )
    
    await update.message.reply_text(msg, parse_mode="Markdown")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle bot errors"""
    logger.error(f"Update {update} caused error {context.error}")

# ==================== MAIN ENTRY POINT ====================
def main():
    """Main entry point - Render compatible"""
    if not TELEGRAM_TOKEN:
        logger.error("❌ TELEGRAM_TOKEN environment variable not set!")
        return
    
    if not ALLOWED_CHAT_IDS:
        logger.warning("⚠️ TELEGRAM_CHAT_IDS not set - bot will accept all users!")
    
    logger.info("🚀 Starting Scalper Bot v2.0...")
    
    try:
        # Create application
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Add handlers
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("check", check_command))
        app.add_handler(CommandHandler("status", status_command))
        app.add_handler(CommandHandler("config", config_command))
        app.add_error_handler(error_handler)
        
        logger.info("✅ Bot handlers registered")
        logger.info("🔄 Starting polling...")
        
        # Run the bot (Render compatible)
        app.run_polling(
            drop_pending_updates=True,
            close_loop=False
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
