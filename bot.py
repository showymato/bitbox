"""
Production Crypto Trading Bot - Pandas-Free Version
Compatible with Python 3.13, uses only basic libraries
"""

import os
import json
import logging
import math
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import ccxt
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Configuration
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
LIMIT = 500
SESSION_FILTER = True
SESSION_START_UTC = 7
SESSION_END_UTC = 20
MAX_DAILY_TRADES = 10
STATE_FILE = "bot_state.json"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ALLOWED_CHAT_IDS = [int(cid.strip()) for cid in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if cid.strip()]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ScalperBot")

def load_state():
    """Load bot state from file"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
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

def save_state(state):
    """Save bot state to file"""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving state: {e}")

bot_state = load_state()

class TechnicalAnalysis:
    """Technical analysis without pandas"""
    
    @staticmethod
    def ema(prices: List[float], period: int) -> List[float]:
        """Calculate EMA without pandas"""
        if len(prices) < period:
            return [None] * len(prices)
        
        multiplier = 2 / (period + 1)
        ema_values = [None] * (period - 1)
        
        # Start with SMA for first value
        ema_values.append(sum(prices[:period]) / period)
        
        for i in range(period, len(prices)):
            ema_values.append((prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier)))
        
        return ema_values
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[float]:
        """Calculate RSI without pandas"""
        if len(prices) < period + 1:
            return [None] * len(prices)
        
        rsi_values = [None] * period
        
        # Calculate initial gains and losses
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        
        # Calculate initial averages
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))
        
        # Calculate remaining RSI values
        for i in range(period + 1, len(prices)):
            gain = gains[i-1]
            loss = losses[i-1]
            
            avg_gain = ((avg_gain * (period - 1)) + gain) / period
            avg_loss = ((avg_loss * (period - 1)) + loss) / period
            
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))
        
        return rsi_values
    
    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
        """Calculate ATR without pandas"""
        if len(highs) < period + 1:
            return [None] * len(highs)
        
        atr_values = [None]
        
        # Calculate true ranges
        true_ranges = []
        for i in range(1, len(highs)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            true_ranges.append(max(hl, hc, lc))
        
        # Calculate initial ATR (simple average)
        initial_atr = sum(true_ranges[:period]) / period
        atr_values.extend([None] * (period - 1))
        atr_values.append(initial_atr)
        
        # Calculate remaining ATR values (smoothed)
        for i in range(period, len(true_ranges)):
            new_atr = (atr_values[-1] * (period - 1) + true_ranges[i]) / period
            atr_values.append(new_atr)
        
        return atr_values
    
    @staticmethod
    def vwap(highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> List[float]:
        """Calculate VWAP without pandas"""
        vwap_values = []
        cumulative_pv = 0
        cumulative_volume = 0
        
        for i in range(len(closes)):
            typical_price = (highs[i] + lows[i] + closes[i]) / 3
            pv = typical_price * volumes[i]
            
            cumulative_pv += pv
            cumulative_volume += volumes[i]
            
            if cumulative_volume > 0:
                vwap_values.append(cumulative_pv / cumulative_volume)
            else:
                vwap_values.append(closes[i])
        
        return vwap_values
    
    @staticmethod
    def sma(values: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average"""
        sma_values = [None] * (period - 1)
        
        for i in range(period - 1, len(values)):
            sma_values.append(sum(values[i-period+1:i+1]) / period)
        
        return sma_values

class ScalperBot:
    """Main trading bot class"""
    
    def __init__(self):
        self.exchange = None
        self.ta = TechnicalAnalysis()
        
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
            
            if len(ohlcv) < 200:
                raise ValueError(f"Insufficient data: {len(ohlcv)} candles (need 200+)")
            
            # Convert to separate lists for easier processing
            data = {
                'timestamps': [candle[0] for candle in ohlcv],
                'opens': [candle[1] for candle in ohlcv],
                'highs': [candle[2] for candle in ohlcv],
                'lows': [candle[3] for candle in ohlcv],
                'closes': [candle[4] for candle in ohlcv],
                'volumes': [candle[5] for candle in ohlcv]
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            raise
    
    def calculate_indicators(self, data):
        """Calculate all technical indicators"""
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        volumes = data['volumes']
        
        indicators = {
            'ema9': self.ta.ema(closes, 9),
            'ema21': self.ta.ema(closes, 21),
            'ema50': self.ta.ema(closes, 50),
            'ema200': self.ta.ema(closes, 200),
            'rsi': self.ta.rsi(closes, 14),
            'atr': self.ta.atr(highs, lows, closes, 14),
            'vwap': self.ta.vwap(highs, lows, closes, volumes),
            'volume_avg': self.ta.sma(volumes, 20)
        }
        
        return indicators
    
    def is_session_active(self):
        """Check if we're in active trading session"""
        if not SESSION_FILTER:
            return True
        
        current_hour = datetime.now(timezone.utc).hour
        return SESSION_START_UTC <= current_hour <= SESSION_END_UTC
    
    def validate_indicators(self, indicators, index):
        """Validate that indicators have values at given index"""
        required_fields = ['ema9', 'ema21', 'ema50', 'vwap', 'rsi', 'volume_avg', 'atr']
        return all(
            indicators[field][index] is not None 
            for field in required_fields 
            if index < len(indicators[field])
        )
    
    def check_entry_signal(self, data, indicators):
        """Check for entry signals"""
        if len(data['closes']) < 200 or not self.is_session_active():
            return None
        
        index = -1  # Last candle
        
        if not self.validate_indicators(indicators, index):
            return None
        
        # Check daily trade limit
        today = datetime.now(timezone.utc).date().isoformat()
        if bot_state.get("last_trade_date") != today:
            bot_state["daily_trades"] = 0
            bot_state["last_trade_date"] = today
            save_state(bot_state)
        
        if bot_state.get("daily_trades", 0) >= MAX_DAILY_TRADES:
            return None
        
        close = data['closes'][index]
        volume = data['volumes'][index]
        ema9 = indicators['ema9'][index]
        ema21 = indicators['ema21'][index]
        ema50 = indicators['ema50'][index]
        vwap = indicators['vwap'][index]
        rsi = indicators['rsi'][index]
        volume_avg = indicators['volume_avg'][index]
        atr = indicators['atr'][index]
        
        # Long signal
        if (ema9 > ema21 and 
            ema21 > ema50 and
            close > vwap and 
            40 <= rsi <= 65 and 
            volume > volume_avg):
            
            return {
                "side": "BUY",
                "price": float(close),
                "atr": float(atr),
                "timestamp": datetime.fromtimestamp(data['timestamps'][index]/1000, tz=timezone.utc).isoformat(),
                "rsi": float(rsi)
            }
        
        # Short signal
        if (ema9 < ema21 and 
            ema21 < ema50 and
            close < vwap and 
            35 <= rsi <= 60 and 
            volume > volume_avg):
            
            return {
                "side": "SELL",
                "price": float(close),
                "atr": float(atr),
                "timestamp": datetime.fromtimestamp(data['timestamps'][index]/1000, tz=timezone.utc).isoformat(),
                "rsi": float(rsi)
            }
        
        return None
    
    def check_exit_signal(self, data, indicators):
        """Check for exit signals"""
        trade = bot_state.get("open_trade")
        if not trade or len(data['closes']) < 1:
            return None
        
        index = -1
        current_price = float(data['closes'][index])
        rsi = indicators['rsi'][index] if indicators['rsi'][index] is not None else 50
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
            if rsi > 75:
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
            if rsi < 25:
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
            data = self.fetch_ohlcv()
            indicators = self.calculate_indicators(data)
            
            current_price = float(data['closes'][-1])
            current_rsi = indicators['rsi'][-1] if indicators['rsi'][-1] is not None else None
            
            # Check for exit signal first
            exit_signal = self.check_exit_signal(data, indicators)
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
                entry_signal = self.check_entry_signal(data, indicators)
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
bot = ScalperBot()

# Telegram handlers (same as before)
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if ALLOWED_CHAT_IDS and update.effective_chat.id not in ALLOWED_CHAT_IDS:
        await update.message.reply_text("❌ Unauthorized access")
        return
    
    welcome_msg = (
        "🤖 **Scalper Bot v2.0 Active!**\n\n"
        "📊 **Commands:**\n"
        "/check - Check signals and position\n"
        "/status - Trading statistics\n"
        "/config - Bot configuration\n\n"
        f"📈 Trading: {SYMBOL} ({TIMEFRAME})\n"
        f"⏰ Session: {SESSION_START_UTC}:00-{SESSION_END_UTC}:00 UTC"
    )
    
    await update.message.reply_text(welcome_msg, parse_mode="Markdown")

async def check_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if ALLOWED_CHAT_IDS and update.effective_chat.id not in ALLOWED_CHAT_IDS:
        return
    
    try:
        result = bot.process_signals()
        
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
    if ALLOWED_CHAT_IDS and update.effective_chat.id not in ALLOWED_CHAT_IDS:
        return
    
    total_trades = bot_state.get("total_trades", 0)
    total_pnl = bot_state.get("total_pnl", 0.0)
    daily_trades = bot_state.get("daily_trades", 0)
    open_trade = bot_state.get("open_trade")
    
    avg_pnl = f"${total_pnl / total_trades:.2f}" if total_trades > 0 else "N/A"
    
    msg = (
        f"📈 **TRADING STATISTICS**\n"
        f"Total Trades: {total_trades}\n"
        f"Total PnL: ${total_pnl:+.2f}\n"
        f"Avg PnL/Trade: {avg_pnl}\n"
        f"Daily Trades: {daily_trades}/{MAX_DAILY_TRADES}\n"
        f"Position: {'Open' if open_trade else 'Closed'}"
    )
    
    await update.message.reply_text(msg, parse_mode="Markdown")

async def config_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if ALLOWED_CHAT_IDS and update.effective_chat.id not in ALLOWED_CHAT_IDS:
        return
    
    current_hour = datetime.now(timezone.utc).hour
    in_session = SESSION_START_UTC <= current_hour <= SESSION_END_UTC
    
    msg = (
        f"⚙️ **BOT CONFIGURATION**\n"
        f"Symbol: {SYMBOL}\n"
        f"Timeframe: {TIMEFRAME}\n"
        f"Session Filter: {'ON' if SESSION_FILTER else 'OFF'}\n"
        f"Trading Hours: {SESSION_START_UTC}:00-{SESSION_END_UTC}:00 UTC\n"
        f"Current Time: {current_hour}:00 UTC\n"
        f"In Session: {'✅' if in_session else '❌'}\n"
        f"Max Daily Trades: {MAX_DAILY_TRADES}"
    )
    
    await update.message.reply_text(msg, parse_mode="Markdown")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")

def main():
    if not TELEGRAM_TOKEN:
        logger.error("❌ TELEGRAM_TOKEN environment variable not set!")
        return
    
    if not ALLOWED_CHAT_IDS:
        logger.warning("⚠️ TELEGRAM_CHAT_IDS not set - bot will accept all users!")
    
    logger.info("🚀 Starting Scalper Bot v2.0...")
    
    try:
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("check", check_command))
        app.add_handler(CommandHandler("status", status_command))
        app.add_handler(CommandHandler("config", config_command))
        app.add_error_handler(error_handler)
        
        logger.info("✅ Bot handlers registered")
        logger.info("🔄 Starting polling...")
        
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
