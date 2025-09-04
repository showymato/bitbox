
import asyncio
import json
import os
import logging
import websocket
import requests
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import math
import ccxt
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from cryptography.fernet import Fernet
import warnings
warnings.filterwarnings('ignore')

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    symbol: str = "BTCUSDT"
    timeframe: str = "5m"
    risk_per_trade_pct: float = 1.0
    max_positions: int = 3
    max_consecutive_losses: int = 3
    emergency_rsi_high: float = 80.0
    emergency_rsi_low: float = 20.0
    use_testnet: bool = True
    telegram_token: str = ""
    binance_api_key: str = ""
    binance_secret: str = ""
    encryption_key: str = ""
    allowed_chat_ids: List[int] = None
    coingecko_api_key: str = ""

    def __post_init__(self):
        if self.allowed_chat_ids is None:
            self.allowed_chat_ids = []

@dataclass
class Signal:
    timestamp: datetime
    signal_type: str  # "LONG", "SHORT", "CLOSE"
    strategy: str
    price: float
    confidence: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_amount: float = 0.0
    atr_value: float = 0.0

@dataclass
class Position:
    id: str
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    trailing_stop: float
    pnl: float
    timestamp: datetime
    strategy: str
    consecutive_losses: int = 0

@dataclass
class MarketData:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class EncryptedStorage:
    def __init__(self, encryption_key: str):
        if encryption_key and len(encryption_key) >= 32:
            self.cipher = Fernet(encryption_key.encode() if len(encryption_key) == 44 else Fernet.generate_key())
        else:
            self.cipher = Fernet(Fernet.generate_key())
        self.data_dir = "state"
        os.makedirs(self.data_dir, exist_ok=True)

    def save(self, filename: str, data: dict):
        try:
            json_data = json.dumps(data, default=str, indent=2)
            encrypted_data = self.cipher.encrypt(json_data.encode())
            filepath = os.path.join(self.data_dir, f"{filename}.enc")

            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                f.write(encrypted_data)
            os.replace(temp_path, filepath)
            logger.info(f"Saved encrypted data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")

    def load(self, filename: str) -> dict:
        try:
            filepath = os.path.join(self.data_dir, f"{filename}.enc")
            if not os.path.exists(filepath):
                return {}

            with open(filepath, 'rb') as f:
                encrypted_data = f.read()

            decrypted_data = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return {}

class TechnicalIndicators:
    @staticmethod
    def ema(prices: List[float], period: int, adjust: bool = False) -> List[float]:
        """Calculate EMA with adjust=False for deterministic results"""
        if len(prices) < period:
            return [0.0] * len(prices)

        multiplier = 2 / (period + 1)
        ema_values = [0.0] * len(prices)

        # Initialize with SMA
        sma = sum(prices[:period]) / period
        ema_values[period - 1] = sma

        # Calculate EMA
        for i in range(period, len(prices)):
            if not adjust:
                ema_values[i] = (prices[i] * multiplier) + (ema_values[i-1] * (1 - multiplier))
            else:
                ema_values[i] = (prices[i] * multiplier) + (ema_values[i-1] * (1 - multiplier))

        return ema_values

    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[float]:
        """Calculate RSI using Wilder smoothing"""
        if len(prices) < period + 1:
            return [50.0] * len(prices)

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            gains.append(max(change, 0))
            losses.append(abs(min(change, 0)))

        rsi_values = [50.0] * len(prices)

        if len(gains) < period:
            return rsi_values

        # Initial average gain/loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        for i in range(period, len(gains)):
            # Wilder smoothing
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi_values[i + 1] = 100 - (100 / (1 + rs))
            else:
                rsi_values[i + 1] = 100

        return rsi_values

    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
        """Calculate ATR (Average True Range)"""
        if len(closes) < 2:
            return [0.0] * len(closes)

        true_ranges = [0.0]

        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            true_ranges.append(max(high_low, high_close, low_close))

        atr_values = [0.0] * len(closes)

        if len(true_ranges) >= period:
            # Initial ATR
            atr_values[period - 1] = sum(true_ranges[:period]) / period

            # Wilder smoothing for subsequent values
            for i in range(period, len(true_ranges)):
                atr_values[i] = (atr_values[i-1] * (period - 1) + true_ranges[i]) / period

        return atr_values

    @staticmethod
    def vwap(highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> List[float]:
        """Calculate VWAP (Volume Weighted Average Price)"""
        vwap_values = []
        cum_volume = 0
        cum_typical_volume = 0

        for i in range(len(closes)):
            typical_price = (highs[i] + lows[i] + closes[i]) / 3
            cum_typical_volume += typical_price * volumes[i]
            cum_volume += volumes[i]

            if cum_volume > 0:
                vwap_values.append(cum_typical_volume / cum_volume)
            else:
                vwap_values.append(closes[i])

        return vwap_values

    @staticmethod
    def sma(prices: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average"""
        sma_values = [0.0] * len(prices)

        for i in range(period - 1, len(prices)):
            sma_values[i] = sum(prices[i - period + 1:i + 1]) / period

        return sma_values

class StrategyEngine:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.data: List[MarketData] = []
        self.consecutive_losses = 0

    def update_data(self, kline_data: dict):
        """Update market data with new kline"""
        try:
            new_data = MarketData(
                timestamp=datetime.fromtimestamp(kline_data['timestamp'] / 1000),
                open=float(kline_data['open']),
                high=float(kline_data['high']),
                low=float(kline_data['low']),
                close=float(kline_data['close']),
                volume=float(kline_data['volume'])
            )

            self.data.append(new_data)

            # Keep only last 1000 bars
            if len(self.data) > 1000:
                self.data = self.data[-1000:]

        except Exception as e:
            logger.error(f"Error updating data: {e}")

    def calculate_indicators(self) -> Dict:
        """Calculate all technical indicators"""
        if len(self.data) < 200:
            return {}

        closes = [d.close for d in self.data]
        highs = [d.high for d in self.data]
        lows = [d.low for d in self.data]
        volumes = [d.volume for d in self.data]

        indicators = {
            'ema9': TechnicalIndicators.ema(closes, 9, adjust=False),
            'ema21': TechnicalIndicators.ema(closes, 21, adjust=False),
            'ema50': TechnicalIndicators.ema(closes, 50, adjust=False),
            'ema200': TechnicalIndicators.ema(closes, 200, adjust=False),
            'rsi': TechnicalIndicators.rsi(closes, 14),
            'atr': TechnicalIndicators.atr(highs, lows, closes, 14),
            'vwap': TechnicalIndicators.vwap(highs, lows, closes, volumes),
            'volume_sma20': TechnicalIndicators.sma(volumes, 20)
        }

        return indicators

    def canonical_hybrid_strategy(self, indicators: Dict) -> Optional[Signal]:
        """Canonical Hybrid Strategy - Main strategy"""
        if len(self.data) < 200:
            return None

        current_idx = -1
        prev_idx = -2

        current = self.data[current_idx]
        prev = self.data[prev_idx]

        # Get indicator values
        ema9_curr = indicators['ema9'][current_idx]
        ema9_prev = indicators['ema9'][prev_idx]
        ema21_curr = indicators['ema21'][current_idx]
        ema21_prev = indicators['ema21'][prev_idx]
        ema200_curr = indicators['ema200'][current_idx]
        vwap_curr = indicators['vwap'][current_idx]
        rsi_curr = indicators['rsi'][current_idx]
        atr_curr = indicators['atr'][current_idx]
        volume_sma20_curr = indicators['volume_sma20'][current_idx]

        # EMA crossover detection
        ema9_cross_up = ema9_prev <= ema21_prev and ema9_curr > ema21_curr
        ema9_cross_down = ema9_prev >= ema21_prev and ema9_curr < ema21_curr

        # Trend filters
        above_ema200 = current.close > ema200_curr
        above_vwap = current.close > vwap_curr

        # RSI conditions
        rsi_long_zone = 50 <= rsi_curr <= 70
        rsi_short_zone = 30 <= rsi_curr <= 50

        # Volume condition
        volume_confirm = current.volume > volume_sma20_curr

        # ATR chop filter
        recent_atr = [indicators['atr'][i] for i in range(-6, 0)]
        chop_filter = max(recent_atr) > (current.close * 0.002)  # 0.2% minimum volatility

        # Emergency RSI check
        if rsi_curr > self.config.emergency_rsi_high or rsi_curr < self.config.emergency_rsi_low:
            return None

        # LONG signal
        if (ema9_cross_up and above_ema200 and above_vwap and 
            rsi_long_zone and volume_confirm and chop_filter):

            stop_loss = current.close - (atr_curr * 1.0)
            take_profit = current.close + (atr_curr * 1.5)

            return Signal(
                timestamp=datetime.now(),
                signal_type="LONG",
                strategy="Canonical_Hybrid",
                price=current.close,
                confidence=0.8,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr_value=atr_curr
            )

        # SHORT signal
        elif (ema9_cross_down and not above_ema200 and not above_vwap and 
              rsi_short_zone and volume_confirm and chop_filter):

            stop_loss = current.close + (atr_curr * 1.0)
            take_profit = current.close - (atr_curr * 1.5)

            return Signal(
                timestamp=datetime.now(),
                signal_type="SHORT",
                strategy="Canonical_Hybrid",
                price=current.close,
                confidence=0.8,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr_value=atr_curr
            )

        return None

    def vwap_pullback_strategy(self, indicators: Dict) -> Optional[Signal]:
        """VWAP Pullback Mean-Reversion Strategy"""
        if len(self.data) < 50:
            return None

        current = self.data[-1]
        prev = self.data[-2]

        vwap_curr = indicators['vwap'][-1]
        atr_curr = indicators['atr'][-1]
        volume_sma20_curr = indicators['volume_sma20'][-1]

        # Price pullback to VWAP
        vwap_touch = abs(current.low - vwap_curr) / vwap_curr < 0.002  # 0.2% tolerance

        # Bullish rejection pattern (hammer/doji)
        body_size = abs(current.close - current.open)
        lower_shadow = current.open - current.low if current.close > current.open else current.close - current.low
        upper_shadow = current.high - current.close if current.close > current.open else current.high - current.open

        is_hammer = (lower_shadow > body_size * 2) and (upper_shadow < body_size * 0.5)
        is_bullish_engulfing = current.close > prev.high and current.open < prev.low

        # Volume spike
        volume_spike = current.volume > volume_sma20_curr * 1.2

        # Above VWAP trend
        above_vwap_trend = current.close > vwap_curr

        if vwap_touch and (is_hammer or is_bullish_engulfing) and volume_spike and above_vwap_trend:
            stop_loss = current.close - atr_curr
            take_profit = current.close + (atr_curr * 1.25)

            return Signal(
                timestamp=datetime.now(),
                signal_type="LONG",
                strategy="VWAP_Pullback",
                price=current.close,
                confidence=0.7,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr_value=atr_curr
            )

        return None

    def atr_breakout_strategy(self, indicators: Dict) -> Optional[Signal]:
        """ATR Breakout Trend-Following Strategy"""
        if len(self.data) < 50:
            return None

        current = self.data[-1]

        atr_curr = indicators['atr'][-1]
        ema200_curr = indicators['ema200'][-1]

        # ATR contraction check
        recent_atr = indicators['atr'][-10:]
        atr_contraction = atr_curr < (sum(recent_atr) / len(recent_atr)) * 0.8

        # Price breakout
        recent_highs = [d.high for d in self.data[-20:]]
        recent_lows = [d.low for d in self.data[-20:]]
        recent_high = max(recent_highs)
        recent_low = min(recent_lows)

        breakout_up = current.close > recent_high and current.close > ema200_curr
        breakout_down = current.close < recent_low and current.close < ema200_curr

        if atr_contraction and breakout_up:
            stop_loss = current.close - (atr_curr * 1.5)
            take_profit = current.close + (atr_curr * 2.0)

            return Signal(
                timestamp=datetime.now(),
                signal_type="LONG",
                strategy="ATR_Breakout",
                price=current.close,
                confidence=0.75,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr_value=atr_curr
            )

        elif atr_contraction and breakout_down:
            stop_loss = current.close + (atr_curr * 1.5)
            take_profit = current.close - (atr_curr * 2.0)

            return Signal(
                timestamp=datetime.now(),
                signal_type="SHORT",
                strategy="ATR_Breakout",
                price=current.close,
                confidence=0.75,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr_value=atr_curr
            )

        return None

    def rsi_divergence_strategy(self, indicators: Dict) -> Optional[Signal]:
        """RSI Divergence Reversal Strategy"""
        if len(self.data) < 50:
            return None

        current = self.data[-1]

        # Look for divergence over last 10 bars
        price_data = [d.close for d in self.data[-10:]]
        rsi_data = indicators['rsi'][-10:]

        # Simple divergence detection
        price_trend = price_data[-1] - price_data[0]
        rsi_trend = rsi_data[-1] - rsi_data[0]

        # Bullish divergence: price down, RSI up
        bullish_divergence = price_trend < 0 and rsi_trend > 0
        # Bearish divergence: price up, RSI down  
        bearish_divergence = price_trend > 0 and rsi_trend < 0

        ema200_curr = indicators['ema200'][-1]
        vwap_curr = indicators['vwap'][-1]
        atr_curr = indicators['atr'][-1]

        if bullish_divergence and current.close > ema200_curr and current.close > vwap_curr:
            stop_loss = current.close - (atr_curr * 0.8)
            take_profit = current.close + (atr_curr * 1.2)

            return Signal(
                timestamp=datetime.now(),
                signal_type="LONG",
                strategy="RSI_Divergence",
                price=current.close,
                confidence=0.6,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr_value=atr_curr
            )

        elif bearish_divergence and current.close < ema200_curr and current.close < vwap_curr:
            stop_loss = current.close + (atr_curr * 0.8)
            take_profit = current.close - (atr_curr * 1.2)

            return Signal(
                timestamp=datetime.now(),
                signal_type="SHORT",
                strategy="RSI_Divergence",
                price=current.close,
                confidence=0.6,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr_value=atr_curr
            )

        return None

    def generate_signals(self) -> List[Signal]:
        """Generate signals from all active strategies"""
        if len(self.data) < 200:
            return []

        indicators = self.calculate_indicators()
        if not indicators:
            return []

        signals = []

        # Run all strategies
        strategies = [
            self.canonical_hybrid_strategy,
            self.vwap_pullback_strategy,
            self.atr_breakout_strategy,
            self.rsi_divergence_strategy
        ]

        for strategy in strategies:
            try:
                signal = strategy(indicators)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error in strategy {strategy.__name__}: {e}")

        return signals

class RiskManager:
    def __init__(self, config: TradingConfig):
        self.config = config

    def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """Calculate position size based on risk management"""
        risk_amount = account_balance * (self.config.risk_per_trade_pct / 100)

        if signal.stop_loss == 0 or signal.price == 0:
            return 0

        price_diff = abs(signal.price - signal.stop_loss)
        if price_diff == 0:
            return 0

        # Position size = Risk Amount / Stop Loss Distance
        position_size = risk_amount / price_diff
        return round(position_size, 6)

    def should_enter_trade(self, signal: Signal, current_positions: List[Position], consecutive_losses: int) -> bool:
        """Determine if we should enter a new trade"""
        # Check max positions
        if len(current_positions) >= self.config.max_positions:
            return False

        # Check consecutive losses
        if consecutive_losses >= self.config.max_consecutive_losses:
            return False

        # Check if already have position in same direction for same strategy
        same_strategy_direction = [p for p in current_positions 
                                 if p.side == signal.signal_type and p.strategy == signal.strategy]
        if len(same_strategy_direction) >= 1:
            return False

        return True

class MarketDataManager:
    def __init__(self, symbol: str = "btcusdt"):
        self.symbol = symbol.lower()
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@kline_5m"
        self.callbacks = []
        self.running = False
        self.ws = None

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def on_message(self, ws, message):
        """Handle WebSocket message"""
        try:
            data = json.loads(message)
            kline = data.get('k', {})

            # Only process completed klines
            if kline.get('x'):  # kline is closed
                kline_data = {
                    'timestamp': kline['t'],
                    'open': kline['o'],
                    'high': kline['h'],
                    'low': kline['l'],
                    'close': kline['c'],
                    'volume': kline['v']
                }

                # Notify all callbacks
                for callback in self.callbacks:
                    try:
                        asyncio.run_coroutine_threadsafe(callback(kline_data), asyncio.get_event_loop())
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket connection closed")
        if self.running:
            # Reconnect after 5 seconds
            threading.Timer(5.0, self.start).start()

    def on_open(self, ws):
        logger.info(f"Connected to Binance WebSocket for {self.symbol.upper()}")

    def start(self):
        """Start WebSocket connection in a separate thread"""
        if self.running:
            return

        self.running = True
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )

            # Run in a separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()

        except Exception as e:
            logger.error(f"Error starting WebSocket: {e}")

    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()

class TradingBot:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.storage = EncryptedStorage(config.encryption_key)
        self.strategy_engine = StrategyEngine(config)
        self.risk_manager = RiskManager(config)
        self.market_data = MarketDataManager(config.symbol)

        # Initialize exchange (for future live trading)
        try:
            self.exchange = ccxt.binance({
                'apiKey': config.binance_api_key,
                'secret': config.binance_secret,
                'sandbox': config.use_testnet,
                'enableRateLimit': True,
            }) if config.binance_api_key else None
        except Exception as e:
            logger.warning(f"Exchange not initialized: {e}")
            self.exchange = None

        # State
        self.positions: List[Position] = []
        self.running = False
        self.account_balance = 10000.0  # Demo balance
        self.consecutive_losses = 0

        # Load saved state
        self.load_state()

        # Setup market data callback
        self.market_data.add_callback(self.on_new_kline)

    async def on_new_kline(self, kline_data):
        """Handle new kline data"""
        try:
            logger.info(f"New 5m candle: {kline_data['close']} | Volume: {kline_data['volume']}")

            self.strategy_engine.update_data(kline_data)

            # Generate signals
            signals = self.strategy_engine.generate_signals()

            for signal in signals:
                await self.process_signal(signal)

            # Update existing positions
            await self.update_positions(float(kline_data['close']))

        except Exception as e:
            logger.error(f"Error processing kline: {e}")

    async def process_signal(self, signal: Signal):
        """Process trading signal"""
        try:
            if not self.risk_manager.should_enter_trade(signal, self.positions, self.consecutive_losses):
                logger.info(f"Signal rejected by risk manager: {signal.strategy}")
                return

            position_size = self.risk_manager.calculate_position_size(signal, self.account_balance)

            if position_size > 0:
                logger.info(f"Processing {signal.signal_type} signal from {signal.strategy}")
                await self.simulate_trade(signal, position_size)

        except Exception as e:
            logger.error(f"Error processing signal: {e}")

    async def simulate_trade(self, signal: Signal, position_size: float):
        """Simulate trade execution (paper trading)"""
        position = Position(
            id=f"{signal.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=self.config.symbol,
            side=signal.signal_type,
            size=position_size,
            entry_price=signal.price,
            current_price=signal.price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            trailing_stop=signal.stop_loss,
            pnl=0.0,
            timestamp=datetime.now(),
            strategy=signal.strategy
        )

        self.positions.append(position)
        self.save_state()

        # Send Telegram notification
        await self.send_telegram_message(
            f"🟢 **NEW POSITION**\n"
            f"Strategy: {signal.strategy}\n"
            f"Side: {signal.signal_type}\n"
            f"Price: ${signal.price:.2f}\n"
            f"Size: {position_size:.6f} BTC\n"
            f"SL: ${signal.stop_loss:.2f}\n"
            f"TP: ${signal.take_profit:.2f}\n"
            f"Confidence: {signal.confidence:.1%}"
        )

    async def update_positions(self, current_price: float):
        """Update existing positions with current prices and manage exits"""
        if not self.positions:
            return

        for position in self.positions[:]:  # Copy to avoid modification during iteration
            position.current_price = current_price

            # Calculate PnL
            if position.side == "LONG":
                position.pnl = (current_price - position.entry_price) * position.size
            else:
                position.pnl = (position.entry_price - current_price) * position.size

            # Update trailing stop
            if position.side == "LONG":
                if position.pnl >= 0.75 * position.strategy_engine.calculate_indicators().get('atr', [0])[-1]:
                    # Move trailing stop to breakeven after 0.75x ATR profit
                    position.trailing_stop = max(position.trailing_stop, position.entry_price)

                    # Then trail at current_price - 0.5 * ATR
                    atr_current = position.strategy_engine.calculate_indicators().get('atr', [0])[-1] if hasattr(position, 'strategy_engine') else 0
                    if atr_current > 0:
                        new_trailing = current_price - (0.5 * atr_current)
                        position.trailing_stop = max(position.trailing_stop, new_trailing)

            # Check exit conditions
            should_close = False
            close_reason = ""

            if position.side == "LONG":
                if current_price <= position.trailing_stop:
                    should_close = True
                    close_reason = "Trailing Stop"
                elif current_price >= position.take_profit:
                    should_close = True
                    close_reason = "Take Profit"
            else:  # SHORT
                if current_price >= position.stop_loss:
                    should_close = True
                    close_reason = "Stop Loss"
                elif current_price <= position.take_profit:
                    should_close = True
                    close_reason = "Take Profit"

            if should_close:
                self.positions.remove(position)
                self.account_balance += position.pnl

                # Track consecutive losses
                if position.pnl < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0

                await self.send_telegram_message(
                    f"🔴 **POSITION CLOSED**\n"
                    f"Strategy: {position.strategy}\n"
                    f"Reason: {close_reason}\n"
                    f"Entry: ${position.entry_price:.2f}\n"
                    f"Exit: ${current_price:.2f}\n"
                    f"PnL: ${position.pnl:.2f}\n"
                    f"Balance: ${self.account_balance:.2f}\n"
                    f"Consecutive Losses: {self.consecutive_losses}"
                )

        self.save_state()

    def load_state(self):
        """Load bot state from encrypted storage"""
        try:
            state = self.storage.load("bot_state")
            if state:
                self.account_balance = state.get("account_balance", 10000.0)
                self.consecutive_losses = state.get("consecutive_losses", 0)
                positions_data = state.get("positions", [])
                self.positions = []
                for pos_data in positions_data:
                    if isinstance(pos_data, dict):
                        pos_data['timestamp'] = datetime.fromisoformat(pos_data['timestamp']) if isinstance(pos_data['timestamp'], str) else pos_data['timestamp']
                        self.positions.append(Position(**pos_data))
        except Exception as e:
            logger.error(f"Error loading state: {e}")

    def save_state(self):
        """Save bot state to encrypted storage"""
        try:
            state = {
                "account_balance": self.account_balance,
                "consecutive_losses": self.consecutive_losses,
                "positions": [asdict(pos) for pos in self.positions],
                "last_updated": datetime.now().isoformat()
            }
            self.storage.save("bot_state", state)
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    async def send_telegram_message(self, message: str):
        """Send message to Telegram"""
        try:
            bot = Bot(token=self.config.telegram_token)
            for chat_id in self.config.allowed_chat_ids:
                await bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode='MarkdownV2'
                )
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")

    def start(self):
        """Start the trading bot"""
        self.running = True
        logger.info("🚀 Starting Ultimate Bitcoin Trading Bot...")

        # Start market data in a separate thread
        self.market_data.start()

        logger.info("✅ Bot started successfully!")
        logger.info(f"📊 Account Balance: ${self.account_balance:.2f}")
        logger.info(f"📈 Open Positions: {len(self.positions)}")
        logger.info(f"🎯 Strategies Active: Canonical Hybrid, VWAP Pullback, ATR Breakout, RSI Divergence")

    def stop(self):
        """Stop the trading bot"""
        self.running = False
        self.market_data.stop()
        logger.info("🛑 Trading Bot stopped")

class TelegramHandler:
    def __init__(self, bot: TradingBot):
        self.trading_bot = bot

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        if update.effective_chat.id not in self.trading_bot.config.allowed_chat_ids:
            await update.message.reply_text("❌ Unauthorized access")
            return

        self.trading_bot.start()

        await update.message.reply_text(
            "🤖 **Ultimate Bitcoin Trading Bot Started**\n\n"
            "📊 **Active Strategies:**\n"
            "• Canonical Hybrid \(EMA9/21\+VWAP\+RSI\)\n"
            "• VWAP Pullback \(Mean Reversion\)\n"
            "• ATR Breakout \(Volatility\)\n"
            "• RSI Divergence \(Reversal\)\n\n"
            "📱 **Commands:**\n"
            "/status \- Bot status & balance\n"
            "/positions \- Open positions\n"
            "/balance \- Account balance\n"
            "/stop \- Stop bot\n"
            "/help \- Show commands\n\n"
            "🔄 **Collecting market data\.\.\.**",
            parse_mode='MarkdownV2'
        )

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        if update.effective_chat.id not in self.trading_bot.config.allowed_chat_ids:
            return

        status = "🟢 Running" if self.trading_bot.running else "🔴 Stopped"
        positions_count = len(self.trading_bot.positions)
        data_points = len(self.trading_bot.strategy_engine.data)

        await update.message.reply_text(
            f"📊 **Bot Status**\n\n"
            f"Status: {status}\n"
            f"Balance: ${self.trading_bot.account_balance:.2f}\n"
            f"Open Positions: {positions_count}\n"
            f"Data Points: {data_points}\n"
            f"Consecutive Losses: {self.trading_bot.consecutive_losses}\n"
            f"Symbol: {self.trading_bot.config.symbol}\n"
            f"Risk per Trade: {self.trading_bot.config.risk_per_trade_pct}%",
            parse_mode='MarkdownV2'
        )

    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command"""
        if update.effective_chat.id not in self.trading_bot.config.allowed_chat_ids:
            return

        if not self.trading_bot.positions:
            await update.message.reply_text("📈 No open positions")
            return

        message = "📈 **Open Positions:**\n\n"
        for i, pos in enumerate(self.trading_bot.positions, 1):
            pnl_emoji = "🟢" if pos.pnl >= 0 else "🔴"
            message += (
                f"{i}\. {pos.strategy}\n"
                f"Side: {pos.side}\n"
                f"Size: {pos.size:.6f} BTC\n"
                f"Entry: ${pos.entry_price:.2f}\n"
                f"Current: ${pos.current_price:.2f}\n"
                f"{pnl_emoji} PnL: ${pos.pnl:.2f}\n"
                f"SL: ${pos.stop_loss:.2f}\n"
                f"TP: ${pos.take_profit:.2f}\n"
                f"────────────────\n"
            )

        await update.message.reply_text(message, parse_mode='MarkdownV2')

    async def balance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command"""
        if update.effective_chat.id not in self.trading_bot.config.allowed_chat_ids:
            return

        total_pnl = sum(pos.pnl for pos in self.trading_bot.positions)

        await update.message.reply_text(
            f"💰 **Account Balance**\n\n"
            f"Balance: ${self.trading_bot.account_balance:.2f}\n"
            f"Unrealized PnL: ${total_pnl:.2f}\n"
            f"Total Equity: ${self.trading_bot.account_balance + total_pnl:.2f}\n"
            f"Open Positions: {len(self.trading_bot.positions)}",
            parse_mode='MarkdownV2'
        )

    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command"""
        if update.effective_chat.id not in self.trading_bot.config.allowed_chat_ids:
            return

        self.trading_bot.stop()
        await update.message.reply_text("🛑 **Trading Bot Stopped**")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        if update.effective_chat.id not in self.trading_bot.config.allowed_chat_ids:
            return

        await update.message.reply_text(
            "📚 **Available Commands:**\n\n"
            "/start \- Initialize and start bot\n"
            "/status \- View bot status & metrics\n"
            "/positions \- Check open positions\n"
            "/balance \- Account balance info\n"
            "/stop \- Emergency stop\n"
            "/help \- Show this help\n\n"
            "🎯 **Trading Strategies:**\n"
            "• Canonical Hybrid\n"
            "• VWAP Pullback\n"
            "• ATR Breakout\n"
            "• RSI Divergence\n\n"
            "⚠️ **Note:** Bot runs in testnet mode by default",
            parse_mode='MarkdownV2'
        )

    def setup_handlers(self, application):
        """Setup command handlers"""
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("status", self.status_command))
        application.add_handler(CommandHandler("positions", self.positions_command))
        application.add_handler(CommandHandler("balance", self.balance_command))
        application.add_handler(CommandHandler("stop", self.stop_command))
        application.add_handler(CommandHandler("help", self.help_command))

async def main():
    """Main function"""
    # Load configuration from environment variables
    config = TradingConfig(
        telegram_token=os.getenv("TELEGRAM_TOKEN", ""),
        binance_api_key=os.getenv("BINANCE_API_KEY", ""),
        binance_secret=os.getenv("BINANCE_SECRET", ""),
        encryption_key=os.getenv("ENCRYPTION_KEY", ""),
        allowed_chat_ids=[int(x) for x in os.getenv("ALLOWED_CHAT_IDS", "").split(",") if x.strip()],
        coingecko_api_key=os.getenv("COINGECKO_API_KEY", "")
    )

    if not config.telegram_token:
        logger.error("❌ TELEGRAM_TOKEN environment variable required")
        return

    if not config.allowed_chat_ids:
        logger.error("❌ ALLOWED_CHAT_IDS environment variable required")
        return

    logger.info("🚀 Starting Ultimate Bitcoin Trading Bot...")
    logger.info(f"📊 Config: {config.symbol} | Risk: {config.risk_per_trade_pct}% | Testnet: {config.use_testnet}")

    # Create and setup bot
    trading_bot = TradingBot(config)
    telegram_handler = TelegramHandler(trading_bot)

    # Setup Telegram application
    app = Application.builder().token(config.telegram_token).build()
    telegram_handler.setup_handlers(app)

    # Start Telegram bot
    try:
        await app.initialize()
        await app.start()

        logger.info("✅ Telegram bot ready!")
        logger.info("📱 Message /start to begin trading")

        # Start polling
        await app.updater.start_polling()

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    finally:
        trading_bot.stop()
        if app.updater.running:
            await app.updater.stop()
        await app.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
