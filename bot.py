import asyncio
import os
import logging
import requests
import ccxt
import numpy as np
from datetime import datetime
from telegram import Bot
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ScalperBot5Min:
    def __init__(self, telegram_token: str, allowed_chat_ids: List[int]):
        self.token = telegram_token
        self.allowed_chat_ids = allowed_chat_ids
        self.ccxt_binance = ccxt.binance()
        self.symbol = "BTC/USDT"
        self.timeframe = "5m"
        self.closes = []
        self.highs = []
        self.lows = []
        self.volumes = []
        self.in_position = False
        self.position_side = None
        self.entry_price = None

    # ---------- Data Fetching ----------
    def fetch_historical_data(self):
        try:
            ohlcv = self.ccxt_binance.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=200)
            self.closes = [c[4] for c in ohlcv]
            self.highs = [c[2] for c in ohlcv]
            self.lows = [c[3] for c in ohlcv]
            self.volumes = [c[5] for c in ohlcv]
            logger.info("Fetched historical data from Binance")
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")

    def fetch_coingecko_price(self):
        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()["bitcoin"]["usd"]
        except Exception as e:
            logger.error(f"Error fetching price from CoinGecko: {e}")
            return None

    # ---------- Indicators ----------
    def ema(self, prices: List[float], period: int) -> List[float]:
        ema_values = []
        k = 2 / (period + 1)
        for i in range(len(prices)):
            if i < period - 1:
                ema_values.append(None)
            elif i == period - 1:
                sma = sum(prices[:period]) / period
                ema_values.append(sma)
            else:
                ema = (prices[i] - ema_values[-1]) * k + ema_values[-1]
                ema_values.append(ema)
        return ema_values

    def rsi(self, prices: List[float], period: int = 14) -> List[float]:
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            upval = max(delta, 0)
            downval = -min(delta, 0)
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        return rsi

    def atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            trs.append(tr)
        atr = [0] * period
        for i in range(period, len(trs)):
            atr.append(np.mean(trs[i - period:i]))
        return atr

    def volume_spike(self, volumes, lookback=20):
        avg_vol = np.mean(volumes[-lookback:])
        return volumes[-1] > avg_vol

    def vwap(self):
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(self.highs, self.lows, self.closes)]
        cum_tp_vol = np.cumsum([tp * v for tp, v in zip(typical_prices, self.volumes)])
        cum_vol = np.cumsum(self.volumes)
        return cum_tp_vol[-1] / cum_vol[-1]

    # ---------- Alerts ----------
    async def send_telegram(self, message: str):
        try:
            bot = Bot(token=self.token)
            for chat_id in self.allowed_chat_ids:
                await bot.send_message(chat_id=chat_id, text=message)
        except Exception as e:
            logger.error(f"Failed to send telegram message: {e}")

    # ---------- Strategy ----------
    async def analyze_and_trade(self):
        if len(self.closes) < 200:
            return

        ema9 = self.ema(self.closes, 9)[-1]
        ema21 = self.ema(self.closes, 21)[-1]
        ema50 = self.ema(self.closes, 50)[-1]
        ema200 = self.ema(self.closes, 200)[-1]
        rsi_last = self.rsi(self.closes)[-1]
        atr_last = self.atr(self.highs, self.lows, self.closes)[-1]
        vwap = self.vwap()
        price = self.closes[-1]
        vol_ok = self.volume_spike(self.volumes)

        # ---------- Entry ----------
        if not self.in_position:
            # Long setup
            if ema9 > ema21 and ema21 > ema50 and price > vwap and 40 < rsi_last < 65 and vol_ok:
                sl = price - atr_last
                tp = price + atr_last * 2
                self.in_position = True
                self.position_side = "BUY"
                self.entry_price = price
                await self.send_telegram(f"🟢 BUY {price:.2f} | SL {sl:.2f} | TP {tp:.2f}")

            # Short setup
            elif ema9 < ema21 and ema21 < ema50 and price < vwap and 35 < rsi_last < 60 and vol_ok:
                sl = price + atr_last
                tp = price - atr_last * 2
                self.in_position = True
                self.position_side = "SELL"
                self.entry_price = price
                await self.send_telegram(f"🔴 SELL {price:.2f} | SL {sl:.2f} | TP {tp:.2f}")

        # ---------- Exit ----------
        elif self.in_position:
            if self.position_side == "BUY":
                if price <= self.entry_price - atr_last:
                    self.in_position = False
                    await self.send_telegram(f"🔺 EXIT BUY {price:.2f} | SL hit")
                elif price >= self.entry_price + atr_last * 2:
                    self.in_position = False
                    await self.send_telegram(f"✅ EXIT BUY {price:.2f} | TP hit")
                elif rsi_last > 75:
                    self.in_position = False
                    await self.send_telegram(f"⚡ EXIT BUY {price:.2f} | RSI Overbought")

            elif self.position_side == "SELL":
                if price >= self.entry_price + atr_last:
                    self.in_position = False
                    await self.send_telegram(f"🔺 EXIT SELL {price:.2f} | SL hit")
                elif price <= self.entry_price - atr_last * 2:
                    self.in_position = False
                    await self.send_telegram(f"✅ EXIT SELL {price:.2f} | TP hit")
                elif rsi_last < 25:
                    self.in_position = False
                    await self.send_telegram(f"⚡ EXIT SELL {price:.2f} | RSI Oversold")


# ---------- Runner ----------
async def main():
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
    ALLOWED_CHAT_IDS = os.getenv("ALLOWED_CHAT_IDS", "")

    if not TELEGRAM_TOKEN or not ALLOWED_CHAT_IDS:
        logger.error("Missing TELEGRAM_TOKEN or ALLOWED_CHAT_IDS environment variables")
        return

    allowed_ids = [int(x) for x in ALLOWED_CHAT_IDS.split(",")]
    bot = ScalperBot5Min(TELEGRAM_TOKEN, allowed_ids)

    bot.fetch_historical_data()

    while True:
        bot.fetch_historical_data()  # refresh candles
        await bot.analyze_and_trade()
        await asyncio.sleep(300)  # run every 5 min


if __name__ == "__main__":
    asyncio.run(main())
