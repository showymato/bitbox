import asyncio
import json
import os
import logging
import requests
import ccxt
import numpy as np
from datetime import datetime
from telegram import Bot
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScalperBot5Min:
    def __init__(self, telegram_token: str, allowed_chat_ids: List[int]):
        self.token = telegram_token
        self.allowed_chat_ids = allowed_chat_ids
        # ✅ Use Bybit instead of Binance (works in restricted regions)
        self.exchange = ccxt.bybit()
        self.symbol = 'BTC/USDT'
        self.timeframe = '5m'
        self.closes = []
        self.highs = []
        self.lows = []
        self.volumes = []
        self.in_position = False
        self.position_side = None
        self.entry_price = None

    def fetch_historical_data(self):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=200)
            self.closes = [candle[4] for candle in ohlcv]
            self.highs = [candle[2] for candle in ohlcv]
            self.lows = [candle[3] for candle in ohlcv]
            self.volumes = [candle[5] for candle in ohlcv]
            logger.info('✅ Fetched historical data from Bybit')
        except Exception as e:
            logger.error(f'Error fetching historical data: {e}')

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
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            trs.append(tr)
        atr = [0] * (period)
        for i in range(period, len(trs)):
            atr.append(np.mean(trs[i-period:i]))
        return atr

    async def send_telegram(self, message: str):
        try:
            bot = Bot(token=self.token)
            for chat_id in self.allowed_chat_ids:
                await bot.send_message(chat_id=chat_id, text=message)
        except Exception as e:
            logger.error(f'Failed to send telegram message: {e}')

    async def analyze_and_trade(self):
        if len(self.closes) < 50:
            return
        ema7 = self.ema(self.closes, 7)[-1]
        ema21 = self.ema(self.closes, 21)[-1]
        rsi_last = self.rsi(self.closes)[-1]
        atr_last = self.atr(self.highs, self.lows, self.closes)[-1]
        price = self.closes[-1]

        # Entry Logic
        if not self.in_position:
            if ema7 and ema21 and rsi_last:
                if ema7 > ema21 and rsi_last < 60:
                    sl = price - atr_last
                    tp = price + atr_last * 1.5
                    self.in_position = True
                    self.position_side = 'BUY'
                    self.entry_price = price
                    msg = f"🟢 BUY {price:.2f} SL {sl:.2f} TP {tp:.2f}"
                    await self.send_telegram(msg)
                elif ema7 < ema21 and rsi_last > 40:
                    sl = price + atr_last
                    tp = price - atr_last * 1.5
                    self.in_position = True
                    self.position_side = 'SELL'
                    self.entry_price = price
                    msg = f"🔴 SELL {price:.2f} SL {sl:.2f} TP {tp:.2f}"
                    await self.send_telegram(msg)

        # Exit Logic
        elif self.in_position:
            if self.position_side == 'BUY' and price <= self.entry_price - atr_last:
                self.in_position = False
                msg = f"🔺 EXIT BUY at {price:.2f} - Stop Loss hit"
                await self.send_telegram(msg)
            elif self.position_side == 'BUY' and price >= self.entry_price + atr_last * 1.5:
                self.in_position = False
                msg = f"✅ EXIT BUY at {price:.2f} - Take Profit hit"
                await self.send_telegram(msg)
            elif self.position_side == 'SELL' and price >= self.entry_price + atr_last:
                self.in_position = False
                msg = f"🔺 EXIT SELL at {price:.2f} - Stop Loss hit"
                await self.send_telegram(msg)
            elif self.position_side == 'SELL' and price <= self.entry_price - atr_last * 1.5:
                self.in_position = False
                msg = f"✅ EXIT SELL at {price:.2f} - Take Profit hit"
                await self.send_telegram(msg)


async def main():
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
    ALLOWED_CHAT_IDS = os.getenv('ALLOWED_CHAT_IDS', '')
    if not TELEGRAM_TOKEN or not ALLOWED_CHAT_IDS:
        logger.error('Missing TELEGRAM_TOKEN or ALLOWED_CHAT_IDS environment variables')
        return

    allowed_ids = [int(x) for x in ALLOWED_CHAT_IDS.split(",")]
    bot = ScalperBot5Min(TELEGRAM_TOKEN, allowed_ids)

    bot.fetch_historical_data()

    while True:
        await bot.analyze_and_trade()
        await asyncio.sleep(300)  # Every 5 minutes

if __name__ == '__main__':
    asyncio.run(main())
