#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bot de Trading Algorítmico para SOLUSDT.
- Opera en Binance Futures usando el 95% del balance.
- Estrategia basada en cruce del RSI con su SMA, filtrado por EMA100 y trailing stop basado en ATR.
- Comandos vía Telegram: /estado, /start, /stop.
- Flags: --forcelong y --forceshort para pruebas.
"""

import os
import time
import threading
import logging
import argparse
import requests
import asyncio
import math
import pandas as pd
import ta

from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            logging.error(f"Error al enviar mensaje a Telegram: {response.text}")
    except Exception as e:
        logging.error(f"Excepción al enviar mensaje a Telegram: {e}")

class TradingBot:
    def __init__(self):
        self.client = Client(API_KEY, API_SECRET)
        try:
            server_time = self.client.futures_time()['serverTime']
            local_time = int(time.time() * 1000)
            self.client.time_offset = local_time - server_time
            logging.info(f"Time offset ajustado: {self.client.time_offset} ms")
        except Exception as e:
            logging.error(f"Error al sincronizar tiempo con Binance: {e}")

        self.symbol = "SOLUSDT"
        self.leverage = 10
        self.balance_pct = 0.95
        self.timeframe = "15m"
        self.rsi_period = 14
        self.ema_period = 100
        self.atr_period = 14
        self.atr_multiplier_sl = 1.5
        self.last_candle_time = None

        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.position_qty = None

        self.active = True
        self.status_info = "Bot iniciado. Sin posición abierta."
        send_telegram_message("✅ Bot de Trading iniciado.")

        try:
            self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
            logging.info(f"Leverage configurado a x{self.leverage} para {self.symbol}")
        except BinanceAPIException as e:
            logging.error(f"Error al configurar el apalancamiento: {e}")

    def adjust_quantity(self, quantity: float) -> float:
        try:
            info = self.client.futures_exchange_info()
            symbol_info = next((s for s in info["symbols"] if s["symbol"] == self.symbol), None)
            if symbol_info is None:
                return quantity
            step_size = next((float(f["stepSize"]) for f in symbol_info["filters"] if f["filterType"] == "LOT_SIZE"), None)
            if not step_size:
                return quantity
            adjusted_qty = math.floor(quantity / step_size) * step_size
            precision = int(round(-math.log(step_size, 10), 0))
            return float(format(adjusted_qty, f'.{precision}f'))
        except Exception as e:
            logging.error(f"Error ajustando la cantidad: {e}")
            return quantity

    def get_futures_balance(self) -> float:
        try:
            balance_info = self.client.futures_account_balance()
            for asset in balance_info:
                if asset['asset'] == "USDT":
                    return float(asset['balance'])
        except BinanceAPIException as e:
            logging.error(f"Error al obtener balance: {e}")
        return 0.0

    def fetch_candle_data(self, limit: int = 150) -> pd.DataFrame:
        try:
            klines = self.client.futures_klines(symbol=self.symbol, interval=self.timeframe, limit=limit)
            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            cols = ["open", "high", "low", "close", "volume"]
            df[cols] = df[cols].astype(float)
            return df
        except BinanceAPIException as e:
            logging.error(f"Error al obtener datos de velas: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['RSI'] = ta.momentum.RSIIndicator(close=df['close'], window=self.rsi_period).rsi()
        df['RSI_SMA'] = df['RSI'].rolling(window=self.rsi_period).mean()
        df['EMA100'] = ta.trend.EMAIndicator(close=df['close'], window=self.ema_period).ema_indicator()
        df['ATR'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=self.atr_period).average_true_range()
        return df

    def calculate_position_size(self, price: float) -> float:
        balance = self.get_futures_balance()
        trade_capital = balance * self.balance_pct * self.leverage
        position_size = trade_capital / price
        logging.info(f"Tamaño de posición (sin ajuste): {position_size}")
        return position_size

    def compute_stop_loss(self, entry_price: float, atr: float, side: str) -> float:
        return entry_price - (atr * self.atr_multiplier_sl) if side == "long" else entry_price + (atr * self.atr_multiplier_sl)

    def check_entry_signal(self, df: pd.DataFrame) -> (bool, str):
        last = df.iloc[-2]
        if self.last_candle_time and last['close_time'] <= self.last_candle_time:
            return (False, "")
        self.last_candle_time = last['close_time']

        rsi_prev = df.iloc[-3]['RSI']
        sma_prev = df.iloc[-3]['RSI_SMA']
        rsi_current = last['RSI']
        sma_current = last['RSI_SMA']

        if rsi_prev < sma_prev and rsi_current > sma_current and last['close'] > last['EMA100']:
            logging.info("Señal de entrada LONG detectada.")
            return (True, "long")
        if rsi_prev > sma_prev and rsi_current < sma_current and last['close'] < last['EMA100']:
            logging.info("Señal de entrada SHORT detectada.")
            return (True, "short")
        return (False, "")

    def check_exit_signal(self, df: pd.DataFrame) -> bool:
        last = df.iloc[-2]
        rsi_prev = df.iloc[-3]['RSI']
        sma_prev = df.iloc[-3]['RSI_SMA']
        rsi_current = last['RSI']
        sma_current = last['RSI_SMA']

        if self.position == "long" and (rsi_prev > sma_prev and rsi_current < sma_current):
            logging.info("Señal de salida LONG detectada (cruce RSI).")
            return True
        if self.position == "short" and (rsi_prev < sma_prev and rsi_current > sma_current):
            logging.info("Señal de salida SHORT detectada (cruce RSI).")
            return True
        return False

    def open_position(self, side: str, entry_price: float, atr: float):
        raw_qty = self.calculate_position_size(entry_price)
        qty = self.adjust_quantity(raw_qty)
        try:
            self.client.futures_create_order(
                symbol=self.symbol,
                side="BUY" if side == "long" else "SELL",
                type="MARKET",
                quantity=qty
            )
            self.position = side
            self.entry_price = entry_price
            self.position_qty = qty
            self.stop_loss = self.compute_stop_loss(entry_price, atr, side)
            self.status_info = f"Posición {side.upper()} abierta a {entry_price:.4f}. Trailing Stop: {self.stop_loss:.4f}"
            logging.info(self.status_info)
            send_telegram_message(self.status_info)
        except BinanceAPIException as e:
            logging.error(f"Error al abrir posición: {e}")
            send_telegram_message(f"❌ Error al abrir posición: {e}")

    def close_position(self, current_price: float):
        if not self.position or not self.position_qty:
            return
        side = "SELL" if self.position == "long" else "BUY"
        try:
            self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type="MARKET",
                quantity=self.position_qty
            )
            msg = f"Posición {self.position.upper()} cerrada a {current_price:.4f}."
            logging.info(msg)
            send_telegram_message(msg)
            self.position = None
            self.entry_price = None
            self.stop_loss = None
            self.position_qty = None
            self.status_info = "Sin posición abierta."
        except BinanceAPIException as e:
            logging.error(f"Error al cerrar posición: {e}")
            send_telegram_message(f"❌ Error al cerrar posición: {e}")

    def update_trailing_stop(self, current_price: float, atr: float):
        if self.position == "long":
            new_sl = current_price - (atr * self.atr_multiplier_sl)
            if new_sl > self.stop_loss:
                logging.info(f"Actualizando trailing SL LONG: {self.stop_loss:.4f} -> {new_sl:.4f}")
                self.stop_loss = new_sl
        elif self.position == "short":
            new_sl = current_price + (atr * self.atr_multiplier_sl)
            if new_sl < self.stop_loss:
                logging.info(f"Actualizando trailing SL SHORT: {self.stop_loss:.4f} -> {new_sl:.4f}")
                self.stop_loss = new_sl

    def run(self):
        logging.info("Iniciando bucle principal...")
        while True:
            if not self.active:
                time.sleep(1)
                continue

            df = self.fetch_candle_data()
            if df.empty:
                logging.warning("No se obtuvieron datos. Reintentando...")
                time.sleep(60)
                continue

            df = self.calculate_indicators(df)
            current_price = df.iloc[-2]['close']
            atr = df.iloc[-2]['ATR']

            if self.position is None:
                signal, side = self.check_entry_signal(df)
                if signal:
                    self.open_position(side, current_price, atr)
            else:
                self.update_trailing_stop(current_price, atr)
                if self.check_exit_signal(df):
                    self.close_position(current_price)
                else:
                    if (self.position == "long" and current_price <= self.stop_loss) or \
                       (self.position == "short" and current_price >= self.stop_loss):
                        logging.info("Trailing Stop alcanzado.")
                        self.close_position(current_price)
            time.sleep(60)

    def start_trading(self):
        self.active = True
        self.status_info = "Trading activado."
        logging.info(self.status_info)
        send_telegram_message(self.status_info)

    def stop_trading(self):
        self.active = False
        self.status_info = "Trading pausado."
        logging.info(self.status_info)
        send_telegram_message(self.status_info)

# Integración con Telegram
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

class TelegramBotThread(threading.Thread):
    def __init__(self, trading_bot: TradingBot):
        threading.Thread.__init__(self)
        self.trading_bot = trading_bot
        self.application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.application.add_handler(CommandHandler('estado', self.estado))
        self.application.add_handler(CommandHandler('start', self.cmd_start))
        self.application.add_handler(CommandHandler('stop', self.cmd_stop))
        self.application.run_polling(stop_signals=None)

    async def estado(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = (
            f"🤖 Estado del Bot de Trading\n\n"
            f"🔹 Estado actual: {self.trading_bot.status_info}\n"
            f"🔹 Par operado: {self.trading_bot.symbol} 💱\n"
            f"🔹 Apalancamiento: x{self.trading_bot.leverage} ⚖️\n"
            f"🔹 Timeframe: {self.trading_bot.timeframe} ⏰\n"
            f"🔹 Trading activo: {'✅ Sí' if self.trading_bot.active else '❌ No'}\n\n"
        )
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.trading_bot.start_trading()
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Trading activado.")

    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.trading_bot.stop_trading()
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Trading pausado.")

def simulate_trade(trading_bot: TradingBot, side: str):
    logging.info(f"Simulación {side.upper()} iniciada.")
    df = trading_bot.fetch_candle_data()
    if df.empty:
        logging.error("No se pudo obtener datos para simular.")
        return
    df = trading_bot.calculate_indicators(df)
    current_price = df.iloc[-2]['close']
    atr = df.iloc[-2]['ATR']
    trading_bot.open_position(side, current_price, atr)
    time.sleep(5)
    trading_bot.close_position(current_price)
    logging.info(f"Simulación {side.upper()} completada.")

def main():
    parser = argparse.ArgumentParser(description="Bot de Trading para SOLUSDT")
    parser.add_argument("--forcelong", action="store_true", help="Forzar operación LONG de prueba")
    parser.add_argument("--forceshort", action="store_true", help="Forzar operación SHORT de prueba")
    args = parser.parse_args()

    trading_bot = TradingBot()
    telegram_thread = TelegramBotThread(trading_bot)
    telegram_thread.daemon = True
    telegram_thread.start()

    if args.forcelong:
        simulate_trade(trading_bot, "long")
        return
    if args.forceshort:
        simulate_trade(trading_bot, "short")
        return

    try:
        trading_bot.run()
    except KeyboardInterrupt:
        logging.info("Interrupción manual. Cerrando bot...")
        send_telegram_message("⚠️ Bot detenido manualmente.")

if __name__ == '__main__':
    main()
