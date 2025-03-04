#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bot de Trading Algorítmico para operar en SOLUSDT con apalancamiento x5.
Características:
    - Opera en Binance Futures usando el 95% del balance disponible.
    - Realiza mediciones minuto a minuto, confirmando la entrada al cierre de cada vela de 15 minutos.
    - Señal de entrada basada en el cruce del RSI con su SMA (ambos con 14 períodos).
    - Filtro de tendencia basado en la EMA100.
    - Gestión dinámica de Stop Loss y Take Profit basados en ATR.
    - Comandos vía Telegram: /estado, /start y /stop.
    - Flags de línea de comandos --forcelong y --forceshort para probar la operación.
    
Este script está estructurado de forma modular y sigue las mejores prácticas para trading algorítmico.
"""

import os
import time
import datetime
import threading
import logging
import argparse
import requests
import asyncio
import math

import pandas as pd
import numpy as np
import ta  # Librería de análisis técnico

from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def send_telegram_message(message: str):
    """
    Envía un mensaje a Telegram usando el bot configurado.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            logging.error(f"Error al enviar mensaje a Telegram: {response.text}")
    except Exception as e:
        logging.error(f"Excepción al enviar mensaje a Telegram: {e}")


class TradingBot:
    """
    Clase principal que implementa el bot de trading.
    """
    def __init__(self):
        # Inicialización del cliente de Binance para futuros
        self.client = Client(API_KEY, API_SECRET)
        # Configuración del bot
        self.symbol = "SOLUSDT"
        self.leverage = 15
        self.balance_pct = 0.95  # Usar el 95% del balance disponible
        self.timeframe = "15m"   # Timeframe para vela confirmada
        self.rsi_period = 14
        self.ema_period = 100
        self.atr_period = 14
        self.atr_multiplier_sl = 1.5  # Multiplicador para Stop Loss
        self.atr_multiplier_tp = 3.0  # Multiplicador para Take Profit
        self.last_candle_time = None  # Para evitar señales múltiples en la misma vela
        
        # Estado de la posición: None, "long" o "short"
        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.position_qty = None  # Cantidad abierta
        
        # Flag para iniciar o pausar trading
        self.active = True
        
        # Parámetros y logs para el comando /estado
        self.status_info = "Bot iniciado. Sin posición abierta."
        send_telegram_message("✅ Bot de Trading iniciado.")

        # Configurar el apalancamiento en Binance Futures
        try:
            self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
            logging.info(f"Leverage configurado a x{self.leverage} para {self.symbol}")
        except BinanceAPIException as e:
            logging.error(f"Error al configurar el apalancamiento: {e}")

    def adjust_quantity(self, quantity: float) -> float:
        """
        Ajusta la cantidad (position size) según el stepSize definido para el par en futuros.
        Se obtiene la información del símbolo usando futures_exchange_info y se redondea la cantidad hacia abajo.
        """
        try:
            exchange_info = self.client.futures_exchange_info()
            symbol_info = None
            for s in exchange_info["symbols"]:
                if s["symbol"] == self.symbol:
                    symbol_info = s
                    break
            if symbol_info is None:
                return quantity
            step_size = None
            for f in symbol_info["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    step_size = float(f["stepSize"])
                    break
            if step_size is None or step_size == 0:
                return quantity
            adjusted_qty = math.floor(quantity / step_size) * step_size
            # Determinar la cantidad de decimales permitida según el stepSize
            precision = int(round(-math.log(step_size, 10), 0))
            adjusted_qty = float(format(adjusted_qty, f'.{precision}f'))
            return adjusted_qty
        except Exception as e:
            logging.error(f"Error ajustando la cantidad: {e}")
            return quantity

    def get_futures_balance(self) -> float:
        """
        Obtiene el balance disponible en la billetera de futuros (en USDT).
        """
        try:
            balance_info = self.client.futures_account_balance()
            for asset in balance_info:
                if asset['asset'] == "USDT":
                    return float(asset['balance'])
        except BinanceAPIException as e:
            logging.error(f"Error al obtener balance: {e}")
        return 0.0

    def fetch_candle_data(self, limit: int = 150) -> pd.DataFrame:
        """
        Descarga datos históricos de velas para el par y timeframe especificado.
        """
        try:
            klines = self.client.futures_klines(symbol=self.symbol, interval=self.timeframe, limit=limit)
            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            numeric_cols = ["open", "high", "low", "close", "volume"]
            df[numeric_cols] = df[numeric_cols].astype(float)
            return df
        except BinanceAPIException as e:
            logging.error(f"Error al obtener datos de velas: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula indicadores técnicos: RSI, SMA del RSI, EMA100 y ATR.
        """
        df['RSI'] = ta.momentum.RSIIndicator(close=df['close'], window=self.rsi_period).rsi()
        df['RSI_SMA'] = df['RSI'].rolling(window=self.rsi_period).mean()
        df['EMA100'] = ta.trend.EMAIndicator(close=df['close'], window=self.ema_period).ema_indicator()
        df['ATR'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=self.atr_period).average_true_range()
        return df

    def calculate_position_size(self, price: float) -> float:
        """
        Calcula el tamaño de posición basado en el 95% del balance, apalancamiento y precio actual.
        """
        balance = self.get_futures_balance()
        logging.info(f"Balance obtenido: {balance} USDT")
        trade_capital = balance * self.balance_pct * self.leverage
        position_size = trade_capital / price
        logging.info(f"Tamaño de posición calculado (sin ajuste): {position_size} {self.symbol[:-4]}")
        return position_size

    def compute_stop_loss_take_profit(self, entry_price: float, atr: float, side: str):
        """
        Calcula los niveles dinámicos de Stop Loss y Take Profit en función del ATR.
        """
        if side == "long":
            sl = entry_price - (atr * self.atr_multiplier_sl)
            tp = entry_price + (atr * self.atr_multiplier_tp)
        else:
            sl = entry_price + (atr * self.atr_multiplier_sl)
            tp = entry_price - (atr * self.atr_multiplier_tp)
        return sl, tp

    def check_entry_signal(self, df: pd.DataFrame) -> (bool, str):
        """
        Verifica si se cumple la condición de entrada basada en el cruce del RSI con su SMA,
        y filtra la tendencia mediante la EMA100.
        Retorna una tupla (señal, lado) donde 'lado' es 'long' o 'short'.
        """
        last = df.iloc[-2]
        candle_close_time = last['close_time']
        if self.last_candle_time is not None and candle_close_time <= self.last_candle_time:
            return (False, "")
        self.last_candle_time = candle_close_time

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
        """
        Verifica la señal de salida basada en el cruce opuesto del RSI.
        """
        last = df.iloc[-2]
        rsi_prev = df.iloc[-3]['RSI']
        sma_prev = df.iloc[-3]['RSI_SMA']
        rsi_current = last['RSI']
        sma_current = last['RSI_SMA']
        
        if self.position == "long" and (rsi_prev > sma_prev and rsi_current < sma_current):
            logging.info("Señal de salida LONG detectada (cierre de posición).")
            return True
        if self.position == "short" and (rsi_prev < sma_prev and rsi_current > sma_current):
            logging.info("Señal de salida SHORT detectada (cierre de posición).")
            return True
        return False

    def open_position(self, side: str, entry_price: float, atr: float):
        """
        Abre una posición (long o short) calculando el tamaño de posición y estableciendo SL y TP dinámicos.
        Ajusta la cantidad según el stepSize permitido y almacena la cantidad real usada.
        """
        raw_qty = self.calculate_position_size(entry_price)
        qty = self.adjust_quantity(raw_qty)
        try:
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side="BUY" if side == "long" else "SELL",
                type="MARKET",
                quantity=qty
            )
            self.position = side
            self.entry_price = entry_price
            self.position_qty = qty
            self.stop_loss, self.take_profit = self.compute_stop_loss_take_profit(entry_price, atr, side)
            self.status_info = (f"Posición {side.upper()} abierta a {entry_price:.4f}. "
                                f"SL: {self.stop_loss:.4f}, TP: {self.take_profit:.4f}")
            logging.info(self.status_info)
            send_telegram_message(self.status_info)
        except BinanceAPIException as e:
            logging.error(f"Error al abrir posición: {e}")
            send_telegram_message(f"❌ Error al abrir posición: {e}")

    def close_position(self, current_price: float):
        """
        Cierra la posición abierta utilizando la cantidad almacenada (position_qty).
        """
        if not self.position or not self.position_qty:
            return

        side = "SELL" if self.position == "long" else "BUY"
        try:
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type="MARKET",
                quantity=self.position_qty
            )
            msg = (f"Posición {self.position.upper()} cerrada a {current_price:.4f}. Orden de salida ejecutada.")
            logging.info(msg)
            send_telegram_message(msg)
            self.position = None
            self.entry_price = None
            self.stop_loss = None
            self.take_profit = None
            self.position_qty = None
            self.status_info = "Sin posición abierta."
        except BinanceAPIException as e:
            logging.error(f"Error al cerrar posición: {e}")
            send_telegram_message(f"❌ Error al cerrar posición: {e}")

    def update_trailing_stop(self, current_price: float, atr: float):
        """
        Actualiza el stop loss de forma dinámica (trailing stop) para proteger ganancias.
        """
        if self.position == "long":
            new_sl = current_price - (atr * self.atr_multiplier_sl)
            if new_sl > self.stop_loss:
                logging.info(f"Actualizando trailing SL de LONG: {self.stop_loss:.4f} -> {new_sl:.4f}")
                self.stop_loss = new_sl
        elif self.position == "short":
            new_sl = current_price + (atr * self.atr_multiplier_sl)
            if new_sl < self.stop_loss:
                logging.info(f"Actualizando trailing SL de SHORT: {self.stop_loss:.4f} -> {new_sl:.4f}")
                self.stop_loss = new_sl

    def run(self):
        """
        Bucle principal del bot: evalúa señales y ejecuta órdenes cada minuto,
        respetando la bandera 'active' para iniciar/pausar el trading.
        """
        logging.info("Iniciando bucle principal del bot...")
        while True:
            if not self.active:
                time.sleep(1)
                continue

            df = self.fetch_candle_data()
            if df.empty:
                logging.warning("No se pudieron obtener datos de velas. Reintentando...")
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
                    if self.position == "long" and current_price <= self.stop_loss:
                        logging.info("Stop Loss alcanzado en posición LONG.")
                        self.close_position(current_price)
                    elif self.position == "long" and current_price >= self.take_profit:
                        logging.info("Take Profit alcanzado en posición LONG.")
                        self.close_position(current_price)
                    elif self.position == "short" and current_price >= self.stop_loss:
                        logging.info("Stop Loss alcanzado en posición SHORT.")
                        self.close_position(current_price)
                    elif self.position == "short" and current_price <= self.take_profit:
                        logging.info("Take Profit alcanzado en posición SHORT.")
                        self.close_position(current_price)

            time.sleep(60)

    def start_trading(self):
        """
        Activa el trading.
        """
        self.active = True
        self.status_info = "Trading activado."
        logging.info(self.status_info)
        send_telegram_message(self.status_info)

    def stop_trading(self):
        """
        Pausa el trading.
        """
        self.active = False
        self.status_info = "Trading pausado."
        logging.info(self.status_info)
        send_telegram_message(self.status_info)


# Integración del bot de Telegram usando ApplicationBuilder (python-telegram-bot v20+)
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

class TelegramBotThread(threading.Thread):
    def __init__(self, trading_bot: TradingBot):
        threading.Thread.__init__(self)
        self.trading_bot = trading_bot
        self.application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    def run(self):
        # Crear y asignar un nuevo event loop en este hilo
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.application.add_handler(CommandHandler('estado', self.estado))
        self.application.add_handler(CommandHandler('start', self.cmd_start))
        self.application.add_handler(CommandHandler('stop', self.cmd_stop))
        # Se desactivan los signal handlers para evitar el error al correr en un hilo secundario
        self.application.run_polling(stop_signals=None)

    async def estado(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Envía al usuario el estado actual del bot.
        """
        msg = (f"Estado del Bot:\n{self.trading_bot.status_info}\n"
               f"Par: {self.trading_bot.symbol}\n"
               f"Apalancamiento: x{self.trading_bot.leverage}\n"
               f"Timeframe: {self.trading_bot.timeframe}\n"
               f"Trading activo: {'Sí' if self.trading_bot.active else 'No'}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando para iniciar o reanudar el trading.
        """
        self.trading_bot.start_trading()
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Trading activado.")

    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando para pausar el trading.
        """
        self.trading_bot.stop_trading()
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Trading pausado.")


def simulate_trade(trading_bot: TradingBot, side: str):
    """
    Ejecuta una operación forzada de prueba:
    Abre una posición y luego la cierra para corroborar el funcionamiento.
    """
    logging.info(f"Simulación forzada de operación {side.upper()} iniciada.")
    df = trading_bot.fetch_candle_data()
    if df.empty:
        logging.error("No se pudo obtener datos para simular el trade.")
        return
    df = trading_bot.calculate_indicators(df)
    current_price = df.iloc[-2]['close']
    atr = df.iloc[-2]['ATR']
    trading_bot.open_position(side, current_price, atr)
    time.sleep(5)  # Simular que la posición estuvo abierta por unos segundos
    trading_bot.close_position(current_price)
    logging.info(f"Simulación forzada de operación {side.upper()} completada.")


def main():
    parser = argparse.ArgumentParser(description="Bot de Trading Algorítmico para SOLUSDT")
    parser.add_argument("--forcelong", action="store_true", help="Forzar una operación LONG de prueba")
    parser.add_argument("--forceshort", action="store_true", help="Forzar una operación SHORT de prueba")
    args = parser.parse_args()

    trading_bot = TradingBot()

    # Iniciar el hilo del bot de Telegram para comandos
    telegram_thread = TelegramBotThread(trading_bot)
    telegram_thread.daemon = True
    telegram_thread.start()

    # Ejecutar operaciones de prueba si se usan los flags correspondientes
    if args.forcelong:
        simulate_trade(trading_bot, "long")
        return
    if args.forceshort:
        simulate_trade(trading_bot, "short")
        return

    # Bucle principal de trading
    try:
        trading_bot.run()
    except KeyboardInterrupt:
        logging.info("Interrupción manual. Cerrando bot...")
        send_telegram_message("⚠️ Bot detenido manualmente.")


if __name__ == '__main__':
    main()
