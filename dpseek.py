#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bot de Trading Algorítmico Avanzado para SOLUSDT
Versión 4.1 - Julio 2024 (Corregida y Verificada)
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
import numpy as np
from datetime import datetime, timedelta

from binance.client import Client
from binance.exceptions import BinanceAPIException
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv

# Configuración inicial
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def send_telegram_message(message: str):
    """Función mejorada para enviar mensajes con formato Markdown"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Error enviando mensaje: {e}")

class EnhancedTradingBot:
    def __init__(self):
        self.symbol = "SOLUSDT"
        self.leverage = 13
        self.risk_per_trade = 0.95
        self.timeframe = "15m"
        self.ema_fast = 50
        self.ema_slow = 200
        self.rsi_period = 14
        self.atr_period = 14
        self.tp_multiplier = 1.8
        self.sl_multiplier = 2.0
        self.volume_threshold = 1.2

        self.client = Client(API_KEY, API_SECRET)
        self.configure_binance()
        
        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.position_qty = None
        self.entry_time = None
        self.max_drawdown = 0.20
        
        self.active = True
        self.last_execution = datetime.now()
        self.status_info = "🟢 Bot iniciado - Esperando señales"
        send_telegram_message("🚀 *Bot Mejorado Iniciado*")

    def configure_binance(self):
        try:
            self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
            try:
                self.client.futures_change_margin_type(symbol=self.symbol, marginType='ISOLATED')
            except BinanceAPIException as e:
                if "No need to change margin type" not in str(e):
                    raise
            logging.info(f"Configurado apalancamiento x{self.leverage}")
        except BinanceAPIException as e:
            logging.error(f"Error configuración Binance: {e}")
            send_telegram_message(f"❌ *Error de Configuración:* {e}")

    def get_precision(self):
        info = self.client.futures_exchange_info()
        symbol_info = next(s for s in info['symbols'] if s['symbol'] == self.symbol)
        return symbol_info['pricePrecision'], symbol_info['quantityPrecision']

    def get_futures_balance(self) -> float:
        try:
            balance_info = self.client.futures_account_balance()
            for asset in balance_info:
                if asset['asset'] == "USDT":
                    return float(asset['balance'])
        except BinanceAPIException as e:
            logging.error(f"Error al obtener balance: {e}")
        return 0.0

    def adjust_quantity(self, quantity: float) -> float:
        _, qty_precision = self.get_precision()
        step_size = 10 ** -qty_precision
        return round(math.floor(quantity / step_size) * step_size, qty_precision)

    def calculate_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if len(df) < 200:
                logging.error("Se requieren mínimo 200 velas históricas")
                return pd.DataFrame()

            high = df['high']
            low = df['low']
            close = df['close']
            
            # ATR
            tr = np.maximum(high - low, 
                          np.maximum(abs(high - close.shift()), 
                                     abs(low - close.shift())))
            df['ATR'] = tr.rolling(window=self.atr_period).mean()
            
            # EMAs
            df['EMA_50'] = close.ewm(span=self.ema_fast, adjust=False).mean()
            df['EMA_200'] = close.ewm(span=self.ema_slow, adjust=False).mean()
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['RSI_MA'] = df['RSI'].rolling(window=9).mean()
            
            # Volumen y VWAP
            df['Volume_MA'] = df['volume'].rolling(window=20).mean()
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['VWAP'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            
            # MACD
            exp12 = close.ewm(span=12, adjust=False).mean()
            exp26 = close.ewm(span=26, adjust=False).mean()
            df['MACD'] = exp12 - exp26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

            return df.dropna()
        
        except Exception as e:
            logging.error(f"Error en cálculo de indicadores: {str(e)}")
            return pd.DataFrame()

    def generate_signal(self, df: pd.DataFrame) -> dict:
        required_columns = ['EMA_50', 'EMA_200', 'RSI', 'VWAP', 'MACD', 'MACD_Signal']
        if not all(col in df.columns for col in required_columns):
            return {'direction': None, 'strength': 0}
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        signal = {'direction': None, 'strength': 0}
        
        # Condiciones LONG ajustadas
        if (current['EMA_50'] > current['EMA_200'] and
            current['close'] > current['VWAP'] and
            current['RSI'] > 50 and
            current['MACD'] > current['MACD_Signal'] and
            prev['MACD'] < prev['MACD_Signal']):
            
            signal['direction'] = 'long'
            signal['strength'] = self.calculate_signal_strength(df, 'long')
        
        # Condiciones SHORT ajustadas
        elif (current['EMA_50'] < current['EMA_200'] and
            current['close'] < current['VWAP'] and
            current['RSI'] < 50 and
            current['MACD'] < current['MACD_Signal'] and
            prev['MACD'] > prev['MACD_Signal']):
            
            signal['direction'] = 'short'
            signal['strength'] = self.calculate_signal_strength(df, 'short')
            
        return signal

    def calculate_signal_strength(self, df: pd.DataFrame, direction: str) -> float:
        try:
            current = df.iloc[-1]
            volume_strength = min(current['volume'] / current['Volume_MA'], 2.0)
            ema_distance = abs(current['EMA_50'] - current['EMA_200']) / current['EMA_200']
            rsi_strength = (current['RSI'] - 30)/70 if direction == 'long' else (70 - current['RSI'])/70
            return round((volume_strength * 0.4) + (ema_distance * 0.3) + (rsi_strength * 0.3), 2)
        except:
            return 0.0

    def calculate_required_margin(self, qty: float, price: float) -> float:
        """
        Calcula el margen requerido para abrir una posición.
        En Binance Futures, el margen inicial es aproximadamente qty * price / leverage.
        """
        try:
            return (qty * price) / self.leverage
        except Exception as e:
            logging.error(f"Error calculando margen requerido: {str(e)}")
            return float('inf')  # Valor grande para indicar error

    def dynamic_position_sizing(self, df: pd.DataFrame):
        try:
            current_price = df['close'].iloc[-1]
            atr = df['ATR'].iloc[-1]
            signal = self.generate_signal(df)
            
            balance = self.get_futures_balance()
            if balance <= 0:
                logging.error("Balance insuficiente para operar")
                return 0.0
            
            # Calcular el capital en riesgo
            risk_capital = balance * self.risk_per_trade * (1 + signal['strength'])
            
            # Ajustar por volatilidad
            if (atr / current_price * 100) > 5:
                risk_capital *= 0.7
            
            # Calcular tamaño de posición máximo basado en el balance y apalancamiento
            max_position_size = (balance * self.leverage) / current_price
            position_size = min((risk_capital * self.leverage) / current_price, max_position_size)
            
            # Ajustar la cantidad a la precisión del símbolo
            qty = self.adjust_quantity(position_size)
            
            # Verificar si el margen requerido es menor o igual al balance disponible
            required_margin = self.calculate_required_margin(qty, current_price)
            if required_margin > balance:
                logging.warning(f"Margen requerido ({required_margin:.2f}) excede el balance ({balance:.2f})")
                qty = self.adjust_quantity((balance * self.leverage) / current_price)
                logging.info(f"Ajustando qty a {qty} para no exceder el balance")
            
            return qty if qty > 0 else 0.0
        except Exception as e:
            logging.error(f"Error en cálculo de tamaño de posición: {str(e)}")
            return 0.0

    def execute_entry(self, signal: dict, df: pd.DataFrame):
        try:
            required_columns = ['ATR', 'close']
            if not all(col in df.columns for col in required_columns):
                return

            current_price = df['close'].iloc[-1]
            atr = df['ATR'].iloc[-1]
            qty = self.dynamic_position_sizing(df)
            
            if qty <= 0:
                logging.warning("Cantidad calculada <= 0, no se abre posición")
                return

            # Calcular SL y TP tentativos para logging
            tentative_sl = self._calculate_initial_sl(signal['direction'], atr)
            tentative_tp = self._calculate_initial_tp(signal['direction'], atr)
            
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side='BUY' if signal['direction'] == 'long' else 'SELL',
                type='MARKET',
                quantity=qty
            )
            
            if self.check_order_fill(order['orderId']):
                self.position = signal['direction']
                self.entry_price = float(order['avgPrice'])
                self.entry_time = datetime.now()
                self.stop_loss = tentative_sl
                self.take_profit = tentative_tp
                self.position_qty = qty
                
                msg = (f"📈 *Posición {signal['direction'].upper()} ABIERTA*\n"
                    f"Entrada: {self.entry_price:.4f}\n"
                    f"Tamaño: {qty:.2f} contratos\n"
                    f"SL: {self.stop_loss:.4f}\n"
                    f"TP: {self.take_profit:.4f}")
                send_telegram_message(msg)
                
        except BinanceAPIException as e:
            if "Margin is insufficient" in str(e):
                logging.error(f"Error: Margen insuficiente para abrir posición")
                msg = (f"❌ *Error: Margen Insuficiente*\n"
                    f"Señal: {signal['direction'].upper()}\n"
                    f"Precio de Entrada Tentativo: {current_price:.4f}\n"
                    f"Tamaño Tentativo: {qty:.2f} contratos\n"
                    f"SL Tentativo: {tentative_sl:.4f}\n"
                    f"TP Tentativo: {tentative_tp:.4f}\n"
                    f"Balance Actual: {self.get_futures_balance():.2f} USDT")
                send_telegram_message(msg)
            else:
                logging.error(f"Error entrada: {e}")
                send_telegram_message(f"❌ *Error apertura:* {e}")
            self._reset_position()
        except Exception as e:
            logging.error(f"Error general entrada: {str(e)}")
            self._reset_position()

    def _calculate_initial_sl(self, direction: str, atr: float):
        if direction == 'long':
            return self.entry_price - (atr * self.sl_multiplier)
        return self.entry_price + (atr * self.sl_multiplier)

    def _calculate_initial_tp(self, direction: str, atr: float):
        if direction == 'long':
            return self.entry_price + (atr * self.tp_multiplier)
        return self.entry_price - (atr * self.tp_multiplier)

    def check_order_fill(self, order_id: int, timeout=300) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                order_status = self.client.futures_get_order(
                    symbol=self.symbol,
                    orderId=order_id
                )
                if order_status['status'] == 'FILLED':
                    return True
                time.sleep(5)
            except BinanceAPIException as e:
                logging.error(f"Error verificando orden: {e}")
                return False
        return False

    def execute_exit(self, price: float):
        try:
            if not self.position or not self.position_qty:
                logging.error("No hay posición activa")
                return

            self.client.futures_create_order(
                symbol=self.symbol,
                side='SELL' if self.position == 'long' else 'BUY',
                type='MARKET',
                quantity=self.position_qty
            )
            
            pnl = (price - self.entry_price) * self.position_qty * (-1 if self.position == 'short' else 1)
            duration = datetime.now() - self.entry_time
            
            msg = (f"📉 *Posición {self.position.upper()} CERRADA*\n"
                   f"Salida: {price:.4f}\n"
                   f"Resultado: {pnl:.2f} USDT\n"
                   f"Duración: {duration}")
            send_telegram_message(msg)
            
            self._reset_position()
            
        except BinanceAPIException as e:
            logging.error(f"Error salida: {e}")
            send_telegram_message(f"❌ *Error cierre:* {e}")
        except Exception as e:
            logging.error(f"Error general salida: {str(e)}")
            self._reset_position()

    def _reset_position(self):
        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.position_qty = None
        self.entry_time = None

    def manage_exits(self, current_price: float, df: pd.DataFrame):
        if not self.position:
            return
            
        try:
            atr = df['ATR'].iloc[-1]
            price_precision, _ = self.get_precision()
            
            if self.position == 'long':
                new_sl = current_price - (atr * self.sl_multiplier)
                self.stop_loss = max(self.stop_loss, new_sl) if self.stop_loss else new_sl
                
                if current_price >= self.entry_price * (1 + (self.tp_multiplier * atr/self.entry_price)):
                    self.take_profit = current_price - (atr * 0.5)
                    
            elif self.position == 'short':
                new_sl = current_price + (atr * self.sl_multiplier)
                self.stop_loss = min(self.stop_loss, new_sl) if self.stop_loss else new_sl
                
                if current_price <= self.entry_price * (1 - (self.tp_multiplier * atr/self.entry_price)):
                    self.take_profit = current_price + (atr * 0.5)
            
            self.stop_loss = round(self.stop_loss, price_precision)
            if self.take_profit:
                self.take_profit = round(self.take_profit, price_precision)
                
        except Exception as e:
            logging.error(f"Error gestionando SL/TP: {str(e)}")

    def execute_strategy(self):
        while self.active:
            try:
                if (datetime.now() - self.last_execution).seconds < 300:
                    time.sleep(60)
                    continue
                
                df = self.fetch_enhanced_data(limit=300)
                df = self.calculate_enhanced_indicators(df)
                
                if df.empty:
                    continue
                
                signal = self.generate_signal(df)
                current_price = df['close'].iloc[-1]
                
                self.manage_exits(current_price, df)
                
                if self.position:
                    self.monitor_positions(current_price)
                elif signal['direction'] and signal['strength'] >= 0.5:  # Umbral ajustado de 0.6 a 0.5
                    self.execute_entry(signal, df)
                    
                self.last_execution = datetime.now()
                time.sleep(60)
            
            except Exception as e:
                logging.error(f"Error en estrategia: {str(e)}")
                time.sleep(300)

    def fetch_enhanced_data(self, limit=300):
        tries = 3
        for _ in range(tries):
            try:
                klines = self.client.futures_klines(
                    symbol=self.symbol,
                    interval=self.timeframe,
                    limit=limit
                )
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            except Exception as e:
                logging.warning(f"Error datos (intento {_+1}): {e}")
                time.sleep(5)
        return pd.DataFrame()

    def monitor_positions(self, current_price: float):
        if (self.position == 'long' and (current_price <= self.stop_loss or current_price >= self.take_profit)) or \
           (self.position == 'short' and (current_price >= self.stop_loss or current_price <= self.take_profit)):
            self.execute_exit(current_price)

class TelegramBotThread(threading.Thread):
    def __init__(self, trading_bot: EnhancedTradingBot):
        super().__init__()
        self.bot = trading_bot
        self.application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        self.setup_handlers()

    def setup_handlers(self):
        self.application.add_handler(CommandHandler('estado', self.estado))
        self.application.add_handler(CommandHandler('start', self.start_bot))
        self.application.add_handler(CommandHandler('stop', self.stop_bot))

    async def estado(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        status_text = (
            f"🔍 *Estado del Bot*\n"
            f"• Par: {self.bot.symbol}\n"
            f"• Posición: {self.bot.position or 'Ninguna'}\n"
            f"• Balance: {self.bot.get_futures_balance():.2f} USDT\n"
            f"• Última acción: {self.bot.last_execution.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        await update.message.reply_text(status_text, parse_mode='Markdown')

    async def start_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.bot.active = True
        await update.message.reply_text("✅ *Trading activado*", parse_mode='Markdown')

    async def stop_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.bot.active = False
        if self.bot.position:
            self.bot.execute_exit(self.get_current_price())
        await update.message.reply_text("🛑 *Trading detenido*", parse_mode='Markdown')

    def get_current_price(self):
        df = self.bot.fetch_enhanced_data(limit=1)
        return df['close'].iloc[-1] if not df.empty else 0.0

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Configuración especial para evitar conflictos de señales
            loop.run_until_complete(
                self.application.run_polling(
                    close_loop=False,
                    stop_signals=None,
                    allowed_updates=Update.ALL_TYPES
                )
            )
        finally:
            loop.close()

def main():
    parser = argparse.ArgumentParser(description="Bot de Trading Avanzado para SOLUSDT")
    parser.add_argument("--simulate", choices=['long', 'short'], help="Simular una operación")
    args = parser.parse_args()

    bot = EnhancedTradingBot()
    telegram_thread = TelegramBotThread(bot)
    telegram_thread.daemon = True
    telegram_thread.start()

    if args.simulate:
        logging.info(f"Iniciando simulación {args.simulate.upper()}...")
        try:
            df = bot.fetch_enhanced_data(limit=300)
            df = bot.calculate_enhanced_indicators(df)
            
            if not df.empty and all(col in df.columns for col in ['ATR', 'close', 'EMA_50', 'EMA_200', 'RSI']):
                signal = {'direction': args.simulate, 'strength': 0.8}
                bot.execute_entry(signal, df)
                time.sleep(2)
                bot.execute_exit(df['close'].iloc[-1])
            else:
                missing = [col for col in ['ATR', 'close', 'EMA_50', 'EMA_200', 'RSI'] if col not in df.columns]
                logging.error(f"Columnas faltantes: {missing}")
                
        except Exception as e:
            logging.error(f"Fallo en simulación: {str(e)}")
        return

    try:
        bot.execute_strategy()
    except KeyboardInterrupt:
        send_telegram_message("🛑 Bot detenido manualmente")
        logging.info("Ejecución finalizada")

if __name__ == '__main__':
    main()