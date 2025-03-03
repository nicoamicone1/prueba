import ccxt
import telebot
import argparse
import time
import pandas as pd
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv

# Configuración inicial
# Reemplaza con tus propias claves API y TELEGRAM_CHAT_ID de Telegram
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

# Conexión al exchange (Binance en este caso)
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},  # Operar en futuros
})

# Conexión a Telegram
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Configuración de argumentos de línea de comandos
parser = argparse.ArgumentParser(description='Bot de trading RSI')
parser.add_argument('--forcelong', action='store_true', help='Forzar apertura de posición larga')
parser.add_argument('--forceshort', action='store_true', help='Forzar apertura de posición corta')
args = parser.parse_args()

# Parámetros globales
SYMBOL = 'ETH/USDT'
TIMEFRAME = '15m'
RSI_PERIOD = 14
OVERSOLD = 30
OVERBOUGHT = 70
LEVERAGE = 5
BALANCE_PERCENTAGE = 0.95
SL_PERCENTAGE = 0.01  # 1% stop loss
TP_PERCENTAGE = 0.02  # 2% take profit

# Configurar apalancamiento en el exchange
exchange.set_leverage(LEVERAGE, SYMBOL)

# Función para obtener el balance disponible en la billetera de futuros
def get_futures_balance():
    balance = exchange.fetch_balance({'type': 'future'})
    return balance['USDT']['free']

# Función para obtener datos OHLCV
def fetch_ohlcv():
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Función para calcular el RSI
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Función de la estrategia RSI
def rsi_strategy(df):
    df['rsi'] = calculate_rsi(df, RSI_PERIOD)
    if df['rsi'].iloc[-2] < OVERSOLD and df['rsi'].iloc[-1] >= OVERSOLD:
        return 'buy'
    elif df['rsi'].iloc[-2] > OVERBOUGHT and df['rsi'].iloc[-1] <= OVERBOUGHT:
        return 'sell'
    return None

# Función para calcular el tamaño de la posición
def calculate_position_size(price):
    balance = get_futures_balance()
    position_value = balance * BALANCE_PERCENTAGE * LEVERAGE
    position_size = position_value / price
    return position_size

# Función para abrir una posición larga
def open_long(price):
    size = calculate_position_size(price)
    sl = price * (1 - SL_PERCENTAGE)
    tp = price * (1 + TP_PERCENTAGE)
    exchange.create_market_buy_order(SYMBOL, size)
    bot.send_message(TELEGRAM_CHAT_ID, f"Long abierto: {size:.4f} {SYMBOL} a {price}\nSL: {sl:.2f}\nTP: {tp:.2f}")
    return {'side': 'long', 'entry_price': price, 'size': size, 'sl': sl, 'tp': tp}

# Función para abrir una posición corta
def open_short(price):
    size = calculate_position_size(price)
    sl = price * (1 + SL_PERCENTAGE)
    tp = price * (1 - TP_PERCENTAGE)
    exchange.create_market_sell_order(SYMBOL, size)
    bot.send_message(TELEGRAM_CHAT_ID, f"Short abierto: {size:.4f} {SYMBOL} a {price}\nSL: {sl:.2f}\nTP: {tp:.2f}")
    return {'side': 'short', 'entry_price': price, 'size': size, 'sl': sl, 'tp': tp}

# Función para cerrar una posición
def close_position(position, price):
    if position['side'] == 'long':
        exchange.create_market_sell_order(SYMBOL, position['size'])
        bot.send_message(TELEGRAM_CHAT_ID, f"Long cerrado: {position['size']:.4f} {SYMBOL} a {price}")
    elif position['side'] == 'short':
        exchange.create_market_buy_order(SYMBOL, position['size'])
        bot.send_message(TELEGRAM_CHAT_ID, f"Short cerrado: {position['size']:.4f} {SYMBOL} a {price}")

# Bucle principal del bot
position = None
print("Bot iniciado...")
bot.send_message(TELEGRAM_CHAT_ID, "Bot de trading RSI iniciado")

while True:
    try:
        # Obtener datos y precio actual
        df = fetch_ohlcv()
        current_price = df['close'].iloc[-1]

        # Comandos manuales
        if args.forcelong and position is None:
            position = open_long(current_price)
            args.forcelong = False  # Resetear el flag
        elif args.forceshort and position is None:
            position = open_short(current_price)
            args.forceshort = False  # Resetear el flag

        # Estrategia RSI automática
        elif position is None:
            signal = rsi_strategy(df)
            if signal == 'buy':
                position = open_long(current_price)
            elif signal == 'sell':
                position = open_short(current_price)

        # Comprobar stop loss y take profit si hay posición abierta
        elif position:
            if position['side'] == 'long':
                if current_price <= position['sl']:
                    bot.send_message(TELEGRAM_CHAT_ID, f"Stop Loss alcanzado en Long a {current_price}")
                    close_position(position, current_price)
                    position = None
                elif current_price >= position['tp']:
                    bot.send_message(TELEGRAM_CHAT_ID, f"Take Profit alcanzado en Long a {current_price}")
                    close_position(position, current_price)
                    position = None
            elif position['side'] == 'short':
                if current_price >= position['sl']:
                    bot.send_message(TELEGRAM_CHAT_ID, f"Stop Loss alcanzado en Short a {current_price}")
                    close_position(position, current_price)
                    position = None
                elif current_price <= position['tp']:
                    bot.send_message(TELEGRAM_CHAT_ID, f"Take Profit alcanzado en Short a {current_price}")
                    close_position(position, current_price)
                    position = None

        # Esperar hasta el próximo intervalo (15 minutos)
        time.sleep(900)

    except Exception as e:
        bot.send_message(TELEGRAM_CHAT_ID, f"Error: {str(e)}")
        time.sleep(60)  # Esperar 1 minuto antes de reintentar