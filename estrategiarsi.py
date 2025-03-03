import ccxt
import telebot
import argparse
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv

# Configuración inicial
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

# Conexión al exchange (Binance)
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
BASE_ASSET = SYMBOL.split('/')[0]  # 'ETH'
TIMEFRAME = '15m'
RSI_PERIOD = 14
OVERSOLD = 30
OVERBOUGHT = 70
LEVERAGE = 5
BALANCE_PERCENTAGE = 0.95
SL_PERCENTAGE = 0.01  # 1% stop loss
TP_PERCENTAGE = 0.02  # 2% take profit
LOG_FILE = "trading_log.txt"

# Obtener información del símbolo para precisión
markets = exchange.fetch_markets()
symbol_info = next((m for m in markets if m['symbol'] == SYMBOL), None)
if symbol_info is None:
    raise ValueError(f"No se encontró información para el símbolo {SYMBOL}")

# Configurar apalancamiento
exchange.set_leverage(LEVERAGE, SYMBOL)

# Función para obtener el balance disponible en futuros
def get_futures_balance():
    balance = exchange.fetch_balance({'type': 'future'})
    return balance['USDT']['free']

# Función para obtener datos OHLCV
def fetch_ohlcv():
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # Convertir a datetime con zona horaria UTC
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
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
    last_rsi = df['rsi'].iloc[-1]
    prev_rsi = df['rsi'].iloc[-2]

    if prev_rsi < OVERSOLD and last_rsi >= OVERSOLD:
        return 'buy', last_rsi
    elif prev_rsi > OVERBOUGHT and last_rsi <= OVERBOUGHT:
        return 'sell', last_rsi
    return None, last_rsi

# Función para calcular el tamaño de la posición
def calculate_position_size(price):
    balance = get_futures_balance()
    position_value = balance * BALANCE_PERCENTAGE * LEVERAGE
    position_size = position_value / price
    return position_size

# Función para ajustar precisión de cantidad y precio
def adjust_precision(value, precision_type):
    if precision_type == 'amount':
        return exchange.amount_to_precision(SYMBOL, value)
    elif precision_type == 'price':
        return exchange.price_to_precision(SYMBOL, value)
    else:
        raise ValueError("precision_type debe ser 'amount' o 'price'")

# Función para abrir una posición larga
def open_long(price):
    size = calculate_position_size(price)
    size = adjust_precision(size, 'amount')
    sl = adjust_precision(price * (1 - SL_PERCENTAGE), 'price')
    tp = adjust_precision(price * (1 + TP_PERCENTAGE), 'price')
    
    exchange.create_market_buy_order(SYMBOL, size)
    exchange.create_order(SYMBOL, 'stop_market', 'sell', size, None, {'stopPrice': sl, 'reduceOnly': True})
    exchange.create_order(SYMBOL, 'take_profit_market', 'sell', size, None, {'stopPrice': tp, 'reduceOnly': True})
    
    message = f"🟢 Posición LARGA abierta en {SYMBOL}\nPrecio: {price} USDT\nSL: {sl} USDT\nTP: {tp} USDT"
    bot.send_message(TELEGRAM_CHAT_ID, message)
    return {'side': 'long', 'entry_price': price, 'size': size, 'sl': sl, 'tp': tp}

# Función para abrir una posición corta
def open_short(price):
    size = calculate_position_size(price)
    size = adjust_precision(size, 'amount')
    sl = adjust_precision(price * (1 + SL_PERCENTAGE), 'price')
    tp = adjust_precision(price * (1 - TP_PERCENTAGE), 'price')
    
    exchange.create_market_sell_order(SYMBOL, size)
    exchange.create_order(SYMBOL, 'stop_market', 'buy', size, None, {'stopPrice': sl, 'reduceOnly': True})
    exchange.create_order(SYMBOL, 'take_profit_market', 'buy', size, None, {'stopPrice': tp, 'reduceOnly': True})
    
    message = f"🔴 Posición CORTA abierta en {SYMBOL}\nPrecio: {price} USDT\nSL: {sl} USDT\nTP: {tp} USDT"
    bot.send_message(TELEGRAM_CHAT_ID, message)
    return {'side': 'short', 'entry_price': price, 'size': size, 'sl': sl, 'tp': tp}

# Función para escribir en el log
def write_log(entry):
    with open(LOG_FILE, "a") as file:
        file.write(f"{datetime.now(timezone.utc)} - {entry}\n")

# Bucle principal del bot
position = None
print("Bot iniciado...")
bot.send_message(TELEGRAM_CHAT_ID, "🤖 Bot de trading RSI iniciado")

while True:
    try:
        df = fetch_ohlcv()
        last_timestamp = df['timestamp'].iloc[-1]

        # Esperar hasta un minuto después del cierre de la última vela
        wait_time = (last_timestamp + timedelta(minutes=1)) - datetime.now(timezone.utc)
        if wait_time.total_seconds() > 0:
            time.sleep(wait_time.total_seconds())

        # Obtener la señal
        signal, rsi_value = rsi_strategy(df)
        current_price = df['close'].iloc[-1]

        log_entry = f"Precio: {current_price}, RSI: {rsi_value}, Señal: {signal}"
        print(log_entry)
        write_log(log_entry)

        if position is None and signal:
            if signal == 'buy':
                position = open_long(current_price)
            elif signal == 'sell':
                position = open_short(current_price)

        time.sleep(900)  # Esperar 15 minutos antes del próximo ciclo

    except Exception as e:
        error_message = f"⚠️ Error: {str(e)}"
        bot.send_message(TELEGRAM_CHAT_ID, error_message)
        write_log(error_message)
        time.sleep(60)
