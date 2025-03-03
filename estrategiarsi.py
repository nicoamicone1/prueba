import ccxt
import time
import pandas as pd
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv
import logging

# Configuración inicial
load_dotenv()
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

# Conexión al exchange (Binance)
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
})

# Parámetros globales
SYMBOL = 'ETH/USDT'
TIMEFRAME = '15m'
RSI_PERIOD = 14
OVERSOLD = 30
OVERBOUGHT = 70

# Configuración de logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

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
    rsi_current = df['rsi'].iloc[-1]
    rsi_previous = df['rsi'].iloc[-2]
    
    # Registrar los valores del RSI
    log_message = f"RSI actual: {rsi_current:.2f}, RSI anterior: {rsi_previous:.2f}"
    logging.info(log_message)
    
    # Determinar la señal
    if rsi_previous < OVERSOLD and rsi_current >= OVERSOLD:
        signal = 'compra'
    elif rsi_previous > OVERBOUGHT and rsi_current <= OVERBOUGHT:
        signal = 'venta'
    else:
        signal = 'ninguna'
    
    # Registrar la señal
    signal_message = f"Señal tomada: {signal}"
    logging.info(signal_message)
    
    return signal

# Bucle principal del bot
print("Bot iniciado...")
logging.info("Bot de trading RSI iniciado")

while True:
    try:
        # Obtener datos y calcular señal
        df = fetch_ohlcv()
        signal = rsi_strategy(df)

        # Esperar 60 segundos para la próxima iteración
        time.sleep(60)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        logging.error(error_message)
        time.sleep(60)  # Esperar 1 minuto antes de reintentar