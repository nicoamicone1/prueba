import os
import time
import argparse
import requests
import pandas as pd
import numpy as np
import ta  # pip install ta
from binance.client import Client
from binance.enums import *
from dotenv import load_dotenv

# ============================ Argumentos de línea de comandos ============================
parser = argparse.ArgumentParser(description="Bot de trading con bandera --demo")
parser.add_argument("--demo", action="store_true", help="Modo demo: no ejecuta órdenes reales, solo envía señales a Telegram")
args = parser.parse_args()
DEMO = args.demo

# ============================ Carga de variables de entorno ============================
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

# ============================ Inicialización del cliente de Binance ============================
client = Client(API_KEY, API_SECRET)
symbol = 'SOLUSDT'
timeframe = '15m'
leverage = 10

# Configurar apalancamiento en Binance Futures
if not DEMO:
    client.futures_change_leverage(symbol=symbol, leverage=leverage)
else:
    print("Modo DEMO activado. No se enviarán órdenes reales.")

# ============================ Funciones de notificación ============================
def send_telegram_message(message):
    """Envía notificaciones a Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    params = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
    try:
        requests.get(url, params=params)
    except Exception as e:
        print("Error enviando Telegram:", e)

# ============================ Función para obtener candles ============================
def get_candles(symbol, interval, limit=150):
    """Obtiene datos históricos de velas de Binance Futures."""
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=['open_time','open','high','low','close','volume','close_time',
                                         'quote_asset_volume','number_of_trades','taker_buy_base_volume',
                                         'taker_buy_quote_volume','ignore'])
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    return df

# ============================ Cálculo de indicadores con la librería ta ============================
def calculate_indicators(df):
    """Calcula EMAs, MACD, RSI, ATR, ADX y Bollinger Bands usando la librería ta."""
    # EMAs
    df['EMA12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['EMA26'] = ta.trend.ema_indicator(df['close'], window=26)
    df['EMA50'] = ta.trend.ema_indicator(df['close'], window=50)
    # MACD
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    # ATR
    atr_indicator = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['ATR'] = atr_indicator.average_true_range()
    # ADX
    adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['ADX'] = adx_indicator.adx()
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    return df

# ============================ Decisión de señal ============================
def decide_signal(df):
    """
    Determina la señal de entrada:
      - Para LONG:
          • Precio > EMA50.
          • Cruce alcista del MACD (de debajo a encima de la señal).
          • ADX > 25.
          • RSI entre 40 y 70.
          • Precio <= 1% por encima de la banda inferior.
      - Para SHORT:
          • Precio < EMA50.
          • Cruce bajista del MACD (de encima a debajo de la señal).
          • ADX > 25.
          • RSI entre 30 y 60.
          • Precio >= 0.99 * BB_upper.
    """
    if len(df) < 50:
        return None
    prev_macd = df['MACD'].iloc[-2]
    prev_signal = df['MACD_signal'].iloc[-2]
    curr_macd = df['MACD'].iloc[-1]
    curr_signal = df['MACD_signal'].iloc[-1]
    
    curr_close = df['close'].iloc[-1]
    ema50 = df['EMA50'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    adx = df['ADX'].iloc[-1]
    bb_lower = df['BB_lower'].iloc[-1]
    bb_upper = df['BB_upper'].iloc[-1]
    
    # Señal LONG
    if (curr_close > ema50 and
        prev_macd < prev_signal and curr_macd > curr_signal and
        adx > 25 and 40 < rsi < 70 and
        curr_close <= bb_lower * 1.01):
        return 'LONG'
    
    # Señal SHORT
    if (curr_close < ema50 and
        prev_macd > prev_signal and curr_macd < curr_signal and
        adx > 25 and 30 < rsi < 60 and
        curr_close >= bb_upper * 0.99):
        return 'SHORT'
    
    return None

# ============================ Cálculo de cantidad ============================
def calculate_quantity(usdt_amount):
    """Calcula la cantidad a operar basándose en el monto en USDT, precio actual y apalancamiento."""
    ticker = client.futures_symbol_ticker(symbol=symbol)
    price = float(ticker['price'])
    quantity = (usdt_amount * leverage) / price
    return round(quantity, 3)

# ============================ Ejecución de la orden ============================
def place_order(signal, usdt_amount):
    """
    Ejecuta la orden de entrada y coloca órdenes de SL y TP basados en ATR.
    En modo demo solo se envían notificaciones a Telegram sin realizar operaciones reales.
    """
    ticker = client.futures_symbol_ticker(symbol=symbol)
    entry_price = float(ticker['price'])
    quantity = calculate_quantity(usdt_amount)
    
    # Recalcular ATR
    df = get_candles(symbol, timeframe)
    df = calculate_indicators(df)
    atr = df['ATR'].iloc[-1]
    
    if signal == 'LONG':
        sl = round(entry_price - 1.5 * atr, 2)
        tp = round(entry_price + 2.0 * atr, 2)
        order_side = SIDE_BUY
        exit_side = SIDE_SELL
    elif signal == 'SHORT':
        sl = round(entry_price + 1.5 * atr, 2)
        tp = round(entry_price - 2.0 * atr, 2)
        order_side = SIDE_SELL
        exit_side = SIDE_BUY
    else:
        return None

    if DEMO:
        # En modo demo solo se notifica la señal
        send_telegram_message(f"[DEMO] Señal {signal} detectada a {entry_price}.\nCantidad simulada: {quantity}\nSL: {sl} | TP: {tp}")
        # Simulamos la posición con un diccionario
        return {
            'entry_price': entry_price,
            'quantity': quantity,
            'sl': sl,
            'tp': tp,
            'signal': signal
        }
    else:
        try:
            # Ejecutar orden de mercado real
            order = client.futures_create_order(
                symbol=symbol,
                side=order_side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            send_telegram_message(f"Orden {signal} ejecutada a {entry_price} por {quantity} unidades.")
        except Exception as e:
            send_telegram_message(f"Error al ejecutar orden {signal}: {e}")
            return None
        
        try:
            # Colocar orden Stop Loss
            sl_order = client.futures_create_order(
                symbol=symbol,
                side=exit_side,
                type=ORDER_TYPE_STOP_MARKET,
                stopPrice=sl,
                closePosition=True
            )
            # Colocar orden Take Profit
            tp_order = client.futures_create_order(
                symbol=symbol,
                side=exit_side,
                type=ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=tp,
                closePosition=True
            )
            send_telegram_message(f"SL configurado a {sl} y TP a {tp}.")
        except Exception as e:
            send_telegram_message(f"Error al configurar SL/TP: {e}")
        return {
            'entry_price': entry_price,
            'quantity': quantity,
            'sl': sl,
            'tp': tp,
            'signal': signal
        }

# ============================ Actualización del trailing stop ============================
def update_trailing_stop(position_info, atr):
    """
    Actualiza de forma dinámica el stop loss (trailing stop) en función del ATR.
    Para LONG, eleva el SL si el precio sube; para SHORT, lo reduce si baja.
    """
    factor_trail = 1.0
    try:
        ticker = client.futures_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        signal = position_info['signal']
        current_sl = position_info['sl']
        new_sl_candidate = None
        
        if signal == 'LONG':
            new_sl_candidate = round(current_price - atr * factor_trail, 2)
            if new_sl_candidate > current_sl:
                trailing_stop = new_sl_candidate
        elif signal == 'SHORT':
            new_sl_candidate = round(current_price + atr * factor_trail, 2)
            if new_sl_candidate < current_sl:
                trailing_stop = new_sl_candidate
        
        if new_sl_candidate and ((signal == 'LONG' and new_sl_candidate > current_sl) or 
                                  (signal == 'SHORT' and new_sl_candidate < current_sl)):
            if not DEMO:
                # Cancelar órdenes abiertas y reinsertar SL actualizado
                client.futures_cancel_all_open_orders(symbol=symbol)
                if signal == 'LONG':
                    client.futures_create_order(
                        symbol=symbol,
                        side=SIDE_SELL,
                        type=ORDER_TYPE_STOP_MARKET,
                        stopPrice=new_sl_candidate,
                        closePosition=True
                    )
                elif signal == 'SHORT':
                    client.futures_create_order(
                        symbol=symbol,
                        side=SIDE_BUY,
                        type=ORDER_TYPE_STOP_MARKET,
                        stopPrice=new_sl_candidate,
                        closePosition=True
                    )
            send_telegram_message(f"Trailing SL actualizado a {new_sl_candidate}.")
            position_info['sl'] = new_sl_candidate
    except Exception as e:
        send_telegram_message(f"Error al actualizar trailing stop: {e}")

# ============================ Monitoreo de posiciones ============================
def monitor_positions(position_info):
    """Monitorea la posición y actualiza el trailing stop; notifica si hay ganancias o pérdidas relevantes."""
    try:
        positions = client.futures_position_information(symbol=symbol)
        for pos in positions:
            if float(pos['positionAmt']) != 0:
                profit = float(pos['unrealizedProfit'])
                df = get_candles(symbol, timeframe)
                df = calculate_indicators(df)
                atr = df['ATR'].iloc[-1]
                update_trailing_stop(position_info, atr)
                if abs(profit) > 5:
                    send_telegram_message(f"Posición {position_info['signal']} en {symbol} con PnL: {profit} USDT.")
    except Exception as e:
        send_telegram_message(f"Error en monitorización: {e}")

# ============================ Bucle principal ============================
def main():
    usdt_amount = 20  # Monto en USDT a operar (ajusta entre 10 y 100 según tu necesidad)
    position_info = None
    while True:
        try:
            df = get_candles(symbol, timeframe)
            df = calculate_indicators(df)
            signal = decide_signal(df)
            if signal and position_info is None:
                send_telegram_message(f"Señal detectada: {signal}. Ejecutando operación.")
                position_info = place_order(signal, usdt_amount)
            if position_info:
                monitor_positions(position_info)
        except Exception as e:
            send_telegram_message(f"Error en ciclo principal: {e}")
        time.sleep(60)

if __name__ == '__main__':
    main()
