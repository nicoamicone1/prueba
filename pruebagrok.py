import os
import time
import math
import threading
import requests
import pandas as pd
import argparse
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Inicialización de Colorama para impresión en consola
init(autoreset=True)

# Funciones de log para mejorar la legibilidad en consola
def log_info(message):
    print(Fore.CYAN + "[INFO] " + message + Style.RESET_ALL)

def log_success(message):
    print(Fore.GREEN + "[SUCCESS] " + message + Style.RESET_ALL)

def log_error(message):
    print(Fore.RED + "[ERROR] " + message + Style.RESET_ALL)

# Variables globales para seguimiento de la posición y el precio favorable
current_side = None    # "LONG", "SHORT" o None
best_price = None      # Precio máximo (para LONG) o mínimo (para SHORT) alcanzado desde la entrada

# Carga de variables de entorno
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

# Inicializa el cliente de Binance (Futures)
client = Client(API_KEY, API_SECRET)

# Parámetros de trading
SYMBOL = 'ETHUSDT'
TIMEFRAME = '5m'
LEVERAGE = 5

# Porcentajes para Stop Loss y Take Profit
SL_PERCENT = 0.02  # 2%
TP_PERCENT = 0.04  # 4%

def send_telegram_message(message):
    """Envía mensajes de notificación a Telegram y loggea en consola."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    params = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
    try:
        requests.get(url, params=params)
        log_info(f"Telegram: {message}")
    except Exception as e:
        log_error("Error al enviar Telegram: " + str(e))

def get_symbol_precisions(symbol):
    """
    Consulta exchangeInfo para obtener la precisión de cantidad (LOT_SIZE)
    y la precisión de precio (PRICE_FILTER) del símbolo.
    Retorna: (quantity_precision, price_precision)
    """
    log_info(f"Obteniendo precisiones para el símbolo {symbol}")
    try:
        exchange_info = client.futures_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                quantity_precision = None
                price_precision = None
                for f in s['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        step_size = float(f['stepSize'])
                        quantity_precision = int(round(-math.log10(step_size)))
                    if f['filterType'] == 'PRICE_FILTER':
                        tick_size = float(f['tickSize'])
                        price_precision = int(round(-math.log10(tick_size)))
                if quantity_precision is None:
                    quantity_precision = 3
                if price_precision is None:
                    price_precision = 2
                return quantity_precision, price_precision
    except Exception as e:
        send_telegram_message("Error al obtener precisiones: " + str(e))
    return 3, 2  # Defaults

def get_klines(symbol, interval, limit=50):
    """Obtiene datos históricos (velas) desde Binance Futures."""
    log_info(f"Obteniendo klines para {symbol} en intervalo {interval}")
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        data = []
        for k in klines:
            data.append({
                'open_time': k[0],
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5])
            })
        df = pd.DataFrame(data)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df
    except BinanceAPIException as e:
        send_telegram_message("Error al obtener klines: " + str(e))
        return None

def calculate_emas(df):
    """Calcula las EMAs de 9 y 21 periodos sobre el cierre."""
    log_info("Calculando EMAs (9 y 21 periodos)")
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    return df

def get_position():
    """
    Consulta la posición actual para SYMBOL.
    Retorna: (cantidad, precio_entrada)
    """
    log_info(f"Obteniendo posición actual para {SYMBOL}")
    try:
        positions = client.futures_position_information(symbol=SYMBOL)
        for pos in positions:
            if pos['symbol'] == SYMBOL:
                position_amt = float(pos['positionAmt'])
                entry_price = float(pos['entryPrice'])
                return position_amt, entry_price
        return 0, 0
    except BinanceAPIException as e:
        send_telegram_message("Error al obtener posición: " + str(e))
        return 0, 0

def get_current_price(symbol):
    """Obtiene el precio actual de mercado para el símbolo."""
    log_info(f"Obteniendo precio actual para {symbol}")
    try:
        ticker = client.futures_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except BinanceAPIException as e:
        send_telegram_message("Error al obtener precio: " + str(e))
        return None

def get_futures_wallet_balance():
    """
    Obtiene el balance disponible de USDT en la billetera de futuros.
    Retorna el balance total como float.
    """
    log_info("Obteniendo balance de futuros")
    try:
        balance_info = client.futures_account_balance()
        for b in balance_info:
            if b['asset'] == 'USDT':
                return float(b['balance'])
        return 0.0
    except BinanceAPIException as e:
        send_telegram_message("Error al obtener balance de futuros: " + str(e))
        return 0.0

def get_order_quantity():
    """
    Calcula la cantidad a operar usando el 50% del balance de USDT en la billetera de futuros.
    La cantidad se calcula como: (balance * 0.5 * LEVERAGE) / precio_actual.
    Se redondea según la precisión requerida para el par.
    """
    log_info("Calculando cantidad de orden")
    balance = get_futures_wallet_balance()
    current_price = get_current_price(SYMBOL)
    if current_price is None or balance == 0:
        return None
    quantity = (balance * 0.5 * LEVERAGE) / current_price
    quantity_precision, _ = get_symbol_precisions(SYMBOL)
    return round(quantity, quantity_precision)

def close_position(side):
    """
    Cierra la posición actual.
    Para posición LONG se vende (SELL) y para posición SHORT se compra (BUY).
    """
    global current_side, best_price
    log_info(f"Cerrando posición {side}")
    try:
        pos_amt, _ = get_position()
        if pos_amt == 0:
            log_info("No hay posición abierta para cerrar")
            return
        order_side = 'SELL' if side == 'LONG' else 'BUY'
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=order_side,
            type='MARKET',
            quantity=abs(pos_amt)
        )
        send_telegram_message(f"Posición {side} cerrada con orden: {order}")
        log_success(f"Posición {side} cerrada correctamente")
        current_side = None
        best_price = None
    except BinanceAPIException as e:
        send_telegram_message("Error al cerrar posición: " + str(e))
        log_error("Error al cerrar posición: " + str(e))

def set_leverage():
    """Establece el apalancamiento para el par."""
    log_info(f"Estableciendo apalancamiento {LEVERAGE} para {SYMBOL}")
    try:
        client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
    except BinanceAPIException as e:
        send_telegram_message("Error al fijar apalancamiento: " + str(e))
        log_error("Error al fijar apalancamiento: " + str(e))

def place_order(side):
    """
    Ejecuta una orden de mercado utilizando el 50% del balance disponible y coloca SL y TP.
    side: 'BUY' para LONG, 'SELL' para SHORT.
    """
    global best_price, current_side
    op = "LONG" if side == 'BUY' else "SHORT"
    log_info(f"Colocando orden {op} para {SYMBOL}")
    quantity = get_order_quantity()
    if quantity is None or quantity <= 0:
        send_telegram_message("No se pudo calcular la cantidad de orden.")
        log_error("Cantidad de orden no válida.")
        return
    try:
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        if 'avgPrice' in order and order['avgPrice']:
            entry_price = float(order['avgPrice'])
        else:
            entry_price = float(order['fills'][0]['price'])
        send_telegram_message(f"Orden {op} ejecutada a {entry_price} con cantidad {quantity}")
        log_success(f"Orden {op} ejecutada a {entry_price}")
        best_price = entry_price
        update_sl_tp(entry_price)
        current_side = op
    except BinanceAPIException as e:
        send_telegram_message("Error al colocar orden: " + str(e))
        log_error("Error al colocar orden: " + str(e))

def check_for_signals():
    """
    Revisa el cruce de las EMAs:
      - Retorna 'LONG' si la EMA9 cruza por encima de la EMA21.
      - Retorna 'SHORT' si la EMA9 cruza por debajo de la EMA21.
      - Retorna None si no hay señal.
    """
    df = get_klines(SYMBOL, TIMEFRAME, limit=50)
    if df is None or len(df) < 2:
        return None
    df = calculate_emas(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signal = None
    if prev['ema9'] <= prev['ema21'] and last['ema9'] > last['ema21']:
        signal = 'LONG'
    elif prev['ema9'] >= prev['ema21'] and last['ema9'] < last['ema21']:
        signal = 'SHORT'
    if signal:
        log_info(f"Señal detectada: {signal}")
    return signal

def update_sl_tp(reference_price):
    """
    Coloca (o actualiza) las órdenes de Stop Loss y Take Profit en base al precio de referencia.
    Para LONG: SL = referencia*(1-4%), TP = referencia*(1+8%)
    Para SHORT: SL = referencia*(1+4%), TP = referencia*(1-8%)
    Se cancelan todas las órdenes abiertas para el par antes de colocar las nuevas.
    """
    global current_side
    _, price_precision = get_symbol_precisions(SYMBOL)
    log_info(f"Actualizando SL/TP para {current_side} a partir de referencia {reference_price}")
    try:
        client.futures_cancel_all_open_orders(symbol=SYMBOL)
        if current_side == "LONG":
            new_sl = reference_price * (1 - SL_PERCENT)
            new_tp = reference_price * (1 + TP_PERCENT)
            stop_side = 'SELL'
        elif current_side == "SHORT":
            new_sl = reference_price * (1 + SL_PERCENT)
            new_tp = reference_price * (1 - TP_PERCENT)
            stop_side = 'BUY'
        else:
            return
        new_sl = round(new_sl, price_precision)
        new_tp = round(new_tp, price_precision)
        client.futures_create_order(
            symbol=SYMBOL,
            side=stop_side,
            type='STOP_MARKET',
            stopPrice=new_sl,
            closePosition=True
        )
        client.futures_create_order(
            symbol=SYMBOL,
            side=stop_side,
            type='TAKE_PROFIT_MARKET',
            stopPrice=new_tp,
            closePosition=True
        )
        send_telegram_message(f"SL/TP actualizados: SL en {new_sl}, TP en {new_tp}")
        log_success(f"SL/TP actualizados: SL en {new_sl}, TP en {new_tp}")
    except BinanceAPIException as e:
        send_telegram_message("Error al actualizar SL/TP: " + str(e))
        log_error("Error al actualizar SL/TP: " + str(e))

def monitor_trailing():
    """
    Cada 2 minutos, si hay posición abierta, revisa el precio actual y,
    si se mueve a favor (mayor para LONG, menor para SHORT), actualiza el 'best_price'
    y vuelve a colocar SL/TP en base al nuevo precio.
    """
    global best_price, current_side
    log_info("Iniciando monitoreo de trailing SL/TP")
    while True:
        if current_side is not None:
            current_price = get_current_price(SYMBOL)
            if current_price is None:
                time.sleep(120)
                continue
            if current_side == "LONG" and current_price > best_price:
                best_price = current_price
                update_sl_tp(best_price)
            elif current_side == "SHORT" and current_price < best_price:
                best_price = current_price
                update_sl_tp(best_price)
        time.sleep(120)

def send_status_update():
    """
    Envía un resumen del estado actual cada hora:
      - Posición actual
      - Precio de entrada
      - Precio actual
      - P/L en porcentaje
    """
    pos_amt, entry_price = get_position()
    current_price = get_current_price(SYMBOL)
    if pos_amt == 0 or current_price is None:
        message = "Sin posición abierta."
    else:
        if current_side == "LONG":
            profit = (current_price - entry_price) / entry_price * 100
        elif current_side == "SHORT":
            profit = (entry_price - current_price) / entry_price * 100
        message = (f"Posición: {current_side}\nEntrada: {entry_price}\n"
                   f"Precio actual: {current_price}\nP/L: {profit:.2f}%")
    send_telegram_message("Estatus horario:\n" + message)
    log_info("Enviada actualización horaria de estado.")

def monitor_status():
    """Cada 1 hora envía un status actualizado."""
    while True:
        send_status_update()
        time.sleep(3600)

def main():
    global current_side, best_price
    set_leverage()
    pos_amt, _ = get_position()
    if pos_amt > 0:
        current_side = "LONG"
    elif pos_amt < 0:
        current_side = "SHORT"
    send_telegram_message("Bot de trading iniciado.")
    log_success("Bot de trading iniciado.")

    # Inicia hilos para monitoreo de trailing SL/TP y estado
    threading.Thread(target=monitor_trailing, daemon=True).start()
    threading.Thread(target=monitor_status, daemon=True).start()

    # Bucle principal: revisa señales cada 1 minuto
    while True:
        signal = check_for_signals()
        if signal:
            log_info(f"Señal detectada: {signal}")
            # Si hay posición y la señal es contraria, cierra la posición
            if current_side is not None and current_side != signal:
                close_position(current_side)
                time.sleep(1)
            # Si no hay posición, abre la nueva en la dirección indicada
            if current_side is None:
                if signal == "LONG":
                    place_order('BUY')
                elif signal == "SHORT":
                    place_order('SELL')
        time.sleep(60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trading Bot con logs y pruebas forzadas.')
    parser.add_argument('--forcelong', action='store_true', help='Forzar operación LONG de prueba')
    parser.add_argument('--forceshort', action='store_true', help='Forzar operación SHORT de prueba')
    args = parser.parse_args()

    # Si se especifica alguna flag de prueba, ejecuta la operación forzada y termina.
    if args.forcelong:
        set_leverage()
        log_info("Ejecutando operación forzada LONG...")
        pos_amt, _ = get_position()
        if pos_amt != 0:
            close_position(current_side)
            time.sleep(1)
        place_order('BUY')
        time.sleep(5)  # Espera para simular operación
        close_position("LONG")
        log_success("Operación forzada LONG completada.")
    elif args.forceshort:
        set_leverage()
        log_info("Ejecutando operación forzada SHORT...")
        pos_amt, _ = get_position()
        if pos_amt != 0:
            close_position(current_side)
            time.sleep(1)
        place_order('SELL')
        time.sleep(5)
        close_position("SHORT")
        log_success("Operación forzada SHORT completada.")
    else:
        main()
