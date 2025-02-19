#!/usr/bin/env python3
import os
import sys
import time
import math
import json
import pandas as pd
import numpy as np
import ta
import psutil
from binance.client import Client
from websocket import create_connection
from threading import Thread
from colorama import Fore, Style, init
from datetime import datetime

# ===========================
# CONFIGURACIÓN INICIAL
# ===========================
init(autoreset=True)

# Configuración de colores
STATUS_COLOR = Fore.CYAN
ERROR_COLOR = Fore.RED
WARNING_COLOR = Fore.YELLOW
SUCCESS_COLOR = Fore.GREEN
BANNER_COLOR = Fore.MAGENTA

def print_separator():
    print(f"\n{BANNER_COLOR}{'='*60}{Style.RESET_ALL}")

def print_banner():
    os.system('cls' if os.name == 'nt' else 'clear')
    print_separator()
    print(f"{BANNER_COLOR}🛠️  INICIALIZANDO BOT DE TRADING")
    print(f"{STATUS_COLOR}📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{STATUS_COLOR}🐍 Python: {sys.version.split()[0]}")
    print(f"{STATUS_COLOR}📂 Directorio: {os.getcwd()}")
    print_separator()

print_banner()

# ===========================
# CONFIGURACIÓN DE BINANCE
# ===========================
try:
    from dotenv import load_dotenv
    load_dotenv()
    print(f"{STATUS_COLOR}🔑 Variables de entorno cargadas")
except Exception as e:
    print(f"{ERROR_COLOR}❌ Error cargando variables: {e}")
    exit(1)

try:
    client = Client(os.getenv('BINANCE_API'), os.getenv('BINANCE_SECRET'))
    print(f"{SUCCESS_COLOR}✅ Conexión exitosa con Binance Futures")
except Exception as e:
    print(f"{ERROR_COLOR}❌ Error de conexión: {e}")
    exit(1)

# Parámetros principales
SYMBOL = "BNBUSDT"
TIMEFRAMES = ['4h', '1h', '15m', '5m', '1m']
WEIGHTS = [0.3, 0.25, 0.2, 0.15, 0.1]

# ===========================
# PARÁMETROS DE TRADING
# ===========================
TRADING_PARAMS = {
    'risk_per_trade': 1,
    'atr_period': 14,
    'ichimoku_params': (9, 26, 52),
    'adx_threshold': 25,
    'max_daily_loss': -0.05,
    'trailing_stop_multiplier': 1.5,
    'profit_ratios': [0.5, 0.3, 0.2]
}

# ===========================
# CLASE DE MONITOR DE PRECIOS
# ===========================
class PriceMonitor(Thread):
    def __init__(self, symbol):
        Thread.__init__(self)
        self.symbol = symbol
        self.bid = None
        self.ask = None
        self.running = True
        self.ws = None
        print(f"{STATUS_COLOR}📡 Iniciando monitor de precios para {symbol}...")

    def run(self):
        ws_url = f"wss://fstream.binance.com/ws/{self.symbol.lower()}@bookTicker"
        reconnect_attempts = 0
        
        while self.running:
            try:
                self.ws = create_connection(ws_url)
                print(f"{SUCCESS_COLOR}🌐 WebSocket conectado")
                reconnect_attempts = 0
                
                while self.running:
                    try:
                        data = json.loads(self.ws.recv())
                        self.bid = float(data.get('b', 0)) if data.get('b') else None
                        self.ask = float(data.get('a', 0)) if data.get('a') else None
                    except Exception as e:
                        print(f"{WARNING_COLOR}⚠️ Error recibiendo datos: {e}")
                        break
                        
            except Exception as e:
                print(f"{WARNING_COLOR}⚠️ Error WS: {e} | Reconectando...")
                reconnect_attempts += 1
                time.sleep(min(2 ** reconnect_attempts, 30))
                
            finally:
                if self.ws:
                    self.ws.close()

    def stop(self):
        print(f"{WARNING_COLOR}🛑 Deteniendo monitor...")
        self.running = False

# ===========================
# FUNCIONES AUXILIARES
# ===========================
def get_system_stats():
    try:
        return (
            f"{STATUS_COLOR}💻 CPU: {psutil.cpu_percent()}% | "
            f"💾 Memoria: {psutil.virtual_memory().percent}%"
        )
    except Exception as e:
        return f"{WARNING_COLOR}⚠️ No se pudo obtener stats del sistema"

def log_status(signal, score, price, capital):
    try:
        status = [
            f"{STATUS_COLOR}🕒 {datetime.now().strftime('%H:%M:%S')}",
            f"{SUCCESS_COLOR}💰 Precio: {price:.4f}" if price is not None else f"{WARNING_COLOR}💰 Precio: N/A",
            f"{BANNER_COLOR}📊 Señal: {signal} (Puntaje: {score:.2f})" if score is not None else f"{WARNING_COLOR}📊 Señal: N/A",
            f"{SUCCESS_COLOR}💵 Capital disponible: {capital:.2f} USDT",
            get_system_stats()
        ]
        print("\n".join(status))
        print_separator()
    except Exception as e:
        print(f"{ERROR_COLOR}❌ Error en logging: {e}")

def get_historical_data(symbol, interval, limit=100):
    try:
        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
    except Exception as e:
        print(f"{ERROR_COLOR}❌ Error obteniendo datos: {e}")
        return None

def calculate_indicators(df):
    try:
        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(
            df['high'], df['low'],
            window1=TRADING_PARAMS['ichimoku_params'][0],
            window2=TRADING_PARAMS['ichimoku_params'][1],
            window3=TRADING_PARAMS['ichimoku_params'][2]
        )
        
        df['tenkan'] = ichimoku.ichimoku_conversion_line()
        df['kijun'] = ichimoku.ichimoku_base_line()
        df['senkou_a'] = ichimoku.ichimoku_a()
        df['senkou_b'] = ichimoku.ichimoku_b()
        
        # ADX y ATR
        df['adx'] = ta.trend.ADXIndicator(
            df['high'], df['low'], df['close'], 
            window=TRADING_PARAMS['atr_period']
        ).adx()
        
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], 
            window=TRADING_PARAMS['atr_period']
        ).average_true_range()
        
        return df
    except Exception as e:
        print(f"{ERROR_COLOR}❌ Error calculando indicadores: {e}")
        return None

def analyze_market(df):
    try:
        last = df.iloc[-1]
        score = 0
        
        # Estrategia Ichimoku
        if last['close'] > last['senkou_a'] and last['close'] > last['senkou_b']:
            score += 2.5
        
        # Fuerza de tendencia
        if last['adx'] > TRADING_PARAMS['adx_threshold']:
            score += 1.5 if last['close'] > last['open'] else -1.5
        
        # Momentum
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]
        macd = ta.trend.MACD(df['close']).macd_diff().iloc[-1]
        
        if rsi < 45 and macd > 0:
            score += 1.0
        elif rsi > 55 and macd < 0:
            score -= 1.0
            
        return 'BULLISH' if score >= 2.5 else 'BEARISH' if score <= -2.5 else 'NEUTRAL', score
    except Exception as e:
        print(f"{ERROR_COLOR}❌ Error en análisis: {e}")
        return 'NEUTRAL', 0

def get_market_signal():
    total_score = 0
    for tf, weight in zip(TIMEFRAMES, WEIGHTS):
        try:
            df = get_historical_data(SYMBOL, tf)
            if df is not None:
                df = calculate_indicators(df)
                signal, score = analyze_market(df)
                total_score += score * weight
                print(f"{STATUS_COLOR}📊 {tf}: {signal} ({score:.2f})")
        except Exception as e:
            print(f"{ERROR_COLOR}❌ Error en timeframe {tf}: {e}")
    
    if total_score >= 2.5:
        return 'BULLISH', total_score
    elif total_score <= -2.5:
        return 'BEARISH', total_score
    return 'NEUTRAL', total_score

# ===========================
# GESTIÓN DE RIESGO
# ===========================
def calculate_position_size(price, atr):
    try:
        balance = float(client.futures_account_balance()[0]['balance'])
        risk_amount = balance * TRADING_PARAMS['risk_per_trade']
        atr_value = atr * price
        size = risk_amount / (TRADING_PARAMS['trailing_stop_multiplier'] * atr_value)
        return min(size, balance * 0.5)  # Máximo 50% del balance
    except Exception as e:
        print(f"{ERROR_COLOR}❌ Error calculando tamaño: {e}")
        return 0

def adjust_precision(value, precision):
    try:
        return math.floor(value * 10**precision) / 10**precision
    except:
        return value

# ===========================
# EJECUCIÓN DE ÓRDENES
# ===========================
def execute_order(signal, price, atr):
    try:
        symbol_info = client.futures_exchange_info()['symbols']
        symbol_info = next(s for s in symbol_info if s['symbol'] == SYMBOL)
        
        # Precisión decimal
        price_precision = int(symbol_info['pricePrecision'])
        qty_precision = int(symbol_info['quantityPrecision'])
        
        # Cálculo de tamaño
        size = calculate_position_size(price, atr)
        if size <= 0:
            return False
            
        quantity = adjust_precision(size, qty_precision)
        
        # Dirección de la operación
        side = 'BUY' if signal == 'BULLISH' else 'SELL'
        stop_price = price * (1 - TRADING_PARAMS['trailing_stop_multiplier'] * atr/price) if side == 'BUY' else price * (1 + TRADING_PARAMS['trailing_stop_multiplier'] * atr/price)
        stop_price = adjust_precision(stop_price, price_precision)
        
        # Ejecutar orden
        print(f"{SUCCESS_COLOR}🚀 Ejecutando orden {side} de {quantity} {SYMBOL}")
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        
        # Colocar stop loss
        client.futures_create_order(
            symbol=SYMBOL,
            side='SELL' if side == 'BUY' else 'BUY',
            type='STOP_MARKET',
            stopPrice=stop_price,
            quantity=quantity,
            timeInForce='GTC'
        )
        
        return True
    except Exception as e:
        print(f"{ERROR_COLOR}❌ Error ejecutando orden: {e}")
        return False

        # En la función main_loop():

def get_available_balance():
    try:
        balances = client.futures_account_balance()
        usdt_balance = next(
            (item for item in balances if item['asset'] == 'USDT'), 
            {'balance': '0.0'}
        )
        return float(usdt_balance['balance'])
    except Exception as e:
        print(f"{ERROR_COLOR}❌ Error obteniendo balance: {e}")
        return 0.0

# ===========================
# BUCLE PRINCIPAL
# ===========================
def main_loop():
    monitor = PriceMonitor(SYMBOL)
    monitor.start()
    last_update = 0
    trade_active = False
    
    try:
        while True:
            current_time = time.time()
            
            try:
                # Esperar inicialización del monitor
                if monitor.bid is None or monitor.ask is None:
                    time.sleep(1)
                    continue
                    
                # Actualización cada 60 segundos
                if current_time - last_update >= 60:
                    signal, score = get_market_signal()
                    price = monitor.ask if signal == 'BULLISH' else monitor.bid
                    try:
                        balance = get_available_balance()
                    except:
                        balance = 0.0
                    
                    log_status(signal, score, price, balance)
                    last_update = current_time
                    
                    if not trade_active and signal != 'NEUTRAL':
                        df = get_historical_data(SYMBOL, '1h')
                        atr = df['atr'].iloc[-1] if df is not None else 0
                        if execute_order(signal, price, atr):
                            trade_active = True
                            print(f"{SUCCESS_COLOR}🎉 Operación activa: {signal}")
                            
                time.sleep(5)
                
            except Exception as e:
                print(f"{ERROR_COLOR}⛔ Error en bucle principal: {e}")
                time.sleep(10)
                
    except KeyboardInterrupt:
        print(f"\n{WARNING_COLOR}🛑 Deteniendo bot...")
        monitor.stop()
        monitor.join()
        print(f"{SUCCESS_COLOR}✅ Bot detenido correctamente")

if __name__ == "__main__":
    try:
        main_loop()
    except Exception as e:
        print(f"{ERROR_COLOR}⛔ Error crítico: {e}")
        exit(1)