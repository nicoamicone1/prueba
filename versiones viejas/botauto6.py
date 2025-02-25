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
from threading import Thread, Lock
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
POSITION_COLOR = Fore.BLUE

def print_separator():
    print(f"\n{BANNER_COLOR}{'='*60}{Style.RESET_ALL}")

def print_banner():
    os.system('cls' if os.name == 'nt' else 'clear')
    print_separator()
    print(f"{BANNER_COLOR}🚀 BOT DE TRADING AVANZADO CON GESTIÓN DINÁMICA DE RIESGO")
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
SYMBOL = "ETHUSDT"
TIMEFRAMES = ['4h', '1h', '15m', '5m', '1m']
WEIGHTS = [0.3, 0.25, 0.2, 0.15, 0.1]

# ===========================
# PARÁMETROS DE TRADING MEJORADOS
# ===========================
TRADING_PARAMS = {
    'risk_per_trade': 1.0,
    'atr_period': 14,
    'ichimoku_params': (9, 26, 52),
    'adx_threshold': 25,
    'max_daily_loss': -0.05,
    'trailing_stop_multiplier': 2,
    'take_profit_multipliers': [1.5, 3.0],  # Multiplicadores de ATR para TP
    'profit_ratios': [0.6, 0.4],             # Distribución de TPs
    'stop_loss_multiplier': 1.8,             # Multiplicador para SL
    'min_notional': 20,
    'max_leverage': 10,
    'position_refresh_interval': 60         # Segundos entre actualizaciones
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
        self.lock = Lock()
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
                        with self.lock:
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
# GESTIÓN DE POSICIONES Y ÓRDENES
# ===========================
class PositionManager:
    def __init__(self):
        self.positions = {}
        self.orders = {}
        self.last_update = 0
        self.lock = Lock()
        
    def update_positions(self):
        try:
            with self.lock:
                self.positions = {}
                positions = client.futures_position_information()
                for p in positions:
                    if float(p['positionAmt']) != 0:
                        self.positions[p['symbol']] = {
                            'size': float(p['positionAmt']),
                            'entry_price': float(p['entryPrice']),
                            'mark_price': float(p['markPrice']),
                            'unrealized_pnl': float(p['unRealizedProfit'])
                        }
                
                self.orders = {}
                orders = client.futures_get_open_orders(symbol=SYMBOL)
                for o in orders:
                    self.orders[o['orderId']] = {
                        'type': o['type'],
                        'side': o['side'],
                        'price': float(o['price']),
                        'stop_price': float(o['stopPrice']) if o['stopPrice'] else None,
                        'quantity': float(o['origQty'])
                    }
                
                self.last_update = time.time()
                return True
        except Exception as e:
            print(f"{ERROR_COLOR}❌ Error actualizando posiciones: {e}")
            return False

    def get_position_info(self):
        self.update_positions()
        return self.positions.get(SYMBOL, None)

    def get_open_orders(self):
        self.update_positions()
        return self.orders

# ===========================
# FUNCIONES AUXILIARES MEJORADAS
# ===========================
position_manager = PositionManager()

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
        position = position_manager.get_position_info()
        orders = position_manager.get_open_orders()
        
        status = [
            f"{STATUS_COLOR}🕒 {datetime.now().strftime('%H:%M:%S')}",
            f"{SUCCESS_COLOR}💰 Precio: {price:.4f}" if price else f"{WARNING_COLOR}💰 Precio: N/A",
            f"{BANNER_COLOR}📊 Señal: {signal} (Puntaje: {score:.2f})" if score else f"{WARNING_COLOR}📊 Señal: N/A",
            f"{SUCCESS_COLOR}💵 Capital disponible: {capital:.2f} USDT"
        ]
        
        if position:
            pos_side = 'LONG' if position['size'] > 0 else 'SHORT'
            status.append(
                f"{POSITION_COLOR}📊 Posición {pos_side} | "
                f"Tamaño: {abs(position['size']):.4f} | "
                f"Entrada: {position['entry_price']:.2f} | "
                f"P&L: {position['unrealized_pnl']:.2f} USDT"
            )
            
            sl_orders = [o for o in orders.values() if o['type'] in ['STOP_MARKET', 'TRAILING_STOP_MARKET']]
            tp_orders = [o for o in orders.values() if o['type'] == 'TAKE_PROFIT']
            
            for o in sl_orders:
                status.append(f"{ERROR_COLOR}⛔ SL: {o['stop_price']:.2f}" if o['stop_price'] else f"{ERROR_COLOR}⛔ SL Dinámico")
            for o in tp_orders:
                status.append(f"{SUCCESS_COLOR}🎯 TP: {o['price']:.2f}")

        status.append(get_system_stats())
        print("\n".join(status))
        print_separator()
    except Exception as e:
        print(f"{ERROR_COLOR}❌ Error en logging: {e}")

def get_available_balance():
    try:
        balances = client.futures_account_balance()
        for asset in balances:
            if asset['asset'] == 'USDT':
                return float(asset['balance'])
        return 0.0
    except Exception as e:
        print(f"{ERROR_COLOR}❌ Error obteniendo balance: {e}")
        return 0.0

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
        return df[['timestamp','open','high','low','close','volume']].astype(float)
    except Exception as e:
        print(f"{ERROR_COLOR}❌ Error obteniendo datos: {e}")
        return None

# ===========================
# CÁLCULO DE INDICADORES ROBUSTO
# ===========================
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
        ).average_true_range().fillna(0)
        
        if df['atr'].iloc[-1] <= 0:
            raise ValueError("ATR no válido")
            
        return df
    except Exception as e:
        print(f"{ERROR_COLOR}❌ Error calculando indicadores: {e}")
        return None

# ===========================
# ANÁLISIS DE MERCADO (FUNCIÓN FALTANTE AGREGADA)
# ===========================
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
                if df is not None:
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
# GESTIÓN DE RIESGO MEJORADA
# ===========================
def set_leverage_and_mode():
    try:
        # Configurar modo de margen aislado
        client.futures_change_margin_type(
            symbol=SYMBOL,
            marginType='ISOLATED'
        )
        
        # Configurar apalancamiento
        client.futures_change_leverage(
            symbol=SYMBOL,
            leverage=TRADING_PARAMS['max_leverage']
        )
        print(f"{SUCCESS_COLOR}✅ Modo ISOLATED | Apalancamiento {TRADING_PARAMS['max_leverage']}x")
    except Exception as e:
        if 'No need to change margin type' in str(e):
            print(f"{WARNING_COLOR}⚠️ El modo de margen ya está configurado")
        else:
            print(f"{ERROR_COLOR}❌ Error configurando apalancamiento/modo: {e}")

def calculate_position_size(price, atr):
    try:
        set_leverage_and_mode()
        balance = get_available_balance()
        
        if balance < TRADING_PARAMS['min_notional']:
            raise ValueError(f"Balance insuficiente: {balance:.2f} < {TRADING_PARAMS['min_notional']}")
        
        risk_amount = balance * TRADING_PARAMS['risk_per_trade']
        position_size = (risk_amount / (TRADING_PARAMS['stop_loss_multiplier'] * atr)) 
        position_size *= TRADING_PARAMS['max_leverage']
        
        # Obtener parámetros del símbolo
        symbol_info = client.futures_exchange_info()['symbols']
        symbol_info = next(s for s in symbol_info if s['symbol'] == SYMBOL)
        lot_size_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE')
        
        step_size = float(lot_size_filter['stepSize'])
        min_qty = float(lot_size_filter['minQty'])
        max_qty = float(lot_size_filter['maxQty'])
        
        # Ajustar precisión
        precision = int(abs(math.log10(step_size)))
        size = round(position_size - (position_size % step_size), precision)
        size = max(min(size, max_qty), min_qty)
        
        # Verificar notional mínimo
        notional = size * price
        if notional < TRADING_PARAMS['min_notional']:
            raise ValueError(f"Notional insuficiente: {notional:.2f} < {TRADING_PARAMS['min_notional']}")
            
        return size
    except Exception as e:
        print(f"{ERROR_COLOR}❌ Error calculando tamaño: {e}")
        return 0

# ===========================
# EJECUCIÓN DE ÓRDENES MEJORADA
# ===========================
def execute_order(signal, price, atr):
    try:
        if atr <= 0:
            raise ValueError("ATR no válido")
            
        symbol_info = client.futures_exchange_info()['symbols']
        symbol_info = next(s for s in symbol_info if s['symbol'] == SYMBOL)
        price_precision = int(symbol_info['pricePrecision'])
        qty_precision = int(symbol_info['quantityPrecision'])
        
        size = calculate_position_size(price, atr)
        if size <= 0:
            return False
            
        quantity = round(size, qty_precision)
        side = 'BUY' if signal == 'BULLISH' else 'SELL'
        stop_loss_price = price - (TRADING_PARAMS['stop_loss_multiplier'] * atr * (-1 if side == 'SELL' else 1))
        
        print(f"{SUCCESS_COLOR}🚀 Ejecutando orden {side} de {quantity} {SYMBOL}")
        
        # Orden de mercado principal
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        
        entry_price = float(order['avgPrice'])
        
        # Stop Loss dinámico
        sl_order = client.futures_create_order(
            symbol=SYMBOL,
            side='SELL' if side == 'BUY' else 'BUY',
            type='STOP_MARKET',
            quantity=quantity,
            stopPrice=round(stop_loss_price, price_precision),
            closePosition=True,
            workingType='MARK_PRICE'
        )
        print(f"{ERROR_COLOR}⛔ SL colocado en {stop_loss_price:.2f}")

        # Take Profit escalonado
        for i, multiplier in enumerate(TRADING_PARAMS['take_profit_multipliers']):
            ratio = TRADING_PARAMS['profit_ratios'][i]
            tp_qty = round(quantity * ratio, qty_precision)
            tp_price = entry_price + (multiplier * atr * (1 if side == 'BUY' else -1))
            
            client.futures_create_order(
                symbol=SYMBOL,
                side='SELL' if side == 'BUY' else 'BUY',
                type='TAKE_PROFIT',
                timeInForce='GTC',
                quantity=tp_qty,
                price=round(tp_price, price_precision),
                stopPrice=round(tp_price * (0.999 if side == 'BUY' else 1.001), price_precision),
                workingType='MARK_PRICE'
            )
            print(f"{SUCCESS_COLOR}🎯 TP{i+1} colocado en {tp_price:.2f}")

        # Trailing Stop
        trailing_order = client.futures_create_order(
            symbol=SYMBOL,
            side='SELL' if side == 'BUY' else 'BUY',
            type='TRAILING_STOP_MARKET',
            quantity=quantity,
            activationPrice=round(price * (1.005 if side == 'BUY' else 0.995), price_precision),
            callbackRate=str(round(TRADING_PARAMS['trailing_stop_multiplier'] * (atr / price) * 100, 1)),
            workingType='MARK_PRICE'
        )
        print(f"{SUCCESS_COLOR}🔒 Trailing Stop activo desde {trailing_order['activationPrice']}")

        return True
        
    except Exception as e:
        print(f"{ERROR_COLOR}❌ Error ejecutando orden: {e}")
        return False

# ===========================
# BUCLE PRINCIPAL MEJORADO
# ===========================
def main_loop():
    monitor = PriceMonitor(SYMBOL)
    monitor.start()
    last_signal_check = 0
    last_position_check = 0
    
    try:
        while True:
            current_time = time.time()
            
            try:
                # Actualizar posiciones regularmente
                if current_time - last_position_check > TRADING_PARAMS['position_refresh_interval']:
                    position_manager.update_positions()
                    last_position_check = current_time
                
                if monitor.bid is None or monitor.ask is None:
                    time.sleep(1)
                    continue
                    
                if current_time - last_signal_check >= 60:
                    signal, score = get_market_signal()
                    price = monitor.ask if signal == 'BULLISH' else monitor.bid
                    balance = get_available_balance()
                    
                    df = get_historical_data(SYMBOL, '1h')
                    atr_value = 0
                    if df is not None:
                        df = calculate_indicators(df)
                        if df is not None and 'atr' in df.columns:
                            atr_value = df['atr'].iloc[-1]
                    
                    log_status(signal, score, price, balance)
                    last_signal_check = current_time
                    
                    # Verificar si ya hay posición abierta
                    position = position_manager.get_position_info()
                    
                    if not position and signal != 'NEUTRAL':
                        if atr_value > 0:
                            execute_order(signal, price, atr_value)
                        else:
                            print(f"{WARNING_COLOR}⚠️ ATR inválido, omitiendo operación")
                            
                time.sleep(5)
                
            except Exception as e:
                print(f"{ERROR_COLOR}⛔ Error en bucle principal: {str(e)}")
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