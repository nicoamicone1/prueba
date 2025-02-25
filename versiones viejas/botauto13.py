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
import tensorflow as tf
from binance.client import Client
from websocket import create_connection
from threading import Thread, Lock
from colorama import Fore, Style, init
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from dotenv import load_dotenv

# ===========================
# CONFIGURACIÓN INICIAL
# ===========================
init(autoreset=True)

COLORES = {
    'status': Fore.CYAN,
    'error': Fore.RED,
    'warning': Fore.YELLOW,
    'success': Fore.GREEN,
    'banner': Fore.MAGENTA,
    'position': Fore.BLUE,
    'ai': Fore.LIGHTMAGENTA_EX
}

def print_separador():
    print(f"\n{COLORES['banner']}{'='*60}{Style.RESET_ALL}")

def print_encabezado():
    os.system('cls' if os.name == 'nt' else 'clear')
    print_separador()
    print(f"{COLORES['banner']}🔥 BOT DE SCALPING CON IA - APALANCAMIENTO 10X")
    print(f"{COLORES['status']}📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{COLORES['status']}🐍 Python: {sys.version.split()[0]}")
    print(f"{COLORES['status']}🧠 TensorFlow: {tf.__version__}")
    print_separador()

print_encabezado()

# ===========================
# CONFIGURACIÓN DE BINANCE
# ===========================
try:
    load_dotenv()
    print(f"{COLORES['status']}🔑 Variables de entorno cargadas")
except Exception as e:
    print(f"{COLORES['error']}❌ Error cargando variables: {e}")
    sys.exit(1)

try:
    client = Client(os.getenv('BINANCE_API'), os.getenv('BINANCE_SECRET'))
    print(f"{COLORES['success']}✅ Conexión exitosa con Binance Futures")
except Exception as e:
    print(f"{COLORES['error']}❌ Error de conexión: {e}")
    sys.exit(1)

# ===========================
# PARÁMETROS DE SCALPING
# ===========================
SIMBOLO = "ETHUSDT"
TIMEFRAMES = ['1m', '3m', '5m']
PESOS = [0.4, 0.4, 0.2]

PARAMETROS = {
    'riesgo_por_operacion': 1.0,
    'periodo_atr': 14,
    'apalancamiento_maximo': 10,
    'notional_minimo': 20,
    'stop_loss_pct': 0.5,  # Se reemplaza por niveles calculados con ATR
    'take_profit_pct': [0.5, 1.0],
    'ratios_profit': [0.7, 0.3],
    'intervalo_actualizacion': 60,        # Chequeo de señales cada 1 minuto
    'intervalo_actualizacion_modelo': 180,  # Reentrenamiento del modelo cada 3 minutos
    'longitud_secuencia_lstm': 60,
    'ventana_prediccion_lstm': 5,
    'umbral_adx': 15,
    'umbral_rsi_compra': 45,
    'umbral_rsi_venta': 55,
    'umbral_macd': 0.1,
    'trailing_factor': 0.5,              # Factor para trailing stop (múltiplo de ATR)
    'tp_multiplier': 2.0,                # Multiplicador para Take Profit
    'sl_multiplier': 1.0                 # Multiplicador para Stop Loss
}

# Variables globales para seguimiento de trade
trade_entry_time = None
trade_entry_price = None

# ===========================
# MODELO LSTM MEJORADO
# ===========================
class PredictorLSTM:
    def __init__(self):
        self.modelo = self.construir_modelo()
        self.escalador = MinMaxScaler(feature_range=(0, 1))
    
    def construir_modelo(self):
        modelo = Sequential([
            Input(shape=(PARAMETROS['longitud_secuencia_lstm'], 7)),
            LSTM(96, return_sequences=True),
            Dropout(0.4),
            LSTM(64),
            Dropout(0.3),
            Dense(PARAMETROS['ventana_prediccion_lstm'] * 3)
        ])
        modelo.compile(optimizer='adam', loss='mse')
        return modelo
    
    def preparar_datos(self, df):
        try:
            df = df.dropna()
            # Seleccionamos las columnas necesarias
            datos = df[['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']].values
            datos_escalados = self.escalador.fit_transform(datos)
            X, y = [], []
            total_length = PARAMETROS['longitud_secuencia_lstm'] + PARAMETROS['ventana_prediccion_lstm']
            for i in range(len(datos_escalados) - total_length):
                X.append(datos_escalados[i:i+PARAMETROS['longitud_secuencia_lstm']])
                y.append(datos_escalados[i+PARAMETROS['longitud_secuencia_lstm']:
                                         i+total_length, 1:4].flatten())
            return np.array(X), np.array(y)
        except Exception as e:
            print(f"{COLORES['error']}❌ Error preparando datos: {e}")
            return None, None
    
    def entrenar(self, df):
        X, y = self.preparar_datos(df)
        if X is not None and y is not None and len(X) > 0:
            try:
                print(f"{COLORES['ai']}🧠 Entrenando modelo LSTM con {len(X)} muestras...")
                early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
                self.modelo.fit(X, y, epochs=20, batch_size=32, verbose=0, callbacks=[early_stop])
                print(f"{COLORES['success']}✅ Modelo entrenado correctamente")
            except Exception as e:
                print(f"{COLORES['error']}❌ Error entrenando modelo: {e}")
        else:
            print(f"{COLORES['error']}❌ No hay datos suficientes para entrenar el modelo")
    
    def predecir(self, df):
        try:
            df_tail = df[['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']].tail(PARAMETROS['longitud_secuencia_lstm']).dropna()
            if len(df_tail) < PARAMETROS['longitud_secuencia_lstm']:
                raise ValueError("No hay suficientes datos limpios para predecir")
            datos = df_tail.values
            datos_escalados = self.escalador.transform(datos)
            X = datos_escalados.reshape(1, PARAMETROS['longitud_secuencia_lstm'], 7)
            pred = self.modelo.predict(X, verbose=0)[0]
            pred = pred.reshape(PARAMETROS['ventana_prediccion_lstm'], 3)
            # Desescalamos la predicción usando el rango del escalador para las columnas high, low, close
            inv_pred = (pred - self.escalador.min_[1:4]) / self.escalador.scale_[1:4]
            return inv_pred
        except Exception as e:
            print(f"{COLORES['error']}❌ Error prediciendo: {e}")
            return None

predictor_lstm = PredictorLSTM()

# ===========================
# MONITOR DE PRECIOS MEJORADO
# ===========================
class MonitorPrecios(Thread):
    def __init__(self, simbolo):
        Thread.__init__(self)
        self.simbolo = simbolo
        self.bid = None
        self.ask = None
        self.ejecutando = True
        self.ws = None
        self.bloqueo = Lock()
        
    def run(self):
        url_ws = f"wss://fstream.binance.com/ws/{self.simbolo.lower()}@bookTicker"
        while self.ejecutando:
            try:
                self.ws = create_connection(url_ws)
                with self.bloqueo:
                    print(f"{COLORES['success']}🌐 WebSocket conectado")
                while self.ejecutando:
                    try:
                        datos = json.loads(self.ws.recv())
                        with self.bloqueo:
                            self.bid = float(datos.get('b', self.bid or 0))
                            self.ask = float(datos.get('a', self.ask or 0))
                    except Exception as e:
                        print(f"{COLORES['error']}⚠️ Error recibiendo datos: {e}")
                        break
            except Exception as e:
                print(f"{COLORES['error']}⚠️ Error WS: {e}")
                time.sleep(3)
            finally:
                if self.ws:
                    self.ws.close()
                    
    def detener(self):
        self.ejecutando = False

# ===========================
# FUNCIONES DE GESTIÓN DE ÓRDENES Y POSICIONES
# ===========================
def cancelar_todas_las_ordenes():
    try:
        result = client.futures_cancel_all_open_orders(symbol=SIMBOLO)
        print(f"{COLORES['success']}Órdenes canceladas: {result}")
    except Exception as e:
        print(f"{COLORES['error']}Error cancelando órdenes: {e}")

def cerrar_posicion():
    global trade_entry_time, trade_entry_price
    try:
        positions = client.futures_position_information(symbol=SIMBOLO)
        for p in positions:
            amt = float(p['positionAmt'])
            if amt != 0:
                side = "SELL" if amt > 0 else "BUY"
                order = client.futures_create_order(
                    symbol=SIMBOLO,
                    side=side,
                    type="MARKET",
                    quantity=abs(amt)
                )
                print(f"{COLORES['success']}Posición cerrada: {order}")
        trade_entry_time = None
        trade_entry_price = None
    except Exception as e:
        print(f"{COLORES['error']}Error cerrando posición: {e}")

def hay_posicion_abierta():
    try:
        positions = client.futures_position_information(symbol=SIMBOLO)
        for p in positions:
            if float(p['positionAmt']) != 0:
                return True
        return False
    except Exception as e:
        print(f"{COLORES['error']}❌ Error obteniendo posición: {e}")
        return False

def obtener_direccion_posicion():
    try:
        positions = client.futures_position_information(symbol=SIMBOLO)
        for p in positions:
            amt = float(p['positionAmt'])
            if amt != 0:
                return 'BULLISH' if amt > 0 else 'BEARISH'
        return None
    except Exception as e:
        print(f"{COLORES['error']}❌ Error obteniendo dirección de posición: {e}")
        return None

def ejecutar_orden(simbolo, direccion, precio, df):
    global trade_entry_time, trade_entry_price
    try:
        atr = df['atr'].iloc[-1]
    except Exception as e:
        print(f"{COLORES['error']}❌ No se pudo obtener ATR para calcular SL/TP: {e}")
        return

    notional = PARAMETROS['notional_minimo']
    leverage = PARAMETROS['apalancamiento_maximo']
    qty = (notional * leverage) / precio
    qty = round(qty, 3)

    if direccion == 'BULLISH':
        SL = precio - atr * PARAMETROS['sl_multiplier']
        TP = precio + atr * PARAMETROS['tp_multiplier']
        side_entry = Client.SIDE_BUY
        side_exit = Client.SIDE_SELL
    else:
        SL = precio + atr * PARAMETROS['sl_multiplier']
        TP = precio - atr * PARAMETROS['tp_multiplier']
        side_entry = Client.SIDE_SELL
        side_exit = Client.SIDE_BUY

    print(f"{COLORES['success']}Ejecutando orden {direccion}: Entrada={precio}, SL={SL:.2f}, TP={TP:.2f}, Cantidad={qty}")

    try:
        entry_order = client.futures_create_order(
            symbol=simbolo,
            side=side_entry,
            type="MARKET",
            quantity=qty
        )
        print(f"{COLORES['success']}Orden de entrada ejecutada: {entry_order}")
        trade_entry_time = time.time()
        trade_entry_price = precio
    except Exception as e:
        print(f"{COLORES['error']}Error en orden de entrada: {e}")
        return

    try:
        sl_order = client.futures_create_order(
            symbol=simbolo,
            side=side_exit,
            type="STOP_MARKET",
            stopPrice=round(SL, 2),
            closePosition=True
        )
        print(f"{COLORES['success']}Orden SL ejecutada: {sl_order}")
    except Exception as e:
        print(f"{COLORES['error']}Error en orden SL: {e}")

    try:
        tp_order = client.futures_create_order(
            symbol=simbolo,
            side=side_exit,
            type="TAKE_PROFIT_MARKET",
            stopPrice=round(TP, 2),
            closePosition=False,
            quantity=qty
        )
        print(f"{COLORES['success']}Orden TP ejecutada: {tp_order}")
    except Exception as e:
        print(f"{COLORES['error']}Error en orden TP: {e}")

def actualizar_ordenes_dinamicamente():
    try:
        positions = client.futures_position_information(symbol=SIMBOLO)
        pos = None
        for p in positions:
            if float(p['positionAmt']) != 0:
                pos = p
                break
        if pos is None:
            return

        trade_direction = "BULLISH" if float(pos['positionAmt']) > 0 else "BEARISH"
        df = obtener_datos_historicos(SIMBOLO, '1m', 500)
        if df is None:
            return
        df = calcular_indicadores_completos(df)
        atr = df['atr'].iloc[-1]
        precio_actual = df['close'].iloc[-1]
        if trade_direction == "BULLISH":
            new_SL = precio_actual - atr * PARAMETROS['sl_multiplier']
            new_TP = precio_actual + atr * PARAMETROS['tp_multiplier']
            side_exit = "SELL"
        else:
            new_SL = precio_actual + atr * PARAMETROS['sl_multiplier']
            new_TP = precio_actual - atr * PARAMETROS['tp_multiplier']
            side_exit = "BUY"
        cancelar_todas_las_ordenes()
        sl_order = client.futures_create_order(
            symbol=SIMBOLO,
            side=side_exit,
            type="STOP_MARKET",
            stopPrice=round(new_SL, 2),
            closePosition=True
        )
        pos_amt = float(pos['positionAmt'])
        tp_order = client.futures_create_order(
            symbol=SIMBOLO,
            side=side_exit,
            type="TAKE_PROFIT_MARKET",
            stopPrice=round(new_TP, 2),
            closePosition=False,
            quantity=abs(pos_amt)
        )
        print(f"{COLORES['success']}Órdenes actualizadas: SL={round(new_SL,2)}, TP={round(new_TP,2)}")
    except Exception as e:
        print(f"{COLORES['error']}Error actualizando órdenes dinámicamente: {e}")

# ===========================
# FUNCIONES PARA OBTENER DATOS E INDICADORES
# ===========================
def obtener_datos_historicos(simbolo, intervalo, limite=500):
    try:
        velas = client.futures_klines(symbol=simbolo, interval=intervalo, limit=limite)
        df = pd.DataFrame(velas, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df[['timestamp', 'open', 'high', 'low', 'close', 'volume']] = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"{COLORES['error']}❌ Error obteniendo datos: {e}")
        return None

def calcular_indicadores_completos(df):
    try:
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd_diff()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        return df
    except Exception as e:
        print(f"{COLORES['error']}❌ Error calculando indicadores: {e}")
        return None

def analizar_con_lstm(df):
    try:
        if len(df) < 300:
            return 'NEUTRAL', None
        
        prediccion = predictor_lstm.predecir(df)
        if prediccion is None:
            return 'NEUTRAL', None
        
        precio_actual = df['close'].iloc[-1]
        precio_predicho = prediccion[-1][2]  # Usamos la predicción del "close"
        umbral_lstm = 0.02  # Diferencia mínima del 2%
        
        if precio_predicho > precio_actual * (1 + umbral_lstm):
            return 'BULLISH', prediccion
        elif precio_predicho < precio_actual * (1 - umbral_lstm):
            return 'BEARISH', prediccion
        return 'NEUTRAL', prediccion
    except Exception as e:
        print(f"{COLORES['error']}❌ Error en IA: {e}")
        return 'NEUTRAL', None

def obtener_señal_efectiva():
    try:
        puntaje_total = 0
        for tf_interval, peso in zip(TIMEFRAMES, PESOS):
            df = obtener_datos_historicos(SIMBOLO, tf_interval, 500)
            if df is not None:
                df = calcular_indicadores_completos(df)
                señal_lstm, prediccion = analizar_con_lstm(df)
                if señal_lstm == 'BULLISH':
                    puntaje_total += 5 * peso
                elif señal_lstm == 'BEARISH':
                    puntaje_total -= 5 * peso

                rsi = df['rsi'].iloc[-1]
                macd = df['macd'].iloc[-1]
                adx = df['adx'].iloc[-1]

                if adx > PARAMETROS['umbral_adx']:
                    puntaje_total += 2 * peso
                if rsi < PARAMETROS['umbral_rsi_compra'] and macd > PARAMETROS['umbral_macd']:
                    puntaje_total += 3 * peso
                elif rsi > PARAMETROS['umbral_rsi_venta'] and macd < -PARAMETROS['umbral_macd']:
                    puntaje_total -= 3 * peso

                print(f"{COLORES['ai']}📊 [{tf_interval}] | RSI: {rsi:.1f} | MACD: {macd:.4f} | ADX: {adx:.1f}")
                if prediccion is not None:
                    print(f"{COLORES['ai']}   🎯 LSTM Predicción (último high): {prediccion[-1][0]:.2f}")
                    log_recursos()

        print(f"{COLORES['ai']}🔥 Puntaje Total: {puntaje_total:.2f}")
        if puntaje_total >= 3:
            return 'BULLISH'
        elif puntaje_total <= -3:
            return 'BEARISH'
        return 'NEUTRAL'
    except Exception as e:
        print(f"{COLORES['error']}❌ Error en análisis: {e}")
        return 'NEUTRAL'

def log_recursos():
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    proceso = psutil.Process()
    uso_memoria = proceso.memory_info().rss / (1024 * 1024)
    log = f"🖥️ CPU: {cpu}% | RAM Total: {ram}% | RAM Bot: {uso_memoria:.2f} MB"
    print(log)
    return log

def actualizar_modelo():
    try:
        print(f"{COLORES['ai']}🧠 Actualizando modelo LSTM con datos recientes...")
        datos_historicos = obtener_datos_historicos(SIMBOLO, '1m', 1000)
        if datos_historicos is not None:
            datos_historicos = calcular_indicadores_completos(datos_historicos)
            predictor_lstm.entrenar(datos_historicos)
            print(f"{COLORES['success']}✅ Modelo LSTM actualizado")
        if hay_posicion_abierta():
            actualizar_ordenes_dinamicamente()
    except Exception as e:
        print(f"{COLORES['error']}❌ Error actualizando el modelo: {e}")

# ===========================
# EJECUCIÓN PRINCIPAL
# ===========================
def ejecucion_principal():
    monitor = MonitorPrecios(SIMBOLO)
    monitor.start()
    
    ultimo_entrenamiento = time.time()
    intervalo_actualizacion_modelo = PARAMETROS['intervalo_actualizacion_modelo']

    try:
        while True:
            if time.time() - ultimo_entrenamiento > intervalo_actualizacion_modelo:
                actualizar_modelo()
                ultimo_entrenamiento = time.time()
            
            señal = obtener_señal_efectiva()
            with monitor.bloqueo:
                precio = monitor.ask if señal == 'BULLISH' else monitor.bid
            
            if señal != 'NEUTRAL' and precio is not None:
                posicion = obtener_direccion_posicion()
                if posicion is None:
                    print(f"{COLORES['success']}🚀 Señal detectada: {señal} | Precio: {precio} (Sin posición abierta)")
                    cancelar_todas_las_ordenes()
                    df_actual = obtener_datos_historicos(SIMBOLO, '1m', 500)
                    if df_actual is not None:
                        df_actual = calcular_indicadores_completos(df_actual)
                        ejecutar_orden(SIMBOLO, señal, precio, df_actual)
                elif posicion == señal:
                    print(f"{COLORES['warning']}Posición abierta en misma dirección ({posicion}). Actualizando órdenes de protección...")
                    actualizar_ordenes_dinamicamente()
                else:
                    print(f"{COLORES['warning']}Posición abierta en dirección opuesta ({posicion}). Cerrando posición y abriendo nueva operación...")
                    cerrar_posicion()
                    cancelar_todas_las_ordenes()
                    df_actual = obtener_datos_historicos(SIMBOLO, '1m', 500)
                    if df_actual is not None:
                        df_actual = calcular_indicadores_completos(df_actual)
                        ejecutar_orden(SIMBOLO, señal, precio, df_actual)
            
            time.sleep(PARAMETROS['intervalo_actualizacion'])
            
    except KeyboardInterrupt:
        print(f"{COLORES['warning']}🛑 Deteniendo bot...")
        monitor.detener()
        monitor.join()

if __name__ == "__main__":
    print(f"{COLORES['ai']}🧠 Cargando datos históricos para entrenamiento inicial (1000 velas)...")
    datos_historicos = obtener_datos_historicos(SIMBOLO, '1m', 1000)
    if datos_historicos is not None:
        datos_historicos = calcular_indicadores_completos(datos_historicos)
        predictor_lstm.entrenar(datos_historicos)
    
    ejecucion_principal()
