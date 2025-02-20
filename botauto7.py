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
    'ai': Fore.YELLOW
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
    from dotenv import load_dotenv
    load_dotenv()
    print(f"{COLORES['status']}🔑 Variables de entorno cargadas")
except Exception as e:
    print(f"{COLORES['error']}❌ Error cargando variables: {e}")
    exit(1)

try:
    client = Client(os.getenv('BINANCE_API'), os.getenv('BINANCE_SECRET'))
    print(f"{COLORES['success']}✅ Conexión exitosa con Binance Futures")
except Exception as e:
    print(f"{COLORES['error']}❌ Error de conexión: {e}")
    exit(1)

# ===========================
# PARÁMETROS DE SCALPING
# ===========================
SIMBOLO = "ETHUSDT"
TIMEFRAMES = ['5m', '3m', '1m']
PESOS = [0.4, 0.4, 0.2]

PARAMETROS = {
    'riesgo_por_operacion': 1.0,
    'periodo_atr': 14,
    'apalancamiento_maximo': 10,
    'notional_minimo': 20,
    'stop_loss_pct': 0.5,
    'take_profit_pct': [2.0, 5.0],
    'ratios_profit': [0.7, 0.3],
    'duracion_maxima_operacion': 300,
    'intervalo_actualizacion': 15,
    'longitud_secuencia_lstm': 60,
    'ventana_prediccion_lstm': 5,
    'umbral_adx': 20,
    'umbral_rsi_compra': 45,
    'umbral_rsi_venta': 55,
    'umbral_macd': 0.3
}

# ===========================
# MODELO LSTM MEJORADO
# ===========================
class PredictorLSTM:
    def __init__(self):
        self.modelo = self.construir_modelo()
        self.escalador = MinMaxScaler(feature_range=(0, 1))
        
    def construir_modelo(self):
        modelo = Sequential([
            Input(shape=(PARAMETROS['longitud_secuencia_lstm'], 7)),  # Incluye open, high, low, close, volume, rsi, macd
            LSTM(96, return_sequences=True),
            Dropout(0.4),
            LSTM(64),
            Dropout(0.3),
            Dense(PARAMETROS['ventana_prediccion_lstm'] * 3)  # 5 periodos x 3 columnas (high, low, close)
        ])
        modelo.compile(optimizer='adam', loss='mse')
        return modelo
    
    def preparar_datos(self, df):
        try:
            # Eliminar filas con nulos para evitar problemas en el escalado y entrenamiento
            df = df.dropna()
            datos = df[['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']].values
            datos_escalados = self.escalador.fit_transform(datos)
            
            X, y = [], []
            for i in range(len(datos_escalados) - PARAMETROS['longitud_secuencia_lstm'] - PARAMETROS['ventana_prediccion_lstm']):
                X.append(datos_escalados[i:i+PARAMETROS['longitud_secuencia_lstm']])
                y.append(datos_escalados[i+PARAMETROS['longitud_secuencia_lstm']:
                                         i+PARAMETROS['longitud_secuencia_lstm']+PARAMETROS['ventana_prediccion_lstm'], 1:4].flatten())
            
            return np.array(X), np.array(y)
        except Exception as e:
            print(f"{COLORES['error']}❌ Error preparando datos: {e}")
            return None, None
    
    def entrenar(self, df):
        try:
            X, y = self.preparar_datos(df)
            if X is not None and y is not None:
                self.modelo.fit(X, y, epochs=50, batch_size=32, verbose=0)
        except Exception as e:
            print(f"{COLORES['error']}❌ Error entrenando modelo: {e}")
    
    def predecir(self, df):
        try:
            # Seleccionar las últimas filas y eliminar nulos
            df_tail = df[['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']].tail(PARAMETROS['longitud_secuencia_lstm']).dropna()
            if len(df_tail) < PARAMETROS['longitud_secuencia_lstm']:
                raise ValueError("No hay suficientes datos limpios para predecir")
            datos = df_tail.values
            datos_escalados = self.escalador.transform(datos)
            X = datos_escalados.reshape(1, PARAMETROS['longitud_secuencia_lstm'], 7)
            pred = self.modelo.predict(X, verbose=0)[0]
            # Reestructurar la predicción a (ventana_prediccion_lstm, 3)
            pred = pred.reshape(PARAMETROS['ventana_prediccion_lstm'], 3)
            # Inversa para las columnas high, low, close usando la fórmula: X_original = (X_scaled - min_) / scale_
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
                            self.bid = float(datos['b'])
                            self.ask = float(datos['a'])
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
# NUEVA LÓGICA DE SEÑALES
# ===========================
def obtener_señal_efectiva():
    try:
        puntaje_total = 0
        
        for tf, peso in zip(TIMEFRAMES, PESOS):
            df = obtener_datos_historicos(SIMBOLO, tf, 500)
            if df is not None:
                df = calcular_indicadores_completos(df)
                
                # Señal LSTM
                señal, prediccion = analizar_con_lstm(df)
                if señal == 'BULLISH':
                    puntaje_total += 3 * peso  # Mayor peso a señales de IA
                elif señal == 'BEARISH':
                    puntaje_total -= 3 * peso
                
                # Condiciones técnicas
                rsi = df['rsi'].iloc[-1]
                macd = df['macd'].iloc[-1]
                adx = df['adx'].iloc[-1]
                
                # Puntaje por condiciones
                if adx > PARAMETROS['umbral_adx']:
                    puntaje_total += 1.5 * peso  # Mercado con tendencia
                if rsi < PARAMETROS['umbral_rsi_compra'] and macd > PARAMETROS['umbral_macd']:
                    puntaje_total += 2 * peso
                elif rsi > PARAMETROS['umbral_rsi_venta'] and macd < -PARAMETROS['umbral_macd']:
                    puntaje_total -= 2 * peso
                
                # Logs detallados
                print(f"{COLORES['ai']}📊 {tf} | RSI: {rsi:.1f} | MACD: {macd:.4f} | ADX: {adx:.1f}")
                if prediccion is not None:
                    print(f"{COLORES['ai']}   🎯 Predicción: {prediccion[-1][2]:.2f}")
                    log_recursos()
        
        # Validación final
        print(f"{COLORES['ai']}🔥 Puntaje Total: {puntaje_total:.2f}")
        if puntaje_total >= 2.5:
            return 'BULLISH'
        elif puntaje_total <= -2.5:
            return 'BEARISH'
        return 'NEUTRAL'
    
    except Exception as e:
        print(f"{COLORES['error']}❌ Error en análisis: {e}")
        return 'NEUTRAL'

# ===========================
# FUNCIONES AUXILIARES
# ===========================
def obtener_datos_historicos(simbolo, intervalo, limite=500):
    try:
        velas = client.futures_klines(
            symbol=simbolo,
            interval=intervalo,
            limit=limite
        )
        df = pd.DataFrame(velas, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
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
        
        predictor_lstm.entrenar(df)
        prediccion = predictor_lstm.predecir(df)
        
        if prediccion is None:
            return 'NEUTRAL', None
        
        precio_actual = df['close'].iloc[-1]
        precio_predicho = prediccion[-1][2]
        
        if precio_predicho > precio_actual * 1.01:
            return 'BULLISH', prediccion
        elif precio_predicho < precio_actual * 0.99:
            return 'BEARISH', prediccion
        return 'NEUTRAL', prediccion
    except Exception as e:
        print(f"{COLORES['error']}❌ Error en IA: {e}")
        return 'NEUTRAL', None

def log_recursos():
    cpu = psutil.cpu_percent(interval=1)  # Uso de CPU en porcentaje
    ram = psutil.virtual_memory().percent  # Uso de RAM en porcentaje
    proceso = psutil.Process()  # Obtener proceso actual
    uso_memoria = proceso.memory_info().rss / (1024 * 1024)  # RAM usada en MB

    log = f"🖥️ CPU: {cpu}% | RAM Total: {ram}% | RAM Bot: {uso_memoria:.2f} MB"
    print(log)  # Puedes guardarlo en un archivo si quieres
    return log

# ===========================
# EJECUCIÓN PRINCIPAL
# ===========================
def ejecucion_principal():
    monitor = MonitorPrecios(SIMBOLO)
    monitor.start()
    
    try:
        while True:
            señal = obtener_señal_efectiva()
            precio = monitor.ask if señal == 'BULLISH' else monitor.bid
            
            if señal != 'NEUTRAL':
                print(f"{COLORES['success']}🚀 Señal detectada: {señal}")
                # Aquí se implementaría la lógica de ejecución de órdenes
            
            time.sleep(PARAMETROS['intervalo_actualizacion'])
            
    except KeyboardInterrupt:
        print(f"{COLORES['warning']}🛑 Deteniendo bot...")
        monitor.detener()
        monitor.join()

if __name__ == "__main__":
    print(f"{COLORES['ai']}🧠 Cargando datos históricos (1000 velas)...")
    datos_historicos = obtener_datos_historicos(SIMBOLO, '1m', 1000)
    if datos_historicos is not None:
        datos_historicos = calcular_indicadores_completos(datos_historicos)
        predictor_lstm.entrenar(datos_historicos)
    
    ejecucion_principal()
