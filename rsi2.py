import ccxt
import telebot
import argparse
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os
import json
from dotenv import load_dotenv

# Configuración inicial
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

# Conexión al exchange
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
})

# Conexión a Telegram
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Configuración de argumentos
parser = argparse.ArgumentParser(description='Bot de trading mejorado')
parser.add_argument('--forcelong', action='store_true')
parser.add_argument('--forceshort', action='store_true')
args = parser.parse_args()

# Parámetros configurables
SYMBOL = 'ETH/USDT'
TIMEFRAME = '15m'
RSI_PERIOD = 14
EMA_PERIOD = 200
ATR_PERIOD = 14
RISK_PER_TRADE = 0.5  # 2% por operación
LEVERAGE = 5
LOG_FILE = "trading_log.csv"

class TradingBot:
    def __init__(self):
        self.position = None
        self.params = {
            'symbol': SYMBOL,
            'leverage': LEVERAGE,
            'risk': RISK_PER_TRADE,
            'rsi_period': RSI_PERIOD,
            'atr_period': ATR_PERIOD
        }
        
        # Inicializar exchange
        self.exchange = exchange
        self.exchange.load_markets()
        self.exchange.set_leverage(LEVERAGE, SYMBOL)
        
        # Cargar posición existente, si la hay
        self.position = self.initialize_position()
        
        # Log inicial de parámetros
        init_msg = f"Parámetros iniciales: SYMBOL={SYMBOL}, TIMEFRAME={TIMEFRAME}, RSI_PERIOD={RSI_PERIOD}, EMA_PERIOD={EMA_PERIOD}, ATR_PERIOD={ATR_PERIOD}, LEVERAGE={LEVERAGE}, RISK_PER_TRADE={RISK_PER_TRADE}"
        print(init_msg)
        self.write_log({'event': 'inicialización', 'message': init_msg})

    # --------------------------
    # Funciones principales
    # --------------------------
    
    def run(self):
        """Bucle principal de trading"""
        self.send_telegram("🚀 Bot de trading iniciado")
        print("Iniciando bucle principal...")
        self.write_log({'event': 'inicio', 'message': 'Bucle principal iniciado'})
        
        while True:
            try:
                # Obtener datos OHLCV
                df = self.fetch_ohlcv()
                current_price = df['close'].iloc[-1]
                last_candle_time = df['timestamp'].iloc[-1]
                log_msg = f"[{datetime.now(timezone.utc).isoformat()}] Datos recibidos: Último candle en {last_candle_time}, Precio actual: {current_price:.2f}"
                print(log_msg)
                self.write_log({'event': 'datos_ohlcv', 'timestamp_candle': last_candle_time.isoformat(), 'price': current_price})
                
                # Verificar si se ha cerrado un nuevo candle
                if self.is_new_candle(df):
                    self.write_log({'event': 'nuevo_candle', 'message': 'Nuevo candle detectado'})
                    print("Nuevo candle detectado.")
                    
                    # Calcular indicadores y generar señal
                    signal, rsi = self.generate_signal(df)
                    trend = self.check_trend(df)
                    atr = self.calculate_atr(df)
                    volatility = self.check_volatility(df)
                    
                    # Log de indicadores calculados
                    log_params = f"Indicadores calculados -> RSI: {rsi:.2f}, Señal: {signal}, Trend (EMA): {trend}, ATR: {atr:.4f}, Volatilidad: {volatility:.4f}"
                    print(log_params)
                    self.write_log({'event': 'indicadores', 'rsi': rsi, 'signal': signal, 'trend': trend, 'atr': atr, 'volatility': volatility})
                    
                    # Lógica de trading: ejecutar operación si no hay posición abierta
                    if not self.position:
                        if args.forcelong:
                            print("Forzando operación LONG por argumento.")
                            self.write_log({'event': 'trade_forzado', 'signal': 'buy', 'message': 'Forzando LONG'})
                            self.execute_trade('buy', current_price, atr)
                        elif args.forceshort:
                            print("Forzando operación SHORT por argumento.")
                            self.write_log({'event': 'trade_forzado', 'signal': 'sell', 'message': 'Forzando SHORT'})
                            self.execute_trade('sell', current_price, atr)
                        elif signal and trend:
                            print(f"Ejecutando trade: {signal.upper()} a precio {current_price:.2f}")
                            self.write_log({'event': 'trade_ejecución', 'signal': signal, 'price': current_price})
                            self.execute_trade(signal, current_price, atr)
                        else:
                            print("No se cumple la condición para operar.")
                            self.write_log({'event': 'trade_no_ejecutado', 'message': 'No se cumplen condiciones de señal o tendencia'})
                    else:
                        print("Ya existe una posición abierta. No se ejecuta nueva operación.")
                        self.write_log({'event': 'posicion_abierta', 'message': 'Posición abierta, sin nueva operación'})
                
                else:
                    print("No se detecta nuevo candle. Esperando...")
                    self.write_log({'event': 'espera', 'message': 'No se detecta nuevo candle'})
                
                # Verificar cierres o ajustes de posiciones (lógica pendiente)
                self.check_positions(current_price)
                
                time.sleep(60)  # Verificar cada minuto
                
            except Exception as e:
                self.handle_error(e)
                time.sleep(300)

    # --------------------------
    # Funciones de mercado
    # --------------------------
    
    def fetch_ohlcv(self, limit=100):
        """Obtener datos OHLCV"""
        ohlcv = self.exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return df
    
    def get_balance(self):
        """Obtener balance disponible en futuros"""
        balance = self.exchange.fetch_balance({'type': 'future'})
        return balance['USDT']['free']
    
    def get_open_positions(self):
        """Obtener posiciones abiertas para el símbolo"""
        positions = self.exchange.fetch_positions([SYMBOL])
        return [p for p in positions if float(p['contracts']) > 0]
    
    # --------------------------
    # Indicadores técnicos
    # --------------------------
    
    def calculate_rsi(self, df):
        """Calcular RSI usando EMA de las ganancias y pérdidas"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/self.params['rsi_period'], adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/self.params['rsi_period'], adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_atr(self, df):
        """Calcular ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(self.params['atr_period']).mean().iloc[-1]
    
    def check_trend(self, df):
        """Confirmar tendencia usando EMA"""
        df['ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
        return df['close'].iloc[-1] > df['ema'].iloc[-1]
    
    def check_volatility(self, df):
        """Evaluar volatilidad: ATR relativo al precio actual"""
        atr = self.calculate_atr(df)
        current_price = df['close'].iloc[-1]
        return atr / current_price
    
    # --------------------------
    # Lógica de trading
    # --------------------------
    
    def generate_signal(self, df):
        """Generar señal de trading basándose en RSI"""
        df['rsi'] = self.calculate_rsi(df)
        last_rsi = df['rsi'].iloc[-1]
        prev_rsi = df['rsi'].iloc[-2]
        
        if prev_rsi < 30 and last_rsi >= 30:
            return 'buy', last_rsi
        elif prev_rsi > 70 and last_rsi <= 70:
            return 'sell', last_rsi
        return None, last_rsi
    
    def execute_trade(self, signal, price, atr):
        """Ejecutar operación de trading según la señal"""
        size = self.calculate_position_size(price, atr)
        sl_price = price - (1.5 * atr) if signal == 'buy' else price + (1.5 * atr)
        tp_price = price + (3 * atr) if signal == 'buy' else price - (3 * atr)
        
        try:
            if signal == 'buy':
                self.exchange.create_market_buy_order(SYMBOL, size)
                self.create_sl_order(sl_price, size, 'sell')
                self.create_tp_order(tp_price, size, 'sell')
            else:
                self.exchange.create_market_sell_order(SYMBOL, size)
                self.create_sl_order(sl_price, size, 'buy')
                self.create_tp_order(tp_price, size, 'buy')
                
            self.position = {
                'side': signal,
                'size': size,
                'entry_price': price,
                'sl': sl_price,
                'tp': tp_price,
                'timestamp': datetime.now(timezone.utc)
            }
            
            trade_msg = f"🎯 Nueva operación {'LARGA' if signal == 'buy' else 'CORTA'}\nPrecio: {price:.2f}\nTamaño: {size:.4f}\nSL: {sl_price:.2f}\nTP: {tp_price:.2f}"
            print(trade_msg)
            self.write_log({'event': 'trade_ejecutado', 'signal': signal, 'price': price, 'size': size, 'sl': sl_price, 'tp': tp_price})
            self.send_telegram(trade_msg)
            
        except Exception as e:
            self.handle_error(e)
    
    def calculate_position_size(self, price, atr):
        """Calcular tamaño de posición basado en el riesgo asignado"""
        balance = self.get_balance()
        risk_amount = balance * self.params['risk']
        risk_per_contract = atr * price
        size = risk_amount / risk_per_contract
        return self.exchange.amount_to_precision(SYMBOL, size)
    
    # --------------------------
    # Gestión de órdenes
    # --------------------------
    
    def create_sl_order(self, price, size, side):
        """Crear orden de Stop Loss"""
        return self.exchange.create_order(
            SYMBOL, 'STOP_MARKET', side, size, None, {
                'stopPrice': self.exchange.price_to_precision(SYMBOL, price),
                'reduceOnly': True
            })
    
    def create_tp_order(self, price, size, side):
        """Crear orden de Take Profit"""
        return self.exchange.create_order(
            SYMBOL, 'TAKE_PROFIT_MARKET', side, size, None, {
                'stopPrice': self.exchange.price_to_precision(SYMBOL, price),
                'reduceOnly': True
            })
    
    # --------------------------
    # Utilidades
    # --------------------------
    
    def is_new_candle(self, df):
        """Determina si ya se cerró un nuevo candle"""
        last_candle_time = df['timestamp'].iloc[-1]
        now = datetime.now(timezone.utc)
        # Se considera nuevo candle si el tiempo transcurrido supera el período de la vela
        return (now - last_candle_time) > timedelta(minutes=int(TIMEFRAME[:-1]))
    
    def initialize_position(self):
        """Cargar posición existente al iniciar el bot, si hay alguna"""
        positions = self.get_open_positions()
        if positions:
            pos = positions[0]
            return {
                'side': pos['side'],
                'size': pos['contracts'],
                'entry_price': pos['entryPrice'],
                'sl': pos['stopLossPrice'],
                'tp': pos['takeProfitPrice']
            }
        return None
    
    def write_log(self, data):
        """Registrar datos en un archivo CSV"""
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **data
        }
        df = pd.DataFrame([log_entry])
        df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
    
    def handle_error(self, error):
        """Manejar errores enviando notificación y registrándolos"""
        error_msg = f"⚠️ Error: {str(error)}"
        print(error_msg)
        self.send_telegram(error_msg)
        self.write_log({'event': 'error', 'error': error_msg})
    
    def send_telegram(self, message):
        """Enviar mensaje a Telegram"""
        bot.send_message(TELEGRAM_CHAT_ID, message)
    
    def check_positions(self, current_price):
        """Verificar cierre o ajustes de posiciones abiertas (lógica pendiente)"""
        if self.position:
            # Aquí se puede implementar lógica para cerrar o ajustar posiciones según evolución del precio.
            log_pos = f"Posición abierta: {self.position['side']} a {self.position['entry_price']}, precio actual: {current_price}"
            print(log_pos)
            self.write_log({'event': 'check_positions', 'position': self.position, 'current_price': current_price})

# --------------------------
# Comandos de Telegram
# --------------------------

@bot.message_handler(commands=['estado'])
def handle_status(message):
    bot_instance = TradingBot()
    balance = bot_instance.get_balance()
    positions = bot_instance.get_open_positions()
    
    response = f"💰 Balance: {balance:.2f} USDT\n"
    response += f"📊 Posiciones abiertas: {len(positions)}"
    
    if positions:
        pos = positions[0]
        response += f"\n\n📈 Posición {pos['side'].upper()}\n"
        response += f"Tamaño: {pos['contracts']}\n"
        response += f"Precio entrada: {pos['entryPrice']}\n"
        response += f"SL: {pos['stopLossPrice']}\n"
        response += f"TP: {pos['takeProfitPrice']}"
    
    bot.reply_to(message, response)

@bot.message_handler(commands=['cerrar'])
def handle_close(message):
    bot_instance = TradingBot()
    positions = bot_instance.get_open_positions()
    
    if positions:
        pos = positions[0]
        # Determinar la orden de cierre según el lado de la posición
        if pos['side'] == 'buy':
            bot_instance.exchange.create_market_sell_order(SYMBOL, pos['contracts'])
        else:
            bot_instance.exchange.create_market_buy_order(SYMBOL, pos['contracts'])
        bot.reply_to(message, "✅ Posición cerrada")
    else:
        bot.reply_to(message, "⚠️ No hay posiciones abiertas")

# --------------------------
# Ejecución principal
# --------------------------

if __name__ == "__main__":
    # Iniciar bot de Telegram en segundo plano
    from threading import Thread
    Thread(target=bot.infinity_polling).start()
    
    # Iniciar bot de trading
    trading_bot = TradingBot()
    trading_bot.run()
