import sys
import asyncio
import pandas as pd
import numpy as np
import time
import logging
import os
import requests
import locale
from decimal import Decimal, ROUND_DOWN
from datetime import datetime
from binance import Client, ThreadedWebsocketManager
from ta.momentum import StochasticOscillator, RSIIndicator
from ta.trend import IchimokuIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from typing import Dict
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Configurar locale para compatibilidad con Windows
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Inicializar colorama
init(autoreset=True)

# Configuración avanzada de logging con manejo de emojis
class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: Fore.GREEN,
        logging.INFO: Fore.CYAN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }
    
    EMOJI_MAP = {
        '🚀': '[ENTRADA]',
        '🔻': '[SALIDA]',
        '🤖': '[BOT]',
        '💰': '[GANANCIA]',
        '⚡': '[ALERTA]',
        '🔴': '[ERROR]',
        '🛑': '[DETENIDO]'
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, "")
        original_msg = super().format(record)
        
        # Convertir emojis para consola
        console_msg = original_msg
        for emoji, text in self.EMOJI_MAP.items():
            console_msg = console_msg.replace(emoji, text)
        
        # Mantener emojis originales para archivo
        if record.name == __name__ and self._fmt == file_formatter._fmt:
            return f"{original_msg}"
        return f"{color}{console_msg}{Style.RESET_ALL}"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Handler para consola
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

# Handler para archivo
file_handler = logging.FileHandler('eth_trading_bot.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Cargar variables de entorno
load_dotenv()

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.chat_id = chat_id
        self.validate_chat_id()

    def validate_chat_id(self):
        """Valida que el chat ID exista"""
        try:
            test_url = f"{self.base_url}/getChat?chat_id={self.chat_id}"
            response = requests.get(test_url, timeout=10)
            if not response.json().get('ok'):
                logger.error(f"Chat ID inválido: {response.json().get('description')}")
                raise ValueError("Chat ID de Telegram inválido")
        except Exception as e:
            logger.error(f"Error validando chat ID: {str(e)}")
            raise

    def send(self, message: str):
        """Envía mensaje con manejo de errores mejorado"""
        try:
            url = f"{self.base_url}/sendMessage"
            params = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, params=params, timeout=15)
            if response.status_code != 200:
                logger.error(f"Error Telegram: {response.text}")
        except Exception as e:
            logger.error(f"Error enviando a Telegram: {str(e)}")

class EnhancedETHTradingBot:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        telegram: TelegramNotifier,
        risk_per_trade: float = 0.02,
        base_trailing_stop: float = 0.0075,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 3
    ):
        self.client = Client(api_key, api_secret)
        self.symbol = symbol
        self.risk_per_trade = risk_per_trade
        self.base_trailing_stop = base_trailing_stop
        self.timeframe = Client.KLINE_INTERVAL_15MINUTE
        self.positions: Dict = {}
        self.telegram = telegram
        self.metrics = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0
        }
        self.error_count = 0
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.max_retries = max_retries
        self.last_prices = []
        self.ws_manager = None

        # Parámetros optimizados
        self.stoch_window = 14
        self.adx_window = 14
        self.bb_window = 20
        self.atr_window = 14

        self.update_market_info()
        self.validate_environment()
        self.start_price_stream()

    def update_market_info(self):
        """Actualiza información de mercado con reintentos"""
        attempts = 0
        max_attempts = 5
        base_delay = 1
        
        while attempts < max_attempts:
            try:
                info = self.client.futures_exchange_info()
                symbol_info = next(s for s in info['symbols'] if s['symbol'] == self.symbol)
                
                # Obtener parámetros con valores por defecto
                self.tick_size = float(next(
                    f for f in symbol_info['filters'] 
                    if f['filterType'] == 'PRICE_FILTER'
                )['tickSize'])
                
                self.lot_size = float(next(
                    f for f in symbol_info['filters'] 
                    if f['filterType'] == 'LOT_SIZE'
                )['stepSize'])
                
                self.min_notional = float(next(
                    f for f in symbol_info['filters'] 
                    if f['filterType'] == 'MIN_NOTIONAL'
                )['minNotional'])
                
                logger.info("Parámetros de mercado actualizados:")
                logger.info(f"• Tick Size: {self.tick_size}")
                logger.info(f"• Lot Size: {self.lot_size}")
                logger.info(f"• Min Notional: {self.min_notional}")
                return
                
            except StopIteration:
                logger.warning("Usando valores por defecto para parámetros de mercado")
                self.tick_size = 0.01
                self.lot_size = 0.001
                self.min_notional = 5.0
                return
            except Exception as e:
                attempts += 1
                delay = base_delay * 2 ** attempts
                logger.error(f"Error actualizando mercado (Intento {attempts}): {str(e)}")
                time.sleep(delay)
        
        raise ConnectionError("No se pudo obtener información del mercado")

    def validate_environment(self):
        """Valida el entorno de ejecución"""
        required_vars = ['API_KEY', 'API_SECRET', 'TELEGRAM_TOKEN', 'TELEGRAM_CHAT_ID']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise EnvironmentError(f"Variables faltantes: {', '.join(missing)}")
        
        try:
            # Test de conexión a Binance
            start = time.time()
            self.client.futures_ping()
            latency = (time.time() - start) * 1000
            logger.info(f"Conexión a Binance exitosa | Latencia: {latency:.2f}ms")
            
            # Verificar permisos de trading
            account_info = self.client.futures_account()
            if not account_info['canTrade']:
                raise PermissionError("La API no tiene permisos para operar")
                
        except Exception as e:
            logger.critical(f"Error de validación: {str(e)}")
            raise

    def get_data(self, limit: int = 100) -> pd.DataFrame:
        """Obtiene datos históricos con manejo de errores"""
        for attempt in range(self.max_retries):
            try:
                klines = self.client.futures_klines(
                    symbol=self.symbol,
                    interval=self.timeframe,
                    limit=limit
                )
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ]).astype({
                    'open': float, 'high': float, 'low': float, 
                    'close': float, 'volume': float
                })
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            except Exception as e:
                logger.warning(f"Error obteniendo datos (Intento {attempt+1}): {str(e)}")
                time.sleep(1)
        raise ConnectionError("No se pudieron obtener datos históricos")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos con validación"""
        try:
            # Momentum
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
            df['stoch_k'] = StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14,
                smooth_window=3
            ).stoch()
            
            # Tendencia
            ichimoku = IchimokuIndicator(
                high=df['high'],
                low=df['low'],
                window1=9,   # Línea de conversión
                window2=26,  # Línea base
                window3=52   # Línea retardada
            )
            df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
            df['ichimoku_base'] = ichimoku.ichimoku_base_line()
            df['ichimoku_span_a'] = ichimoku.ichimoku_a()
            df['ichimoku_span_b'] = ichimoku.ichimoku_b()
            
            # Volatilidad
            df['adx'] = ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.adx_window
            ).adx()
            
            bb = BollingerBands(df['close'], window=self.bb_window)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            
            df['atr'] = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.atr_window
            ).average_true_range()
            
            # Volumen
            df['obv'] = OnBalanceVolumeIndicator(
                close=df['close'],
                volume=df['volume']
            ).on_balance_volume()
            
            return df.dropna()
        except Exception as e:
            logger.error(f"Error calculando indicadores: {str(e)}")
            raise

    def generate_signal(self, df: pd.DataFrame) -> str:
        """Genera señal de trading con múltiples estrategias"""
        try:
            current = df.iloc[-1]
            prev = df.iloc[-2]

            # Condiciones de compra
            buy_conditions = [
                current['rsi'] < 45 and current['stoch_k'] > 20,
                current['close'] > current['ichimoku_span_a'],
                current['adx'] > 25,
                current['obv'] > prev['obv'],
                current['close'] > current['bb_upper']
            ]

            # Condiciones de venta
            sell_conditions = [
                current['rsi'] > 55 and current['stoch_k'] < 80,
                current['close'] < current['ichimoku_span_b'],
                current['adx'] > 30,
                current['obv'] < prev['obv'],
                current['close'] < current['bb_lower']
            ]

            if sum(buy_conditions) >= 3:
                return 'BUY'
            elif sum(sell_conditions) >= 3:
                return 'SELL'
            return 'HOLD'
        except Exception as e:
            logger.error(f"Error generando señal: {str(e)}")
            return 'HOLD'

    def dynamic_risk_management(self) -> float:
        """Ajusta el riesgo según la volatilidad"""
        if len(self.last_prices) < 50:
            return self.risk_per_trade
            
        returns = np.diff(self.last_prices[-50:]) / self.last_prices[-51:-1]
        volatility = np.std(returns)
        
        if volatility > 0.05:
            return self.risk_per_trade * 0.5
        elif volatility < 0.02:
            return self.risk_per_trade * 1.5
        return self.risk_per_trade

    def calculate_position_size(self, price: float, stop_loss: float) -> float:
        """Calcula el tamaño de posición con precisión"""
        try:
            balance = self.get_portfolio_balance()
            if balance <= 0:
                raise ValueError("Balance no válido")
            
            risk = self.dynamic_risk_management()
            risk_amount = balance * risk
            risk_per_unit = abs(price - stop_loss)
            
            if risk_per_unit <= 0:
                raise ValueError("Stop Loss inválido")
            
            raw_size = risk_amount / risk_per_unit
            position_size = Decimal(str(raw_size)).quantize(
                Decimal(str(self.lot_size)),
                rounding=ROUND_DOWN
            )
            
            if position_size * price < self.min_notional:
                raise ValueError("Tamaño de posición muy pequeño")
                
            return float(position_size)
        except Exception as e:
            logger.error(f"Error cálculo posición: {str(e)}")
            return 0.0

    def execute_trade(self, signal: str, price: float):
        """Ejecuta operaciones con gestión de errores"""
        try:
            if signal == 'BUY' and self.symbol not in self.positions:
                df = self.get_data(100)
                atr = df['atr'].iloc[-1]
                stop_loss = price - (atr * 1.5)
                take_profit = price + (atr * 3)
                
                size = self.calculate_position_size(price, stop_loss)
                if size <= 0:
                    return
                
                order_msg = (
                    f"🚀 **Nueva Entrada**\n"
                    f"• Par: `{self.symbol}`\n"
                    f"• Precio: `{price:.2f}`\n"
                    f"• Tamaño: `{size:.4f}`\n"
                    f"• SL: `{stop_loss:.2f}`\n"
                    f"• TP: `{take_profit:.2f}`"
                )
                
                for attempt in range(self.max_retries):
                    try:
                        order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=Client.SIDE_BUY,
                            type=Client.ORDER_TYPE_LIMIT,
                            timeInForce='GTC',
                            quantity=size,
                            price=str(round(price, int(-np.log10(self.tick_size))),
                            recvWindow=5000
                        ))
                        
                        if order['status'] == 'FILLED':
                            self.positions[self.symbol] = {
                                'entry': price,
                                'sl': stop_loss,
                                'tp': take_profit,
                                'size': size,
                                'ts': self.base_trailing_stop,
                                'time': datetime.now()
                            }
                            self.metrics['trades'] += 1
                            self.telegram.send(order_msg)
                            logger.info("Posición comprada exitosamente")
                            break
                            
                    except Exception as e:
                        logger.error(f"Error orden compra (Intento {attempt+1}): {str(e)}")
                        time.sleep(1)
                        
            elif signal == 'SELL' and self.symbol in self.positions:
                position = self.positions[self.symbol]
                size = position['size']
                
                order_msg = (
                    f"🔻 **Cierre de Posición**\n"
                    f"• Par: `{self.symbol}`\n"
                    f"• Precio: `{price:.2f}`\n"
                    f"• Tamaño: `{size:.4f}`"
                )
                
                for attempt in range(self.max_retries):
                    try:
                        order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=Client.SIDE_SELL,
                            type=Client.ORDER_TYPE_MARKET,
                            quantity=size,
                            recvWindow=5000
                        )
                        
                        if order['status'] == 'FILLED':
                            pnl = (price - position['entry']) * size
                            self.metrics['total_pnl'] += pnl
                            self.metrics['wins' if pnl > 0 else 'losses'] += 1
                            del self.positions[self.symbol]
                            
                            profit_msg = (
                                f"💰 **Resultado**\n"
                                f"• PnL: `${pnl:.2f}`\n"
                                f"• Duración: `{(datetime.now() - position['time']).seconds // 60}m`\n"
                                f"• Balance: `${self.get_portfolio_balance():.2f}`"
                            )
                            
                            self.telegram.send(order_msg)
                            self.telegram.send(profit_msg)
                            logger.info("Posición vendida exitosamente")
                            break
                            
                    except Exception as e:
                        logger.error(f"Error orden venta (Intento {attempt+1}): {str(e)}")
                        time.sleep(1)
                        
        except Exception as e:
            logger.error(f"Error ejecutando orden: {str(e)}")
            self.error_count += 1

    def get_portfolio_balance(self) -> float:
        """Obtiene el balance en USDT con validación"""
        for _ in range(self.max_retries):
            try:
                balances = self.client.futures_account_balance()
                usdt_balance = next(
                    float(b['balance']) for b in balances if b['asset'] == 'USDT'
                )
                return usdt_balance
            except Exception as e:
                logger.error(f"Error obteniendo balance: {str(e)}")
                time.sleep(1)
        return 0.0

    def start_price_stream(self):
        """Inicia el stream de precios en tiempo real"""
        try:
            self.ws_manager = ThreadedWebsocketManager(
                api_key=self.client.API_KEY,
                api_secret=self.client.API_SECRET
            )
            self.ws_manager.start()
            self.ws_manager.start_symbol_ticker_socket(
                callback=self.handle_price_update,
                symbol=self.symbol
            )
            logger.info("WebSocket de precios iniciado")
        except Exception as e:
            logger.error(f"Error iniciando WebSocket: {str(e)}")

    def handle_price_update(self, msg):
        """Actualiza precios y trailing stop"""
        try:
            if 'c' in msg:
                price = float(msg['c'])
                self.last_prices.append(price)
                self.last_prices = self.last_prices[-100:]
                self.update_trailing_stop(price)
        except Exception as e:
            logger.error(f"Error actualizando precio: {str(e)}")

    def update_trailing_stop(self, current_price: float):
        """Actualiza el stop loss dinámico"""
        if self.symbol in self.positions:
            position = self.positions[self.symbol]
            new_stop = current_price * (1 - position['ts'])
            
            # Ajustar trailing stop según ganancias
            price_change = current_price / position['entry']
            if price_change > 1.02:
                position['ts'] = self.base_trailing_stop * 0.8
            if price_change > 1.05:
                position['ts'] = self.base_trailing_stop * 0.5
                
            if new_stop > position['sl']:
                position['sl'] = new_stop
                logger.info(f"Trailing stop actualizado: {new_stop:.2f}")
                
            # Verificar triggers
            if current_price <= position['sl'] or current_price >= position['tp']:
                self.execute_trade('SELL', current_price)

    def check_circuit_breaker(self):
        """Verifica condiciones de parada de emergencia"""
        if self.error_count >= self.circuit_breaker_threshold:
            self.telegram.send("🔴 **CIRCUIT BREAKER ACTIVADO**")
            raise RuntimeError("Demasiados errores consecutivos")
            
        if len(self.last_prices) > 10:
            returns = np.diff(self.last_prices[-10:]) / self.last_prices[-11:-1]
            if np.std(returns) > 0.1:
                self.telegram.send("⚡ **Alta volatilidad - Operaciones detenidas**")
                raise RuntimeError("Volatilidad extrema detectada")

    def run(self):
        """Bucle principal de operación"""
        logger.info("Iniciando bot de trading")
        self.telegram.send("🤖 **Bot iniciado**")
        
        try:
            while True:
                try:
                    self.check_circuit_breaker()
                    
                    df = self.get_data(100)
                    df = self.calculate_indicators(df)
                    current_price = df['close'].iloc[-1]
                    
                    signal = self.generate_signal(df)
                    self.execute_trade(signal, current_price)
                    
                    # Actualizar métricas
                    self.metrics['win_rate'] = self.metrics['wins'] / self.metrics['trades'] if self.metrics['trades'] > 0 else 0
                    logger.info(f"Métricas actualizadas: {self.metrics}")
                    
                    time.sleep(60)
                    
                except Exception as e:
                    logger.error(f"Error en bucle principal: {str(e)}")
                    time.sleep(30)
                    
        except KeyboardInterrupt:
            self.telegram.send("🛑 **Bot detenido manualmente**")
            logger.info("Detención manual recibida")
        except Exception as e:
            self.telegram.send(f"🔥 **Error crítico**: `{str(e)}`")
            logger.critical(f"Error fatal: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Configurar Telegram
        telegram = TelegramNotifier(
            token=os.getenv('TELEGRAM_TOKEN'),
            chat_id=os.getenv('TELEGRAM_CHAT_ID')
        )
        
        # Inicializar bot
        bot = EnhancedETHTradingBot(
            api_key=os.getenv('API_KEY'),
            api_secret=os.getenv('API_SECRET'),
            symbol='ETHUSDT',
            telegram=telegram,
            risk_per_trade=0.02,
            base_trailing_stop=0.0075
        )
        
        # Iniciar operaciones
        bot.run()
        
    except Exception as e:
        logger.critical(f"Error inicial: {str(e)}")