import sys
import asyncio
from datetime import datetime, timedelta
import time
import logging
import os
import requests
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_DOWN
from ta.momentum import StochasticOscillator, RSIIndicator
from ta.trend import IchimokuIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from typing import Dict
from dotenv import load_dotenv
from colorama import init, Fore, Style
import argparse
import threading

# Inicializar colorama
init(autoreset=True)

# Configuración avanzada de logging
class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: Fore.GREEN,
        logging.INFO: Fore.CYAN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }
    
    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, "")
        msg = super().format(record)
        return f"{color}{msg}{Style.RESET_ALL}"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler('eth_trading_bot.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Configurar la salida estándar para usar UTF-8 (soluciona problemas de emojis en Windows)
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from binance import Client, ThreadedWebsocketManager

# Cargar variables de entorno
load_dotenv()

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.chat_id = chat_id
        
    def send(self, message: str):
        try:
            url = f"{self.base_url}/sendMessage"
            params = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, params=params, timeout=10)
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
        risk_per_trade: float = 1,  # 1 = 100% del balance (usar con precaución)
        base_trailing_stop: float = 0.0075,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 3
    ):
        self.client = Client(api_key, api_secret)
        self.symbol = symbol
        self.risk_per_trade = risk_per_trade
        self.base_trailing_stop = base_trailing_stop
        self.timeframe = Client.KLINE_INTERVAL_15MINUTE
        self.positions: Dict = {}  # Se guardará la posición activa con su 'side'
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
        self.sma_window = 50
        self.ichimoku_conversion = 9
        self.ichimoku_base = 26
        self.ichimoku_lagging = 52
        self.adx_window = 14
        self.bb_window = 20
        self.atr_window = 14

        # Configurar apalancamiento
        self.leverage = 5

        # Variables para el status periódico
        self.start_time = datetime.now()
        self.last_status_time = datetime.now()

        self.update_market_info()
        self.validate_environment()
        self.set_leverage(self.leverage)
        self.start_price_stream()

        # Iniciar hilo para enviar status cada 30 minutos
        status_thread = threading.Thread(target=self.status_update_loop, daemon=True)
        status_thread.start()

    def set_leverage(self, leverage: int):
        """Fuerza el apalancamiento al valor indicado para el símbolo."""
        try:
            response = self.client.futures_change_leverage(symbol=self.symbol, leverage=leverage)
            logger.info(f"Apalancamiento configurado a x{leverage}: {response}")
        except Exception as e:
            logger.error(f"Error configurando apalancamiento: {e}")

    def update_market_info(self):
        """Actualiza información de mercado"""
        for _ in range(self.max_retries):
            try:
                info = self.client.futures_exchange_info()
                symbol_info = next(s for s in info['symbols'] if s['symbol'] == self.symbol)
                
                price_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER')
                self.tick_size = float(price_filter['tickSize'])
                
                lot_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE')
                self.lot_size = float(lot_filter['stepSize'])
                
                notional_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'MIN_NOTIONAL')
                self.min_notional = float(notional_filter.get('minNotional', 10.0))
                
                logger.info(f"Market Params | Tick: {self.tick_size} | Lot: {self.lot_size} | Min Notional: {self.min_notional}")
                return
            except Exception as e:
                logger.error(f"Error mercado: {str(e)}")
                time.sleep(2)
        raise ConnectionError("Failed to get market info")

    def validate_environment(self):
        """Valida variables de entorno"""
        required_vars = ['API_KEY', 'API_SECRET', 'TELEGRAM_TOKEN', 'TELEGRAM_CHAT_ID']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise EnvironmentError(f"Missing: {', '.join(missing)}")
        
        try:
            self.client.futures_ping()
            logger.info("✅ Conexión Binance OK")
        except Exception as e:
            raise ConnectionError(f"Binance error: {str(e)}")

    def get_data(self, limit: int = 100) -> pd.DataFrame:
        """Obtiene datos históricos"""
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
                logger.warning(f"Retry {attempt+1} data fetch: {str(e)}")
                time.sleep(1)
        raise ConnectionError("Failed to get data")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos mejorados"""
        try:
            # Momentum
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
            df['stoch_k'] = StochasticOscillator(
                df['high'], df['low'], df['close'], window=14, smooth_window=3
            ).stoch()
            
            # Trend: parámetros para Ichimoku
            ichimoku = IchimokuIndicator(df['high'], df['low'], self.ichimoku_conversion, self.ichimoku_base, self.ichimoku_lagging)
            df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
            df['ichimoku_base'] = ichimoku.ichimoku_base_line()
            df['ichimoku_cloud'] = ichimoku.ichimoku_a() - ichimoku.ichimoku_b()
            
            df['adx'] = ADXIndicator(
                df['high'], df['low'], df['close'], window=self.adx_window
            ).adx()
            
            # Volatilidad
            bb = BollingerBands(df['close'], window=self.bb_window)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            
            df['atr'] = AverageTrueRange(
                df['high'], df['low'], df['close'], window=self.atr_window
            ).average_true_range()
            
            # Volumen
            df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            
            return df.dropna()
        except Exception as e:
            logger.error(f"Indicator error: {str(e)}")
            raise

    def generate_signal(self, df: pd.DataFrame) -> str:
        """
        Genera señal basada en múltiples estrategias.
        La señal 'BUY' indica LONG y 'SELL' indica SHORT.
        Se ha ajustado para disparar una operación si se cumple al menos 1 condición.
        """
        try:
            current = df.iloc[-1]
            previous = df.iloc[-2]

            # Condiciones para LONG
            trend_strength = current['adx'] > 25
            cloud_bullish = current['ichimoku_conversion'] > current['ichimoku_base']
            price_above_cloud = current['close'] > current['ichimoku_base']
            obv_bullish = current['obv'] > previous['obv']
            
            buy_condition1 = all([
                current['rsi'] < 45,
                current['stoch_k'] > 20,
                cloud_bullish,
                price_above_cloud,
                trend_strength
            ])
            
            buy_condition2 = current['close'] > current['bb_upper']
            buy_condition3 = all([obv_bullish, current['close'] > current['ichimoku_conversion'], current['adx'] > 30])
            
            # Condiciones para SHORT
            cloud_bearish = current['ichimoku_conversion'] < current['ichimoku_base']
            price_below_cloud = current['close'] < current['ichimoku_base']
            obv_bearish = current['obv'] < previous['obv']
            
            sell_condition1 = all([
                current['rsi'] > 55,
                current['stoch_k'] < 80,
                cloud_bearish,
                price_below_cloud,
                trend_strength
            ])
            
            sell_condition2 = current['close'] < current['bb_lower']
            sell_condition3 = all([obv_bearish, current['close'] < current['ichimoku_base'], current['adx'] > 25])
            
            buy_signals = sum([buy_condition1, buy_condition2, buy_condition3])
            sell_signals = sum([sell_condition1, sell_condition2, sell_condition3])
            
            if buy_signals >= 1:
                return 'BUY'
            elif sell_signals >= 1:
                return 'SELL'
            
            return 'HOLD'
        except Exception as e:
            logger.error(f"Signal error: {str(e)}")
            return 'HOLD'

    def dynamic_risk_management(self) -> float:
        """Ajusta riesgo basado en volatilidad"""
        if len(self.last_prices) < 50:
            return self.risk_per_trade
            
        volatility = np.std(self.last_prices[-50:])
        if volatility > 0.05:
            return self.risk_per_trade * 0.5
        elif volatility < 0.02:
            return self.risk_per_trade * 1.5
        return self.risk_per_trade

    def round_down_quantity(self, quantity: float) -> float:
        """Redondea hacia abajo la cantidad de acuerdo a la precisión permitida (lot size)"""
        return float(Decimal(str(quantity)).quantize(Decimal(str(self.lot_size)), rounding=ROUND_DOWN))

    def calculate_position_size(self, price: float, stop_loss: float) -> float:
        """
        Calcula el tamaño de posición usando el balance disponible y un factor de riesgo.
        Se aplica un factor de seguridad al tamaño máximo permitido para evitar usar el 100% del margen.
        """
        try:
            balance = self.get_portfolio_balance()
            if balance <= 0:
                logger.error("Balance inválido")
                return 0.0
                
            risk = self.dynamic_risk_management()
            risk_amount = balance * risk
            risk_per_unit = abs(price - stop_loss)
            
            if risk_per_unit <= 0:
                logger.error("Stop loss inválido")
                return 0.0
                
            raw_size = risk_amount / risk_per_unit
            margin_safety_factor = 0.95  # Deja un 5% de colchón
            max_size = balance * self.leverage * margin_safety_factor / price
            final_size = min(raw_size, max_size)
            final_size = self.round_down_quantity(final_size)
            
            if float(final_size) * price < self.min_notional:
                logger.warning("Posición muy pequeña")
                return 0.0
                
            logger.info(f"Tamaño posición calculado: {final_size}")
            return float(final_size)
        except Exception as e:
            logger.error(f"Error cálculo tamaño: {str(e)}")
            return 0.0

    def open_position(self, side: str, price: float, atr: float):
        """
        Abre una posición según el lado.
        Para LONG: orden BUY, SL = precio - (atr*1.5) y TP = precio + (atr*2).
        Para SHORT: orden SELL, SL = precio + (atr*1.5) y TP = precio - (atr*2).
        """
        if side == 'LONG':
            stop_loss = price - (atr * 1.5)
            take_profit = price + (atr * 2)
            order_side = Client.SIDE_BUY
        elif side == 'SHORT':
            stop_loss = price + (atr * 1.5)
            take_profit = price - (atr * 2)
            order_side = Client.SIDE_SELL
        else:
            logger.error("Lado de posición desconocido")
            return

        size = self.calculate_position_size(price, stop_loss)
        if size <= 0:
            return

        quantity = self.round_down_quantity(size)
        quantity_str = format(quantity, 'f')
        price_str = str(round(price, int(-np.log10(self.tick_size))))
        
        order_msg = (
            f"🚀 *Nueva Entrada ({side})* \n"
            f"• Par: {self.symbol}\n"
            f"• Tipo: LIMITE\n"
            f"• Precio: {price:.2f}\n"
            f"• Tamaño: {quantity_str}\n"
            f"• SL: {stop_loss:.2f}\n"
            f"• TP: {take_profit:.2f}"
        )
        for attempt in range(self.max_retries):
            try:
                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=order_side,
                    type=Client.ORDER_TYPE_LIMIT,
                    timeInForce='GTC',
                    quantity=quantity_str,
                    price=price_str,
                    recvWindow=5000
                )
                if order['status'] in ['FILLED', 'NEW']:
                    self.positions[self.symbol] = {
                        'entry': price,
                        'sl': stop_loss,
                        'tp': take_profit,
                        'size': size,
                        'ts': self.base_trailing_stop,
                        'time': datetime.now(),
                        'side': side
                    }
                    self.metrics['trades'] += 1
                    self.telegram.send(order_msg)
                    logger.info(f"Posición {side} abierta exitosamente")
                    break
            except Exception as e:
                logger.error(f"Error orden {order_side} al abrir posición: {str(e)}")
                time.sleep(1)

    def close_position(self, closing_signal: str, price: float):
        """
        Cierra la posición abierta.
        Si la posición es LONG se envía una orden SELL; si es SHORT se envía una orden BUY.
        Se consulta la posición real desde la cuenta y se utiliza esa cantidad para cerrar.
        En caso de obtener el error "ReduceOnly Order is rejected" (que indica que la posición ya se cerró),
        se ignora ese error y se actualizan las métricas.
        """
        if self.symbol not in self.positions:
            return
        pos = self.positions[self.symbol]
        order_side = Client.SIDE_SELL if pos['side'] == 'LONG' else Client.SIDE_BUY

        # Consultar la posición actual desde la cuenta
        try:
            pos_info = self.client.futures_position_information(symbol=self.symbol)
            for p in pos_info:
                if p['symbol'] == self.symbol:
                    pos_amt = abs(float(p['positionAmt']))
                    break
            else:
                pos_amt = pos['size']
        except Exception as e:
            logger.error(f"Error obteniendo información de posición: {str(e)}")
            pos_amt = pos['size']

        quantity = self.round_down_quantity(pos_amt)
        quantity_str = format(quantity, 'f')

        order_msg = (
            f"🔻 *Cierre de Posición ({pos['side']})* \n"
            f"• Par: {self.symbol}\n"
            f"• Precio: {price:.2f}\n"
            f"• Tamaño: {quantity_str}\n"
            f"• Razón: Señal de reversión ({closing_signal})"
        )
        for attempt in range(self.max_retries):
            try:
                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=order_side,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity_str,
                    reduceOnly=True,
                    recvWindow=5000
                )
                if order['status'] == 'FILLED':
                    pnl = (price - pos['entry']) * pos['size'] if pos['side'] == 'LONG' else (pos['entry'] - price) * pos['size']
                    self.metrics['total_pnl'] += pnl
                    if pnl > 0:
                        self.metrics['wins'] += 1
                    else:
                        self.metrics['losses'] += 1
                    del self.positions[self.symbol]
                    profit_msg = (
                        f"💰 *Resultado Operación* \n"
                        f"• PnL: ${pnl:.2f}\n"
                        f"• Duración: {(datetime.now() - pos['time']).seconds // 60} mins\n"
                        f"• Balance Actual: ${self.get_portfolio_balance():.2f}"
                    )
                    self.telegram.send(order_msg)
                    self.telegram.send(profit_msg)
                    logger.info("Posición cerrada exitosamente")
                    break
            except Exception as e:
                if "ReduceOnly Order is rejected" in str(e):
                    logger.info("La posición se cerró, error 'ReduceOnly' ignorado.")
                    pnl = (price - pos['entry']) * pos['size'] if pos['side'] == 'LONG' else (pos['entry'] - price) * pos['size']
                    self.metrics['total_pnl'] += pnl
                    if pnl > 0:
                        self.metrics['wins'] += 1
                    else:
                        self.metrics['losses'] += 1
                    del self.positions[self.symbol]
                    profit_msg = (
                        f"💰 *Resultado Operación* \n"
                        f"• PnL: ${pnl:.2f}\n"
                        f"• Duración: {(datetime.now() - pos['time']).seconds // 60} mins\n"
                        f"• Balance Actual: ${self.get_portfolio_balance():.2f}"
                    )
                    self.telegram.send(order_msg)
                    self.telegram.send(profit_msg)
                    break
                else:
                    logger.error(f"Error al cerrar posición: {str(e)}")
                    time.sleep(1)

    def execute_trade(self, signal: str, price: float):
        """
        Ejecuta la operación según la señal:
          - 'BUY': Si no hay posición abre LONG; si hay posición SHORT, la cierra y abre LONG.
          - 'SELL': Si no hay posición abre SHORT; si hay posición LONG, la cierra y abre SHORT.
        """
        logger.info(f"🔎 Procesando señal {signal} a {price:.2f}")
        df = self.calculate_indicators(self.get_data(100))
        atr = df['atr'].iloc[-1]
        if signal == 'BUY':
            if self.symbol in self.positions:
                if self.positions[self.symbol]['side'] == 'SHORT':
                    self.close_position('BUY', price)
                    self.open_position('LONG', price, atr)
                else:
                    # Actualizar trailing stop o mantener posición
                    pass
            else:
                self.open_position('LONG', price, atr)
        elif signal == 'SELL':
            if self.symbol in self.positions:
                if self.positions[self.symbol]['side'] == 'LONG':
                    self.close_position('SELL', price)
                    self.open_position('SHORT', price, atr)
                else:
                    # Actualizar trailing stop o mantener posición
                    pass
            else:
                self.open_position('SHORT', price, atr)

    def update_trailing_stop(self, current_price: float):
        """Actualiza el trailing stop según la dirección de la posición"""
        if self.symbol in self.positions:
            pos = self.positions[self.symbol]
            if pos['side'] == 'LONG':
                new_stop = current_price * (1 - pos['ts'])
                if new_stop > pos['sl']:
                    pos['sl'] = new_stop
                    logger.info(f"Trailing stop LONG actualizado: {new_stop:.2f}")
                if current_price <= pos['sl'] or current_price >= pos['tp']:
                    self.close_position('SELL', current_price)
            else:  # SHORT
                new_stop = current_price * (1 + pos['ts'])
                if new_stop < pos['sl']:
                    pos['sl'] = new_stop
                    logger.info(f"Trailing stop SHORT actualizado: {new_stop:.2f}")
                if current_price >= pos['sl'] or current_price <= pos['tp']:
                    self.close_position('BUY', current_price)

    def get_portfolio_balance(self) -> float:
        """Obtiene balance de USDT en Futuros"""
        for _ in range(self.max_retries):
            try:
                balances = self.client.futures_account_balance()
                usdt_balance = next((float(b['balance']) for b in balances if b['asset'] == 'USDT'))
                return usdt_balance
            except Exception as e:
                logger.error(f"Error obteniendo balance: {str(e)}")
                time.sleep(1)
        return 0.0

    def start_price_stream(self):
        """Inicia stream de precios en tiempo real"""
        try:
            self.ws_manager = ThreadedWebsocketManager(
                api_key=self.client.API_KEY,
                api_secret=self.client.API_SECRET
            )
            self.ws_manager.start()
            self.ws_manager.start_symbol_ticker_socket(
                callback=self._handle_socket_message,
                symbol=self.symbol
            )
            logger.info("WebSocket iniciado")
        except Exception as e:
            logger.error(f"Error WebSocket: {str(e)}")

    def _handle_socket_message(self, msg):
        """Maneja actualizaciones de precios en tiempo real"""
        try:
            if 'c' in msg:
                price = float(msg['c'])
                self.last_prices.append(price)
                self.last_prices = self.last_prices[-100:]  # Mantener últimos 100 precios
                self.update_trailing_stop(price)
        except Exception as e:
            logger.error(f"Error mensaje WebSocket: {str(e)}")

    def check_circuit_breaker(self):
        """Verifica condiciones de parada de emergencia"""
        if self.error_count >= self.circuit_breaker_threshold:
            self.telegram.send("🔴 CIRCUIT BREAKER ACTIVADO")
            raise RuntimeError("Demasiados errores consecutivos")
            
        if len(self.last_prices) >= 11:
            recent_prices = np.array(self.last_prices[-11:])
            returns = np.diff(recent_prices) / recent_prices[:-1]
            if np.std(returns) > 0.1:
                self.telegram.send("⚡ Alta volatilidad - Circuit breaker")
                raise RuntimeError("Volatilidad extrema detectada")

    def send_status_update(self):
        """Envía un informe de status al Telegram"""
        try:
            uptime = datetime.now() - self.start_time
            balance = self.get_portfolio_balance()
            status_msg = (
                f"📊 *Status del Bot*\n"
                f"• Tiempo activo: {str(uptime).split('.')[0]}\n"
                f"• Trades ejecutados: {self.metrics['trades']}\n"
                f"• Ganadoras: {self.metrics['wins']}\n"
                f"• Perdedoras: {self.metrics['losses']}\n"
                f"• Win Rate: {self.metrics['win_rate']*100:.2f}%\n"
                f"• PnL Total: ${self.metrics['total_pnl']:.2f}\n"
                f"• Balance Actual: ${balance:.2f}\n"
                f"• Errores consecutivos: {self.error_count}"
            )
            self.telegram.send(status_msg)
            logger.info("Informe de status enviado al Telegram")
        except Exception as e:
            logger.error(f"Error enviando status: {str(e)}")

    def status_update_loop(self):
        """Bucle que envía el status cada 30 minutos"""
        while True:
            time.sleep(1800)  # Espera 30 minutos
            self.send_status_update()

    def run(self):
        """Bucle principal"""
        logger.info("🚀 Iniciando ETH Trading Bot")
        self.telegram.send("🤖 *Bot Iniciado*")
        
        try:
            while True:
                try:
                    self.check_circuit_breaker()
                    
                    df = self.get_data(100)
                    df = self.calculate_indicators(df)
                    current_price = df['close'].iloc[-1]
                    
                    signal = self.generate_signal(df)
                    self.execute_trade(signal, current_price)
                    self.update_trailing_stop(current_price)
                    
                    self.metrics['win_rate'] = self.metrics['wins'] / self.metrics['trades'] if self.metrics['trades'] > 0 else 0
                    logger.info(f"Métricas actualizadas: {self.metrics}")
                    
                    time.sleep(60)
                except Exception as e:
                    logger.error(f"Error bucle principal: {str(e)}")
                    time.sleep(30)
        except KeyboardInterrupt:
            self.telegram.send("🛑 Bot detenido manualmente")
            logger.info("Detención manual")
        except Exception as e:
            self.telegram.send(f"🔥 Error crítico: {str(e)}")
            logger.critical(f"Error fatal: {str(e)}")
            raise

def main():
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Enhanced ETH Trading Bot")
    parser.add_argument("--forcebuy", action="store_true", help="Forzar compra para testing.")
    parser.add_argument("--forcesell", action="store_true", help="Forzar venta para testing.")
    args = parser.parse_args()

    # Cargar variables de entorno y configurar el notificador de Telegram
    telegram = TelegramNotifier(
        token=os.getenv('TELEGRAM_TOKEN'),
        chat_id=os.getenv('TELEGRAM_CHAT_ID')
    )

    # Inicializar el bot
    bot = EnhancedETHTradingBot(
        api_key=os.getenv('API_KEY'),
        api_secret=os.getenv('API_SECRET'),
        symbol='ETHUSDT',
        telegram=telegram,
        risk_per_trade=1,          # Usar el 100% del balance para testing (ajusta según convenga)
        base_trailing_stop=0.0075
    )

    # Obtener datos actuales para poder calcular indicadores y extraer el precio y el ATR
    df = bot.get_data(100)
    df = bot.calculate_indicators(df)
    current_price = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]

    if args.forcebuy:
        logger.info("Forzando orden de compra (LONG) para testing.")
        bot.open_position("LONG", current_price, atr)
        return

    if args.forcesell:
        if bot.symbol not in bot.positions:
            logger.info("No existe posición abierta. Abriendo posición LONG para luego forzar venta.")
            bot.open_position("LONG", current_price, atr)
            time.sleep(2)
        logger.info("Forzando orden de venta (cierre de posición) para testing.")
        bot.close_position("SELL", current_price)
        return

    bot.run()

if __name__ == "__main__":
    main()
