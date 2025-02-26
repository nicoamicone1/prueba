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

file_handler = logging.FileHandler('bnb_trading_bot.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Configurar salida estándar para UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from binance import Client, ThreadedWebsocketManager

# Cargar variables de ambiente
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

class EnhancedBNBTradingBot:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        telegram: TelegramNotifier,
        base_trailing_stop: float = 0.0075,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 3,
        max_drawdown: float = 0.2
    ):
        self.client = Client(api_key, api_secret)
        self.symbol = symbol
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
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'peak_balance': 0.0
        }
        self.error_count = 0
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.max_retries = max_retries
        self.last_prices = []
        self.ws_manager = None
        self.max_drawdown = max_drawdown

        # Parámetros optimizados para BNB
        self.stoch_window = 14
        self.sma_window = 50
        self.ichimoku_conversion = 7
        self.ichimoku_base = 22
        self.ichimoku_lagging = 44
        self.adx_window = 14
        self.bb_window = 20
        self.atr_window = 14

        self.leverage = 10  # Apalancamiento
        self.POSITION_TOLERANCE = 1e-4  # Tolerancia para considerar posición cerrada
        self.trailing_tp_factor = 0.5  # Factor para actualizar TP de forma trailing

        self.start_time = datetime.now()
        self.last_status_time = datetime.now()

        self.update_market_info()
        self.validate_environment()
        self.set_leverage(self.leverage)
        self.start_price_stream()

        status_thread = threading.Thread(target=self.status_update_loop, daemon=True)
        status_thread.start()

    def set_leverage(self, leverage: int):
        try:
            response = self.client.futures_change_leverage(symbol=self.symbol, leverage=leverage)
            logger.info(f"Apalancamiento configurado a x{leverage}: {response}")
        except Exception as e:
            logger.error(f"Error configurando apalancamiento: {e}")

    def update_market_info(self):
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
        try:
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
            df['stoch_k'] = StochasticOscillator(
                df['high'], df['low'], df['close'], window=14, smooth_window=3
            ).stoch()
            
            ichimoku = IchimokuIndicator(df['high'], df['low'], self.ichimoku_conversion, self.ichimoku_base, self.ichimoku_lagging)
            df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
            df['ichimoku_base'] = ichimoku.ichimoku_base_line()
            df['ichimoku_cloud'] = ichimoku.ichimoku_a() - ichimoku.ichimoku_b()
            
            df['adx'] = ADXIndicator(
                df['high'], df['low'], df['close'], window=self.adx_window
            ).adx()
            
            bb = BollingerBands(df['close'], window=self.bb_window)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            
            df['atr'] = AverageTrueRange(
                df['high'], df['low'], df['close'], window=self.atr_window
            ).average_true_range()
            
            df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            
            return df.dropna()
        except Exception as e:
            logger.error(f"Indicator error: {str(e)}")
            raise

    def generate_signal(self, df: pd.DataFrame) -> str:
        try:
            current = df.iloc[-1]
            previous = df.iloc[-2]

            weights = {
                'rsi': 0.25,
                'stoch': 0.15,
                'ichimoku': 0.3,
                'adx': 0.15,
                'bb': 0.1,
                'obv': 0.05
            }

            buy_signals = {
                'rsi': current['rsi'] < 30,
                'stoch': current['stoch_k'] > 20,
                'ichimoku': current['close'] > current['ichimoku_base'],
                'adx': current['adx'] > 25,
                'bb': current['close'] > current['bb_upper'],
                'obv': current['obv'] > previous['obv']
            }

            sell_signals = {
                'rsi': current['rsi'] > 70,
                'stoch': current['stoch_k'] < 80,
                'ichimoku': current['close'] < current['ichimoku_base'],
                'adx': current['adx'] > 25,
                'bb': current['close'] < current['bb_lower'],
                'obv': current['obv'] < previous['obv']
            }

            buy_score = sum(weights[key] for key, value in buy_signals.items() if value)
            sell_score = sum(weights[key] for key, value in sell_signals.items() if value)

            if buy_score > sell_score and buy_score >= 0.6:
                return 'BUY'
            elif sell_score > buy_score and sell_score >= 0.6:
                return 'SELL'
            else:
                return 'HOLD'
        except Exception as e:
            logger.error(f"Signal error: {str(e)}")
            return 'HOLD'

    def round_down_quantity(self, quantity: float) -> float:
        if quantity <= 0:
            return 0.0
        precision = int(-np.log10(self.lot_size))
        return float(Decimal(str(quantity)).quantize(Decimal('0.' + '0'*precision), rounding=ROUND_DOWN))

    def calculate_position_size(self, price: float) -> float:
        balance = self.get_portfolio_balance()
        # Con apalancamiento, el tamaño notional disponible es: balance * 0.95 * leverage
        max_notional = balance * 0.95 * self.leverage
        raw_size = max_notional / price
        final_size = self.round_down_quantity(raw_size)
        if final_size * price < self.min_notional:
            logger.warning("Posición muy pequeña para el notional mínimo")
            return 0.0
        logger.info(f"Tamaño posición calculado: {final_size}")
        return final_size

    def get_current_position(self):
        try:
            pos_info = self.client.futures_position_information(symbol=self.symbol)
            for p in pos_info:
                if p['symbol'] == self.symbol:
                    pos_amt = float(p['positionAmt'])
                    # Si la posición es casi cero, se considera cerrada
                    if abs(pos_amt) < self.POSITION_TOLERANCE:
                        if self.symbol in self.positions:
                            del self.positions[self.symbol]
                            logger.info(f"Posición cerrada detectada para {self.symbol}, eliminada del registro")
                    else:
                        side = 'LONG' if pos_amt > 0 else 'SHORT'
                        if self.symbol not in self.positions:
                            self.positions[self.symbol] = {
                                'entry': float(p['entryPrice']),
                                'size': abs(pos_amt),
                                'side': side,
                                'sl': 0.0,
                                'tp': 0.0,
                                'ts': self.base_trailing_stop,
                                'time': datetime.now(),
                                'trailing_active': False
                            }
                            logger.warning(f"Posición {side} encontrada en Binance pero no en registro local para {self.symbol}")
                        elif self.positions[self.symbol]['side'] != side or abs(self.positions[self.symbol]['size'] - abs(pos_amt)) > 0.001:
                            logger.warning(f"Discrepancia en posición para {self.symbol}, actualizando desde Binance")
                            self.positions[self.symbol].update({
                                'entry': float(p['entryPrice']),
                                'size': abs(pos_amt),
                                'side': side
                            })
                    break
        except Exception as e:
            logger.error(f"Error obteniendo posición actual: {str(e)}")

    def open_position(self, side: str, price: float, df: pd.DataFrame):
        self.get_current_position()
        if self.symbol in self.positions:
            logger.info(f"Ya existe una posición abierta ({self.positions[self.symbol]['side']}) para {self.symbol}, no se abre una nueva")
            return
        
        atr = df['atr'].iloc[-1]
        if side == 'LONG':
            stop_loss = price - (atr * 1.5)
            take_profit = price + (atr * 3)
            order_side = Client.SIDE_BUY
        elif side == 'SHORT':
            stop_loss = price + (atr * 1.5)
            take_profit = price - (atr * 3)
            order_side = Client.SIDE_SELL
        else:
            logger.error("Lado desconocido")
            return

        size = self.calculate_position_size(price)
        if size <= 0:
            return

        quantity = self.round_down_quantity(size)
        if quantity <= 0:
            logger.warning("Cantidad de posición inválida después de redondeo")
            return

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
                        'size': quantity,
                        'ts': self.base_trailing_stop,
                        'time': datetime.now(),
                        'side': side,
                        'trailing_active': False
                    }
                    self.metrics['trades'] += 1
                    self.telegram.send(order_msg)
                    logger.info(f"Posición {side} abierta")
                    break
            except Exception as e:
                logger.error(f"Error orden {order_side}: {str(e)}")
                time.sleep(1)

    def close_position(self, closing_signal: str, price: float):
        if self.symbol not in self.positions:
            return
        pos = self.positions[self.symbol]
        order_side = Client.SIDE_SELL if pos['side'] == 'LONG' else Client.SIDE_BUY

        try:
            pos_info = self.client.futures_position_information(symbol=self.symbol)
            for p in pos_info:
                if p['symbol'] == self.symbol:
                    pos_amt = abs(float(p['positionAmt']))
                    break
            else:
                pos_amt = pos['size']
        except Exception as e:
            logger.error(f"Error obteniendo info de posición: {str(e)}")
            pos_amt = pos['size']

        quantity = self.round_down_quantity(pos_amt)
        if quantity <= 0:
            if self.symbol in self.positions:
                del self.positions[self.symbol]
            return

        quantity_str = format(quantity, 'f')

        order_msg = (
            f"🔻 *Cierre de Posición ({pos['side']})* \n"
            f"• Par: {self.symbol}\n"
            f"• Precio: {price:.2f}\n"
            f"• Tamaño: {quantity_str}\n"
            f"• Razón: {closing_signal}"
        )
        for attempt in range(self.max_retries):
            try:
                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=order_side,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity_str,
                    recvWindow=5000
                )
                if order['status'] == 'FILLED':
                    pnl = (price - pos['entry']) * pos['size'] if pos['side'] == 'LONG' else (pos['entry'] - price) * pos['size']
                    self.metrics['total_pnl'] += pnl
                    if pnl > 0:
                        self.metrics['wins'] += 1
                    else:
                        self.metrics['losses'] += 1
                    if self.metrics['trades'] > 0:
                        self.metrics['win_rate'] = self.metrics['wins'] / self.metrics['trades']
                    self.telegram.send(order_msg)
                    profit_msg = (
                        f"💰 *Resultado Operación* \n"
                        f"• PnL: ${pnl:.2f}\n"
                        f"• Duración: {(datetime.now() - pos['time']).seconds // 60} mins\n"
                        f"• Balance: ${self.get_portfolio_balance():.2f}"
                    )
                    self.telegram.send(profit_msg)
                    logger.info("Posición cerrada exitosamente")
                    self.get_current_position()
                    break
            except Exception as e:
                if "Quantity less than or equal to zero" in str(e):
                    logger.info("Posición ya cerrada, error 'Quantity less than or equal to zero' ignorado")
                    if self.symbol in self.positions:
                        del self.positions[self.symbol]
                    break
                else:
                    logger.error(f"Error al cerrar posición: {str(e)}")
                    time.sleep(1)

    def execute_trade(self, signal: str, price: float, df: pd.DataFrame):
        logger.info(f"🔎 Procesando señal {signal} a {price:.2f}")
        self.get_current_position()

        if self.symbol not in self.positions:
            if signal == 'BUY':
                self.open_position('LONG', price, df)
            elif signal == 'SELL':
                self.open_position('SHORT', price, df)
        else:
            pos = self.positions[self.symbol]
            if (pos['side'] == 'LONG' and signal == 'SELL') or (pos['side'] == 'SHORT' and signal == 'BUY'):
                self.close_position(signal, price)
                time.sleep(1)
                self.get_current_position()
                if self.symbol not in self.positions:
                    if signal == 'BUY':
                        self.open_position('LONG', price, df)
                    elif signal == 'SELL':
                        self.open_position('SHORT', price, df)
            else:
                logger.info(f"Señal {signal} coincide con la posición actual ({pos['side']}), no se realiza acción")

    def update_trailing_stop(self, current_price: float):
        if self.symbol in self.positions:
            pos = self.positions[self.symbol]
            # Para posiciones LONG
            if pos['side'] == 'LONG':
                if current_price > pos['entry'] * 1.01 and not pos['trailing_active']:
                    pos['trailing_active'] = True
                if pos['trailing_active']:
                    new_stop = max(pos['sl'], current_price * (1 - self.base_trailing_stop))
                    if new_stop > pos['sl']:
                        pos['sl'] = new_stop
                        logger.info(f"Trailing stop LONG actualizado: {new_stop:.2f}")
                    new_tp = max(pos['tp'], current_price + (current_price - pos['entry']) * self.trailing_tp_factor)
                    if new_tp > pos['tp']:
                        pos['tp'] = new_tp
                        logger.info(f"Trailing TP LONG actualizado: {new_tp:.2f}")
                if current_price <= pos['sl'] or current_price >= pos['tp']:
                    self.close_position('Trailing Stop or TP reached', current_price)
            else:  # Para posiciones SHORT
                if current_price < pos['entry'] * 0.99 and not pos['trailing_active']:
                    pos['trailing_active'] = True
                if pos['trailing_active']:
                    new_stop = min(pos['sl'], current_price * (1 + self.base_trailing_stop))
                    if new_stop < pos['sl']:
                        pos['sl'] = new_stop
                        logger.info(f"Trailing stop SHORT actualizado: {new_stop:.2f}")
                    new_tp = min(pos['tp'], current_price - (pos['entry'] - current_price) * self.trailing_tp_factor)
                    if new_tp < pos['tp']:
                        pos['tp'] = new_tp
                        logger.info(f"Trailing TP SHORT actualizado: {new_tp:.2f}")
                if current_price >= pos['sl'] or current_price <= pos['tp']:
                    self.close_position('Trailing Stop or TP reached', current_price)

    def get_portfolio_balance(self) -> float:
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
        try:
            if 'c' in msg:
                price = float(msg['c'])
                self.last_prices.append(price)
                self.last_prices = self.last_prices[-100:]
                self.update_trailing_stop(price)
        except Exception as e:
            logger.error(f"Error mensaje WebSocket: {str(e)}")

    def check_circuit_breaker(self):
        if self.error_count >= self.circuit_breaker_threshold:
            self.telegram.send("🔴 CIRCUIT BREAKER ACTIVADO")
            raise RuntimeError("Demasiados errores")
            
        if len(self.last_prices) >= 11:
            recent_prices = np.array(self.last_prices[-11:])
            returns = np.diff(recent_prices) / recent_prices[:-1]
            if np.std(returns) > 0.1:
                self.telegram.send("⚡ Alta volatilidad")
                raise RuntimeError("Volatilidad extrema")

        balance = self.get_portfolio_balance()
        if balance > self.metrics['peak_balance']:
            self.metrics['peak_balance'] = balance
        self.metrics['current_drawdown'] = (self.metrics['peak_balance'] - balance) / self.metrics['peak_balance'] if self.metrics['peak_balance'] > 0 else 0
        if self.metrics['current_drawdown'] > self.max_drawdown:
            self.telegram.send("🔴 Drawdown máximo")
            raise RuntimeError("Drawdown máximo alcanzado")

    def send_status_update(self):
        try:
            uptime = datetime.now() - self.start_time
            balance = self.get_portfolio_balance()
            status_msg = (
                f"📊 *Status del Bot*\n"
                f"• Tiempo activo: {str(uptime).split('.')[0]}\n"
                f"• Trades: {self.metrics['trades']}\n"
                f"• Ganadoras: {self.metrics['wins']}\n"
                f"• Perdedoras: {self.metrics['losses']}\n"
                f"• Win Rate: {self.metrics['win_rate']*100:.2f}%\n"
                f"• PnL Total: ${self.metrics['total_pnl']:.2f}\n"
                f"• Balance: ${balance:.2f}\n"
                f"• Drawdown: {self.metrics['current_drawdown']*100:.2f}%\n"
                f"• Errores: {self.error_count}"
            )
            self.telegram.send(status_msg)
            logger.info("Status enviado")
        except Exception as e:
            logger.error(f"Error status: {str(e)}")

    def status_update_loop(self):
        while True:
            time.sleep(1800)
            self.send_status_update()

    def run(self):
        logger.info("🚀 Iniciando BNB Trading Bot")
        self.telegram.send("🤖 *Bot Iniciado*")
        
        try:
            while True:
                try:
                    self.check_circuit_breaker()
                    df = self.get_data(100)
                    df = self.calculate_indicators(df)
                    current_price = df['close'].iloc[-1]
                    signal = self.generate_signal(df)
                    self.execute_trade(signal, current_price, df)
                    time.sleep(60)
                except Exception as e:
                    logger.error(f"Error bucle: {str(e)}")
                    time.sleep(30)
        except KeyboardInterrupt:
            self.telegram.send("🛑 Bot detenido manualmente")
            logger.info("Detención manual")
        except Exception as e:
            self.telegram.send(f"🔥 Error crítico: {str(e)}")
            logger.critical(f"Error fatal: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Enhanced BNB Trading Bot")
    parsers = parser.add_subparsers(dest='command')
    parsers.add_parser('run', help='Ejecutar el bot')
    parsers.add_parser('forcebuy', help='Forzar compra')
    parsers.add_parser('forcesell', help='Forzar venta')
    args = parser.parse_args()

    telegram = TelegramNotifier(
        token=os.getenv('TELEGRAM_TOKEN'),
        chat_id=os.getenv('TELEGRAM_CHAT_ID')
    )

    bot = EnhancedBNBTradingBot(
        api_key=os.getenv('API_KEY'),
        api_secret=os.getenv('API_SECRET'),
        symbol='BNBUSDT',
        telegram=telegram,
        base_trailing_stop=0.0075
    )

    df = bot.get_data(100)
    df = bot.calculate_indicators(df)
    current_price = df['close'].iloc[-1]

    if args.command == 'forcebuy':
        logger.info("Forzando compra (LONG).")
        bot.open_position("LONG", current_price, df)
    elif args.command == 'forcesell':
        if bot.symbol not in bot.positions:
            logger.info("No hay posición. Abriendo LONG para forzar venta.")
            bot.open_position("LONG", current_price, df)
            time.sleep(2)
        logger.info("Forzando venta.")
        bot.close_position("SELL", current_price)
    else:
        bot.run()

if __name__ == "__main__":
    main()
