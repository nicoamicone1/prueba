import os
import time, math, pandas as np
import talib
import pandas as pd
from colorama import Fore, Style, init
from binance.client import Client

# ===========================
#  LECTURA DE CLAVES DESDE KEYS.TXT
# ===========================
api_key = ""
api_secret = ""

try:
    with open("keys.txt", "r") as f:
        lines = f.read().splitlines()
    for line in lines:
        if "API_KEY" in line:
            api_key = line.split("=")[1].strip()
        elif "API_SECRET" in line:
            api_secret = line.split("=")[1].strip()
except Exception as e:
    print(f"{Fore.RED}Error leyendo keys.txt: {e}")
    api_key = "SIN_API_KEY"
    api_secret = "SIN_API_SECRET"

# ===========================
#  CONFIGURACIÓN DE API
# ===========================
client = Client(api_key, api_secret)
symbol = "BNBUSDT"

# Inicializar colorama para logs en color
init(autoreset=True)

# ===========================
#  PARÁMETROS DE TRADING
# ===========================
trailing_stop_pct = 0.005       # Trailing Stop del 0.5%
partial_profit_threshold = 0.02 # Toma parcial a partir de 2% de ganancia
partial_profit_ratio = 0.5      # Se cierra el 50% de la posición en toma parcial
exit_cooldown = 300             # Cooldown de 5 minutos tras cierre por pérdida

# Ajuste de capital y apalancamiento
capital_invertido = 30
apalancamiento = 10
capital_total = capital_invertido * apalancamiento
trade_capital_ratio = 1  # Usar el 100% del capital_total en cada trade

# Stop Loss y Take Profit iniciales
stop_loss_init_pct = 0.01  # 1%
take_profit_init_pct = 0.01 # 1%

# ===========================
#  FUNCIONES DE FORMATEO
# ===========================
def adjust_price(value, tick_size):
    return math.floor(value / tick_size) * tick_size

def adjust_quantity(value, step_size):
    return math.floor(value / step_size) * step_size

def format_price(price, tick_size):
    decimals = len(str(tick_size).split('.')[1])
    return f"{price:.{decimals}f}"

def format_quantity(qty, step_size):
    decimals = len(str(step_size).split('.')[1])
    return f"{qty:.{decimals}f}"

# ===========================
#  TRAILING STOP
# ===========================
def update_trailing_stop_order(symbol, quantity, sl_side, new_sl, step_size, tick_size):
    try:
        # Cancelar órdenes STOP_MARKET existentes
        orders = client.futures_get_open_orders(symbol=symbol)
        for order in orders:
            if order.get("type") == "STOP_MARKET":
                client.futures_cancel_order(symbol=symbol, orderId=order["orderId"])
        new_order = client.futures_create_order(
            symbol=symbol,
            side=sl_side,
            type="STOP_MARKET",
            quantity=format_quantity(quantity, step_size),
            stopPrice=format_price(new_sl, tick_size),
            timeInForce="GTC"
        )
        print(f"{Fore.GREEN}Trailing Stop Loss actualizado a: {format_price(new_sl, tick_size)}")
        return new_order
    except Exception as e:
        print(f"{Fore.RED}Error actualizando Trailing Stop Loss: {e}")
        return None

# ===========================
#  SEÑALES DE IA (INDICADORES)
# ===========================
def get_ai_signal(symbol, interval="5m", lookback=30):
    """
    Calcula una señal (BULLISH o BEARISH) con SMA, RSI, MACD y Stochastic
    usando TA-Lib.
    """
    try:
        req_lookback = max(lookback, 50)
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=req_lookback)
        closes = [float(k[4]) for k in klines]
        close_array = np.array(closes)

        # SMA
        sma_short = talib.SMA(close_array, timeperiod=5)
        sma_long = talib.SMA(close_array, timeperiod=lookback)
        ma_signal = "BULLISH" if sma_short[-1] > sma_long[-1] else "BEARISH"

        # RSI
        rsi = talib.RSI(close_array, timeperiod=14)
        last_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
        rsi_signal = "BULLISH" if last_rsi < 40 else "BEARISH" if last_rsi > 60 else "NEUTRAL"

        # MACD
        macd, macdsignal, macdhist = talib.MACD(close_array, fastperiod=12, slowperiod=26, signalperiod=9)
        last_macdhist = macdhist[-1] if not np.isnan(macdhist[-1]) else 0
        macd_signal = "BULLISH" if last_macdhist > 0 else "BEARISH"

        # Stochastic
        highs = [float(k[2]) for k in klines]
        lows  = [float(k[3]) for k in klines]
        slowk, slowd = talib.STOCH(np.array(highs), np.array(lows), close_array,
                                   fastk_period=14, slowk_period=3, slowd_period=3)
        last_slowk = slowk[-1] if not np.isnan(slowk[-1]) else 50
        last_slowd = slowd[-1] if not np.isnan(slowd[-1]) else 50
        stoch_signal = "BULLISH" if last_slowk > last_slowd else "BEARISH" if last_slowk < last_slowd else "NEUTRAL"

        print(
            f"{Fore.CYAN}[IA {interval}] - SMA: {Fore.GREEN if ma_signal=='BULLISH' else Fore.RED}{ma_signal}{Style.RESET_ALL} | "
            f"RSI: {last_rsi:.2f} ({Fore.GREEN if rsi_signal=='BULLISH' else Fore.RED if rsi_signal=='BEARISH' else Fore.YELLOW}{rsi_signal}{Style.RESET_ALL}) | "
            f"MACD Hist: {last_macdhist:.4f} ({Fore.GREEN if macd_signal=='BULLISH' else Fore.RED}{macd_signal}{Style.RESET_ALL}) | "
            f"STOCH: {last_slowk:.2f}/{last_slowd:.2f} ({Fore.GREEN if stoch_signal=='BULLISH' else Fore.RED if stoch_signal=='BEARISH' else Fore.YELLOW}{stoch_signal}{Style.RESET_ALL})"
        )

        # Lógica de conteo
        if rsi_signal == "NEUTRAL":
            signals = [ma_signal, macd_signal, stoch_signal]
            required = 2
        else:
            signals = [ma_signal, macd_signal, stoch_signal, rsi_signal]
            required = 3

        bullish_count = signals.count("BULLISH")
        return "BULLISH" if bullish_count >= required else "BEARISH"
    except Exception as e:
        print(f"{Fore.RED}Error en get_ai_signal: {e}")
        return "BULLISH"

# ===========================
#  PATRONES CHARTISTAS
# ===========================
def get_chart_pattern(symbol, interval="5m", lookback=50):
    """
    Usa TA-Lib para detectar patrones Engulfing, Hammer y Doji.
    """
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=lookback)
        opens = np.array([float(k[1]) for k in klines])
        highs = np.array([float(k[2]) for k in klines])
        lows  = np.array([float(k[3]) for k in klines])
        closes = np.array([float(k[4]) for k in klines])

        engulfing = talib.CDLENGULFING(opens, highs, lows, closes)
        hammer = talib.CDLHAMMER(opens, highs, lows, closes)
        doji = talib.CDLDOJI(opens, highs, lows, closes)

        pat_engulfing = engulfing[-1]
        pat_hammer = hammer[-1]
        pat_doji = doji[-1]

        bullish_score = 0
        bearish_score = 0
        if pat_engulfing > 0:
            bullish_score += 1
        elif pat_engulfing < 0:
            bearish_score += 1
        if pat_hammer > 0:
            bullish_score += 1
        elif pat_hammer < 0:
            bearish_score += 1

        if bullish_score > bearish_score:
            overall = "BULLISH"
        elif bearish_score > bullish_score:
            overall = "BEARISH"
        else:
            overall = "NEUTRAL"

        print(f"{Fore.MAGENTA}[Chart {interval}] Engulfing: {pat_engulfing}, Hammer: {pat_hammer}, Doji: {pat_doji} -> Overall: {overall}")
        return overall
    except Exception as e:
        print(f"{Fore.RED}Error en get_chart_pattern: {e}")
        return "NEUTRAL"

# ===========================
#  CONFIRMACIÓN MULTI-TIMEFRAME
# ===========================
def get_multi_timeframe_signal(symbol):
    """
    Compara señales en 5m y 15m. Solo retorna BULLISH si ambas son BULLISH,
    BEARISH si ambas son BEARISH, de lo contrario NEUTRAL.
    Además, aplica un filtro chartista en 5m.
    """
    signal_5m = get_ai_signal(symbol, interval="5m", lookback=30)
    signal_15m = get_ai_signal(symbol, interval="15m", lookback=30)

    # Si no coinciden, marcamos NEUTRAL
    if signal_5m == signal_15m:
        # Filtro chartista en 5m
        chart_5m = get_chart_pattern(symbol, interval="5m", lookback=50)
        # Si la señal coincide con el chart pattern, es una señal fuerte
        if chart_5m == signal_5m:
            return signal_5m
        else:
            # Si el chart pattern es neutral, mantenemos la señal AI
            if chart_5m == "NEUTRAL":
                return signal_5m
            else:
                # Si el chart pattern contradice la señal, nos quedamos neutrales
                return "NEUTRAL"
    else:
        return "NEUTRAL"

# ===========================
#  MOSTRAR POSICIONES
# ===========================
def print_position_info(symbol):
    pos_info = client.futures_position_information(symbol=symbol)
    for pos in pos_info:
        amt = float(pos.get("positionAmt", "0"))
        if abs(amt) > 0:
            entry_price_pos = float(pos.get("entryPrice", "0"))
            current_price = float(client.futures_symbol_ticker(symbol=symbol)["price"])
            pnl_pct = (current_price - entry_price_pos) / entry_price_pos if amt > 0 else (entry_price_pos - current_price) / entry_price_pos
            print(f"{Fore.BLUE}Posición: {amt} {symbol} | Entrada: {entry_price_pos} | Actual: {current_price} | PnL: {pnl_pct*100:.2f}%")

# ===========================
#  CERRAR POSICIÓN
# ===========================
def force_close_position(symbol, step_size):
    try:
        pos_info = client.futures_position_information(symbol=symbol)
        closed = False
        for pos in pos_info:
            amt = float(pos.get("positionAmt", "0"))
            if amt != 0:
                side = "SELL" if amt > 0 else "BUY"
                qty = abs(amt)
                print(f"{Fore.YELLOW}Cerrando posición: {side} {qty} {symbol}")
                market_order = client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type="MARKET",
                    quantity=format_quantity(qty, step_size)
                )
                print(f"{Fore.GREEN}Orden de cierre ejecutada:", market_order)
                closed = True
        if not closed:
            print(f"{Fore.CYAN}No hay posiciones abiertas para cerrar.")
    except Exception as e:
        print(f"{Fore.RED}Error al cerrar posición: {e}")

# ===========================
#  OBTENER INFORMACIÓN DEL SÍMBOLO
# ===========================
exchange_info = client.futures_exchange_info()
symbol_info = next(item for item in exchange_info['symbols'] if item['symbol'] == symbol)
tick_size = float(symbol_info['filters'][0]['tickSize'])
step_size = float(symbol_info['filters'][1]['stepSize'])
print(f"{Fore.CYAN}Parámetros del símbolo - Tick size: {tick_size}, Step size: {step_size}")

# ===========================
#  VARIABLES DE ESTADO
# ===========================
check_interval = 120  # Segundos entre chequeos
trade_in_progress = False
initial_signal = None
entry_price = None
trade_side = None
pnl_pct = 0
highest_price = None
lowest_price = None
current_sl = None
partial_profit_taken = False
last_exit_time = None

# ===========================
#  BUCLE PRINCIPAL
# ===========================
while True:
    pos_info = client.futures_position_information(symbol=symbol)
    position_open = any(abs(float(pos.get("positionAmt", "0"))) > 0 for pos in pos_info)
    
    if not position_open:
        # Cooldown tras pérdida
        if last_exit_time is not None and (time.time() - last_exit_time) < exit_cooldown:
            remaining = int(exit_cooldown - (time.time() - last_exit_time))
            print(f"{Fore.YELLOW}Cooldown activo. Esperando {remaining} segundos antes de reingresar.")
            time.sleep(remaining)
            continue
        
        print(f"\n{Fore.YELLOW}=== Iniciando nueva operación ===")
        # Cancelar órdenes residuales
        try:
            cancel_result = client.futures_cancel_all_open_orders(symbol=symbol)
            print(f"{Fore.GREEN}Órdenes canceladas: {cancel_result['msg']}")
        except Exception as e:
            print(f"{Fore.RED}Error al cancelar órdenes: {e}")
        time.sleep(2)
        
        # Precio actual
        try:
            precio_actual = float(client.futures_symbol_ticker(symbol=symbol)["price"])
        except Exception as e:
            print(f"{Fore.RED}Error obteniendo precio actual: {e}")
            continue
        print(f"{Fore.CYAN}Precio actual de {symbol}: {precio_actual}")
    
        # Señal final multi-timeframe
        combined_signal = get_multi_timeframe_signal(symbol)
        initial_signal = combined_signal
        print(f"{Fore.CYAN}Señal final: {combined_signal}")
    
        if combined_signal == "BULLISH":
            trade_side = "BUY"
            sl_side = "SELL"
            calc_sl = lambda p: adjust_price(p * (1 - stop_loss_init_pct), tick_size)
            calc_tp = lambda p: adjust_price(p * (1 + take_profit_init_pct), tick_size)
        elif combined_signal == "BEARISH":
            trade_side = "SELL"
            sl_side = "BUY"
            calc_sl = lambda p: adjust_price(p * (1 + stop_loss_init_pct), tick_size)
            calc_tp = lambda p: adjust_price(p * (1 - take_profit_init_pct), tick_size)
        else:
            print(f"{Fore.YELLOW}Señal NEUTRAL, no se abre operación.")
            time.sleep(check_interval)
            continue
    
        trade_value = capital_total * trade_capital_ratio
        cantidad_operacion = trade_value / precio_actual
        cantidad_operacion_adj = adjust_quantity(cantidad_operacion, step_size)
        print(f"{Fore.CYAN}Operando {format_quantity(cantidad_operacion_adj, step_size)} BNB (≈ USD {trade_value})")
    
        # Orden de entrada
        try:
            entry_order = client.futures_create_order(
                symbol=symbol,
                side=trade_side,
                type="MARKET",
                quantity=format_quantity(cantidad_operacion_adj, step_size)
            )
            print(f"{Fore.GREEN}Orden de entrada ejecutada:", entry_order)
            order_status = entry_order.get("status", "NEW")
            while order_status != "FILLED":
                entry_order = client.futures_get_order(symbol=symbol, orderId=entry_order['orderId'])
                order_status = entry_order.get("status", "NEW")
                time.sleep(1)
            entry_price = float(entry_order.get("avgFillPrice", precio_actual))
            if entry_price == 0:
                entry_price = precio_actual
            print(f"{Fore.GREEN}Precio de entrada establecido: {entry_price}")
            
            trade_in_progress = True
            highest_price = entry_price
            lowest_price = entry_price
            current_sl = calc_sl(entry_price)
            partial_profit_taken = False
        except Exception as e:
            print(f"{Fore.RED}Error al crear orden de entrada: {e}")
            time.sleep(60)
            continue
    
        # Stop Loss y Take Profit
        initial_sl = current_sl
        initial_tp = calc_tp(entry_price)
        print(f"{Fore.CYAN}Stop Loss: {format_price(initial_sl, tick_size)} | Take Profit: {format_price(initial_tp, tick_size)}")
    
        try:
            sl_order = client.futures_create_order(
                symbol=symbol,
                side=sl_side,
                type="STOP_MARKET",
                quantity=format_quantity(cantidad_operacion_adj, step_size),
                stopPrice=format_price(initial_sl, tick_size),
                timeInForce="GTC"
            )
            print(f"{Fore.GREEN}Orden de Stop Loss colocada:", sl_order)
        except Exception as e:
            print(f"{Fore.RED}Error al colocar orden de Stop Loss: {e}")
        try:
            tp_order = client.futures_create_order(
                symbol=symbol,
                side="SELL" if trade_side=="BUY" else "BUY",
                type="LIMIT",
                quantity=format_quantity(cantidad_operacion_adj, step_size),
                price=format_price(initial_tp, tick_size),
                timeInForce="GTC"
            )
            print(f"{Fore.GREEN}Orden de Take Profit colocada:", tp_order)
        except Exception as e:
            print(f"{Fore.RED}Error al colocar orden de Take Profit: {e}")
    else:
        # Si hay una operación en curso
        print(f"{Fore.YELLOW}Operación en curso. Monitoreando posición...")
        print_position_info(symbol)
    
    # Esperar el intervalo de verificación
    time.sleep(check_interval)
    
    # Revisar señales y PnL
    new_signal = get_multi_timeframe_signal(symbol)
    print(f"{Fore.CYAN}[Monitoreo] Nueva señal: {new_signal}")
    
    if (entry_price is None or entry_price == 0) and position_open:
        for pos in pos_info:
            if abs(float(pos.get("positionAmt", "0"))) > 0:
                entry_price = float(pos.get("entryPrice", "0"))
                break
    
    try:
        current_price = float(client.futures_symbol_ticker(symbol=symbol)["price"])
        if entry_price and entry_price != 0:
            if trade_side == "BUY":
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            print(f"{Fore.CYAN}[Monitoreo] PnL actual: {pnl_pct*100:.2f}%")
        else:
            pnl_pct = 0
            print(f"{Fore.YELLOW}Advertencia: Precio de entrada no disponible.")
    except Exception as e:
        pnl_pct = 0
        print(f"{Fore.RED}Error al obtener PnL: {e}")
    
    # Actualizar trailing stop
    if trade_in_progress:
        if trade_side == "BUY":
            highest_price = max(highest_price, current_price)
            new_trailing_sl = adjust_price(highest_price * (1 - trailing_stop_pct), tick_size)
            if new_trailing_sl > current_sl:
                current_sl = new_trailing_sl
                # Obtener cantidad actual
                for pos in client.futures_position_information(symbol=symbol):
                    if abs(float(pos.get("positionAmt", "0"))) > 0:
                        current_qty = abs(float(pos.get("positionAmt", "0")))
                        break
                update_trailing_stop_order(symbol, current_qty, "SELL", current_sl, step_size, tick_size)
        else:
            lowest_price = min(lowest_price, current_price)
            new_trailing_sl = adjust_price(lowest_price * (1 + trailing_stop_pct), tick_size)
            if new_trailing_sl < current_sl:
                current_sl = new_trailing_sl
                for pos in client.futures_position_information(symbol=symbol):
                    if abs(float(pos.get("positionAmt", "0"))) > 0:
                        current_qty = abs(float(pos.get("positionAmt", "0")))
                        break
                update_trailing_stop_order(symbol, current_qty, "BUY", current_sl, step_size, tick_size)
    
    # Toma de beneficio parcial
    if pnl_pct >= partial_profit_threshold and not partial_profit_taken:
        pos_info = client.futures_position_information(symbol=symbol)
        for pos in pos_info:
            if abs(float(pos.get("positionAmt", "0"))) > 0:
                current_qty = abs(float(pos.get("positionAmt", "0")))
                break
        partial_qty = current_qty * partial_profit_ratio
        partial_qty_adj = adjust_quantity(partial_qty, step_size)
        print(f"{Fore.MAGENTA}Tomando beneficio parcial de {format_quantity(partial_qty_adj, step_size)} {symbol}")
        try:
            if trade_side == "BUY":
                close_order = client.futures_create_order(
                    symbol=symbol,
                    side="SELL",
                    type="MARKET",
                    quantity=format_quantity(partial_qty_adj, step_size)
                )
            else:
                close_order = client.futures_create_order(
                    symbol=symbol,
                    side="BUY",
                    type="MARKET",
                    quantity=format_quantity(partial_qty_adj, step_size)
                )
            print(f"{Fore.GREEN}Orden de beneficio parcial ejecutada:", close_order)
            partial_profit_taken = True
            # Cancelar órdenes pendientes y reestablecer trailing stop
            try:
                cancel_result = client.futures_cancel_all_open_orders(symbol=symbol)
                print(f"{Fore.GREEN}Órdenes canceladas tras beneficio parcial: {cancel_result['msg']}")
            except Exception as e:
                print(f"{Fore.RED}Error al cancelar órdenes tras beneficio parcial: {e}")
            remaining_qty = current_qty - partial_qty_adj
            if remaining_qty > 0:
                if trade_side == "BUY":
                    current_sl = adjust_price(current_price * (1 - trailing_stop_pct), tick_size)
                else:
                    current_sl = adjust_price(current_price * (1 + trailing_stop_pct), tick_size)
                new_sl_order = client.futures_create_order(
                    symbol=symbol,
                    side="SELL" if trade_side=="BUY" else "BUY",
                    type="STOP_MARKET",
                    quantity=format_quantity(remaining_qty, step_size),
                    stopPrice=format_price(current_sl, tick_size),
                    timeInForce="GTC"
                )
                print(f"{Fore.GREEN}Nueva orden de trailing stop para cantidad restante:", new_sl_order)
            else:
                print(f"{Fore.YELLOW}No queda cantidad para trailing stop tras beneficio parcial.")
        except Exception as e:
            print(f"{Fore.RED}Error en beneficio parcial: {e}")
    
    # Cierre por reversión negativa o pérdida excesiva
    if (new_signal != initial_signal and pnl_pct < 0):
        print(f"{Fore.MAGENTA}Se detectó reversión en condiciones desfavorables. Cerrando posición.")
        force_close_position(symbol, step_size)
        trade_in_progress = False
        entry_price = None
        last_exit_time = None
        # Evitar re-abrir en el mismo ciclo
        continue
    elif pnl_pct < -0.01:
        print(f"{Fore.MAGENTA}Pérdida excesiva. Cerrando posición y activando cooldown.")
        force_close_position(symbol, step_size)
        trade_in_progress = False
        entry_price = None
        last_exit_time = time.time()
        # Evitar re-abrir en el mismo ciclo
        continue
    else:
        print(f"{Fore.GREEN}La operación se mantiene en curso sin cambios.")

    time.sleep(60)
