
api_key = "RkavHxWrM3OMfVtTKmaJjwl7rKYrIhXMMU2IXaNXhqMXUMBKrkz472Bm04yHrsc8"
api_secret = "qdk5W5OOGIo214HfXSrKt51FuGxnuh03vQ9s5JipzHyn7d2sBFXmrOS9K50CeFR5"

import os
import time, math, pandas as pd, numpy as np, talib
from binance.client import Client
from colorama import Fore, Style, init

# ----------------------------
# Configuración de API y Conexión
# ----------------------------
# Puedes obtener las claves desde las variables de entorno o definirlas aquí:
# api_key = os.getenv("BINANCE_API_KEY")
# api_secret = os.getenv("BINANCE_API_SECRET")

client = Client(api_key, api_secret)
symbol = "BNBUSDT"

# Inicializar colorama para logs en color
init(autoreset=True)

# ----------------------------
# Parámetros de mejora para maximizar ganancia y patrones chartistas
# ----------------------------
trailing_stop_pct = 0.005          # Trailing Stop del 0.5%
partial_profit_threshold = 0.02    # Toma parcial a partir de 2% de ganancia
partial_profit_ratio = 0.5         # Se cierra el 50% de la posición en toma parcial
exit_cooldown = 300                # Cooldown de 5 minutos tras cierre por pérdida

# ----------------------------
# Funciones de ajuste y formateo
# ----------------------------
def adjust_price(value, tick_size):
    """Redondea hacia abajo el precio al múltiplo permitido por tick_size."""
    return math.floor(value / tick_size) * tick_size

def adjust_quantity(value, step_size):
    """Redondea hacia abajo la cantidad al múltiplo permitido por step_size."""
    return math.floor(value / step_size) * step_size

def format_price(price, tick_size):
    decimals = len(str(tick_size).split('.')[1])
    return f"{price:.{decimals}f}"

def format_quantity(qty, step_size):
    decimals = len(str(step_size).split('.')[1])
    return f"{qty:.{decimals}f}"

# ----------------------------
# Función para actualizar la orden de trailing stop
# ----------------------------
def update_trailing_stop_order(symbol, quantity, sl_side, new_sl, step_size, tick_size):
    try:
        # Cancelar órdenes STOP_MARKET existentes para el símbolo
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

# ----------------------------
# Función de señal "IA" usando TA‑Lib (indicadores técnicos)
# ----------------------------
def get_ai_signal(symbol, interval="5m", lookback=30):
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

        # MACD usando el histograma
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
            f"{Fore.CYAN}[IA] Señal - SMA: {Fore.GREEN if ma_signal=='BULLISH' else Fore.RED}{ma_signal}{Style.RESET_ALL} | "
            f"RSI: {last_rsi:.2f} ({Fore.GREEN if rsi_signal=='BULLISH' else Fore.RED if rsi_signal=='BEARISH' else Fore.YELLOW}{rsi_signal}{Style.RESET_ALL}) | "
            f"MACD Hist: {last_macdhist:.4f} ({Fore.GREEN if macd_signal=='BULLISH' else Fore.RED}{macd_signal}{Style.RESET_ALL}) | "
            f"STOCH: {last_slowk:.2f}/{last_slowd:.2f} ({Fore.GREEN if stoch_signal=='BULLISH' else Fore.RED if stoch_signal=='BEARISH' else Fore.YELLOW}{stoch_signal}{Style.RESET_ALL})"
        )
        
        # Lógica de señales modificada:
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

# ----------------------------
# Función para detectar patrones chartistas (candlestick patterns) usando TA‑Lib
# ----------------------------
def get_chart_pattern(symbol, interval="5m", lookback=50):
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=lookback)
        opens = np.array([float(k[1]) for k in klines])
        highs = np.array([float(k[2]) for k in klines])
        lows  = np.array([float(k[3]) for k in klines])
        closes = np.array([float(k[4]) for k in klines])
        # Detectar algunos patrones candlestick:
        engulfing = talib.CDLENGULFING(opens, highs, lows, closes)
        hammer = talib.CDLHAMMER(opens, highs, lows, closes)
        doji = talib.CDLDOJI(opens, highs, lows, closes)
        
        # Obtener los últimos valores de cada patrón:
        pat_engulfing = engulfing[-1]
        pat_hammer = hammer[-1]
        pat_doji = doji[-1]
        
        # Asumimos:
        #   - Engulfing: > 0 indica patrón alcista, < 0 bajista.
        #   - Hammer: > 0 indica patrón alcista, < 0 bajista.
        #   - Doji se considera neutral.
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
        
        # Definir señal global basada en patrones:
        if bullish_score > bearish_score:
            overall = "BULLISH"
        elif bearish_score > bullish_score:
            overall = "BEARISH"
        else:
            overall = "NEUTRAL"
        
        print(f"{Fore.MAGENTA}[Chart Patterns] Engulfing: {pat_engulfing}, Hammer: {pat_hammer}, Doji: {pat_doji} -> Overall: {overall}")
        return overall
    except Exception as e:
        print(f"{Fore.RED}Error en get_chart_pattern: {e}")
        return "NEUTRAL"

# ----------------------------
# Función para combinar señales de IA y patrones chartistas
# ----------------------------
def get_final_signal(symbol, ai_interval="5m", ai_lookback=30, chart_interval="5m", chart_lookback=50):
    ai_signal = get_ai_signal(symbol, interval=ai_interval, lookback=ai_lookback)
    chart_signal = get_chart_pattern(symbol, interval=chart_interval, lookback=chart_lookback)
    signals = []
    if ai_signal != "NEUTRAL":
        signals.append(ai_signal)
    if chart_signal != "NEUTRAL":
        signals.append(chart_signal)
    if len(signals) == 0:
        final = ai_signal
    else:
        if signals.count("BULLISH") > signals.count("BEARISH"):
            final = "BULLISH"
        elif signals.count("BEARISH") > signals.count("BULLISH"):
            final = "BEARISH"
        else:
            final = ai_signal  # desempata con la señal de IA
    print(f"{Fore.CYAN}Señal combinada: {final} (AI: {ai_signal}, Chart: {chart_signal})")
    return final

# ----------------------------
# Función para imprimir información de posiciones abiertas
# ----------------------------
def print_position_info(symbol):
    pos_info = client.futures_position_information(symbol=symbol)
    for pos in pos_info:
        amt = float(pos.get("positionAmt", "0"))
        if abs(amt) > 0:
            entry_price_pos = float(pos.get("entryPrice", "0"))
            current_price = float(client.futures_symbol_ticker(symbol=symbol)["price"])
            if amt > 0:
                pnl_pct = (current_price - entry_price_pos) / entry_price_pos
            else:
                pnl_pct = (entry_price_pos - current_price) / entry_price_pos
            print(f"{Fore.BLUE}Posición: {amt} {symbol} | Entrada: {entry_price_pos} | Actual: {current_price} | PnL: {pnl_pct*100:.2f}%")

# ----------------------------
# Función para forzar el cierre de posiciones abiertas
# ----------------------------
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

# ----------------------------
# Parámetros generales de la operación
# ----------------------------
check_interval = 120  # Verificar cada 2 minutos

# Capital y tamaño de operación
capital_invertido = 30        # USD base invertidos
apalancamiento = 10
capital_total = capital_invertido * apalancamiento  # Valor total en USD
trade_capital_ratio = 1    # Usar el 100% del capital_total en cada trade (ajusta según tu estrategia)

# Parámetros iniciales para SL y TP (porcentaje respecto al precio de entrada)
stop_loss_init_pct = 0.01    # 1%
take_profit_init_pct = 0.01  # 1%

# Obtener información de precisión para el símbolo
exchange_info = client.futures_exchange_info()
symbol_info = next(item for item in exchange_info['symbols'] if item['symbol'] == symbol)
tick_size = float(symbol_info['filters'][0]['tickSize'])
step_size = float(symbol_info['filters'][1]['stepSize'])
print(f"{Fore.CYAN}Parámetros del símbolo - Tick size: {tick_size}, Step size: {step_size}")

# Variables de estado globales
trade_in_progress = False
initial_signal = None  # Esta será la señal combinada (AI + Chart Patterns) al momento de abrir la operación
entry_price = None
trade_side = None  
pnl_pct = 0  

# Variables para trailing stop y beneficio parcial (se definen al abrir operación)
highest_price = None
lowest_price = None
current_sl = None
partial_profit_taken = False

# Variable para cooldown tras cierre por pérdida
last_exit_time = None

# ----------------------------
# Bucle principal: operaciones y monitoreo cada 2 minutos
# ----------------------------
while True:
    # Verificar si hay posición abierta
    pos_info = client.futures_position_information(symbol=symbol)
    position_open = any(abs(float(pos.get("positionAmt", "0"))) > 0 for pos in pos_info)
    
    # Si no hay posición abierta, revisar cooldown
    if not position_open:
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
        
        # Obtener precio actual
        try:
            precio_actual = float(client.futures_symbol_ticker(symbol=symbol)["price"])
        except Exception as e:
            print(f"{Fore.RED}Error obteniendo precio actual: {e}")
            continue
        print(f"{Fore.CYAN}Precio actual de {symbol}: {precio_actual}")
    
        # Obtener señales de IA y patrones chartistas y combinarlas
        combined_signal = get_final_signal(symbol, ai_interval="5m", ai_lookback=30, chart_interval="5m", chart_lookback=50)
        initial_signal = combined_signal
        print(f"{Fore.CYAN}Señal combinada inicial: {combined_signal}")
    
        # Definir dirección y niveles según la señal combinada
        if combined_signal == "BULLISH":
            trade_side = "BUY"     # Abrir posición long
            sl_side = "SELL"       # Orden para cerrar long (Stop Loss y TP)
            calc_sl = lambda price: adjust_price(price * (1 - stop_loss_init_pct), tick_size)
            calc_tp = lambda price: adjust_price(price * (1 + take_profit_init_pct), tick_size)
        else:
            trade_side = "SELL"    # Abrir posición short
            sl_side = "BUY"        # Orden para cerrar short (Stop Loss y TP)
            calc_sl = lambda price: adjust_price(price * (1 + stop_loss_init_pct), tick_size)
            calc_tp = lambda price: adjust_price(price * (1 - take_profit_init_pct), tick_size)
    
        # Calcular monto a operar (capital_total * trade_capital_ratio)
        trade_value = capital_total * trade_capital_ratio
        cantidad_operacion = trade_value / precio_actual  # en BNB
        cantidad_operacion_adj = adjust_quantity(cantidad_operacion, step_size)
        print(f"{Fore.CYAN}Operando {format_quantity(cantidad_operacion_adj, step_size)} BNB (valor ≈ USD {trade_value})")
    
        # Ejecutar orden de entrada (MARKET)
        try:
            entry_order = client.futures_create_order(
                symbol=symbol,
                side=trade_side,
                type="MARKET",
                quantity=format_quantity(cantidad_operacion_adj, step_size)
            )
            print(f"{Fore.GREEN}Orden de entrada ejecutada:", entry_order)
            # Esperar a que se complete la orden
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
            # Inicializar variables para trailing stop y toma parcial
            highest_price = entry_price
            lowest_price = entry_price
            current_sl = calc_sl(entry_price)
            partial_profit_taken = False
        except Exception as e:
            print(f"{Fore.RED}Error al crear orden de entrada: {e}")
            time.sleep(60)
            continue
    
        # Calcular niveles iniciales de SL y TP
        initial_sl = current_sl
        initial_tp = calc_tp(entry_price)
        print(f"{Fore.CYAN}Niveles establecidos - Stop Loss: {format_price(initial_sl, tick_size)} | Take Profit: {format_price(initial_tp, tick_size)}")
    
        # Colocar órdenes de Stop Loss y Take Profit
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
        print(f"{Fore.YELLOW}Operación en curso. Monitoreando posición...")
        print_position_info(symbol)
    
    # Esperar el intervalo de verificación (2 minutos)
    time.sleep(check_interval)
    
    # Durante el monitoreo, obtener nuevas señales (IA y Chart) y combinarlas
    new_combined_signal = get_final_signal(symbol, ai_interval="5m", ai_lookback=30, chart_interval="5m", chart_lookback=50)
    print(f"{Fore.CYAN}[Monitoreo] Nueva señal combinada: {new_combined_signal}")
    
    # Recuperar entry_price si no está definido
    if (entry_price is None or entry_price == 0) and position_open:
        for pos in pos_info:
            if abs(float(pos.get("positionAmt", "0"))) > 0:
                entry_price = float(pos.get("entryPrice", "0"))
                break
    
    try:
        current_price = float(client.futures_symbol_ticker(symbol=symbol)["price"])
        if entry_price is not None and entry_price != 0:
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
    
    # Actualizar trailing stop dinámico
    if trade_in_progress:
        if trade_side == "BUY":
            highest_price = max(highest_price, current_price)
            new_trailing_sl = adjust_price(highest_price * (1 - trailing_stop_pct), tick_size)
            if new_trailing_sl > current_sl:
                current_sl = new_trailing_sl
                # Obtener cantidad actual de la posición
                for pos in client.futures_position_information(symbol=symbol):
                    if abs(float(pos.get("positionAmt", "0"))) > 0:
                        current_qty = abs(float(pos.get("positionAmt", "0")))
                        break
                update_trailing_stop_order(symbol, current_qty, sl_side, current_sl, step_size, tick_size)
        else:  # Para posición short
            lowest_price = min(lowest_price, current_price)
            new_trailing_sl = adjust_price(lowest_price * (1 + trailing_stop_pct), tick_size)
            if new_trailing_sl < current_sl:
                current_sl = new_trailing_sl
                for pos in client.futures_position_information(symbol=symbol):
                    if abs(float(pos.get("positionAmt", "0"))) > 0:
                        current_qty = abs(float(pos.get("positionAmt", "0")))
                        break
                update_trailing_stop_order(symbol, current_qty, sl_side, current_sl, step_size, tick_size)
    
    # Ejecutar toma de beneficio parcial si se supera el umbral y no se ha hecho ya
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
            # Cancelar órdenes pendientes y reestablecer trailing stop para la cantidad restante
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
                    side=sl_side,
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
    
    # Cerrar posición si se detecta reversión o pérdida excesiva
    if (new_combined_signal != initial_signal and pnl_pct < 0):
        print(f"{Fore.MAGENTA}Se detectó reversión en condiciones desfavorables. Procediendo al cierre de la posición.")
        force_close_position(symbol, step_size)
        trade_in_progress = False
        entry_price = None  # Reiniciar para la siguiente operación
        last_exit_time = None  # Permite reingresar de inmediato con la señal de reversión
    elif pnl_pct < -0.01:
        print(f"{Fore.MAGENTA}Pérdida excesiva detectada. Procediendo al cierre de la posición y activando cooldown.")
        force_close_position(symbol, step_size)
        trade_in_progress = False
        entry_price = None
        last_exit_time = time.time()
    else:
        print(f"{Fore.GREEN}La operación se mantiene en curso sin cambios.")

    # Esperar 1 minuto antes del siguiente ciclo de verificación
    time.sleep(60)
