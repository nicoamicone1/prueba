
api_key = "RkavHxWrM3OMfVtTKmaJjwl7rKYrIhXMMU2IXaNXhqMXUMBKrkz472Bm04yHrsc8"
api_secret = "qdk5W5OOGIo214HfXSrKt51FuGxnuh03vQ9s5JipzHyn7d2sBFXmrOS9K50CeFR5"

#!/usr/bin/env python3
import os
import time, math, pandas as pd, numpy as np, ta
from binance.client import Client
from colorama import Fore, Style, init

# ----------------------------
# Configuración de API y Conexión
# ----------------------------
# Puedes obtener las claves desde variables de entorno o definirlas aquí:
# api_key = os.getenv("BINANCE_API_KEY")
# api_secret = os.getenv("BINANCE_API_SECRET")

client = Client(api_key, api_secret)
symbol = "BNBUSDT"

# Inicializar colorama para logs en color
init(autoreset=True)

# ----------------------------
# Parámetros de trading (10x apalancamiento) y gestión de riesgo
# ----------------------------
trailing_stop_pct = 0.005          # Trailing stop del 0.5%
partial_profit_threshold = 0.02    # Toma parcial a partir de 2% de ganancia
partial_profit_ratio = 0.5         # Se cierra el 50% de la posición en beneficio parcial
exit_cooldown = 300                # Cooldown de 5 minutos tras cierre por pérdida

# Ajustamos los niveles para buscar mayor rentabilidad:
stop_loss_init_pct = 0.005   # 0.5% de stop loss inicial
take_profit_init_pct = 0.02  # 2% de take profit inicial

# ----------------------------
# Parámetros generales de la operación
# ----------------------------
check_interval = 120  # Verificar cada 2 minutos

# Capital y apalancamiento (10x)
capital_invertido = 30         # USD base invertidos
apalancamiento = 10
capital_total = capital_invertido * apalancamiento  # Valor total en USD
trade_capital_ratio = 1        # Usar el 100% del capital_total en cada trade

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
# Función de señal "IA" usando indicadores de la librería 'ta'
# ----------------------------
def get_ai_signal(symbol, interval="5m", lookback=30):
    try:
        req_lookback = max(lookback, 50)
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=req_lookback)
        df = pd.DataFrame(klines, columns=['open_time','open','high','low','close','volume','close_time',
                                             'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
                                             'taker_buy_quote_asset_volume','ignore'])
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        
        sma_short = ta.trend.SMAIndicator(df['close'], window=5).sma_indicator().iloc[-1]
        sma_long = ta.trend.SMAIndicator(df['close'], window=lookback).sma_indicator().iloc[-1]
        ma_signal = "BULLISH" if sma_short > sma_long else "BEARISH"
        
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]
        rsi_signal = "BULLISH" if rsi < 40 else "BEARISH" if rsi > 60 else "NEUTRAL"
        
        macd_obj = ta.trend.MACD(df['close'])
        macd_hist = macd_obj.macd_diff().iloc[-1]
        macd_signal = "BULLISH" if macd_hist > 0 else "BEARISH"
        
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        last_slowk = stoch.stoch().iloc[-1]
        last_slowd = stoch.stoch_signal().iloc[-1]
        stoch_signal = "BULLISH" if last_slowk > last_slowd else "BEARISH" if last_slowk < last_slowd else "NEUTRAL"
        
        print(f"{Fore.CYAN}[IA] Señal - SMA: {Fore.GREEN if ma_signal=='BULLISH' else Fore.RED}{ma_signal}{Style.RESET_ALL} | "
              f"RSI: {rsi:.2f} ({Fore.GREEN if rsi_signal=='BULLISH' else Fore.RED if rsi_signal=='BEARISH' else Fore.YELLOW}{rsi_signal}{Style.RESET_ALL}) | "
              f"MACD Hist: {macd_hist:.4f} ({Fore.GREEN if macd_signal=='BULLISH' else Fore.RED}{macd_signal}{Style.RESET_ALL}) | "
              f"STOCH: {last_slowk:.2f}/{last_slowd:.2f} ({Fore.GREEN if stoch_signal=='BULLISH' else Fore.RED if stoch_signal=='BEARISH' else Fore.YELLOW}{stoch_signal}{Style.RESET_ALL})")
        
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
# Función para detectar patrones chartistas de forma simple
# ----------------------------
def get_chart_pattern(symbol, interval="5m", lookback=50):
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=lookback)
        df = pd.DataFrame(klines, columns=['open_time','open','high','low','close','volume','close_time',
                                             'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
                                             'taker_buy_quote_asset_volume','ignore'])
        for col in ['open','high','low','close']:
            df[col] = pd.to_numeric(df[col])
        if len(df) < 2:
            return "NEUTRAL"
        last = df.iloc[-1]
        prev = df.iloc[-2]
        bullish_score = 0
        bearish_score = 0
        # Bullish engulfing
        if (prev['close'] < prev['open']) and (last['close'] > last['open']) and (last['close'] > prev['open']) and (last['open'] < prev['close']):
            bullish_score += 1
        # Bearish engulfing
        if (prev['close'] > prev['open']) and (last['close'] < last['open']) and (last['open'] > prev['close']) and (last['close'] < prev['open']):
            bearish_score += 1
        # Hammer (simple): cuerpo pequeño, sombra inferior larga
        body = abs(last['close'] - last['open'])
        total_range = last['high'] - last['low']
        lower_shadow = (last['open'] - last['low']) if last['close'] >= last['open'] else (last['close'] - last['low'])
        upper_shadow = (last['high'] - last['close']) if last['close'] >= last['open'] else (last['high'] - last['open'])
        if total_range > 0 and body < total_range * 0.3 and lower_shadow > body * 2 and upper_shadow < body:
            bullish_score += 1
        
        if bullish_score > bearish_score:
            overall = "BULLISH"
        elif bearish_score > bullish_score:
            overall = "BEARISH"
        else:
            overall = "NEUTRAL"
        print(f"{Fore.MAGENTA}[Chart Patterns] Señal: {overall} (Bullish Score: {bullish_score}, Bearish Score: {bearish_score})")
        return overall
    except Exception as e:
        print(f"{Fore.RED}Error en get_chart_pattern: {e}")
        return "NEUTRAL"

# ----------------------------
# Función para combinar señales de indicadores y patrones chartistas
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
            final = ai_signal
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
            pnl_pct = (current_price - entry_price_pos) / entry_price_pos if amt > 0 else (entry_price_pos - current_price) / entry_price_pos
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
# Parámetros del símbolo
# ----------------------------
print(f"{Fore.CYAN}Obteniendo parámetros del símbolo...")
exchange_info = client.futures_exchange_info()
symbol_info = next(item for item in exchange_info['symbols'] if item['symbol'] == symbol)
tick_size = float(symbol_info['filters'][0]['tickSize'])
step_size = float(symbol_info['filters'][1]['stepSize'])
print(f"{Fore.CYAN}Parámetros del símbolo - Tick size: {tick_size}, Step size: {step_size}")

# Variables de estado globales
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

# ----------------------------
# Bucle principal de operaciones
# ----------------------------
while True:
    pos_info = client.futures_position_information(symbol=symbol)
    position_open = any(abs(float(pos.get("positionAmt", "0"))) > 0 for pos in pos_info)
    
    if not position_open:
        if last_exit_time is not None and (time.time() - last_exit_time) < exit_cooldown:
            remaining = int(exit_cooldown - (time.time() - last_exit_time))
            print(f"{Fore.YELLOW}Cooldown activo. Esperando {remaining} segundos antes de reingresar.")
            time.sleep(remaining)
            continue
        
        print(f"\n{Fore.YELLOW}=== Iniciando nueva operación ===")
        try:
            cancel_result = client.futures_cancel_all_open_orders(symbol=symbol)
            print(f"{Fore.GREEN}Órdenes canceladas: {cancel_result['msg']}")
        except Exception as e:
            print(f"{Fore.RED}Error al cancelar órdenes: {e}")
        time.sleep(2)
        
        try:
            precio_actual = float(client.futures_symbol_ticker(symbol=symbol)["price"])
        except Exception as e:
            print(f"{Fore.RED}Error obteniendo precio actual: {e}")
            continue
        print(f"{Fore.CYAN}Precio actual de {symbol}: {precio_actual}")
    
        combined_signal = get_final_signal(symbol, ai_interval="5m", ai_lookback=30, chart_interval="5m", chart_lookback=50)
        initial_signal = combined_signal
        print(f"{Fore.CYAN}Señal combinada inicial: {combined_signal}")
    
        if combined_signal == "BULLISH":
            trade_side = "BUY"
            sl_side = "SELL"
            calc_sl = lambda price: adjust_price(price * (1 - stop_loss_init_pct), tick_size)
            calc_tp = lambda price: adjust_price(price * (1 + take_profit_init_pct), tick_size)
        else:
            trade_side = "SELL"
            sl_side = "BUY"
            calc_sl = lambda price: adjust_price(price * (1 + stop_loss_init_pct), tick_size)
            calc_tp = lambda price: adjust_price(price * (1 - take_profit_init_pct), tick_size)
    
        trade_value = capital_total * trade_capital_ratio
        cantidad_operacion = trade_value / precio_actual
        cantidad_operacion_adj = adjust_quantity(cantidad_operacion, step_size)
        print(f"{Fore.CYAN}Operando {format_quantity(cantidad_operacion_adj, step_size)} BNB (valor ≈ USD {trade_value})")
    
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
    
        initial_sl = current_sl
        initial_tp = calc_tp(entry_price)
        print(f"{Fore.CYAN}Niveles establecidos - Stop Loss: {format_price(initial_sl, tick_size)} | Take Profit: {format_price(initial_tp, tick_size)}")
    
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
    
    time.sleep(check_interval)
    
    new_combined_signal = get_final_signal(symbol, ai_interval="5m", ai_lookback=30, chart_interval="5m", chart_lookback=50)
    print(f"{Fore.CYAN}[Monitoreo] Nueva señal combinada: {new_combined_signal}")
    
    if (entry_price is None or entry_price == 0) and position_open:
        for pos in pos_info:
            if abs(float(pos.get("positionAmt", "0"))) > 0:
                entry_price = float(pos.get("entryPrice", "0"))
                break
    
    try:
        current_price = float(client.futures_symbol_ticker(symbol=symbol)["price"])
        if entry_price is not None and entry_price != 0:
            pnl_pct = (current_price - entry_price) / entry_price if trade_side == "BUY" else (entry_price - current_price) / entry_price
            print(f"{Fore.CYAN}[Monitoreo] PnL actual: {pnl_pct*100:.2f}%")
        else:
            pnl_pct = 0
            print(f"{Fore.YELLOW}Advertencia: Precio de entrada no disponible.")
    except Exception as e:
        pnl_pct = 0
        print(f"{Fore.RED}Error al obtener PnL: {e}")
    
    if trade_in_progress:
        if trade_side == "BUY":
            highest_price = max(highest_price, current_price)
            new_trailing_sl = adjust_price(highest_price * (1 - trailing_stop_pct), tick_size)
            if new_trailing_sl > current_sl:
                current_sl = new_trailing_sl
                for pos in client.futures_position_information(symbol=symbol):
                    if abs(float(pos.get("positionAmt", "0"))) > 0:
                        current_qty = abs(float(pos.get("positionAmt", "0")))
                        break
                update_trailing_stop_order(symbol, current_qty, sl_side, current_sl, step_size, tick_size)
        else:
            lowest_price = min(lowest_price, current_price)
            new_trailing_sl = adjust_price(lowest_price * (1 + trailing_stop_pct), tick_size)
            if new_trailing_sl < current_sl:
                current_sl = new_trailing_sl
                for pos in client.futures_position_information(symbol=symbol):
                    if abs(float(pos.get("positionAmt", "0"))) > 0:
                        current_qty = abs(float(pos.get("positionAmt", "0")))
                        break
                update_trailing_stop_order(symbol, current_qty, sl_side, current_sl, step_size, tick_size)
    
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
    
    if (new_combined_signal != initial_signal and pnl_pct < 0):
        print(f"{Fore.MAGENTA}Se detectó reversión en condiciones desfavorables. Procediendo al cierre de la posición.")
        force_close_position(symbol, step_size)
        trade_in_progress = False
        entry_price = None
        last_exit_time = None
    elif pnl_pct < -0.01:
        print(f"{Fore.MAGENTA}Pérdida excesiva detectada. Procediendo al cierre de la posición y activando cooldown.")
        force_close_position(symbol, step_size)
        trade_in_progress = False
        entry_price = None
        last_exit_time = time.time()
    else:
        print(f"{Fore.GREEN}La operación se mantiene en curso sin cambios.")
    
    time.sleep(60)
