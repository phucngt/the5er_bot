import MetaTrader5 as mt5
import pandas as pd

class MT5AutoTrading:
    """
    A wrapper to manage MT5 automated trading on a single symbol,
    ensuring only one open order at a time.
    """
    def __init__(self, symbol: str, magic: int = 123456):
        self.symbol = symbol
        self.magic = magic
        self.cards = []  # track executed orders

    def _get_open_positions(self):
        """
        Retrieve current open positions for this symbol and our magic number.
        """
        # filter by symbol and magic to avoid interfering with other orders
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []
        # filter only those with our magic
        return [pos for pos in positions if pos.magic == self.magic]


    def _send_market_order(self, volume: float, direction: int, area: str, comment_suffix: str,
                           sl: float = None, tp: float = None):
        # skip if there’s already an open position
        if self.has_open_position():
            print("[MT5AutoTrading] Skipping order: already have a position")
            return

        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        tick       = mt5.symbol_info_tick(self.symbol)
        price_send = tick.ask if direction == 1 else tick.bid

        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       self.symbol,
            "volume":       float(volume),
            "type":         order_type,
            "price":        price_send,
            "deviation":    10,
            "magic":        self.magic,
            "comment":      f"SMC {area} {comment_suffix}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        # add SL/TP if provided
        if sl is not None:
            req["sl"] = sl
        if tp is not None:
            req["tp"] = tp

        result = mt5.order_send(req)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[ORDER] {comment_suffix} {'BUY' if direction==1 else 'SELL'} @ {price_send}, lot={volume}")
            ticket = result.order
            self.cards.append({
                'ticket':      ticket,
                'time':        pd.Timestamp.now(),
                'entry_price': price_send,
                'direction':   direction,
                'area':        area,
                'lot_size':    volume,
                'taken':       False
            })
            return ticket
        else:
            print(f"[ORDER FAILED] {result.retcode}, {result.comment}")
            return None

    def close_all_positions(self):
        """
        Close any open positions on this symbol for our magic number.
        """
        positions = self._get_open_positions()
        for pos in positions:
            direction = 1 if pos.type == mt5.ORDER_TYPE_SELL else -1
            volume    = pos.volume
            # reverse trade
            tick      = mt5.symbol_info_tick(self.symbol)
            price     = tick.ask if direction == 1 else tick.bid
            req = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       self.symbol,
                "volume":       float(volume),
                "type":         mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL,
                "price":        price,
                "deviation":    10,
                "magic":        self.magic,
                "comment":      "SMC close",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            res = mt5.order_send(req)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[CLOSE] ticket {pos.ticket} closed @ {price}")
            else:
                print(f"[CLOSE FAILED] {res.retcode}, {res.comment}")

    def has_open_position(self) -> bool:
        """
        Return True if there's any open position for our symbol.
        """
        return bool(self._get_open_positions())
