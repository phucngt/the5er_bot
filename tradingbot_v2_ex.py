import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime, timedelta
from smc_indicator_v2 import SmartMoneyConcepts

class MT5SMCbot:
    def __init__(self, symbol="BTCUSDc", primary_lot=0.01, interval=60):
        if not mt5.initialize():
            raise RuntimeError("MT5 initialize() failed")
        self.symbol = symbol
        self.primary_lot = primary_lot
        self.interval = interval
        self.cards = []  # track open positions

    def get_ohlcv(self, timeframe, n):
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, n)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df[['time','open','high','low','close']]

    def detect_daily_zone_signal(self):
        df1d = self.get_ohlcv(mt5.TIMEFRAME_D1, 500)
        smc1d = SmartMoneyConcepts(df1d).run()
        last = smc1d.iloc[-1]
        price = last['close']
        total_rng = last['premium_high'] - last['discount_low']
        pct = (price - last['discount_low']) / total_rng
        if pct >= 0.80:
            return -1  # short only
        if pct <= 0.20:
            return 1   # long only
        return 0

    def detect_sweep_entry(self):
        df1h = self.get_ohlcv(mt5.TIMEFRAME_H1, 200)
        smc1h = SmartMoneyConcepts(df1h)
        smc1h.calculate_atr()
        smc1h.calculate_zones()
        smc1h.swings()
        smc1h.break_of_structure()
        smc1h.detect_liquidity_sweeps()
        df = smc1h.df
        # check penultimate bar for sweep
        bar = df.iloc[-2]
        direction = 0
        if bar['sweep_long']:
            direction = -1
        elif bar['sweep_short']:
            direction = 1
        if direction == 0:
            return 0
        curr = df.iloc[-1]
        if direction == -1 and curr['close'] < bar['close']:
            return -1
        if direction ==  1 and curr['close'] > bar['close']:
            return 1
        return 0

    def compute_atr14_h4(self):
        df4h = self.get_ohlcv(mt5.TIMEFRAME_H4, 20)
        high_low = df4h['high'] - df4h['low']
        high_close = (df4h['high'] - df4h['close'].shift()).abs()
        low_close = (df4h['low']  - df4h['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(14, min_periods=1).mean().iat[-1]

    def sync_cards(self):
        live = mt5.positions_get(symbol=self.symbol) or []
        live_ids = {p.ticket for p in live}
        kept = [c for c in self.cards if c['ticket'] in live_ids]
        # add any new positions
        for p in live:
            if p.ticket not in {c['ticket'] for c in kept}:
                kept.append({'ticket':p.ticket,'entry_time':datetime.fromtimestamp(p.time),
                             'direction':1 if p.type==mt5.POSITION_TYPE_BUY else -1,
                             'entry_price':p.price_open})
        self.cards = kept

    def can_open(self, direction):
        if len(self.cards) >= 2:
            return False
        # only one long and one short at a time
        dirs = [c['direction'] for c in self.cards]
        if direction in dirs:
            return False
        return True

    def open_order(self, direction, stop_loss):
        order_type = mt5.ORDER_TYPE_BUY if direction==1 else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(self.symbol)
        price = tick.ask if direction==1 else tick.bid
        req = {
            "action":mt5.TRADE_ACTION_DEAL,
            "symbol":self.symbol,
            "volume":self.primary_lot,
            "type":order_type,
            "price":price,
            "sl":stop_loss,
            "deviation":10,
            "magic":123456,
            "type_time":mt5.ORDER_TIME_GTC,
            "type_filling":mt5.ORDER_FILLING_IOC
        }
        res = mt5.order_send(req)
        if res.retcode==mt5.TRADE_RETCODE_DONE:
            self.cards.append({'ticket':res.order,'entry_time':datetime.now(),
                               'direction':direction,'entry_price':price})

    def close_position(self, ticket):
        # fetch position and close fully
        pos = mt5.positions_get(ticket=ticket)[0]
        direction = -1 if pos.type==mt5.POSITION_TYPE_BUY else 1
        tick = mt5.symbol_info_tick(self.symbol)
        price = tick.bid if direction==1 else tick.ask
        req = {"action":mt5.TRADE_ACTION_DEAL,"symbol":self.symbol,
               "volume":pos.volume,"type":mt5.ORDER_TYPE_SELL if direction==1 else mt5.ORDER_TYPE_BUY,
               "position":ticket,"price":price,"deviation":10,
               "magic":123456,"type_time":mt5.ORDER_TIME_GTC,"type_filling":mt5.ORDER_FILLING_IOC}
        res = mt5.order_send(req)
        if res.retcode==mt5.TRADE_RETCODE_DONE:
            self.cards = [c for c in self.cards if c['ticket']!=ticket]

    def time_based_tp(self):
        now = datetime.now()
        for c in list(self.cards):
            if now - c['entry_time'] >= timedelta(hours=8):
                self.close_position(c['ticket'])

    def trailing_stop(self):
        df1d = self.get_ohlcv(mt5.TIMEFRAME_D1, 50)
        ema20 = df1d['close'].ewm(span=20, adjust=False).mean().iat[-1]
        tick = mt5.symbol_info_tick(self.symbol)
        price = tick.last
        for c in list(self.cards):
            if c['direction']==1 and price < ema20:
                self.close_position(c['ticket'])
            if c['direction']==-1 and price > ema20:
                self.close_position(c['ticket'])

    def run(self):
        try:
            while True:
                self.sync_cards()
                self.time_based_tp()
                self.trailing_stop()

                # zone & sweep
                zone_dir = self.detect_daily_zone_signal()
                if zone_dir==0 or not self.can_open(zone_dir):
                    time.sleep(self.interval)
                    continue

                sweep_dir = self.detect_sweep_entry()
                if sweep_dir!=zone_dir:
                    time.sleep(self.interval)
                    continue

                # entry
                atr14 = self.compute_atr14_h4()
                tick = mt5.symbol_info_tick(self.symbol)
                price = tick.ask if sweep_dir==1 else tick.bid
                sl = price - sweep_dir * atr14
                self.open_order(sweep_dir, sl)

                time.sleep(self.interval)
        except KeyboardInterrupt:
            mt5.shutdown()

if __name__ == '__main__':
    bot = MT5SMCbot()
    bot.run()
