import MetaTrader5 as mt5
import pandas as pd
import time
from smc_indicator_v2 import SmartMoneyConcepts
from nadaraya_watson_indicator import compute_nadaraya_watson_envelope
from datetime import datetime, timedelta
from itertools import combinations
import math


class MT5SmartScalper:
    """
    A scalping trading bot for MT5 that opens trades based on price zones (premium, discount, equilibrium)
    and closes profitable trades to maintain a net PnL of ~1% and balanced long/short exposure.
    Avoids closing positions with negative PnL to prevent locking in losses.
    Aligns with the faucets-and-buckets analogy: faucets open trades, buckets hold positions,
    and cards close the oldest profitable trades to balance PnL and exposure.
    """
    def __init__(self,
                 symbol="BTCUSDc",
                 timeframe=mt5.TIMEFRAME_M5,
                 atr_len=50,
                 atr_tol=0.0005,
                 primary_lot=0.01,
                 hedge_ratio=0.5,
                 fee_pct=0.005,
                 target_pct=0.0075,
                 max_cards=40,
                 max_per_dir=20,
                 imbalance_thresh=0.01,
                 max_margin_ratio=0.5,
                 min_free_equity_pct=0.5,
                 rebalance_thresh=0.03,  # Lot-based threshold (0.03 lots)
                 min_bucket_lot=0.02,
                 min_hold_minutes=5,
                 open_cooldown_seconds=60,
                 interval=30):
        """
        Initialize the bot with trading parameters.
        """
        if not mt5.initialize():
            raise RuntimeError("MT5 initialize() failed")
        self.symbol = symbol
        self.timeframe = timeframe
        info = mt5.symbol_info(self.symbol)
        if info is None:
            raise RuntimeError(f"Symbol {self.symbol} not found")
        # Exness cent accounts: profit in cents, price in dollars
        if self.symbol.endswith('c'):
            self.contract_size = 1.0
            self.profit_divisor = 1.0
        else:
            self.contract_size = info.trade_contract_size
            self.profit_divisor = 1.0

        self.atr_len = atr_len
        self.atr_tol = atr_tol
        self.primary_lot = primary_lot
        self.hedge_ratio = hedge_ratio
        self.fee_pct = fee_pct
        self.target_pct = target_pct
        self.max_cards = max_cards
        self.max_per_dir = max_per_dir
        self.cards = []
        self.closed_cards = []
        self.imbalance_thresh = imbalance_thresh
        self.max_margin_ratio = max_margin_ratio
        self.min_free_equity_pct = min_free_equity_pct
        self.rebalance_thresh = rebalance_thresh
        self.min_bucket_lot = min_bucket_lot
        self.min_hold_minutes = min_hold_minutes
        self.open_cooldown_seconds = open_cooldown_seconds
        self.last_close_time = None
        self.smc = None
        self.interval = interval

    def compute_global_hedge_ratio(self, price):
        """
        Combines net-long and net-short drawdowns into one ratio [0..∞).
        Prints all intermediate metrics for debugging.
        """
        cs = self.contract_size

        # 1) split into longs vs shorts
        longs  = [c for c in self.cards if c['direction']==1]
        shorts = [c for c in self.cards if c['direction']==-1]

        def pnl_pct_and_notional(group, dirn, side):
            tot = sum(c['entry_price']*c['lot_size']*cs for c in group)
            pnl = sum(dirn*(price-c['entry_price'])*c['lot_size']*cs for c in group)
            pct = (pnl/tot) if tot>0 else 0.0
            print(f"[HEDGE DEBUG] {side:>5}: Notional=${tot:.2f}, PnL=${pnl:.2f}, PnL%={pct:.2%}")
            return pct, tot

        long_pct,  long_not  = pnl_pct_and_notional(longs,  1,  "LONG")
        short_pct, short_not = pnl_pct_and_notional(shorts, -1, "SHORT")

        # 2) pick the side with the worst drawdown
        if long_pct < short_pct:
            draw, notional, side = long_pct,  long_not,  "LONG"
        else:
            draw, notional, side = short_pct, short_not, "SHORT"
        print(f"[HEDGE DEBUG] Worst drawdown on {side}: {draw:.2%} of ${notional:.2f}")

        # 3) map drawdown to a size-ratio
        min_t, max_t = 0.001, 0.0075   # 0.1% .. 0.75%
        dd = -draw                     # positive drawdown
        if dd <= min_t:
            ratio = 1.0
        else:
            ratio = (dd - min_t) / (max_t - min_t)
        ratio = min(ratio, 3.0)
        print(f"[HEDGE DEBUG] Hedge‐ratio = {ratio:.2f}  (mapped from {dd:.2%} drawdown)")

        return ratio

    def get_ohlcv(self, timeframe=None, n=200):
        """
        Fetch OHLCV data from MT5 for the given timeframe and symbol.
        """
        if timeframe is None:
            timeframe = self.timeframe
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, n)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Failed to fetch {timeframe} data for {self.symbol}")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        return df[['time', 'open', 'high', 'low', 'close', 'volume']]

    def add_indicators(self, df):
        """
        Add technical indicators: ATR and EMA slope (for debugging only).
        """
        df['prev_close'] = df['close'].shift(1)
        df['TR'] = df[['high', 'prev_close']].max(axis=1) - df[['low', 'prev_close']].min(axis=1)
        df['ATR'] = df['TR'].rolling(self.atr_len, min_periods=1).mean()
        df['ATR_pct'] = df['ATR'] / df['close']
        df['EMA25'] = df['close'].ewm(span=25, adjust=False).mean()
        df['EMA_slope'] = df['EMA25'].diff()
        print(f"[IND] ATR%={df['ATR_pct'].iat[-1]:.5%}, EMA_slope={df['EMA_slope'].iat[-1]:.5f}")
        return df

    def detect_smc_areas(self, df, price):
        """
        Detect price zones (premium, discount, equilibrium) using range-based checks with a 0.1% buffer for premium/discount,
        compatible with updated smc_indicator_v2.py mimicking LuxAlgo's Pine Script.
        """
        if df.empty or len(df) < 50:
            print("[SMC ERROR] Insufficient data")
            return [], []
        try:
            self.smc.df = df
            out = self.smc.run()
            if out.empty:
                print("[SMC ERROR] SMC run returned empty output")
                return [], []
            last = out.iloc[-1]
            required_cols = {'premium_low', 'premium_high', 'discount_low', 'discount_high', 'equilibrium_low', 'equilibrium_high', 'premium_zone', 'discount_zone'}
            if not required_cols.issubset(out.columns):
                print(f"[SMC ERROR] Missing columns: {required_cols - set(out.columns)}")
                return [], []
            areas = []
            directions = []
            buffer = 0.001  # 0.1% buffer
            # Premium zone, expanded both ends
            p_low  = last['premium_low']  * (1 - buffer)
            p_high = last['premium_high'] * (1 + buffer)
            if p_low <= price <= p_high:
                areas.append('premium')
                directions.append(-1)
            # Discount zone, expanded both ends
            d_low  = last['discount_low']  * (1 - buffer)
            d_high = last['discount_high'] * (1 + buffer)
            if d_low <= price <= d_high:
                areas.append('discount')
                directions.append(1)
            # Equilibrium zone, expanded both ends
            e_low  = last['equilibrium_low']  * (1 - buffer)
            e_high = last['equilibrium_high'] * (1 + buffer)
            if e_low <= price <= e_high:
                areas.append('equilibrium')
                directions.extend([1, -1])
            print(f"[SMC AREAS] {areas}, Directions: {directions}, Price={price:.2f}")
            return areas, directions
        except Exception as e:
            print(f"[SMC ERROR] {e}")
            return [], []

    def can_trade(self):
        """
        Check if trading is allowed based on margin and equity constraints.
        """
        info = mt5.account_info()
        if not info:
            print("[CHECK] Failed to fetch account info")
            return False
        if info.margin_free < info.equity * self.min_free_equity_pct:
            print("[CHECK] Free margin too low")
            return False
        if info.margin / info.equity > self.max_margin_ratio:
            print("[CHECK] Margin ratio too high")
            return False
        return True

    def can_open(self, direction, price, tol_pct=0.01):
        """
        Check if a new trade can be opened, considering max cards, direction limits,
        price proximity, and cooldown.
        """
            # overall cap
        if len(self.cards) >= self.max_cards:
            print("[CHECK] Max cards reached")
            return False

        # count longs and shorts separately
        longs  = sum(1 for c in self.cards if c['direction']==1  and not c.get('taken'))
        shorts = sum(1 for c in self.cards if c['direction']==-1 and not c.get('taken'))

        if direction==1 and longs >= self.max_per_dir:
            print("[CHECK] Max longs reached")
            return False
        if direction==-1 and shorts >= self.max_per_dir:
            print("[CHECK] Max shorts reached")
            return False
    
        if len(self.cards) >= self.max_cards:
            print("[CHECK] Max cards reached")
            return False
        cnt = sum(1 for c in self.cards if c['direction'] == direction and not c.get('taken'))
        if cnt >= self.max_per_dir:
            print(f"[CHECK] Max cards per direction ({direction}) reached")
            return False
        tol = price * tol_pct
        for c in self.cards:
            if not c.get('taken') and c['direction'] == direction:

                if abs(c['entry_price'] - price) <= tol:
                    print(f"[CHECK] Price {price:.5f} within ±{tol_pct*100:.2f}% ({tol:.5f}) of existing {c['entry_price']:.5f}, skipping")
                    return False
        if self.last_close_time:
            time_since_close = (datetime.now() - self.last_close_time).total_seconds()
            if time_since_close < self.open_cooldown_seconds:
                print(f"[CHECK] Cooldown active, {self.open_cooldown_seconds - time_since_close:.0f}s remaining")
                return False
        return True

    def place_orders(self, total_lot, direction, area, comment_suffix):
        """
        Break total_lot into self.primary_lot‐sized chunks, rounding UP to
        the next multiple of primary_lot. Skip if below primary_lot.
        """
        min_lot = self.primary_lot
        # if requested size is below the minimum, skip entirely
        if total_lot < min_lot:
            print(f"[PLACE] Requested {total_lot:.4f} < min lot {min_lot:.2f}, skipping")
            return

        # round UP to nearest 0.01: e.g. 0.055 → 0.06
        chunks = math.ceil(total_lot / min_lot)
        for _ in range(chunks):
            self._send_market_order(min_lot, direction, area, comment_suffix)

    def _send_market_order(self, volume, direction, area, comment_suffix):
        """
        Fire exactly one MT5 market order for `volume` lots in `direction`,
        tag with `area` and `comment_suffix`, then record it in self.cards.
        """
        order_type = mt5.ORDER_TYPE_BUY   if direction == 1 else mt5.ORDER_TYPE_SELL
        tick       = mt5.symbol_info_tick(self.symbol)
        price_send = tick.ask if direction == 1 else tick.bid

        req = {
        "action":     mt5.TRADE_ACTION_DEAL,
        "symbol":     self.symbol,
        "volume":     float(volume),
        "type":       order_type,
        "price":      price_send,
        "deviation":  10,
        "magic":      123456,
        "comment":    f"SMC {area} {comment_suffix}",
        "type_time":  mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(req)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[ORDER] {comment_suffix} {'BUY' if direction==1 else 'SELL'} @{price_send}, lot={volume}")
            self.cards.append({
            'ticket':      res.order,
            'time':        pd.Timestamp.now(),
            'entry_price': price_send,
            'direction':   direction,
            'area':        area,
            'lot_size':    volume,
            'taken':       False
            })
        else:
            print(f"[ORDER] FAILED {comment_suffix}: {res.retcode}, {res.comment}")

    def open_card(self, price, direction, area, hedge_ratio=1.0):
        # 1) basic gating
        if not self.can_open(direction, price):
            return

        # 2) compute scaled lot, snap up to 0.01
        raw = self.primary_lot * hedge_ratio
        lot = math.ceil(raw / self.primary_lot) * self.primary_lot

        print(f"[OPEN DEBUG] hedge_ratio={hedge_ratio:.2f}, raw={raw:.4f}, final lot={lot:.2f}")
        if lot < self.primary_lot:
            return

        # 3) send it in 0.01‐lots (or smaller remainder)
        self.place_orders(lot, direction, area, comment_suffix="open")

        # 4) bucket‐rebalance on top if needed
        long_vol  = sum(c['lot_size'] for c in self.cards if c['direction']==1)
        short_vol = sum(c['lot_size'] for c in self.cards if c['direction']==-1)
        diff = abs(long_vol - short_vol)
        if diff > self.rebalance_thresh:
            # send the extra in the opposite direction
            extra_dir = 1 if short_vol > long_vol else -1
            extra_lot = diff - self.rebalance_thresh
            if extra_lot > 0 and self.can_open(extra_dir, price):
                self.place_orders(extra_lot, extra_dir, area, comment_suffix="rebalance")

    def sync_cards(self):
        """
        Synchronize in-memory cards with live MT5 positions, preventing negative dropped counts.
        """
        live = mt5.positions_get(symbol=self.symbol) or []
        live_tickets = {p.ticket for p in live}
        kept = []
        seen_tickets = set()
        for c in self.cards:
            ticket = c.get('ticket')
            if ticket in live_tickets and not c.get('taken') and ticket not in seen_tickets:
                kept.append(c)
                seen_tickets.add(ticket)
        known = seen_tickets
        for pos in live:
            if pos.ticket not in known and pos.ticket not in seen_tickets:
                kept.append({
                    'ticket': pos.ticket,
                    'entry_price': pos.price_open,
                    'direction': 1 if pos.type == mt5.POSITION_TYPE_BUY else -1,
                    'lot_size': pos.volume,
                    'time': datetime.fromtimestamp(pos.time),
                    'tiers': [],
                    'area': None,
                    'smc_signals': {},
                    'taken': False
                })
                seen_tickets.add(pos.ticket)
        dropped = max(0, len(self.cards) - len(kept))
        if dropped:
            print(f"[SYNC] Dropped {dropped} stale card(s) no longer in MT5")
        if len(self.cards) != len(kept):
            print(f"[SYNC DEBUG] Before: {len(self.cards)} cards, After: {len(kept)} cards, Live tickets: {live_tickets}")
        self.cards = kept

    def _close_position(self, ticket, volume, direction, reason):
        """
        Send a market order to close an MT5 position by ticket.
        """
        close_type = mt5.ORDER_TYPE_SELL if direction == 1 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(self.symbol)
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(volume),
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": 123456,
            "comment": reason,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(req)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[CLOSE] {reason} ticket {ticket} @{price:.5f}")
            self.last_close_time = datetime.now()
            return True
        elif res.retcode == 10013:
            print(f"[CLOSE] Ticket {ticket} already closed on server (10013), marking as done")
            return True
        else:
            print(f"[CLOSE] Failed ticket {ticket}: {res.retcode}, {res.comment}")
            return False

    def _would_balance_if_closed(self, c):
        long_vol  = sum(cd['lot_size'] for cd in self.cards if cd['direction']==1)
        short_vol = sum(cd['lot_size'] for cd in self.cards if cd['direction']==-1)

        # volumes *after* closing c
        if c['direction']==1:
            long_vol -= c['lot_size']
        else:
            short_vol -= c['lot_size']

        return abs(long_vol - short_vol) <= self.rebalance_thresh

    def manage_cards(self, price):
        zones = self.smc.run().iloc[-1]
        p_low, p_high = zones['premium_low'], zones['premium_high']
        d_low, d_high = zones['discount_low'], zones['discount_high']

        now       = pd.Timestamp.now()
        survivors = []

        # helper to check balance if we close these two tickets
        def would_balance_if_closed(c1, c2=None):
            long_vol  = sum(c['lot_size'] for c in self.cards if c['direction']==1 and not c.get('taken'))
            short_vol = sum(c['lot_size'] for c in self.cards if c['direction']==-1 and not c.get('taken'))

            # subtract c1
            if c1['direction']==1:  long_vol  -= c1['lot_size']
            else:                   short_vol -= c1['lot_size']
            # subtract c2 too if provided
            if c2:
                if c2['direction']==1:  long_vol  -= c2['lot_size']
                else:                    short_vol -= c2['lot_size']

            return abs(long_vol - short_vol) <= self.rebalance_thresh

        for c in self.cards:
            age_min = (now - c['time']).total_seconds()/60
            if c.get('taken') or age_min < self.min_hold_minutes:
                survivors.append(c)
                continue

            # calculate pnl & pnl_pct
            pnl      = c['direction'] * (price - c['entry_price']) * c['lot_size'] * self.contract_size
            notional = c['entry_price'] * c['lot_size'] * self.contract_size
            pnl_pct  = pnl / notional

            # need to be profitable enough
            if pnl_pct < self.target_pct:
                survivors.append(c)
                continue

            # zone‐based exit condition
            do_close = (
                (c['direction']==1 and price >= p_low) or
                (c['direction']==-1 and price <= d_high)
            )
            if not do_close:
                survivors.append(c)
                continue

            # 1) if closing this one alone stays within rebalance_thresh, just close it
            if would_balance_if_closed(c):
                partner = None
            else:
                # 2) otherwise look for exactly one counter‐trade d to close as well
                partner = None
                for d in self.cards:
                    if d is c or d.get('taken') or d['direction']==c['direction']:
                        continue
                    age_d = (now - d['time']).total_seconds()/3600
                    # require it be older than 1h
                    if age_d < 1:
                        continue

                    pnl_d      = d['direction'] * (price - d['entry_price']) * d['lot_size'] * self.contract_size
                    notional_d = d['entry_price'] * d['lot_size'] * self.contract_size
                    net_not    = notional + notional_d
                    combined_pct = (pnl + pnl_d) / net_not

                    # does closing both still beat target_pct?
                    if combined_pct >= self.target_pct and would_balance_if_closed(c, d):
                        partner = d
                        break

            if partner is None and not would_balance_if_closed(c):
                # we can't close this one without unbalancing
                print(f"[SKIP CLOSE] #{c['ticket']} profitable but no partner found")
                survivors.append(c)
                continue

            # if we get here, we close c (and partner if any)
            to_close = [c] + ([partner] if partner else [])
            for x in to_close:
                success = self._close_position(
                    ticket   = x['ticket'],
                    volume   = x['lot_size'],
                    direction= x['direction'],
                    reason   = "zone_exit"
                )
                if success:
                    pnl_x = x['direction'] * (price - x['entry_price']) * x['lot_size'] * self.contract_size
                    x.update({
                        'exit_price':  price,
                        'profit':      pnl_x,
                        'exit_reason': 'zone_exit',
                        'taken':       True
                    })
                    self.closed_cards.append(x)
                    print(f"[CARD CLOSE] Zone+PnL exit: {x}")

        self.cards = survivors
        print(f"[STATUS] Open cards: {len(self.cards)}, Closed: {len(self.closed_cards)}")

        # after all zone‐exits, you may still want to bucket‐rebalance
        self.rebalance_buckets(price)


    def balance_to_1pct(self, price, target_pct=0.0075):
        """
        Close profitable cards to maintain net PnL around 1%.
        Do not close positions with negative PnL.
        Respect minimum holding period.
        """
        if not self.cards:
            print("[BALANCE] No open cards")
            return
        total_notional = sum(c['entry_price'] * c['lot_size'] * self.contract_size for c in self.cards)
        print(f"[BALANCE DEBUG] Total notional={total_notional:.2f}, Cards={len(self.cards)}")
        if total_notional <= 0:
            print("[BALANCE] Invalid total notional, skipping")
            return
        card_pnls = [(c, c['direction'] * (price - c['entry_price']) * c['lot_size'] * self.contract_size)
                     for c in self.cards]
        net_pnl = sum(p for _, p in card_pnls)
        net_pnl_pct = net_pnl / total_notional if total_notional > 0 else 0
        print(f"[BALANCE] Net PnL%={net_pnl_pct:.5%}, Target={target_pct*100:.1f}%")

        if net_pnl_pct > 0.015:
            now = pd.Timestamp.now()
            candidates = [
                (c, p) for c, p in card_pnls
                if p > 0 and (now - c['time']).total_seconds() / 60 >= self.min_hold_minutes
            ]
            if not candidates:
                print("[BALANCE] No profitable cards eligible for closure")
                return
            candidates.sort(key=lambda x: x[0]['time'])
            target_abs = target_pct * total_notional
            to_close, cum = [], 0
            for c, p in candidates:
                to_close.append((c, p))
                cum += p
                if cum >= (net_pnl - target_abs):
                    break
            for c, p in to_close:
                if self._close_position(c['ticket'], c['lot_size'], c['direction'], "balance_to_1pct"):
                    c['exit_price'] = price
                    c['profit'] = p
                    c['taken'] = True
                    c['exit_reason'] = 'balance_to_1pct'
                    self.closed_cards.append(c)
                    self.cards.remove(c)
                    print(f"[CARD CLOSE] 1%-group: {c}")

    def rebalance_buckets(self, price):
        """
        Close the oldest profitable card in the overexposed direction to balance buckets.
        Respect min_bucket_lot and either:
          • a 15 min minimum hold, or
          • an emergency PnL% threshold (bypass hold time).
        """
        long_vol  = sum(c['lot_size'] for c in self.cards if c['direction']==1)
        short_vol = sum(c['lot_size'] for c in self.cards if c['direction']==-1)
        print(f"[REBAL] Long={long_vol:.2f}, Short={short_vol:.2f}, Diff={abs(long_vol-short_vol):.2f}")
        if abs(long_vol - short_vol) <= self.rebalance_thresh:
            print(f"[REBAL] Volume difference within threshold ({self.rebalance_thresh} lots)")
            return

        over = 1 if long_vol > short_vol else -1
        now = pd.Timestamp.now()

        # parameters
        min_age       = 15.0        # minutes
        emergency_pct = 0.001        # e.g. 0.1% profit lets us ignore the age

        candidates = []
        for c in sorted(self.cards, key=lambda x: x['time']):
            if c['direction'] != over:
                continue

            age = (now - c['time']).total_seconds() / 60
            pnl  = c['direction'] * (price - c['entry_price']) * c['lot_size'] * self.contract_size
            notional = c['entry_price'] * c['lot_size'] * self.contract_size
            pnl_pct  = pnl / notional if notional>0 else 0

            # must be profitable
            if pnl <= 0:
                continue

            # enforce either age OR emergency PnL%
            if age < min_age and pnl_pct < emergency_pct:
                continue

            # also make sure we won't breach min_bucket_lot if we close it
            if over==1 and long_vol - c['lot_size'] < self.min_bucket_lot:
                print(f"[REBAL] Would breach min_bucket_lot closing #{c['ticket']}, skip")
                continue
            if over==-1 and short_vol - c['lot_size'] < self.min_bucket_lot:
                print(f"[REBAL] Would breach min_bucket_lot closing #{c['ticket']}, skip")
                continue

            candidates.append(c)
            break  # only close the oldest one that passes

        print(f"[REBAL DEBUG] Candidates for {'long' if over==1 else 'short'}: {len(candidates)}")
        if not candidates:
            return

        c = candidates[0]
        success = self._close_position(
            ticket   = c['ticket'],
            volume   = c['lot_size'],
            direction= c['direction'],
            reason   = "bucket_balance"
        )
        if success:
            c['exit_price']  = price
            c['profit']      = c['direction']*(price-c['entry_price'])*c['lot_size']*self.contract_size
            c['taken']       = True
            c['exit_reason'] = 'bucket_balance'
            self.closed_cards.append(c)
            self.cards.remove(c)
            print(f"[CARD CLOSE] Bucket balance: {c}")


    def close_aged_flat(self, price, min_age_days=1, tol_pct=0.001):
        """
        Close cards older than `min_age_days` whose PnL% is above -tol_pct (e.g. -0.1%).
        """
        now = pd.Timestamp.now()
        for c in list(self.cards):
            age_days = (now - c['time']).total_seconds() / (60*60*24)
            if age_days <= min_age_days or c.get('taken'):
                continue

            # compute PnL%
            pnl      = c['direction'] * (price - c['entry_price']) * c['lot_size'] * self.contract_size
            notional = c['entry_price'] * c['lot_size'] * self.contract_size
            pnl_pct  = pnl / notional if notional>0 else 0

            if pnl_pct > -tol_pct:
                # force‐close to cap loss at tol_pct
                if self._close_position(
                    ticket=c['ticket'],
                    volume=c['lot_size'],
                    direction=c['direction'],
                    reason="aged_flat_close"
                ):
                    c.update({
                        'exit_price': price,
                        'profit':    pnl,
                        'exit_reason': 'aged_flat_close',
                        'taken':     True
                    })
                    self.closed_cards.append(c)
                    self.cards.remove(c)
                    print(f"[AGED FLAT] Closed ticket {c['ticket']} @ {price:.2f} "
                          f"(age={age_days:.2f}d, PnL%={pnl_pct*100:.2f}%)")

    def close_aged_offsets(self, days=1, target_pct=0.002):
        """
        Offset aged losing positions with profitable opposites, respecting minimum hold time.
        """
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            print("[OFFSET] No positions to offset")
            return
        print(f"[OFFSET DEBUG] Found {len(positions)} positions")
        now = pd.Timestamp.now()
        aged_shorts = []
        aged_longs = []
        prof_longs = []
        prof_shorts = 
        for pos in positions:
            opened = datetime.fromtimestamp(pos.time)
            age = (now - opened).total_seconds() / (60 * 60 * 24)
            pnl = pos.profit
            notional = pos.price_open * pos.volume * self.contract_size
            pnl_pct = pnl / notional if notional > 0 else 0
            print(f"[OFFSET DEBUG] Ticket {pos.ticket}: Age={age:.2f} days, PnL={pnl:.2f}, Notional={notional:.2f}, PnL%={pnl_pct:.2%}")
            if age > days and pnl < 0 and abs(pnl_pct) >= target_pct:
                if pos.type == mt5.POSITION_TYPE_SELL:
                    aged_shorts.append((pos, pnl_pct))
                else:
                    aged_longs.append((pos, pnl_pct))
            elif pnl > 0 and pnl_pct >= target_pct:
                if pos.type == mt5.POSITION_TYPE_BUY:
                    prof_longs.append((pos, pnl_pct))
                else:
                    prof_shorts.append((pos, pnl_pct))

        def _close(pos, reason):
            close_type = mt5.ORDER_TYPE_BUY if pos.type == mt5.POSITION_TYPE_SELL else mt5.ORDER_TYPE_SELL
            tick = mt5.symbol_info_tick(pos.symbol)
            price = tick.ask if close_type == mt5.ORDER_TYPE_BUY else tick.bid
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": float(pos.volume),
                "type": close_type,
                "position": pos.ticket,
                "price": price,
                "deviation": 10,
                "magic": 123456,
                "comment": reason,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            res = mt5.order_send(req)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[OFFSET] Closed ticket {pos.ticket} @{price:.4f} ({reason})")
                self.last_close_time = datetime.now()
                return True
            else:
                print(f"[OFFSET] Failed to close {pos.ticket}: {res.retcode}, {res.comment}")
                return False

        for short_pos, pnl in aged_shorts:
            needed = abs(pnl)
            match = next((p for p, pct in prof_longs if pct >= needed), None)
            if match and _close(match, "offset aged short"):
                _close(short_pos, "aged short")
                continue
            for (p1, p1_pct), (p2, p2_pct) in combinations(prof_longs, 2):
                if p1_pct + p2_pct >= needed:
                    if _close(p1, "offset aged short") and _close(p2, "offset aged short"):
                        _close(short_pos, "aged short")
                    break

        for long_pos, pnl in aged_longs:
            needed = abs(pnl)
            match = next((p for p, pct in prof_shorts if pct >= needed), None)
            if match and _close(match, "offset aged long"):
                _close(long_pos, "aged long")
                continue
            for (p1, p1_pct), (p2, p2_pct) in combinations(prof_shorts, 2):
                if p1_pct + p2_pct >= needed:
                    if _close(p1, "offset aged long") and _close(p2, "offset aged long"):
                        _close(long_pos, "aged long")
                    break
        self.sync_cards()  # Ensure cards reflect MT5 positions

    def adjust_aged_take_profits(self):
        """For any open position older than 4 h, set a 0.25% TP; if ≥8 h, move TP to break-even."""
        positions = mt5.positions_get(symbol=self.symbol) or []
        now = datetime.now()

        for pos in positions:
            opened = datetime.fromtimestamp(pos.time)
            age = (now - opened).total_seconds() / 3600  # in hours

            # too young?
            if age < 4:
                continue

            # compute desired TP
            entry = pos.price_open
            if age >= 8:
                desired_tp = entry
            else:  # 4 ≤ age < 8
                multiplier = 1 + 0.0025 if pos.type == mt5.POSITION_TYPE_BUY else 1 - 0.0025
                desired_tp = entry * multiplier

            # --- GUARD: skip if TP already set ---
            current_tp = pos.tp
            if current_tp:
                # It already has a TP (manually set or previously applied), so leave it alone.
                print(f"[ADJUST TP SKIPPED] #{pos.ticket} age={age:.1f}h, existing TP={current_tp:.2f}")
                continue

            # send modify request
            request = {
                "action":   mt5.TRADE_ACTION_SLTP,
                "position": pos.ticket,
                "symbol":   self.symbol,
                "sl":       pos.sl,
                "tp":       float(desired_tp),
                "magic":    123456,
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[ADJUST TP] #{pos.ticket} age={age:.1f}h, tp set to {desired_tp:.2f}")
            else:
                print(f"[ADJUST TP FAILED] #{pos.ticket}: {result.retcode} {result.comment}")
    def close_excess_risk(self,
                          lookback_days: float = 1.0,
                          risk_pct:      float = 0.20,
                          min_profit:    float = 75.0):
        """
        1) Look up all closed deals for this symbol in the past `lookback_days`.
        2) Sum their PnL; if ≤ min_profit, bail out.
        3) Compute risk_amount = realized * risk_pct.
        4) Find all **open** positions ≥4 h old with **negative** PnL.
        5) Close as many of those (oldest first) as you can without exceeding risk_amount.
        """
        now      = datetime.now()
        since    = now - timedelta(days=lookback_days)
        # fetch all history deals for our symbol in that window
        deals    = mt5.history_deals_get(since, now, group=self.symbol) or []
        realized = sum(d.profit/self.profit_divisor for d in deals if d.symbol == self.symbol)

        if realized <= min_profit:
            print(f"[RISK] Only ${realized:.2f} realized (<${min_profit}), skipping risk-cap")
            return

        risk_amount = realized * risk_pct
        print(f"[RISK] Realized=${realized:.2f}, capping losses to ${risk_amount:.2f}")

        # collect open positions ≥4 h old with negative PnL
        old_losers = []
        for pos in mt5.positions_get(symbol=self.symbol) or []:
            age_h = (now - datetime.fromtimestamp(pos.time)).total_seconds()/3600
            pnl   = pos.profit / self.profit_divisor
            if age_h >= 4 and pnl < 0:
                old_losers.append((pos, pnl, age_h))

        # sort by age descending (oldest first), then by **least** negative PnL first
        old_losers.sort(key=lambda x: (-x[2], x[1]))

        to_close = []
        used     = 0.0
        for pos, pnl, age_h in old_losers:
            loss = -pnl
            if used + loss <= risk_amount:
                to_close.append(pos)
                used += loss

        if not to_close:
            print(f"[RISK] No old losers can be closed without exceeding ${risk_amount:.2f}")
            return

        # close them
        for pos in to_close:
            direction =  1 if pos.type == mt5.POSITION_TYPE_BUY else -1
            self._close_position(
                ticket    = pos.ticket,
                volume    = pos.volume,
                direction = direction,
                reason    = "risk_cap"
            )
        print(f"[RISK] Closed {len(to_close)} old positions, total loss capped at ${used:.2f}")


    def get_journal_df(self):
        """
        Return a DataFrame of closed trades for analysis.
        """
        df = pd.DataFrame(self.closed_cards)
        if not df.empty:
            df['smc_signals'] = df.get('smc_signals', [{}]*len(df)).apply(str)
        print(f"[JOURNAL] {len(df)} closed cards")
        return df

    def debug_cards(self, price):
        """Debug open cards with correct notional and PnL."""
        print("\n===== DEBUG OPEN CARDS =====")
        if not self.cards:
            print("  No open cards")
        else:
            for i, c in enumerate(self.cards, start=1):
                entry = c['entry_price']
                lots = c['lot_size']
                dirn = c['direction']
                units = lots * self.contract_size
                notional = entry * units
                pnl = dirn * (price - entry) * units
                pct = (pnl / notional * 100) if notional > 0 else 0
                print(f" {i}. Ticket: {c.get('ticket','N/A')}, "
                      f"Dir: {'LONG' if dirn==1 else 'SHORT':5}, "
                      f"Entry={entry:.5f}, Lots={lots:.2f}, Units={units:.4f}, "
                      f"PnL=${pnl:.5f} ({pct:.2f}%)")
            total_notional = sum(c['entry_price'] * c['lot_size'] * self.contract_size for c in self.cards)
            net_pnl = sum(c['direction'] * (price - c['entry_price']) * c['lot_size'] * self.contract_size for c in self.cards)
            net_pct = (net_pnl / total_notional * 100) if total_notional > 0 else 0
            print(f"\n Total notional = ${total_notional:,.2f}")
            print(f" Net    PnL     = ${net_pnl:,.2f} ({net_pct:.2f}%)")
        print("============================\n")

    def debug_live_positions(self):
        """Debug live MT5 positions with correct notional and PnL."""
        positions = mt5.positions_get(symbol=self.symbol)
        print("\n===== DEBUG MT5 POSITIONS (Correct Notional) =====")
        if not positions:
            print("  No live MT5 positions for", self.symbol)
            return
        total_notional = 0.0
        net_pnl = 0.0
        for pos in positions:
            dir_str = "LONG" if pos.type == mt5.POSITION_TYPE_BUY else "SHORT"
            # convert profit to USD
            profit_usd = pos.profit / self.profit_divisor
            units = pos.volume * self.contract_size
            notional = units * pos.price_open
            pnl_pct = (profit_usd / notional * 100) if notional > 0 else 0.0
            total_notional += notional
            net_pnl += profit_usd
            print(f" Ticket {pos.ticket}: {dir_str:5} | "
                  f"Open={pos.price_open:.2f} x{pos.volume:.2f} lots (units={units:.4f}) "
                  f"→ Notional=${notional:,.2f} | "
                  f"P/L=${profit_usd:.2f} ({pnl_pct:.2f}%)")
        net_pct = (net_pnl / total_notional * 100) if total_notional > 0 else 0.0
        print(f"\n Total notional = ${total_notional:,.2f}")
        print(f" Net    PnL     = ${net_pnl:,.2f} ({net_pct:.2f}%)")
        print("==================================================\n")
    def take_profit_nadaraya_imbalance(self):
        """
        Close any position whose %PnL ≥ self.target_pct when price
        has touched the Nadaraya-Watson envelope (upper for longs,
        lower for shorts) on the 5m timeframe.
        """
        # 1) Pull the last 500 5m bars
        df5 = self.get_ohlcv(timeframe=mt5.TIMEFRAME_M5, n=500)

        # 2) Compute the Nadaraya-Watson envelope
        upper, lower = compute_nadaraya_watson_envelope(df5)

        # 3) Current price
        price = df5['close'].iat[-1]

        # 4) Scan open cards
        for c in list(self.cards):
            pnl_pct = c['direction'] * (price - c['entry_price']) / c['entry_price']
            # only candidates at or above your target_pct
            if pnl_pct < self.target_pct:
                continue

            # long → price ≥ upper | short → price ≤ lower
            if (c['direction'] == 1 and price >= upper) or \
               (c['direction'] == -1 and price <= lower):

                if self._close_position(
                    ticket   = c['ticket'],
                    volume   = c['lot_size'],
                    direction= c['direction'],
                    reason   = "nwe_exit"
                ):
                    # record it
                    c.update({
                        'exit_price':  price,
                        'profit':      c['direction'] * (price - c['entry_price']) * c['lot_size'],
                        'exit_reason': 'nwe_exit',
                        'taken':       True
                    })
                    self.closed_cards.append(c)
                    self.cards.remove(c)
                    print(f"[NWE EXIT] Closed #{c['ticket']} PnL%={pnl_pct:.2%} @ {price:.2f}")

    def run(self):
        """
        Main trading loop: sync cards, manage positions, balance PnL/buckets, and open trades.
        """
        last_df_hash = None
        try:
            while True:
                if not self.can_trade():
                    print("[RUN] Trading paused due to margin/equity constraints")
                    time.sleep(self.interval)
                    continue
                self.sync_cards()
                self.adjust_aged_take_profits()      # ← new
                df = self.get_ohlcv()
                df_h1 = self.get_ohlcv(timeframe=mt5.TIMEFRAME_H1, n=100)
                df_hash = hash(df.to_string())
                if self.smc is None or df_hash != last_df_hash:
                    self.smc = SmartMoneyConcepts(df)
                    self.smc.multi_timeframe_levels(df_h1, label='H1')
                    last_df_hash = df_hash
                df = self.add_indicators(df)
                price = df['close'].iat[-1]
                self.debug_live_positions()
                self.debug_cards(price)
                hedge_ratio = self.compute_global_hedge_ratio(price)
                areas, directions = self.detect_smc_areas(df, price)
                self.smc.multi_timeframe_levels(df_h1, label='H1')
                out = self.smc.run()
                if not {'premium_zone', 'discount_zone'}.issubset(out.columns):
                    print("[RUN] Skipping iteration due to invalid SMC output")
                    time.sleep(self.interval)
                    continue
                last = out.iloc[-1]
                print(f"[SMC RANGES] Premium=[{last['premium_low']:.2f}, {last['premium_high']:.2f}], "
                      f"Discount=[{last['discount_low']:.2f}, {last['discount_high']:.2f}], "
                      f"Equilibrium=[{last['equilibrium_low']:.2f}, {last['equilibrium_high']:.2f}], "
                      f"Price={price:.2f}")
                cols = [
                    'time', 'open', 'high', 'low', 'close', 'volume',
                    'atr', 'leg_internal', 'leg_swing',
                    'bos_int', 'choch_int', 'bos_swg', 'choch_swg',
                    'ob_int', 'ob_swing', 'fvg_bull', 'fvg_bear',
                    'premium_zone', 'discount_zone',
                    'level_H1_high', 'level_H1_low'
                ]
                print("\n===== SMC DEBUG DUMP =====")
                print(out[cols].tail(5).to_string(index=False))
                print(f" CURRENT PRICE: {df['close'].iat[-1]:.4f}\n")

                # 0) first kill small aged losses
                self.close_aged_flat(price, min_age_days=1, tol_pct=0.001)

                self.close_excess_risk(
                        lookback_days=1.0,
                        risk_pct=0.20,
                        min_profit=75.0)

                self.manage_cards(price)
                self.take_profit_nadaraya_imbalance()
                self.balance_to_1pct(price, target_pct=0.0075)
                self.close_aged_offsets(days=1, target_pct=0.002)
                self.rebalance_buckets(price)

                if 'equilibrium' in areas:
                    # in equilibrium zone, open both a long and a short
                    for direction in directions:
                        if direction != 0 and self.can_open(direction, price):
                            self.open_card(price, direction, 'equilibrium', hedge_ratio)
                else:
                    # in premium or discount, open only the specified direction
                    for area, direction in zip(areas, directions):
                        if direction != 0 and self.can_open(direction, price):
                            self.open_card(price, direction, area, hedge_ratio)

                time.sleep(self.interval)
        except KeyboardInterrupt:
            journal = self.get_journal_df()
            journal.to_csv('trade_journal.csv', index=False)
            print("[EXIT] Journal saved to trade_journal.csv")
            print(journal.head())
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
        finally:
            mt5.shutdown()
            print("[EXIT] MT5 connection closed")

if __name__ == "__main__":
    try:
        trader = MT5SmartScalper(
            symbol="BTCUSDc",
            timeframe=mt5.TIMEFRAME_M5,
            primary_lot=0.01
        )
        trader.run()
    except Exception as e:
        print(f"[MAIN ERROR] {e}")
    finally:
        mt5.shutdown()