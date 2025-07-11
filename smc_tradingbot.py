# tradingbot_updated.py
import sys
import asyncio
import time
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from smc_indicator_v6 import SMCStructuresFVG
from mt5_auto_trading import MT5AutoTrading

from config2 import TERMINAL_PATH, ACCOUNT, PASSWORD, SERVER

# ── FIX FOR WINDOWS + AIODNS ─────────────────────────────────────────────
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ── DISCORD REPORTER ──────────────────────────────────────────────────────
class DiscordReporter:
    """
    Sends plain-text messages to Discord via a webhook.
    """
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send_message(self, message: str):
        import aiohttp
        payload = {"content": message}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as resp:
                    if resp.status in (200, 204):
                        print(f"[DISCORD] Sent: {message}")
                    else:
                        text = await resp.text()
                        print(f"[DISCORD ERROR] {resp.status}: {text}")
        except Exception as e:
            print(f"[DISCORD ERROR] {e}")

# ── MT5 SETUP ─────────────────────────────────────────────────────────────
if not mt5.initialize(
    path=TERMINAL_PATH,
    login=ACCOUNT,
    password=PASSWORD,
    server=SERVER
):
    raise RuntimeError("MT5 init failed")
print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] MT5 connected: Login={ACCOUNT}")

# ── CACHING CONFIG ────────────────────────────────────────────────────────
HISTORY_TFS = ['4h', '1h', '30m', '5m', '1m']
TF_MAP = {
    '4h':  mt5.TIMEFRAME_H4,
    '1h':  mt5.TIMEFRAME_H1,
    '30m': mt5.TIMEFRAME_M30,
    '5m':  mt5.TIMEFRAME_M5,
    '1m':  mt5.TIMEFRAME_M1,
}
HIST_CACHE = {}
LAST_FETCH = {}
FETCH_INTERVAL = 5 * 60  # seconds

# ── UTILITIES ─────────────────────────────────────────────────────────────
def fetch_history(symbol: str, tf_name: str, n: int = 100) -> pd.DataFrame:
    """
    Fetch OHLCV once per FETCH_INTERVAL per timeframe, cache in memory.
    """
    now = time.time()
    last = LAST_FETCH.get(tf_name, 0)
    if tf_name not in HIST_CACHE or (now - last) > FETCH_INTERVAL:
        print(f"[DEBUG] Fetching {tf_name} history @ {datetime.now():%H:%M:%S}")
        rates = mt5.copy_rates_from_pos(symbol, TF_MAP[tf_name], 0, n)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No data for {symbol}@{tf_name}")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df[['time','open','high','low','close','tick_volume']]
        df.rename(columns={'tick_volume':'volume'}, inplace=True)
        HIST_CACHE[tf_name] = df
        LAST_FETCH[tf_name] = now
    return HIST_CACHE[tf_name].copy()

def detect_liquidity_sweep_candle(
    bar,            # a Series with ['open','high','low','close']
    dir4h,          # +1 or -1
    fibo50_4h,      # float
    choch_price,    # float, the choch_bull_price or choch_bear_price you picked
    wick_ratio=1.5
):
    o, h, l, c = bar.open, bar.high, bar.low, bar.close
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    # Bullish sweep on a 4h downtrend
    if dir4h == -1:
        # 1) CHoCH price must exceed the 4h fibo50
        if not (choch_price > fibo50_4h):
            return False
        # 2) wick test: upper_wick must be big enough
        return upper_wick > wick_ratio * lower_wick

    # Bearish sweep on a 4h uptrend
    if dir4h == 1:
        if not (choch_price < fibo50_4h):
            return False
        return lower_wick > wick_ratio * upper_wick

    return False

# ── MODULE-SCOPE STATE (top of file) ────────────────────────────────
zone_origin_idx = None
zone_low = zone_high = None
last_zone_alert = None

# your webhook here
DISCORD_WEBHOOK = (
    "https://discord.com/api/v10/webhooks/"
    "1380480444390441063/"
    "_BfWBWfGPW8hsotBlPE6gbK1bpft8y32JhG2bVIBEdE9Lx5jRoYWG04_dJu5Vgu_QB9p"
)
reporter = DiscordReporter(DISCORD_WEBHOOK)

# at module‐scope, before monitor_loop()
last_long_orig  = None
last_short_orig = None

# at module scope
bot = MT5AutoTrading(symbol='BTCUSDc', magic=123456)

async def monitor_loop(symbol='BTCUSDc', n_bars=100, pause_s=10):
    print(f"[DEBUG] Starting monitor loop @ {datetime.now():%Y-%m-%d %H:%M:%S}")
    global last_long_orig, last_short_orig
    global zone_origin_idx, zone_low, zone_high, last_zone_alert
    try:
        while True:
            print(f"[DEBUG] Tick @ {datetime.now():%H:%M:%S}")
            entry_long = entry_short = False
            this_long_orig = this_short_orig = None
            overlap1 = overlap2 = False
            rt_price = None

            # --- 1) 4h analysis/cache ---
            df4h = fetch_history(symbol, '4h', n_bars)
            # inject live 4h price or fallback to 1m close if unavailable/zero
            # tick4h = mt5.symbol_info_tick(symbol)
            # rt4h = tick4h.last if tick4h and tick4h.last > 0 else None
            # always use the most‐recent 1 m close for your “live” 4 h price
            df1m_tmp = fetch_history(symbol, '1m', n_bars)
            rt4h = df1m_tmp.iloc[-1].close
            if not rt4h:
                # fallback: use last 1m close
                df1m_tmp = fetch_history(symbol, '1m', n_bars)
                rt4h = df1m_tmp.iloc[-1].close
            idx4h = df4h.index[-1]
            df4h.at[idx4h, 'close'] = rt4h
            df4h.at[idx4h, 'high'] = max(df4h.at[idx4h, 'high'], rt4h)
            df4h.at[idx4h, 'low']  = min(df4h.at[idx4h, 'low'], rt4h)
            print(f"[DEBUG][4h tick] price={rt4h:.2f} @ {datetime.now():%H:%M:%S}")
            smc4h, conf4h = SMCStructuresFVG(df4h).run()
            print(f"[DEBUG] 4h bias calc @ {datetime.now():%H:%M:%S}")
            last4h = smc4h.iloc[-1]      # must assign before use

            # 1) Build a dict of fib50 levels on 4h, 1h, 30m 
            fib_tfs = ['4h','1h','30m']
            fib50   = {}
            for tf in fib_tfs:
                df_tf, _  = SMCStructuresFVG(fetch_history(symbol, tf, n_bars)).run()
                fib50[tf] = df_tf.iloc[-1]['fibo_0.5']
                print(f"[DEBUG][{tf}] fibo50={fib50[tf]:.2f}")
            
            # 2) Define your “fair” live price (you already have rt4h from 1m close)
            price = rt4h

            # 3) Compute multi‐TF bias
            #    SHORT only if price > every single Fib50
            #    LONG  only if price < every single Fib50
            # 1) multi-TF bias (4h, 1h, 30m)
            mtf_short = all(price > lvl for lvl in fib50.values())
            mtf_long  = all(price < lvl for lvl in fib50.values())
            print(f"[DEBUG] Multi-TF bias: {'LONG' if mtf_long else 'SHORT' if mtf_short else 'NONE'}")

            # Pullback‐against‐trend bias 4h
            fib50_4h = last4h['fibo_0.5']
            if last4h.structure_direction ==  1:
                # uptrend → long when price < 50% fibo
                pb_long = last4h.close < fib50_4h
                pb_short  = False
            elif last4h.structure_direction == -1:
                # downtrend → short when price > 50% fibo
                pb_long  = False
                pb_short = last4h.close > fib50_4h
            else:
                bias_long = bias_short = False

            print(
                f"[DEBUG][4h pullback] dir={last4h.structure_direction:+.0f} "
                f"fibo50={fib50_4h:.2f} "
                f"pullback={'LONG' if pb_long else 'SHORT' if pb_short else 'NONE'}"
            )

            # 3) final bias = multi-TF AND pullback
            bias_long  = mtf_long  and pb_long
            bias_short = mtf_short and pb_short
            print(f"[DEBUG] Final bias: {'LONG' if bias_long else 'SHORT' if bias_short else 'NONE'}")

            # now print 4h state
            print(
                f"[DEBUG][4h state] time={last4h.time.strftime('%Y-%m-%d %H:%M')}"
                f"[DEBUG][4h pullback] dir={last4h.structure_direction:+.0f} "
                f"struct_hi={last4h.structure_high:.2f} "
                f"struct_lo={last4h.structure_low:.2f} "
                f"fibo50={fib50_4h:.2f} "
                f"bias={'LONG' if bias_long else 'SHORT' if bias_short else 'NONE'}"
            )

            # proceed with con4h_ok tests
            # recompute bias after injecting live price into 4h
            # fib50 = last4h['fibo_0.5']
            # bias_long  = (last4h.structure_direction == 1 and last4h.close < fib50)
            # bias_short = (last4h.structure_direction == -1 and last4h.close > fib50)

            # ── 2) Nearest‐4h + 1h/30m FVG overlap ─────────────────────────────────
            # pick our flag/cols based on bias
            flag_col = 'fvg_bull' if bias_short else 'fvg_bear'
            low_col  = 'fvg_bull_lower' if bias_short else 'fvg_bear_lower'
            high_col = 'fvg_bull_upper' if bias_short else 'fvg_bear_upper'

            # 2a) find the nearest open 4h gap (ifvg_flag = FALSE, fvg_not_valid = FALSE)
            mask4h = (conf4h[flag_col] & ~conf4h['ifvg_flag'] & ~conf4h['fvg_not_valid'])

            # if not mask4h.any():
            #     print(f"[DEBUG] No open 4h {'bull' if bias_long else 'bear'} FVG → skipping")
            #     await asyncio.sleep(pause_s)
            #     continue

            # # compute “distance” of each gap to current price
            # gaps4h = conf4h.loc[mask4h, [low_col, high_col]]
            # # midpoint of each gap
            # mid4h = (gaps4h[low_col] + gaps4h[high_col]) / 2
            # dists = (mid4h - price).abs()
            # nearest_idx = dists.idxmin()
            # range_lo, range_hi = gaps4h.loc[nearest_idx, [low_col, high_col]]
            # print(f"[DEBUG][4h nearest] range=[{range_lo:.2f},{range_hi:.2f}]")

            # # start our overlap list with that 4h band
            # ranges = [(range_lo, range_hi)]

            # ── 2b/2c) intersect the *same* FVG gap on 1h & 30m
            # fetch and confluence for 1h & 30m just once
            df1h, conf1h = SMCStructuresFVG(fetch_history(symbol, '1h', n_bars)).run()
            df30m, conf30m = SMCStructuresFVG(fetch_history(symbol, '30m', n_bars)).run()
            # --- after you do df1h, conf1h = SMCStructuresFVG(fetch_history(...)).run() ---
            # ── 1h SMC run ─────────────────────────────────────────────────────
            fibo4h_50    = last4h['fibo_0.5']
            dir4h        = int(last4h.structure_direction)

            # 1) filter CHoCH candidates (your existing code)…
            if dir4h == -1:
                series1 = df1h['choch_bull_price']
                series2 = df30m['choch_bull_price']
                valid1  = series1[series1 > fibo4h_50]
                valid2  = series2[series2 > fibo4h_50]
            else:
                series1 = df1h['choch_bear_price']
                series2 = df30m['choch_bear_price']
                valid1  = series1[series1 < fibo4h_50]
                valid2  = series2[series2 < fibo4h_50]

            combined = pd.concat([
                valid1.rename('price').to_frame().assign(tf='1h'),
                valid2.rename('price').to_frame().assign(tf='30m'),
            ])

            if combined.empty:
                await asyncio.sleep(pause_s)
                continue

            
            # take the last row
            last_row     = combined.iloc[-1]
            choch_idx    = combined.index[-1]     # the original bar index
            tf_ch        = last_row['tf']
            choch_price  = last_row['price']

            print(f"[DEBUG] Selected {tf_ch} CHoCH at idx {choch_idx}, price={choch_price:.2f}")


            # 3) build/reset the zone from choch_price ↔ 4h structure high/low
            struct_high = last4h['structure_high']
            struct_low  = last4h['structure_low']

            # Reset the sweep zone whenever we see a new CHoCH origin
            if zone_origin_idx != (tf_ch, choch_idx):
                # 1) store the new origin
                zone_origin_idx = (tf_ch, choch_idx)
                last_zone_alert = None

                # 2) grab your 4h structure high/low
                struct_high = last4h['structure_high']
                struct_low  = last4h['structure_low']

                # 3) build the zone from CHoCH to 4h structure
                if dir4h == -1:
                    # 4h downtrend → SHORT zone runs from CHoCH price up to the 4h high
                    zone_low, zone_high = choch_price, struct_high
                else:
                    # 4h uptrend → LONG zone runs from the 4h low up to CHoCH price
                    zone_low, zone_high = struct_low, choch_price

                print(f"[DEBUG] New sweep-zone ({tf_ch}@{choch_idx}): "
                    f"{zone_low:.2f} ↔ {zone_high:.2f}")



            # masks for “still‐open & valid” gaps
            m1 = conf1h[flag_col] & ~conf1h['ifvg_flag'] & ~conf1h['fvg_not_valid']
            m3 = conf30m[flag_col] & ~conf30m['ifvg_flag'] & ~conf30m['fvg_not_valid']

            if not (m1.any() and m3.any()):
                print(f"[DEBUG] No open 1h or 30m gaps → skipping")
                await asyncio.sleep(pause_s)
                continue

            # grab the latest 1h & 30m bounds
            lo1, hi1 = conf1h.loc[m1,  [low_col, high_col]].iloc[-1]
            lo3, hi3 = conf30m.loc[m3, [low_col, high_col]].iloc[-1]
            print(f"[DEBUG][1h] open range=[{lo1:.2f},{hi1:.2f}]")
            print(f"[DEBUG][30m] open range=[{lo3:.2f},{hi3:.2f}]")

            # # intersect with your 4h band:
            # overlap_lo = max(range_lo, lo1, lo3)
            # overlap_hi = min(range_hi, hi1, hi3)
            # if overlap_lo < overlap_hi:
            #     print(f"[DEBUG] 4h/1h/30m overlap = [{overlap_lo:.2f},{overlap_hi:.2f}]")
            #     overlap_zone = (overlap_lo, overlap_hi)
            # else:
            #     print("[DEBUG] No common overlap on 4h+1h+30m")
            #     await asyncio.sleep(pause_s)
            #     continue

            # --- 4) 1m & real-time price for micro entry ---
            print(f"[DEBUG] Fetching 1m & tick @ {datetime.now():%H:%M:%S}")
            df1m = fetch_history(symbol, '1m', n_bars)
            # ALSO fetch 5m & 30m
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                # fair live price as midpoint between bid & ask
                rt_price = (tick.bid + tick.ask) / 2
                df1m.at[df1m.index[-1], 'close'] = rt_price
                df1m.at[df1m.index[-1], 'high'] = max(df1m.at[df1m.index[-1], 'high'], rt_price)
                df1m.at[df1m.index[-1], 'low'] = min(df1m.at[df1m.index[-1], 'low'], rt_price)
            else:
                rt_price = df1m.iloc[-1].close

            smc1m, conf1m = SMCStructuresFVG(df1m).run()
            
            print(f"[DEBUG] Fetching 5m,30m df @ {datetime.now():%H:%M:%S}")
            df5m, _  = SMCStructuresFVG(fetch_history(symbol, '5m', n_bars)).run()
            df30m, _ = SMCStructuresFVG(fetch_history(symbol, '30m', n_bars)).run()

            # grab the latest 5m Fib50 and 30m structure SL/SH
            fibo5m    = df5m.iloc[-1]['fibo_0.5']
            # sl_long   = df30m.iloc[-1]['structure_low']
            # sl_short  = df30m.iloc[-1]['structure_high']


            # # pick the correct BOS direction
            # if bias_long:
            #     mask = smc1m['bos_int'] == -1   # looking for bear-BOS on 1m
            # elif bias_short:
            #     mask = smc1m['bos_int'] == 1    # looking for bull-BOS on 1m
            # else:
            #     mask = pd.Series(False, index=smc1m.index)
            
            entry_long = entry_short = False

            # — LONG bias logic —
            if bias_long and rt_price < fibo5m:
                # 1) Find the newest bear-BOS on 1m
                bear_idxs = smc1m.index[smc1m['bos_bear_price'].notnull()]
                if bear_idxs.any():
                    j = bear_idxs[-1]
                    bos_price = smc1m.at[j, 'bos_bear_price']
                    print(f"[DEBUG][LONG] newest BOS_bear idx={j}, price={bos_price:.2f}")
                    
                    # 2) Check that BOS price is below fibo5m
                    if bos_price < fibo5m:
                        print(f"[DEBUG][LONG] BOS price {bos_price:.2f} < fibo5m {fibo5m:.2f} → pass")
                        
                        # 3) Look for any bullish FVG (ifvg_flag==True) AFTER that BOS
                        after = conf1m.loc[j+1:]
                        valid_fvgs = after[
                            (after['fvg_bull']) &                    # bull‐type gap
                            (after['ifvg_flag']) &                   # it filled
                            (after['fvg_bull_upper'] < fibo5m) &     # gap entirely below fibo5m
                            (after['fvg_bull_lower'] < fibo5m)
                        ]
                        print(f"[DEBUG][LONG] valid FVG origins after idx {j}: {valid_fvgs.index.tolist()}")
                        
                        if not valid_fvgs.empty:
                            orig = valid_fvgs.index[0]
                            # print(f"[DEBUG][LONG] ▶️ ENTRYLONG at origin idx={orig}")
                            # <<< DEBOUNCE BLOCK STARTS HERE >>>
                            this_long_orig = orig
                            if this_long_orig != last_long_orig:
                                # first time seeing this exact origin
                                entry_long = True
                                last_long_orig = this_long_orig
                            # <<< DEBOUNCE BLOCK ENDS HERE >>>
                        
            # — SHORT bias logic —
            if bias_short and rt_price > fibo5m:
                bull_idxs = smc1m.index[smc1m['bos_bull_price'].notnull()]
                if bull_idxs.any():
                    j = bull_idxs[-1]
                    bos_price = smc1m.at[j, 'bos_bull_price']
                    print(f"[DEBUG][SHORT] newest BOS_bull idx={j}, price={bos_price:.2f}")
                    
                    if bos_price > fibo5m:
                        print(f"[DEBUG][SHORT] BOS price {bos_price:.2f} > fibo5m {fibo5m:.2f} → pass")
                        
                        after = conf1m.loc[j+1:]
                        valid_fvgs = after[
                            (after['fvg_bear']) &                    # bear‐type gap
                            (after['ifvg_flag']) &                   # it filled
                            (after['fvg_bear_upper'] > fibo5m) &     # gap entirely above fibo5m
                            (after['fvg_bear_lower'] > fibo5m)
                        ]
                        print(f"[DEBUG][SHORT] valid FVG origins after idx {j}: {valid_fvgs.index.tolist()}")
                        
                        if not valid_fvgs.empty:
                            orig = valid_fvgs.index[0]
                            # print(f"[DEBUG][SHORT] ▶️ ENTRYSHORT at origin idx={orig}")
                            # <<< DEBOUNCE BLOCK STARTS HERE >>>
                            this_short_orig = orig
                            if this_short_orig != last_short_orig:
                                # first time seeing this exact origin
                                entry_short = True
                                last_short_orig = this_short_orig
                            # <<< DEBOUNCE BLOCK ENDS HERE >>>
                            #             
            # # finally, the TP and flag summary
            # if entry_long or entry_short:
            #     tp = overlap_zone[1] if entry_long else overlap_zone[0]
            #     await reporter.send_message(f"[{symbol}][TP] ▶️ Target @ {tp:.2f}")

            # ── RISK/REWARD FILTER & 1h STOP-LOSS OVERRIDE ─────────────────────────
            # fetch 1h structure one last time (or reuse df1h if you already have it)
            df1h, _     = SMCStructuresFVG(fetch_history(symbol, '1h', n_bars)).run()
            sl1h_long   = df1h.iloc[-1]['structure_low']
            sl1h_short  = df1h.iloc[-1]['structure_high']
            # fetch 30 m structure for TP
            tp_long  = df30m.iloc[-1]['structure_high']
            tp_short = df30m.iloc[-1]['structure_low']

            # after entry_long/entry_short and tp_long/tp_short, sl1h_long/sl1h_short, rt_price…

            TARGET_RR = 1.5
            MAX_RR    = 3.0

            if entry_long or entry_short:
                now = datetime.now()
                if zone_origin_idx is not None and zone_low is not None \
                and zone_low <= rt_price <= zone_high \
                and (last_zone_alert is None or (now - last_zone_alert).total_seconds() >= 300):
                # 1) compute reward, risk, rr
                    if entry_long:
                        reward   = tp_long - rt_price
                        raw_risk = rt_price - sl1h_long
                    else:  # short
                        reward   = rt_price - tp_short
                        raw_risk = sl1h_short - rt_price

                    rr = reward / raw_risk if raw_risk > 0 else 0

                    # 2) only accept if rr is between your TARGET and MAX
                    if rr < TARGET_RR or rr > MAX_RR:
                        print(f"[DEBUG] RR {rr:.2f} not in [{TARGET_RR:.2f},{MAX_RR:.2f}] → skipping signal")
                    else:
                        # 3) widen SL if rr > target
                        if entry_long:
                            if rr > TARGET_RR:
                                new_risk   = reward / TARGET_RR
                                sl_to_use  = rt_price - new_risk
                            else:
                                sl_to_use  = sl1h_long
                            tp_to_use = tp_long
                            side       = "LONG"
                        else:  # SHORT
                            if rr > TARGET_RR:
                                new_risk   = reward / TARGET_RR
                                sl_to_use  = rt_price + new_risk
                            else:
                                sl_to_use  = sl1h_short
                            tp_to_use = tp_short
                            side       = "SHORT"

                        # 4) Discord message with adjusted SL and RR
                        msg = (
                            f"📊 **SMC Entry Signal**\n"
                            f"▶️ Side: **{side}**\n"
                            f"💰 Entry Price: {rt_price:.2f}\n"
                            f"🎯 Take Profit (30m): {tp_to_use:.2f}\n"
                            f"🛑 Stop Loss (1h): {sl_to_use:.2f}\n"
                            f"📈 Reward/Risk: {rr:.2f} → {TARGET_RR:.2f}\n"
                            f"⏱ Time: {datetime.now():%Y-%m-%d %H:%M:%S}"
                        )
                        print(f"[DEBUG] Sending message:\n{msg}")
                        await reporter.send_message(msg)

                        # 5) Place market order with the same SL/TP
                        bot._send_market_order(
                            volume=0.01,
                            direction= 1 if entry_long else -1,
                            area='1m',
                            comment_suffix=f"ENTRY{side}",
                            sl=sl_to_use,
                            tp=tp_to_use
                        )
                    # mark that we’ve alerted this zone
                    last_zone_alert = now


            await asyncio.sleep(2)

            flags = []
            if bias_long: flags.append("BIAS=LONG")
            if bias_short: flags.append("BIAS=SHORT")
            if entry_long: flags.append("ENTRY⇧")
            if entry_short: flags.append("ENTRY⇩")
            print(f"[{datetime.now():%H:%M:%S}][1m] {' · '.join(flags)} price={rt_price:.2f}")
 

    except Exception as e:
        await reporter.send_message(f"❗️ Bot error: {e}")
        raise
    finally:
        mt5.shutdown()

async def main():
    mon = asyncio.create_task(monitor_loop())
    await mon

if __name__ == '__main__':
    asyncio.run(main())
