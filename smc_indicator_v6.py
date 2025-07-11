import pandas as pd
import numpy as np

class SMCStructuresFVG:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().reset_index(drop=True)
        self.params = {'struct_lookback': 10, 'fibo_levels': [0.75, 0.5, 0.25]}
        self.structureHigh = self.df.at[0, 'high']
        self.structureLow = self.df.at[0, 'low']
        self.structureHighStart = self.structureLowStart = 0
        self.structureDirection = 0
        self.pending_fvgs = []
        
        # track last break index to extend lookback window
        self.last_break_index = 0
        self.init_columns()

    def init_columns(self):
        # price swings
        self.df['swing_high'] = (
            (self.df['high'] > self.df['high'].shift(1)) &
            (self.df['high'] > self.df['high'].shift(-1))
        )
        self.df['swing_low'] = (
            (self.df['low'] < self.df['low'].shift(1)) &
            (self.df['low'] < self.df['low'].shift(-1))
        )
        self.df['leg_internal'] = np.where(
            self.df['swing_high'], 1,
            np.where(self.df['swing_low'], -1, 0)
        )
        self.df['leg_swing'] = (
            self.df['leg_internal']
               .replace(0, np.nan)
               .ffill()
               .fillna(0)
               .astype(int)
        )
        # structure flags
        self.df['bos_int'] = 0
        self.df['choch_int'] = 0
        self.df['bos_bull_price'] = np.nan
        self.df['bos_bear_price'] = np.nan
        self.df['choch_bull_price'] = np.nan
        self.df['choch_bear_price'] = np.nan
        # FVG flags
        self.df['fvg_bull'] = False
        self.df['fvg_bear'] = False
        self.df['fvg_bull_lower'] = np.nan
        self.df['fvg_bull_upper'] = np.nan
        self.df['fvg_bear_lower'] = np.nan
        self.df['fvg_bear_upper'] = np.nan
        self.df['fvg_bull_index'] = np.nan
        self.df['fvg_bear_index'] = np.nan
        self.df['is_mitigated_bull'] = False
        self.df['is_mitigated_bear'] = False
        self.df['mitigating_candle'] = False
        self.df['ifvg_flag'] = False
        self.df['fvg_not_valid'] = False
        self.df['ifvg_mitigating_candle'] = False

        # Fibonacci levels
        for lvl in self.params['fibo_levels']:
            self.df[f'fibo_{lvl}'] = np.nan
        # record structure per bar
        self.df['structure_high'] = np.nan
        self.df['structure_low'] = np.nan
        self.df['structure_direction'] = np.nan

    def detect_structure_breaks(self):
        lookback = self.params['struct_lookback']
        for i in range(1, len(self.df)):
            prevH, prevL = self.structureHigh, self.structureLow
            #start = max(0, i - lookback + 1)
            # extend window to last break if set, otherwise fallback to lookback
            if self.last_break_index > 0:
                start = self.last_break_index
            else:
                start = max(0, i - lookback + 1)

            highs = self.df['high'].iloc[start:i+1]
            lows = self.df['low'].iloc[start:i+1]
            idxH, idxL = highs.idxmax(), lows.idxmin()
            candH, candL = highs.max(), lows.min()
            c = self.df.at[i, 'close']
            # reset flags
            self.df.loc[i, ['bos_int', 'choch_int']] = [0, 0]
            self.df.loc[i, ['bos_bull_price', 'bos_bear_price', 'choch_bull_price', 'choch_bear_price']] = [np.nan]*4
            fired = False
            if c < prevL and i > self.structureLowStart:
                self.structureLow, self.structureLowStart, self.structureDirection = candL, idxL, -1
                self.df.at[i, 'bos_int'] = -1
                self.df.at[i, 'bos_bear_price'] = candL
                fired = True
            elif c > prevH and i > self.structureHighStart:
                self.structureHigh, self.structureHighStart, self.structureDirection = candH, idxH, 1
                self.df.at[i, 'bos_int'] = 1
                self.df.at[i, 'bos_bull_price'] = candH
                fired = True
            prev_dir = self.df.at[i-1, 'structure_direction'] if i>0 else 0
            if prev_dir and self.structureDirection != prev_dir:
                if self.structureDirection == 1:
                    self.df.at[i, 'choch_int'] = 1
                    self.df.at[i, 'choch_bull_price'] = prevH
                else:
                    self.df.at[i, 'choch_int'] = -1
                    self.df.at[i, 'choch_bear_price'] = prevL
                self.df.loc[i, ['bos_int', 'bos_bull_price', 'bos_bear_price']] = [0, np.nan, np.nan]
                fired = True
            if fired:
                # whenever we break, update last_break_index and reset high/low window
                self.last_break_index = i
                win = self.df.iloc[start:i+1]
                self.structureHigh = win['high'].max()
                self.structureLow = win['low'].min()
                self.structureHighStart = win['high'].idxmax()
                self.structureLowStart = win['low'].idxmin()
            self.df.at[i, 'structure_high'] = self.structureHigh
            self.df.at[i, 'structure_low'] = self.structureLow
            self.df.at[i, 'structure_direction'] = self.structureDirection

    def detect_fvg(self):
        # Step 1: Gather and mark all FVG origins
        origins = []
        bull_idx = bear_idx = 0
        # reset previous FVG markers
        self.df[['fvg_bull','fvg_bear']] = False, False
        self.df[['fvg_bull_lower','fvg_bull_upper','fvg_bear_lower','fvg_bear_upper']] = [np.nan]*4
        self.df[['fvg_bull_index','fvg_bear_index']] = np.nan, np.nan
        # ensure flags exist
        if 'ifvg_mitigating_candle' not in self.df.columns:
            self.df['ifvg_mitigating_candle'] = False

        for i in range(3, len(self.df)):
            # bullish gap-up
            if self.df.at[i-3, 'high'] < self.df.at[i-1, 'low']:
                o = i-2; lowv = self.df.at[i-3, 'high']; upv = self.df.at[i-1, 'low']
                origins.append({'origin_idx': o, 'lower': lowv, 'upper': upv, 'type': 'bull'})
                # mark origin
                self.df.at[o, 'fvg_bull'] = True
                self.df.at[o, 'fvg_bull_lower'] = lowv
                self.df.at[o, 'fvg_bull_upper'] = upv
                self.df.at[o, 'fvg_bull_index'] = bull_idx
                bull_idx += 1
            # bearish gap-down
            elif self.df.at[i-3, 'low'] > self.df.at[i-1, 'high']:
                o = i-2; lowv = self.df.at[i-1, 'high']; upv = self.df.at[i-3, 'low']
                origins.append({'origin_idx': o, 'lower': lowv, 'upper': upv, 'type': 'bear'})
                self.df.at[o, 'fvg_bear'] = True
                self.df.at[o, 'fvg_bear_lower'] = lowv
                self.df.at[o, 'fvg_bear_upper'] = upv
                self.df.at[o, 'fvg_bear_index'] = bear_idx
                bear_idx += 1

        # Step 2: Resolve each origin for wick and close intrusion
        for orig in origins:
            o, lowv, upv, t = orig['origin_idx'], orig['lower'], orig['upper'], orig['type']
            for j in range(o+3, len(self.df)):
                h = self.df.at[j, 'high']; l = self.df.at[j, 'low']; c = self.df.at[j, 'close']
                # wick-based mitigation
                if t == 'bull' and l < upv:
                    self.df.at[j, 'mitigating_candle'] = True
                    self.df.at[o, 'fvg_not_valid'] = True
                elif t == 'bear' and h > lowv:
                    self.df.at[j, 'mitigating_candle'] = True
                    self.df.at[o, 'fvg_not_valid'] = True
                # close-based fill/mitigation
                if t == 'bull' and c < lowv:
                    # full fill
                    self.df.at[o, 'ifvg_flag'] = True
                    self.df.at[o, 'fvg_not_valid'] = True
                    self.df.at[j, 'ifvg_mitigating_candle'] = True
                    break
                if t == 'bear' and c > upv:
                    self.df.at[o, 'ifvg_flag'] = True
                    self.df.at[o, 'fvg_not_valid'] = True
                    self.df.at[j, 'ifvg_mitigating_candle'] = True
                    break

    def compute_fibonacci_levels(self):
        rng = self.df['structure_high'] - self.df['structure_low']
        for lvl in self.params['fibo_levels']:
            up = self.df['structure_low'] + rng*lvl
            dn = self.df['structure_high'] - rng*lvl
            self.df[f'fibo_{lvl}'] = np.where(
                self.df['structure_direction']==1, up,
                np.where(self.df['structure_direction']==-1, dn, np.nan)
            )

    def build_confluence_df(self):
        cols = ['time','fvg_bull','fvg_bear','fvg_bull_lower','fvg_bull_upper',
                'fvg_bear_lower','fvg_bear_upper','is_mitigated_bull',
                'is_mitigated_bear','bos_int','choch_int','ifvg_flag','fvg_not_valid','ifvg_mitigating_candle']
        return self.df[cols].copy()

    def run(self):
        self.detect_structure_breaks()
        self.detect_fvg()
        self.compute_fibonacci_levels()
        i = self.df.index[-1]
        self.df.loc[i,['structure_high','structure_low','structure_direction']] = [self.structureHigh,self.structureLow,self.structureDirection]
        return self.df,self.build_confluence_df()
