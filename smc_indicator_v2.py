import pandas as pd
import numpy as np

class SmartMoneyConcepts:
    """
    Implements Smart Money Concepts (SMC) for trading, including premium, discount, and equilibrium zones as ranges,
    mimicking the zone calculation logic from LuxAlgo's Smart Money Concepts Pine Script.
    """
    def __init__(self, df):
        """
        Initialize with OHLCV DataFrame.
        """
        self.df = df.copy().reset_index(drop=True)
        self.atr_len = 50
        self.run()

    def calculate_zones(self):
        """
        Calculate premium, discount, and equilibrium zones as price ranges, matching Pine Script logic.
        Uses cumulative extremes (not a rolling window) to mirror Pine's updateTrailingExtremes behavior.
        """
        # 1) cumulative trailing extremes
        self.df['trailing_high'] = self.df['high'].cummax()
        self.df['trailing_low']  = self.df['low'].cummin()
        rng = self.df['trailing_high'] - self.df['trailing_low']

        # 2) premium zone (top 5% of range)
        self.df['premium_high'] = self.df['trailing_high']
        self.df['premium_low']  = self.df['trailing_high'] - 0.05 * rng

        # 3) discount zone (bottom 5% of range)
        self.df['discount_low']  = self.df['trailing_low']
        self.df['discount_high'] = self.df['trailing_low'] + 0.05 * rng

        # 4) equilibrium zone (middle 5% of range)
        mid = (self.df['trailing_high'] + self.df['trailing_low']) / 2
        self.df['equilibrium_low']  = mid - 0.025 * rng
        self.df['equilibrium_high'] = mid + 0.025 * rng

        # 5) handle zero or negative range per row
        mask = rng <= 0
        for col in ['premium_high', 'premium_low', 'discount_low', 'discount_high', 'equilibrium_low', 'equilibrium_high']:
            self.df.loc[mask, col] = self.df.loc[mask, 'close']

        # 6) single-point compatibility columns
        self.df['premium_zone']  = self.df['premium_high']
        self.df['discount_zone'] = self.df['discount_low']

    def calculate_atr(self):
        high_low   = self.df['high'] - self.df['low']
        high_close = abs(self.df['high'] - self.df['close'].shift())
        low_close  = abs(self.df['low']  - self.df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['atr'] = tr.rolling(self.atr_len, min_periods=1).mean()

    def swings(self):
        self.df['swing_high'] = (self.df['high'] > self.df['high'].shift(1)) & (self.df['high'] > self.df['high'].shift(-1))
        self.df['swing_low']  = (self.df['low']  < self.df['low'].shift(1)) & (self.df['low']  < self.df['low'].shift(-1))
        self.df['leg_internal'] = np.where(self.df['swing_high'], 1, np.where(self.df['swing_low'], -1, 0))
        # fill forward on NaNs, then cast back to integer explicitly
        s = (
            self.df['leg_internal']
            .replace(0, np.nan)    # use np.nan so dtype stays float
            .ffill()
            .fillna(0)
        )
        self.df['leg_swing'] = s.astype(int)

    def break_of_structure(self):
        self.df['bos_int']  = 0
        self.df['choch_int']= 0
        self.df['bos_swg']  = False
        self.df['choch_swg']= False
        for i in range(2, len(self.df)):
            if self.df['leg_swing'].iat[i-1] == 1 and self.df['low'].iat[i] < self.df['low'].iat[i-2]:
                self.df.iat[i, self.df.columns.get_loc('bos_int')]   = -1
                self.df.iat[i, self.df.columns.get_loc('bos_swg')]   = True
            elif self.df['leg_swing'].iat[i-1] == -1 and self.df['high'].iat[i] > self.df['high'].iat[i-2]:
                self.df.iat[i, self.df.columns.get_loc('bos_int')]   = 1
                self.df.iat[i, self.df.columns.get_loc('bos_swg')]   = True

    def order_block(self):
        self.df['ob_int']   = np.where(self.df['bos_int'] != 0, self.df['bos_int'], 0)
        self.df['ob_swing'] = self.df['close']

    def fair_value_gap(self):
        self.df['fvg_bull'] = (self.df['low'] > self.df['high'].shift(2)) & (self.df['close'] > self.df['open'])
        self.df['fvg_bear'] = (self.df['high'] < self.df['low'].shift(2)) & (self.df['close'] < self.df['open'])

    def multi_timeframe_levels(self, df_higher, label='H1'):
        """
        Add higher timeframe levels for debugging; creates two new columns:
        level_{label}_high and level_{label}_low based on the last row of df_higher.
        """
        if 'high' in df_higher and 'low' in df_higher:
            self.df[f'level_{label}_high'] = df_higher['high'].iloc[-1]
            self.df[f'level_{label}_low']  = df_higher['low'].iloc[-1]

    def run(self):
        """
        Run all SMC calculations and return the processed DataFrame.
        """
        self.calculate_atr()
        self.swings()
        self.break_of_structure()
        self.order_block()
        self.fair_value_gap()
        self.calculate_zones()
        return self.df
