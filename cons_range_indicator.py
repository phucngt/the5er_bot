import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class ConsolidationRangeDetector:
    def __init__(self, length=10, atr_length=100, mult=1.0, slope_threshold_ratio=0.0005):
        self.length = length
        self.atr_length = atr_length
        self.mult = mult
        self.slope_threshold_ratio = slope_threshold_ratio

    def calculate_atr(self, df, period):
        high = df['high']
        low = df['low']
        close = df['close']
        tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift()), np.abs(low - close.shift())))
        atr = pd.Series(tr).rolling(window=period, min_periods=1).mean()
        return atr

    def detect_ranges(self, df):
        df = df.copy()
        df['sma'] = df['close'].rolling(window=self.length, min_periods=self.length).mean()
        df['atr'] = self.calculate_atr(df, self.atr_length) * self.mult

        cons_in_range = []
        upper_cons_price = []
        lower_cons_price = []
        cons_type_list = []

        max_price = None
        min_price = None
        in_range = False
        range_start_idx = None
        current_cons_type = None
        range_data = []

        idx = 0
        while idx < len(df):
            if idx < self.length:
                cons_in_range.append(False)
                upper_cons_price.append(np.nan)
                lower_cons_price.append(np.nan)
                cons_type_list.append(np.nan)
                idx += 1
                continue

            if not in_range:
                window = df.iloc[idx - self.length + 1:idx + 1]
                sma = window['sma'].iloc[-1]
                atr = window['atr'].iloc[-1]
                if np.all(np.abs(window['close'] - sma) <= atr):
                    in_range = True
                    range_start_idx = idx - self.length + 1
                    max_price = sma + atr
                    min_price = sma - atr
                    cons_in_range.append(True)
                    upper_cons_price.append(max_price)
                    lower_cons_price.append(min_price)
                    cons_type_list.append(np.nan)

                    idx += 1
                else:
                    cons_in_range.append(False)
                    upper_cons_price.append(np.nan)
                    lower_cons_price.append(np.nan)
                    cons_type_list.append(np.nan)
                    idx += 1
            else:
                current_close = df.loc[idx, 'close']
                sma = df.loc[idx, 'sma']
                atr = df.loc[idx, 'atr']

                if min_price <= current_close <= max_price:
                    max_price = max(max_price, sma + atr)
                    min_price = min(min_price, sma - atr)
                    cons_in_range.append(True)
                    upper_cons_price.append(max_price)
                    lower_cons_price.append(min_price)
                    cons_type_list.append(np.nan)
                    idx += 1
                else:
                    # Finalize and classify the consolidation range
                    range_df = df.iloc[range_start_idx:idx].copy()
                    range_df['upper'] = max_price
                    range_df['lower'] = min_price
                    range_df['midline'] = (range_df['upper'] + range_df['lower']) / 2
                    range_df['midline_smooth'] = range_df['midline'].rolling(window=3).mean()

                    x = np.arange(len(range_df.dropna())).reshape(-1, 1)
                    y = range_df['midline_smooth'].dropna().values.reshape(-1, 1)
                    if len(x) > 1:
                        model = LinearRegression().fit(x, y)
                        slope = model.coef_[0][0]
                        avg_price = range_df['midline'].mean()
                        threshold = self.slope_threshold_ratio * avg_price
                        if abs(slope) < threshold:
                            current_cons_type = 0
                        elif slope > threshold:
                            current_cons_type = 1
                        else:
                            current_cons_type = -1
                    else:
                        current_cons_type = 0

                    range_df['cons_type'] = current_cons_type
                    range_data.append(range_df[['time', 'cons_type', 'upper', 'lower']])
                    for fill_idx in range(range_start_idx, idx):
                        cons_type_list[fill_idx] = current_cons_type

                    # Reset for new range starting at this candle
                    in_range = False
                    max_price = None
                    min_price = None
                    # DO NOT increment idx — this same candle is now the start for next detection
        # Handle if still in a range at the end
        if in_range:
            range_df = df.iloc[range_start_idx:].copy()
            range_df['upper'] = max_price
            range_df['lower'] = min_price
            range_df['midline'] = (range_df['upper'] + range_df['lower']) / 2
            range_df['midline_smooth'] = range_df['midline'].rolling(window=3).mean()

            x = np.arange(len(range_df.dropna())).reshape(-1, 1)
            y = range_df['midline_smooth'].dropna().values.reshape(-1, 1)
            if len(x) > 1:
                model = LinearRegression().fit(x, y)
                slope = model.coef_[0][0]
                avg_price = range_df['midline'].mean()
                threshold = self.slope_threshold_ratio * avg_price
                if abs(slope) < threshold:
                    current_cons_type = 0
                elif slope > threshold:
                    current_cons_type = 1
                else:
                    current_cons_type = -1
            else:
                current_cons_type = 0

            range_df['cons_type'] = current_cons_type
            range_data.append(range_df[['time', 'cons_type', 'upper', 'lower']])
            for fill_idx in range(range_start_idx, len(df)):
                cons_type_list[fill_idx] = current_cons_type

        df['cons_in_range'] = cons_in_range
        df['upper_cons_price'] = upper_cons_price
        df['lower_cons_price'] = lower_cons_price
        df['cons_type'] = cons_type_list

        range_audit_df = pd.concat(range_data) if range_data else pd.DataFrame()
        return df[['time', 'cons_in_range', 'upper_cons_price', 'lower_cons_price', 'cons_type']], range_audit_df

