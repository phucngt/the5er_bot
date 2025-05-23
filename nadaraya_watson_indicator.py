import numpy as np
import pandas as pd


def compute_nadaraya_watson_envelope(
    df: pd.DataFrame,
    bandwidth: float = 6.0,
    mult: float = 3.0,
    window: int = 500
):
    """
    Compute the Nadaraya-Watson envelope (upper, lower) for the last `window` rows of df.

    Args:
        df: DataFrame with columns ['open','high','low','close','volume'] and datetime index or integer index.
        bandwidth: Gaussian kernel bandwidth (h in Pine).
        mult: Multiplier for the mean absolute error envelope.
        window: Number of bars to use (e.g. 500 for 5m timeframe).

    Returns:
        upper: latest envelope upper band.
        lower: latest envelope lower band.
    """
    # take last `window` closes
    closes = df['close'].iloc[-window:].values
    n = len(closes)
    if n == 0:
        raise ValueError("Not enough data to compute NWE")

    # gaussian weights for end-point method
    # weights w[i] = exp(-i^2/(2*h^2)), i = 0..n-1
    x = np.arange(n)
    w = np.exp(-(x**2) / (2 * bandwidth * bandwidth))

    # compute Nadaraya-Watson (endpoint) -> only latest y_n
    # y_n = sum_{j=0..n-1} close_{n-1-j} * w[j] / sum(w[j])
    # reverse closes to align j=0 with most recent
    rev = closes[::-1]
    num = (rev * w).sum()
    den = w.sum()
    out = num / den

    # mean absolute error of last-window errors
    # mae = sma(abs(close - y_series), window) * mult
    # approximate y_series by out for simplicity (represents smoothed)
    errors = np.abs(closes - out)
    mae = errors.mean() * mult

    upper = out + mae
    lower = out - mae
    return upper, lower
