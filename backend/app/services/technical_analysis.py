import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    def __init__(self):
        pass

    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates hundreds of technical indicators across all categories.
        """
        if df.empty:
            return {}

        # Ensure DataFrame has enough data for long-period indicators
        if len(df) < 50:
            return {"error": "Insufficient data for full technical suite"}

        try:
            # 1. Trend Indicators
            df.ta.sma(length=20, append=True)
            df.ta.ema(length=20, append=True)
            df.ta.wma(length=20, append=True)
            df.ta.hma(length=20, append=True)
            df.ta.dema(length=20, append=True)
            df.ta.tema(length=20, append=True)
            df.ta.kama(length=20, append=True)
            df.ta.vwma(length=20, append=True)
            df.ta.alma(length=20, append=True)
            df.ta.adx(append=True)
            df.ta.vortex(append=True)
            df.ta.aroon(append=True)
            df.ta.supertrend(append=True)
            df.ta.ichimoku(append=True)
            df.ta.linreg(append=True)

            # 2. Momentum Oscillators
            df.ta.rsi(length=14, append=True)
            df.ta.stoch(append=True)
            df.ta.macd(append=True)
            df.ta.cci(append=True)
            df.ta.willr(append=True)
            df.ta.mom(append=True)
            df.ta.roc(append=True)
            df.ta.uo(append=True)
            df.ta.mfi(append=True)
            df.ta.tsi(append=True)
            df.ta.ao(append=True)
            df.ta.fisher(append=True)
            df.ta.cmo(append=True)
            df.ta.ppo(append=True)
            df.ta.bop(append=True)
            df.ta.coppock(append=True)

            # 3. Volatility Indicators
            df.ta.bbands(append=True)
            df.ta.atr(append=True)
            df.ta.kc(append=True)
            df.ta.donchian(append=True)
            df.ta.massi(append=True)
            df.ta.accbands(append=True)
            df.ta.pdist(append=True)

            # 4. Volume Indicators
            df.ta.obv(append=True)
            df.ta.vwap(append=True)
            df.ta.ad(append=True)
            df.ta.cmf(append=True)
            df.ta.eom(append=True)
            df.ta.pvi(append=True)
            df.ta.nvi(append=True)
            df.ta.vpt(append=True)
            df.ta.kvo(append=True)

            # 5. Specialized & Bill Williams
            df.ta.alligator(append=True)
            df.ta.fractals(append=True)

            # Extract last values for the API response
            latest = df.tail(1).to_dict(orient='records')[0]
            
            # 6. Geometric / Fibonacci Logic (Mathematical Support/Resistance)
            latest['fibonacci_levels'] = self.calculate_fibonacci_levels(df)

            # Clean up keys (remove spaces, make serializable)
            clean_latest = {str(k).replace(' ', '_').lower(): v for k, v in latest.items()}
            
            return clean_results(clean_latest)

        except Exception as e:
            logger.error(f"TA calculation error: {str(e)}")
            return {"error": str(e)}

    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates core Fibonacci retracement and extension levels.
        """
        high = df['High'].max()
        low = df['Low'].min()
        diff = high - low
        
        return {
            "r_100": float(high),
            "r_786": float(high - 0.236 * diff),
            "r_618": float(high - 0.382 * diff),
            "r_500": float(high - 0.500 * diff),
            "r_382": float(high - 0.618 * diff),
            "r_236": float(high - 0.786 * diff),
            "r_000": float(low),
            "ext_1618": float(high + 0.618 * diff),
            "ext_2618": float(high + 1.618 * diff)
        }

def clean_results(obj):
    """Deep clean for JSON compliance (NaN -> None)"""
    if isinstance(obj, float):
        return None if np.isnan(obj) or np.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: clean_results(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_results(i) for i in obj]
    return obj
