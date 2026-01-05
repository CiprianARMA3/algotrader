import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class MicrostructureAnalyzer:
    def __init__(self):
        pass

    def calculate_vpin(self, df: pd.DataFrame, n_buckets: int = 50) -> float:
        """
        Volume-Synchronized Probability of Informed Trading
        Detects 'Toxic Flow' by looking at volume imbalance in equal-sized buckets.
        """
        if df.empty or 'Volume' not in df.columns:
            return 0.0
            
        total_vol = df['Volume'].sum()
        bucket_size = total_vol / n_buckets
        
        # Approximate buy/sell volume using tick rule (close vs previous close)
        df['Price_Diff'] = df['Close'].diff()
        df['Side'] = np.sign(df['Price_Diff']).replace(0, method='ffill')
        
        # Aggregate into volume buckets
        df['Cum_Vol'] = df['Volume'].cumsum()
        df['Bucket'] = (df['Cum_Vol'] / bucket_size).astype(int)
        
        buckets = df.groupby('Bucket').apply(
            lambda x: np.abs(
                x[x['Side'] > 0]['Volume'].sum() - x[x['Side'] < 0]['Volume'].sum()
            )
        )
        
        return float(buckets.mean() / bucket_size)

    def calculate_kyles_lambda(self, returns: pd.Series, volume: pd.Series) -> float:
        """
        Market Impact / Inverse Depth
        Higher lambda means the market is less liquid (small volume moves price more).
        """
        if len(returns) < 10:
            return 0.0
        # Regression: Returns = lambda * Signed_Volume
        # Simplified for daily data: abs(returns) / (volume * price)
        impact = np.abs(returns) / (volume + 1e-9)
        return float(impact.mean())

    def calculate_amihud_illiquidity(self, returns: pd.Series, dollar_volume: pd.Series) -> float:
        """
        Amihud Ratio: average(abs(R) / Dollar_Volume)
        """
        illiquidity = np.abs(returns) / (dollar_volume + 1e-9)
        return float(illiquidity.mean())
