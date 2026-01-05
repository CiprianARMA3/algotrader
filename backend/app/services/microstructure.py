import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class MicrostructureAnalyzer:
    def __init__(self):
        pass

    def calculate_vpin(self, df: pd.DataFrame, n_buckets: int = 50) -> float:
        """
        Volume-Synchronized Probability of Informed Trading.
        Detects 'Toxic Flow' by looking at volume imbalance in equal-sized buckets.
        """
        if df.empty or 'Volume' not in df.columns:
            return 0.0
            
        total_vol = df['Volume'].sum()
        if total_vol == 0: return 0.0
        bucket_size = total_vol / n_buckets
        
        # Approximate buy/sell volume using tick rule
        df = df.copy()
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
        Market Impact / Inverse Depth (Kyle's Lambda).
        """
        if len(returns) < 10:
            return 0.0
        impact = np.abs(returns) / (volume + 1e-9)
        return float(impact.mean())

    def calculate_amihud_illiquidity(self, returns: pd.Series, dollar_volume: pd.Series) -> float:
        """
        Amihud Ratio: average(abs(R) / Dollar_Volume).
        """
        illiquidity = np.abs(returns) / (dollar_volume + 1e-9)
        return float(illiquidity.mean())

    def calculate_ofi_proxy(self, df: pd.DataFrame) -> float:
        """
        Order Flow Imbalance (OFI) proxy for OHLCV data.
        Institutional finding: OFI strongly correlates with short-term price impact.
        """
        if len(df) < 2: return 0.0
        
        # OFI logic for OHLC:
        # If Price increases, buying pressure = Volume
        # If Price decreases, selling pressure = Volume
        price_change = df['Close'].diff()
        ofi = (np.sign(price_change) * df['Volume']).fillna(0)
        return float(ofi.sum() / df['Volume'].sum())

    def estimate_pin_mle(self, df: pd.DataFrame) -> float:
        """
        Structural Estimation of Probability of Informed Trading (PIN).
        Uses MLE on buy/sell arrival rates.
        """
        if len(df) < 50: return 0.0
        
        df = df.copy()
        df['Side'] = np.sign(df['Close'].diff()).replace(0, method='ffill')
        buys = df[df['Side'] > 0]['Volume'].values
        sells = df[df['Side'] < 0]['Volume'].values
        
        # Simplified Likelihood function for arrival rates
        # alpha: prob of info event, delta: prob of bad news, mu: informed rate, eb/es: noise rates
        def log_likelihood(params):
            alpha, mu, eb, es = params
            if not (0 <= alpha <= 1 and mu > 0 and eb > 0 and es > 0):
                return 1e10
            
            # Simplified PIN log-likelihood over days
            # This is a proxy for the structural Easley-O'Hara model
            l_buy = np.sum(-eb + buys * np.log(eb)) # noise only
            l_sell = np.sum(-es + sells * np.log(es))
            
            return -(l_buy + l_sell + np.log(alpha * mu + 1e-9))

        try:
            initial_guess = [0.5, np.mean(buys), np.mean(buys)/2, np.mean(sells)/2]
            res = minimize(log_likelihood, initial_guess, method='Nelder-Mead')
            alpha, mu, eb, es = res.x
            pin = (alpha * mu) / (alpha * mu + eb + es)
            return float(pin)
        except:
            return 0.0