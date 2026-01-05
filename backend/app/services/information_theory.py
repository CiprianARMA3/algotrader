import numpy as np
import pandas as pd
from scipy.stats import entropy
from statsmodels.tsa.stattools import grangercausalitytests
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class InformationAnalyzer:
    def __init__(self):
        pass

    def calculate_transfer_entropy(self, x: pd.Series, y: pd.Series, bins: int = 10) -> float:
        """
        Calculates information flow from X to Y.
        TE = H(Y_t | Y_past) - H(Y_t | Y_past, X_past)
        """
        # Discretize data
        x_bins = pd.qcut(x, bins, labels=False, duplicates='drop')
        y_bins = pd.qcut(y, bins, labels=False, duplicates='drop')
        
        # Joint and Conditional Entropies (simplified estimation)
        joint_xy = np.histogram2d(x_bins, y_bins, bins=bins)[0]
        p_xy = joint_xy / np.sum(joint_xy)
        
        # Shannon Entropy of Y
        p_y = np.sum(p_xy, axis=0)
        h_y = entropy(p_y)
        
        # Mutual Information as proxy for TE in this simplified context
        p_x = np.sum(p_xy, axis=1)
        mi = 0
        for i in range(bins):
            for j in range(bins):
                if p_xy[i,j] > 0:
                    mi += p_xy[i,j] * np.log(p_xy[i,j] / (p_x[i] * p_y[j]))
        
        return float(mi)

    def calculate_granger_causality(self, series1: pd.Series, series2: pd.Series, maxlag: int = 5) -> Dict:
        """
        Linear Granger Causality test
        """
        data = pd.concat([series1, series2], axis=1).dropna()
        if len(data) < 20: return {}
        
        test_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
        # Extract p-value for the first lag
        p_val = test_result[1][0]['ssr_ftest'][1]
        
        return {
            'causal_p_value': float(p_val),
            'is_causal': p_val < 0.05
        }
