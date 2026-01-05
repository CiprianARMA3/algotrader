import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from typing import List, Dict, Tuple, Optional
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EconometricAnalyzer:
    def __init__(self):
        self.pca_model = None
        
    def calculate_cointegration(
        self, 
        series1: pd.Series, 
        series2: pd.Series,
        significance_level: float = 0.05
    ) -> Dict:
        """
        Perform Engle-Granger cointegration test with dual stationarity verification (ADF + KPSS).
        """
        aligned = pd.concat([series1, series2], axis=1).dropna()
        if len(aligned) < 20 or aligned.iloc[:, 0].std() == 0 or aligned.iloc[:, 1].std() == 0:
            return {'cointegrated': False}
            
        x = aligned.iloc[:, 0].values
        y = aligned.iloc[:, 1].values
        
        try:
            # Cointegration check
            score, pvalue, _ = coint(x, y)
            
            hedge_ratio = np.polyfit(x, y, 1)[0]
            spread = y - hedge_ratio * x
            
            # 1. ADF (Null: Unit Root / Non-Stationary)
            adf_p = adfuller(spread)[1]
            
            # 2. KPSS (Null: Stationary)
            kpss_p = kpss(spread, regression='c', nlags="auto")[1]
            
            # Robust Stationarity: ADF rejects Non-Stationary AND KPSS fails to reject Stationary
            is_robust = (adf_p < 0.05) and (kpss_p > 0.05)
            
            return {
                'adf_pvalue': float(adf_p),
                'kpss_pvalue': float(kpss_p),
                'cointegration_pvalue': float(pvalue),
                'cointegrated': bool(pvalue < significance_level),
                'robust_stationarity': bool(is_robust),
                'hedge_ratio': float(hedge_ratio),
                'spread_mean': float(np.mean(spread)),
                'spread_std': float(np.std(spread))
            }
        except Exception as e:
            logger.error(f"Coint error: {str(e)}")
            return {'cointegrated': False}

    def estimate_ou_process(self, spread: pd.Series) -> Dict:
        """
        Estimates Ornstein-Uhlenbeck parameters with stability checks.
        """
        if len(spread) < 20: return {}
        
        # y_t = a + b * y_{t-1} + e
        y = spread.values[1:]
        x = spread.values[:-1]
        
        res = stats.linregress(x, y)
        b = res.slope
        a = res.intercept
        
        dt = 1/252
        
        # Stability check: if b >= 1, the process is not mean-reverting (it's a random walk or explosive)
        if b >= 1.0 or b <= 0:
            return {
                'kappa_reversion_speed': 0.0,
                'equilibrium_mu': float(np.mean(spread)),
                'half_life': 999.0
            }
            
        kappa = -np.log(b) / dt
        mu = a / (1 - b)
        
        return {
            'kappa_reversion_speed': float(kappa),
            'equilibrium_mu': float(mu),
            'half_life': float(min(np.log(2) / kappa, 365.0))
        }

    def johansen_cointegration_test(
        self, 
        data: pd.DataFrame, 
        k_ar_diff: int = 1
    ) -> Dict:
        """
        Perform Johansen cointegration test for multiple time series
        """
        try:
            data_clean = data.dropna()
            if data_clean.shape[1] < 2:
                return {}
                
            result = coint_johansen(data_clean, det_order=0, k_ar_diff=k_ar_diff)
            
            return {
                'trace_stats': result.lr1.tolist(),
                'trace_crit': result.cvt.tolist(),
                'max_eig_stats': result.lr2.tolist(),
                'max_eig_crit': result.cvm.tolist(),
                'eigenvalues': result.eig.tolist()
            }
        except Exception as e:
            logger.error(f"Johansen test error: {str(e)}")
            return {}

    def calculate_copula_dependence(self, series1: pd.Series, series2: pd.Series) -> Dict:
        """
        Captures non-linear tail dependence using empirical copulas
        """
        aligned = pd.concat([series1, series2], axis=1).dropna()
        u = stats.rankdata(aligned.iloc[:, 0]) / (len(aligned) + 1)
        v = stats.rankdata(aligned.iloc[:, 1]) / (len(aligned) + 1)
        
        # Tail dependence (Lower and Upper)
        threshold = 0.1
        lower_tail = np.mean((u < threshold) & (v < threshold)) / threshold
        upper_tail = np.mean((u > 1 - threshold) & (v > 1 - threshold)) / threshold
        
        return {
            'kendall_tau': float(stats.kendalltau(aligned.iloc[:, 0], aligned.iloc[:, 1])[0]),
            'lower_tail_dependence': float(lower_tail),
            'upper_tail_dependence': float(upper_tail)
        }

    def perform_pca_analysis(
        self, 
        returns_data: pd.DataFrame,
        n_components: int = 5
    ) -> Dict:
        """
        Perform PCA on returns data using numpy (SVD)
        """
        # Standardize returns
        returns_mean = returns_data.mean()
        returns_std = returns_data.std()
        returns_standardized = (returns_data - returns_mean) / returns_std
        returns_standardized = returns_standardized.dropna()
        X = returns_standardized.values
        
        # Perform PCA via SVD
        # X = U * S * Vt
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        
        # Components (eigenvectors of covariance matrix) are rows of Vt
        components = Vt[:n_components]
        
        # Projected data (Principal Components)
        principal_components = X @ components.T
        
        # Explained variance
        explained_variance = (S ** 2) / (len(X) - 1)
        total_variance = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_variance
        
        # Limit to n_components
        explained_variance_ratio = explained_variance_ratio[:n_components]
        cumulative_variance = np.cumsum(explained_variance_ratio)
        eigenvalues = explained_variance[:n_components]
        
        # Eigenportfolios (weights) - same as components
        eigenportfolios = components.T
        
        # Calculate factor exposures
        factor_exposures = pd.DataFrame(
            components.T,
            index=returns_data.columns,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        return {
            'principal_components': principal_components.tolist(),
            'eigenportfolios': eigenportfolios.tolist(),
            'explained_variance': explained_variance_ratio.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'eigenvalues': eigenvalues.tolist(),
            'factor_exposures': factor_exposures.to_dict(),
            'components': components.tolist()
        }
    
    def calculate_hurst_exponent(self, time_series: np.ndarray, max_lag: int = 50) -> float:
        """
        Calculate Hurst exponent using rescaled range analysis
        """
        lags = range(2, max_lag)
        tau = []
        
        for lag in lags:
            # Divide the series into chunks
            if len(time_series) < lag:
                continue
            chunks = len(time_series) // lag
            if chunks == 0:
                continue
            rs_values = []
            
            for i in range(chunks):
                chunk = time_series[i*lag:(i+1)*lag]
                mean_chunk = np.mean(chunk)
                deviations = chunk - mean_chunk
                z = np.cumsum(deviations)
                r = np.max(z) - np.min(z)
                s = np.std(chunk)
                if s != 0:
                    rs_values.append(r / s)
            
            if rs_values:
                tau.append(np.log(np.mean(rs_values)))
        
        if len(tau) > 1:
            # Calculate Hurst exponent
            hurst = np.polyfit(np.log(list(lags[:len(tau)])), tau, 1)[0]
            return float(hurst)
        else:
            return 0.5  # Return 0.5 for random walk
