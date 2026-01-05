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
        Perform Engle-Granger cointegration test
        """
        # Align series
        aligned = pd.concat([series1, series2], axis=1).dropna()
        x = aligned.iloc[:, 0].values
        y = aligned.iloc[:, 1].values
        
        # Perform cointegration test
        score, pvalue, _ = coint(x, y)
        
        # Calculate hedge ratio using OLS
        hedge_ratio = np.polyfit(x, y, 1)[0]
        
        # Create spread
        spread = y - hedge_ratio * x
        
        # Test spread for stationarity
        adf_result = adfuller(spread)
        kpss_result = kpss(spread, regression='c')
        
        # Calculate half-life of mean reversion
        spread_lag = np.roll(spread, 1)
        spread_lag[0] = spread_lag[1]
        spread_ret = spread - spread_lag
        spread_lag2 = sm.add_constant(spread_lag)
        model = sm.OLS(spread_ret, spread_lag2)
        res = model.fit()
        half_life = -np.log(2) / res.params[1] if res.params[1] < 0 else np.nan
        
        # Calculate Hurst exponent
        hurst = self.calculate_hurst_exponent(spread)
        
        return {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'kpss_statistic': kpss_result[0],
            'kpss_pvalue': kpss_result[1],
            'cointegration_pvalue': pvalue,
            'cointegrated': pvalue < significance_level,
            'hedge_ratio': hedge_ratio,
            'half_life': half_life,
            'hurst_exponent': hurst,
            'spread_mean': np.mean(spread),
            'spread_std': np.std(spread)
        }
    
    def johansen_cointegration_test(
        self, 
        data: pd.DataFrame, 
        k_ar_diff: int = 1
    ) -> Dict:
        """
        Perform Johansen cointegration test for multiple time series
        """
        # Drop NaN values
        data_clean = data.dropna()
        
        # Perform Johansen test
        result = coint_johansen(data_clean, det_order=0, k_ar_diff=k_ar_diff)
        
        # Trace test statistics
        trace_stats = result.lr1
        trace_crit_values = result.cvt
        trace_significant = result.lr1 > result.cvt[:, 1]  # 95% critical value
        
        # Max eigenvalue test statistics
        max_eigen_stats = result.lr2
        max_eigen_crit_values = result.cvm
        max_eigen_significant = result.lr2 > result.cvm[:, 1]
        
        # Cointegrating vectors
        eigenvectors = result.evec
        
        return {
            'trace_statistics': trace_stats.tolist(),
            'trace_critical_values': trace_crit_values.tolist(),
            'trace_significant': trace_significant.tolist(),
            'max_eigen_statistics': max_eigen_stats.tolist(),
            'max_eigen_critical_values': max_eigen_crit_values.tolist(),
            'max_eigen_significant': max_eigen_significant.tolist(),
            'cointegrating_vectors': eigenvectors.tolist(),
            'num_cointegrating_relations': np.sum(trace_significant)
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
            hurst = np.polyfit(np.log(lags[:len(tau)]), tau, 1)[0]
            return hurst
        else:
            return 0.5  # Return 0.5 for random walk
    
    def calculate_orstein_uhlenbeck_params(self, spread: np.ndarray) -> Dict:
        """
        Estimate Ornstein-Uhlenbeck process parameters
        """
        spread_lag = np.roll(spread, 1)
        spread_lag[0] = spread_lag[1]
        spread_ret = spread - spread_lag
        
        # Add constant for OLS
        X = sm.add_constant(spread_lag)
        model = sm.OLS(spread_ret, X)
        res = model.fit()
        
        # OU parameters
        theta = -res.params[1]  # Mean reversion speed
        mu = res.params[0] / theta if theta > 0 else np.mean(spread)  # Long-term mean
        sigma = np.std(res.resid)  # Volatility
        
        # Calculate half-life
        half_life = np.log(2) / theta if theta > 0 else np.nan
        
        return {
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'half_life': half_life,
            'residuals': res.resid.tolist()
        }
    
    def calculate_copula_dependence(self, returns1: np.ndarray, returns2: np.ndarray) -> Dict:
        """
        Calculate copula-based dependence measures
        """
        from scipy.stats import kendalltau, spearmanr
        
        # Rank correlation measures
        kendall_tau, kendall_p = kendalltau(returns1, returns2)
        spearman_rho, spearman_p = spearmanr(returns1, returns2)
        
        # Empirical copula
        n = len(returns1)
        u = stats.rankdata(returns1) / (n + 1)
        v = stats.rankdata(returns2) / (n + 1)
        
        # Calculate tail dependence coefficients
        threshold = 0.1
        lower_tail = np.mean((u < threshold) & (v < threshold)) / threshold
        upper_tail = np.mean((u > 1 - threshold) & (v > 1 - threshold)) / threshold
        
        return {
            'kendall_tau': kendall_tau,
            'spearman_rho': spearman_rho,
            'lower_tail_dependence': lower_tail,
            'upper_tail_dependence': upper_tail,
            'empirical_copula': {
                'u': u.tolist(),
                'v': v.tolist()
            }
        }