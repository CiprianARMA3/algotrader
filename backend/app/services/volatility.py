import numpy as np
import pandas as pd
from arch import arch_model
from typing import List, Dict, Tuple, Optional
import logging
from scipy.stats import norm, skew, kurtosis
from statsmodels.tsa.stattools import acf

logger = logging.getLogger(__name__)

class VolatilityAnalyzer:
    def __init__(self):
        pass
    
    def calculate_realized_volatility(
        self, 
        returns: pd.Series,
        window: int = 20,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate realized volatility (standard deviation of returns)
        """
        realized_vol = returns.rolling(window).std()
        
        if annualize:
            # Annualize assuming 252 trading days
            realized_vol = realized_vol * np.sqrt(252)
        
        return realized_vol
    
    def calculate_parkinson_volatility(
        self, 
        high: pd.Series, 
        low: pd.Series,
        window: int = 20,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate Parkinson volatility estimator using high-low range
        """
        log_hl = np.log(high / low)
        parkinson_vol = np.sqrt((1 / (4 * np.log(2))) * (log_hl ** 2))
        parkinson_vol = parkinson_vol.rolling(window).mean()
        
        if annualize:
            parkinson_vol = parkinson_vol * np.sqrt(252)
        
        return parkinson_vol
    
    def calculate_garman_klass_volatility(
        self,
        open: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate Garman-Klass volatility estimator
        """
        log_hl = np.log(high / low)
        log_co = np.log(close / open)
        
        gk_vol = np.sqrt(0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2))
        gk_vol = gk_vol.rolling(window).mean()
        
        if annualize:
            gk_vol = gk_vol * np.sqrt(252)
        
        return gk_vol
    
    def estimate_garch_models(
        self, 
        returns: pd.Series,
        model_type: str = 'GARCH',
        p: int = 1,
        q: int = 1,
        distribution: str = 'normal'
    ) -> Dict:
        """
        Estimate GARCH family models
        """
        returns_clean = returns.dropna()
        
        try:
            if model_type.upper() == 'GARCH':
                model = arch_model(
                    returns_clean,
                    vol='Garch',
                    p=p,
                    q=q,
                    dist=distribution
                )
            elif model_type.upper() == 'GJR-GARCH':
                model = arch_model(
                    returns_clean,
                    vol='Garch',
                    p=p,
                    o=q,  # asymmetric terms
                    q=q,
                    dist=distribution,
                    power=2.0
                )
            elif model_type.upper() == 'EGARCH':
                model = arch_model(
                    returns_clean,
                    vol='EGARCH',
                    p=p,
                    q=q,
                    dist=distribution
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Fit model
            model_fit = model.fit(disp='off')
            
            # Get forecast
            forecast = model_fit.forecast(horizon=5)
            
            # Calculate leverage effect
            leverage_effect = self.calculate_leverage_effect(returns_clean)
            
            # Calculate volatility of volatility
            volatility_of_vol = returns_clean.rolling(window=20).std().std()
            
            return {
                'model_type': model_type,
                'params': model_fit.params.to_dict(),
                'conditional_volatility': model_fit.conditional_volatility.tolist(),
                'residuals': model_fit.resid.tolist(),
                'forecast': forecast.variance.iloc[-1].values.tolist(),
                'leverage_effect': leverage_effect,
                'volatility_of_vol': volatility_of_vol,
                'aic': model_fit.aic,
                'bic': model_fit.bic,
                'log_likelihood': model_fit.loglikelihood
            }
            
        except Exception as e:
            logger.error(f"Error fitting {model_type} model: {str(e)}")
            return {}
    
    def calculate_leverage_effect(self, returns: pd.Series) -> float:
        """
        Calculate leverage effect (correlation between returns and future volatility)
        """
        returns_clean = returns.dropna()
        future_vol = returns_clean.shift(-1).abs().rolling(window=5).mean()
        
        # Align series
        aligned = pd.concat([returns_clean, future_vol], axis=1).dropna()
        
        if len(aligned) > 10:
            correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            return correlation
        else:
            return 0.0
    
    def calculate_volatility_regime(
        self, 
        volatility: pd.Series,
        n_regimes: int = 3
    ) -> Dict:
        """
        Detect volatility regimes using quantiles
        """
        if volatility.dropna().empty:
            return {}
        
        vol_values = volatility.dropna().values
        
        # Define regimes based on quantiles
        quantiles = np.percentile(vol_values, [33, 67])
        
        regimes = np.zeros(len(vol_values))
        regimes[vol_values <= quantiles[0]] = 0  # Low volatility
        regimes[(vol_values > quantiles[0]) & (vol_values <= quantiles[1])] = 1  # Medium volatility
        regimes[vol_values > quantiles[1]] = 2  # High volatility
        
        # Calculate regime statistics
        regime_stats = {}
        for regime in range(n_regimes):
            mask = regimes == regime
            if np.any(mask):
                regime_stats[f'regime_{regime}'] = {
                    'count': np.sum(mask),
                    'mean_volatility': np.mean(vol_values[mask]),
                    'std_volatility': np.std(vol_values[mask]),
                    'percentage': np.sum(mask) / len(vol_values) * 100
                }
        
        # Calculate regime transition probabilities
        transition_matrix = self.calculate_transition_matrix(regimes, n_regimes)
        
        return {
            'regimes': regimes.tolist(),
            'regime_stats': regime_stats,
            'transition_matrix': transition_matrix.tolist(),
            'quantile_thresholds': quantiles.tolist()
        }
    
    def calculate_transition_matrix(
        self, 
        regimes: np.ndarray,
        n_states: int
    ) -> np.ndarray:
        """
        Calculate Markov transition probability matrix
        """
        transition_matrix = np.zeros((n_states, n_states))
        
        for i in range(len(regimes) - 1):
            current = int(regimes[i])
            next_state = int(regimes[i + 1])
            transition_matrix[current, next_state] += 1
        
        # Normalize rows
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums
        
        return transition_matrix
    
    def calculate_volatility_forecast(
        self,
        returns: pd.Series,
        forecast_horizon: int = 5,
        method: str = 'garch'
    ) -> Dict:
        """
        Forecast volatility using multiple methods
        """
        returns_clean = returns.dropna()
        
        forecasts = {}
        
        # Historical volatility forecast
        historical_vol = returns_clean.std() * np.sqrt(252)
        forecasts['historical'] = [historical_vol] * forecast_horizon
        
        # EWMA forecast
        lambda_param = 0.94
        ewma_vol = self.calculate_ewma_volatility(returns_clean, lambda_param)
        forecasts['ewma'] = [ewma_vol] * forecast_horizon
        
        # GARCH forecast if available
        if method == 'garch' and len(returns_clean) > 100:
            try:
                garch_result = self.estimate_garch_models(returns_clean)
                if garch_result:
                    forecasts['garch'] = garch_result['forecast']
            except:
                pass
        
        # Combined forecast (average)
        all_forecasts = [f for f in forecasts.values() if len(f) == forecast_horizon]
        if all_forecasts:
            combined = np.mean(all_forecasts, axis=0)
            forecasts['combined'] = combined.tolist()
        
        return forecasts
    
    def calculate_ewma_volatility(
        self,
        returns: pd.Series,
        lambda_param: float = 0.94
    ) -> float:
        """
        Calculate EWMA volatility
        """
        returns_clean = returns.dropna()
        n = len(returns_clean)
        
        weights = np.zeros(n)
        for i in range(n):
            weights[i] = (1 - lambda_param) * (lambda_param ** i)
        weights = weights[::-1]  # Reverse for chronological order
        
        weighted_returns = returns_clean.values * weights
        ewma_vol = np.sqrt(np.sum(weighted_returns ** 2))
        
        return ewma_vol * np.sqrt(252)
    
    def calculate_volatility_surface_metrics(
        self,
        returns: pd.Series,
        window: int = 20
    ) -> Dict:
        """
        Calculate various volatility surface metrics
        """
        returns_clean = returns.dropna()
        
        # Volatility term structure (short vs long term)
        short_term_vol = returns_clean.rolling(window=5).std().mean() * np.sqrt(252)
        long_term_vol = returns_clean.rolling(window=60).std().mean() * np.sqrt(252)
        term_structure = short_term_vol / long_term_vol
        
        # Volatility skew (asymmetry)
        positive_returns = returns_clean[returns_clean > 0]
        negative_returns = returns_clean[returns_clean < 0]
        
        if len(positive_returns) > 10 and len(negative_returns) > 10:
            vol_skew = positive_returns.std() - negative_returns.std()
        else:
            vol_skew = 0
        
        # Volatility persistence (autocorrelation)
        vol_series = returns_clean.rolling(window=window).std().dropna()
        if len(vol_series) > 10:
            vol_acf = acf(vol_series, nlags=5, fft=True)
            persistence = vol_acf[1]  # First lag autocorrelation
        else:
            persistence = 0
        
        # Jump intensity
        returns_abs = returns_clean.abs()
        jump_threshold = returns_abs.mean() + 3 * returns_abs.std()
        jump_intensity = np.sum(returns_abs > jump_threshold) / len(returns_abs)
        
        # Rough Volatility (Hurst of Volatility)
        roughness = self.estimate_rough_volatility(returns_clean)
        
        return {
            'short_term_vol': short_term_vol,
            'long_term_vol': long_term_vol,
            'term_structure': term_structure,
            'volatility_skew': vol_skew,
            'volatility_persistence': persistence,
            'jump_intensity': jump_intensity,
            'roughness_exponent': roughness,
            'current_realized_volatility': returns_clean.std() * np.sqrt(252),
            'min_volatility': returns_clean.rolling(window).std().min() * np.sqrt(252),
            'max_volatility': returns_clean.rolling(window).std().max() * np.sqrt(252)
        }

    def estimate_rough_volatility(self, returns: pd.Series) -> float:
        """
        Estimates the 'Roughness' (Hurst Exponent) of the volatility process.
        Institutional finding: H ~ 0.1 (Rough Volatility).
        """
        # Calculate log-volatility proxy
        vol_proxy = returns.abs().rolling(window=5).mean().dropna()
        if len(vol_proxy) < 50: return 0.5
        
        log_vol = np.log(vol_proxy + 1e-9)
        
        # Simple Hurst estimation via rescaled range or log-log regression of increments
        lags = range(2, 20)
        std_diffs = []
        for lag in lags:
            diffs = log_vol.diff(lag).dropna()
            if not diffs.empty:
                std_diffs.append(np.std(diffs))
            else:
                std_diffs.append(1e-9)
        
        # slope of log(lag) vs log(std_diff) gives Hurst H
        h, _ = np.polyfit(np.log(list(lags)), np.log(std_diffs), 1)
        return float(h)

    def calculate_realized_kernel(self, returns: pd.Series) -> float:
        """
        Realized Kernel estimator (Tukey-Hanning) - robust to microstructure noise.
        """
        x = returns.values
        n = len(x)
        # Parzen kernel weights
        def kernel_weight(h, H):
            x = h / H
            if 0 <= x <= 0.5:
                return 1 - 6*x**2 + 6*x**3
            elif 0.5 < x <= 1:
                return 2*(1-x)**3
            return 0

        # Auto-covariances
        gamma = lambda h: np.sum(x[h:] * x[:n-h])
        
        H = int(np.sqrt(n)) # Bandwidth selection proxy
        rk = gamma(0) + 2 * np.sum([kernel_weight(h, H) * gamma(h) for h in range(1, H+1)])
        return float(rk * 252) # Annualized