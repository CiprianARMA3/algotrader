import numpy as np
import pandas as pd
from scipy import signal, fft
import pywt
from typing import List, Dict, Tuple, Optional
import logging
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

class SignalProcessor:
    def __init__(self):
        pass
    
    def fractional_differentiation(
        self, 
        series: pd.Series, 
        d: float = 0.4, 
        threshold: float = 1e-5
    ) -> pd.Series:
        """
        Apply fractional differentiation to a time series
        """
        series = series.dropna()
        n = len(series)
        
        # Calculate weights using binomial expansion
        weights = [1.0]
        for k in range(1, n):
            weight = -weights[-1] * (d - k + 1) / k
            weights.append(weight)
        
        weights = np.array(weights[::-1])
        
        # Apply fractional differentiation
        frac_diff = np.zeros(n)
        for i in range(n):
            frac_diff[i] = np.dot(weights[-(i+1):], series[:i+1])
        
        # Find the index where weights become negligible
        cutoff = np.argmax(np.abs(weights) < threshold)
        
        return pd.Series(frac_diff[cutoff:], index=series.index[cutoff:])
    
    def find_optimal_fractional_d(
        self, 
        series: pd.Series, 
        d_range: Tuple[float, float] = (0, 1),
        step: float = 0.05
    ) -> Dict:
        """
        Find optimal fractional differentiation parameter
        """
        results = []
        d_values = np.arange(d_range[0], d_range[1] + step, step)
        
        for d in d_values:
            if d == 0:
                continue
                
            frac_series = self.fractional_differentiation(series, d)
            
            # Test for stationarity
            if len(frac_series) > 10:
                adf_result = adfuller(frac_series.dropna())
                
                # Calculate correlation with original series
                aligned = pd.concat([series, frac_series], axis=1).dropna()
                correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                
                results.append({
                    'd': d,
                    'adf_statistic': adf_result[0],
                    'adf_pvalue': adf_result[1],
                    'correlation': correlation,
                    'is_stationary': adf_result[1] < 0.05
                })
        
        # Find optimal d (stationary with maximum correlation)
        stationary_results = [r for r in results if r['is_stationary']]
        if stationary_results:
            optimal = max(stationary_results, key=lambda x: x['correlation'])
        else:
            optimal = min(results, key=lambda x: x['adf_pvalue'])
        
        return {
            'optimal_d': optimal['d'],
            'all_results': results,
            'adf_statistic': optimal['adf_statistic'],
            'adf_pvalue': optimal['adf_pvalue'],
            'correlation': optimal['correlation']
        }
    
    def wavelet_transform(
        self, 
        series: pd.Series, 
        wavelet: str = 'db4',
        level: int = 5
    ) -> Dict:
        """
        Perform wavelet decomposition and denoising
        """
        values = series.dropna().values
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(values, wavelet, level=level)
        
        # Calculate threshold for denoising (VisuShrink)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(values)))
        
        # Apply soft thresholding
        coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
        for i in range(1, len(coeffs)):
            coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
        
        # Reconstruct denoised signal
        denoised = pywt.waverec(coeffs_thresh, wavelet)
        
        # Calculate energy distribution
        energy = [np.sum(c**2) for c in coeffs]
        total_energy = np.sum(energy)
        energy_ratio = [e / total_energy for e in energy]
        
        return {
            'original': values.tolist(),
            'denoised': denoised.tolist(),
            'coefficients': [c.tolist() for c in coeffs],
            'energy_distribution': energy_ratio,
            'threshold': threshold,
            'approximation': coeffs[0].tolist(),
            'details': [c.tolist() for c in coeffs[1:]]
        }
    
    def fourier_analysis(
        self, 
        series: pd.Series,
        sampling_rate: float = 1.0
    ) -> Dict:
        """
        Perform Fourier spectral analysis
        """
        values = series.dropna().values
        n = len(values)
        
        # Apply window function
        window = np.hanning(n)
        windowed = values * window
        
        # Perform FFT
        fft_result = fft.fft(windowed)
        frequencies = fft.fftfreq(n, d=1/sampling_rate)
        
        # Calculate power spectrum
        power_spectrum = np.abs(fft_result) ** 2
        power_spectrum_db = 10 * np.log10(power_spectrum)
        
        # Find dominant frequencies
        positive_freq = frequencies[:n//2]
        positive_power = power_spectrum[:n//2]
        
        # Find peaks
        peaks, properties = signal.find_peaks(positive_power, prominence=np.std(positive_power)/10)
        
        dominant_freqs = positive_freq[peaks]
        dominant_powers = positive_power[peaks]
        
        # Sort by power
        sorted_idx = np.argsort(dominant_powers)[::-1]
        dominant_freqs = dominant_freqs[sorted_idx]
        dominant_powers = dominant_powers[sorted_idx]
        
        return {
            'frequencies': frequencies.tolist(),
            'power_spectrum': power_spectrum.tolist(),
            'power_spectrum_db': power_spectrum_db.tolist(),
            'dominant_frequencies': dominant_freqs.tolist(),
            'dominant_powers': dominant_powers.tolist(),
            'peak_properties': properties
        }
    
    def kalman_filter_trend(
        self, 
        series: pd.Series,
        observation_variance: float = 1.0,
        transition_variance: float = 0.1
    ) -> Dict:
        """
        Apply Kalman filter for trend estimation
        """
        values = series.dropna().values
        if len(values) < 2:
            return {'filtered_trend': [], 'smoothed_trend': []}
        
        # Initialize Kalman filter with explicit 2D matrices to avoid alignment issues
        kf = KalmanFilter(
            transition_matrices=np.array([[1]]),
            observation_matrices=np.array([[1]]),
            observation_covariance=observation_variance,
            transition_covariance=np.array([[transition_variance]]),
            initial_state_mean=values[0],
            initial_state_covariance=1.0
        )
        
        # Apply filter
        state_means, state_covariances = kf.filter(values)
        state_means = state_means.flatten()
        
        # Smooth the estimates
        smoothed_means, smoothed_covariances = kf.smooth(values)
        smoothed_means = smoothed_means.flatten()
        
        # Calculate innovation (prediction errors)
        pred_state, pred_cov = kf.filter_update(
            state_means[:-1], state_covariances[:-1], values[1:]
        )
        
        return {
            'original': values.tolist(),
            'filtered_trend': state_means.tolist(),
            'smoothed_trend': smoothed_means.tolist(),
            'confidence_intervals': {
                'upper': (state_means + 1.96 * np.sqrt(state_covariances.flatten())).tolist(),
                'lower': (state_means - 1.96 * np.sqrt(state_covariances.flatten())).tolist()
            },
            'innovation': (values[1:] - state_means[:-1]).tolist(),
            'log_likelihood': kf.loglikelihood(values)
        }
    
    def calculate_trend_strength(
        self, 
        series: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate trend strength using various methods
        """
        prices = series.dropna()
        
        # Method 1: ADX-like trend strength
        high = prices.rolling(window).max()
        low = prices.rolling(window).min()
        true_range = high - low
        normalized_range = true_range / prices.rolling(window).mean()
        
        # Method 2: Slope-based trend strength
        x = np.arange(len(prices))
        slopes = pd.Series(index=prices.index, dtype=float)
        
        for i in range(window, len(prices)):
            y = prices.iloc[i-window:i].values
            slope = np.polyfit(x[:window], y, 1)[0]
            slopes.iloc[i] = slope
        
        # Normalize slopes
        norm_slopes = (slopes - slopes.rolling(window).mean()) / slopes.rolling(window).std()
        
        # Combined trend strength
        trend_strength = (normalized_range * 0.3 + norm_slopes.abs() * 0.7).fillna(0)
        
        return trend_strength
    
    def calculate_instantaneous_metrics(self, series: pd.Series) -> Dict:
        """
        Derives Instantaneous Frequency and Amplitude using Hilbert Transform.
        Institutional finding: HHT detects turning points faster than lagging MAs.
        """
        from scipy.signal import hilbert
        
        signal = series.dropna().values
        if len(signal) < 10: return {}
        
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
        
        return {
            'inst_amplitude': amplitude_envelope.tolist(),
            'inst_frequency': instantaneous_frequency.tolist(),
            'frequency_mean': float(np.mean(instantaneous_frequency)),
            'volatility_spectral': float(np.std(instantaneous_frequency))
        }

    def empirical_mode_decomposition(
        self, 
        series: pd.Series,
        max_imfs: int = 10
    ) -> Dict:
        """
        Perform Empirical Mode Decomposition (simplified version)
        """
        from scipy.signal import hilbert
        
        signal = series.dropna().values
        residue = signal.copy()
        imfs = []
        
        for _ in range(max_imfs):
            h = residue.copy()
            
            while True:
                # Find local maxima and minima
                maxima = signal.argrelmax(h)[0]
                minima = signal.argrelmin(h)[0]
                
                if len(maxima) < 2 or len(minima) < 2:
                    break
                
                # Interpolate envelopes
                max_env = np.interp(range(len(h)), maxima, h[maxima])
                min_env = np.interp(range(len(h)), minima, h[minima])
                
                # Calculate mean envelope
                mean_env = (max_env + min_env) / 2
                
                # Subtract mean
                h = h - mean_env
                
                # Check stopping criterion
                if np.sum(mean_env**2) / np.sum(h**2) < 0.01:
                    break
            
            if np.sum(h**2) < 0.01 * np.sum(signal**2):
                break
            
            imfs.append(h.tolist())
            residue = residue - h
        
        # Calculate instantaneous frequency using Hilbert transform
        instantaneous_freqs = []
        for imf in imfs:
            analytic_signal = hilbert(imf)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
            instantaneous_freqs.append(instantaneous_frequency.tolist())
        
        return {
            'imfs': imfs,
            'residue': residue.tolist(),
            'instantaneous_frequencies': instantaneous_freqs,
            'num_imfs': len(imfs)
        }