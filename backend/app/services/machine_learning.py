import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MachineLearningAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def hidden_markov_model_regime(
        self,
        returns: pd.Series,
        n_states: int = 3,
        n_features: int = 1
    ) -> Dict:
        """
        Fit Hidden Markov Model to detect market regimes
        """
        returns_clean = returns.dropna().values.reshape(-1, 1)
        
        if len(returns_clean) < 100:
            logger.warning("Insufficient data for HMM")
            return {}
        
        try:
            # Fit Gaussian HMM
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=1000,
                random_state=42
            )
            
            model.fit(returns_clean)
            
            # Predict hidden states
            hidden_states = model.predict(returns_clean)
            
            # Calculate state probabilities
            state_probabilities = model.predict_proba(returns_clean)
            
            # Calculate regime statistics
            regime_stats = {}
            for i in range(n_states):
                mask = hidden_states == i
                if np.any(mask):
                    regime_returns = returns_clean[mask]
                    regime_stats[f'regime_{i}'] = {
                        'count': np.sum(mask),
                        'mean_return': float(np.mean(regime_returns)),
                        'std_return': float(np.std(regime_returns)),
                        'probability': np.mean(state_probabilities[:, i]),
                        'percentage': np.sum(mask) / len(returns_clean) * 100
                    }
            
            # Calculate transition matrix
            transition_matrix = model.transmat_
            
            # Calculate persistence (probability of staying in same state)
            persistence = np.diag(transition_matrix)
            
            # Calculate expected duration of each regime
            expected_durations = 1 / (1 - persistence)
            
            return {
                'hidden_states': hidden_states.tolist(),
                'state_probabilities': state_probabilities.tolist(),
                'regime_statistics': regime_stats,
                'transition_matrix': transition_matrix.tolist(),
                'persistence': persistence.tolist(),
                'expected_durations': expected_durations.tolist(),
                'model_score': model.score(returns_clean),
                'means': model.means_.flatten().tolist(),
                'covariances': model.covars_.flatten().tolist()
            }
            
        except Exception as e:
            logger.error(f"Error fitting HMM: {str(e)}")
            return {}
    
    def calculate_transfer_entropy(
        self,
        series_x: pd.Series,
        series_y: pd.Series,
        k: int = 1,
        l: int = 1,
        normalize: bool = True
    ) -> float:
        """
        Calculate transfer entropy from X to Y
        Simplified implementation
        """
        # Align series
        aligned = pd.concat([series_x, series_y], axis=1).dropna()
        x = aligned.iloc[:, 0].values
        y = aligned.iloc[:, 1].values
        
        if len(x) < 100:
            return 0.0
        
        # Discretize series using quantile bins
        n_bins = 10
        x_bins = pd.qcut(x, n_bins, labels=False, duplicates='drop')
        y_bins = pd.qcut(y, n_bins, labels=False, duplicates='drop')
        
        # Calculate conditional entropies
        # Simplified calculation - in practice use specialized library
        te = 0.0
        
        # For each time point t, consider past k values of X and past l values of Y
        for t in range(max(k, l), len(x_bins)):
            # Past of Y
            y_past = tuple(y_bins[t-l:t])
            
            # Past of Y and X
            xy_past = tuple(y_bins[t-l:t]) + tuple(x_bins[t-k:t])
            
            # Calculate probabilities (simplified)
            # This is a placeholder - actual implementation requires probability estimation
            pass
        
        return te
    
    def calculate_mutual_information(
        self,
        series_x: pd.Series,
        series_y: pd.Series,
        n_bins: int = 20
    ) -> float:
        """
        Calculate mutual information between two series
        """
        # Align series
        aligned = pd.concat([series_x, series_y], axis=1).dropna()
        x = aligned.iloc[:, 0].values
        y = aligned.iloc[:, 1].values
        
        if len(x) < 50:
            return 0.0
        
        # Create 2D histogram
        hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)
        
        # Calculate joint probability distribution
        p_xy = hist_2d / np.sum(hist_2d)
        
        # Calculate marginal distributions
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)
        
        # Calculate mutual information
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
        
        return mi
    
    def calculate_entropy_metrics(
        self,
        series: pd.Series,
        window: int = 20
    ) -> Dict:
        """
        Calculate various entropy-based metrics
        """
        returns = series.dropna().values
        
        if len(returns) < window:
            return {}
        
        # Shannon entropy of returns
        hist, bin_edges = np.histogram(returns, bins=50, density=True)
        hist = hist[hist > 0]
        shannon_entropy = entropy(hist)
        
        # Sample entropy (approximate)
        sample_entropy = self.calculate_sample_entropy(returns, m=2, r=0.2)
        
        # Entropy of signs
        signs = np.sign(returns)
        sign_hist = np.bincount((signs + 1).astype(int))  # Convert -1,0,1 to 0,1,2
        sign_hist = sign_hist[sign_hist > 0]
        sign_entropy = entropy(sign_hist)
        
        # Rolling entropy
        rolling_entropies = []
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            window_hist, _ = np.histogram(window_returns, bins=20, density=True)
            window_hist = window_hist[window_hist > 0]
            if len(window_hist) > 1:
                rolling_entropies.append(entropy(window_hist))
        
        return {
            'shannon_entropy': shannon_entropy,
            'sample_entropy': sample_entropy,
            'sign_entropy': sign_entropy,
            'rolling_entropy_mean': np.mean(rolling_entropies) if rolling_entropies else 0,
            'rolling_entropy_std': np.std(rolling_entropies) if rolling_entropies else 0,
            'entropy_trend': self.calculate_entropy_trend(rolling_entropies) if rolling_entropies else 0
        }
    
    def calculate_sample_entropy(
        self,
        time_series: np.ndarray,
        m: int = 2,
        r: float = 0.2
    ) -> float:
        """
        Calculate sample entropy of a time series
        """
        n = len(time_series)
        
        if n < m + 1:
            return 0
        
        # Standardize time series
        std = np.std(time_series)
        if std == 0:
            return 0
        
        time_series_std = (time_series - np.mean(time_series)) / std
        
        # Define tolerance
        r_value = r * std
        
        # Count similar patterns
        def count_similar(vector, matrix, r_val):
            return np.sum(np.max(np.abs(matrix - vector), axis=1) <= r_val)
        
        # Create pattern vectors
        patterns_m = np.array([time_series_std[i:i+m] for i in range(n - m + 1)])
        patterns_m1 = np.array([time_series_std[i:i+m+1] for i in range(n - m)])
        
        # Count matches
        matches_m = 0
        matches_m1 = 0
        
        for i in range(len(patterns_m)):
            # Exclude self-match
            mask = np.ones(len(patterns_m), dtype=bool)
            mask[i] = False
            matches_m += count_similar(patterns_m[i], patterns_m[mask], r_value)
            
            if i < len(patterns_m1):
                mask_m1 = np.ones(len(patterns_m1), dtype=bool)
                mask_m1[i] = False
                matches_m1 += count_similar(patterns_m1[i], patterns_m1[mask_m1], r_value)
        
        # Avoid division by zero
        if matches_m == 0 or matches_m1 == 0:
            return 0
        
        # Calculate sample entropy
        sample_entropy = -np.log(matches_m1 / matches_m)
        
        return sample_entropy
    
    def calculate_entropy_trend(self, entropy_series: List[float]) -> float:
        """
        Calculate trend in entropy (increasing/decreasing)
        """
        if len(entropy_series) < 10:
            return 0
        
        x = np.arange(len(entropy_series))
        y = np.array(entropy_series)
        
        # Fit linear trend
        slope, intercept = np.polyfit(x, y, 1)
        
        return slope
    
    def detect_change_points(
        self,
        series: pd.Series,
        method: str = 'cusum'
    ) -> Dict:
        """
        Detect change points in time series
        """
        values = series.dropna().values
        
        if len(values) < 50:
            return {'change_points': [], 'scores': []}
        
        if method == 'cusum':
            # CUSUM method for change point detection
            mean = np.mean(values)
            std = np.std(values)
            
            cusum_pos = np.zeros(len(values))
            cusum_neg = np.zeros(len(values))
            
            for i in range(1, len(values)):
                cusum_pos[i] = max(0, cusum_pos[i-1] + values[i] - mean - 0.5*std)
                cusum_neg[i] = min(0, cusum_neg[i-1] + values[i] - mean + 0.5*std)
            
            # Find change points where CUSUM exceeds threshold
            threshold = 5 * std
            change_points = []
            scores = []
            
            for i in range(len(values)):
                if abs(cusum_pos[i]) > threshold or abs(cusum_neg[i]) > threshold:
                    change_points.append(i)
                    scores.append(max(abs(cusum_pos[i]), abs(cusum_neg[i])))
            
            return {
                'change_points': change_points,
                'scores': scores,
                'cusum_positive': cusum_pos.tolist(),
                'cusum_negative': cusum_neg.tolist(),
                'threshold': threshold
            }
        
        else:
            # Simple statistical method
            rolling_mean = pd.Series(values).rolling(window=20).mean().values
            rolling_std = pd.Series(values).rolling(window=20).std().values
            
            # Detect changes where value deviates significantly from rolling mean
            z_scores = np.abs((values[20:] - rolling_mean[20:-20]) / (rolling_std[20:-20] + 1e-10))
            
            change_points = np.where(z_scores > 3)[0] + 20
            scores = z_scores[z_scores > 3]
            
            return {
                'change_points': change_points.tolist(),
                'scores': scores.tolist(),
                'z_scores': z_scores.tolist()
            }