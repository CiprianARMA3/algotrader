export interface CointegrationResult {
  pair: string[];
  adf_statistic: number;
  adf_pvalue: number;
  kpss_statistic: number;
  kpss_pvalue: number;
  cointegrated: boolean;
  hedge_ratio: number;
  half_life?: number;
  hurst_exponent?: number;
  spread_mean?: number;
  spread_std?: number;
  cointegration_pvalue?: number; // Inferred from usage in Charts.tsx
}

export interface PCAResult {
  eigenvalues: number[];
  explained_variance: number[];
  eigenportfolios: number[][];
  principal_components: number[][];
  factor_exposures?: Record<string, Record<string, number>>;
  cumulative_variance?: number[];
}

export interface VolatilityResult {
  symbol: string;
  garch_volatility: number[];
  realized_volatility: number[];
  leverage_effect: number;
  vol_of_vol: number;
  short_term_vol: number;
  long_term_vol: number;
  term_structure: number;
  volatility_skew: number;
  volatility_persistence: number;
  jump_intensity: number;
  min_volatility: number;
  max_volatility: number;
}

export interface RegimeResult {
  symbol: string;
  hidden_states?: number[];
  state_probabilities?: number[][];
  regime_statistics?: Record<string, any>;
  transition_matrix?: number[][];
  persistence?: number[];
  expected_durations?: number[];
  model_score?: number;
  means?: number[];
  covariances?: number[];
  volatility_regimes?: number[];
  volatility_regime_stats?: Record<string, any>;
}

export interface TrendingResult {
  symbol: string;
  fractional_d: number;
  hurst_exponent: number;
  trend_strength: number;
  adf_statistic: number;
  is_stationary: boolean;
  wavelet_analysis?: any;
  kalman_trend?: number[];
}

export interface MicrostructureResult {
  symbol: string;
  amihud_illiquidity: number;
  volume_profile: {
    mean_volume: number;
    volume_std: number;
    volume_skew: number;
  };
  volatility_metrics: any;
  order_flow_imbalance?: number[];
}

export interface ExecutionRecommendations {
  market_regime: string;
  volatility_environment: string;
  recommended_strategies: string[];
  risk_warnings: string[];
  position_sizing: string;
  execution_timing: string;
}

export interface AnalysisResponse {
  timestamp: string;
  instruments: string[];
  cointegration?: CointegrationResult[];
  pca?: PCAResult;
  volatility?: VolatilityResult[];
  regimes?: RegimeResult[];
  trending?: TrendingResult[];
  microstructure?: MicrostructureResult[];
  execution_recommendations?: ExecutionRecommendations;
}
