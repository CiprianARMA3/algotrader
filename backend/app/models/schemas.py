from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

class TimeFrame(str, Enum):
    DAILY = "1d"
    WEEKLY = "1wk"
    MONTHLY = "1mo"
    HOURLY = "1h"
    MINUTE_15 = "15m"

class AnalysisType(str, Enum):
    COINTEGRATION = "cointegration"
    PCA = "pca"
    VOLATILITY = "volatility"
    REGIME = "regime"
    TRENDING = "trending"
    MICROSTRUCTURE = "microstructure"
    FULL_ANALYSIS = "full"

class InstrumentRequest(BaseModel):
    symbols: List[str]
    timeframe: TimeFrame = TimeFrame.DAILY
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    lookback_days: int = 365 * 2

class AnalysisRequest(BaseModel):
    instruments: InstrumentRequest
    analysis_types: List[AnalysisType] = [AnalysisType.FULL_ANALYSIS]
    parameters: Optional[Dict[str, Any]] = None

class CointegrationResult(BaseModel):
    pair: List[str]
    adf_statistic: float
    adf_pvalue: float
    kpss_statistic: float
    kpss_pvalue: float
    cointegrated: bool
    hedge_ratio: float
    half_life: Optional[float] = None
    hurst_exponent: Optional[float] = None

class PCAResult(BaseModel):
    eigenvalues: List[float]
    explained_variance: List[float]
    eigenportfolios: List[List[float]]
    principal_components: List[List[float]]

class VolatilityResult(BaseModel):
    symbol: str
    garch_volatility: List[float]
    realized_volatility: List[float]
    leverage_effect: Optional[float] = None
    vol_of_vol: Optional[float] = None

class RegimeResult(BaseModel):
    symbol: str
    regimes: Optional[List[int]] = None
    probabilities: Optional[List[List[float]]] = None
    regime_means: Optional[List[float]] = None
    regime_volatilities: Optional[List[float]] = None
    regime_statistics: Optional[Dict[str, Any]] = None

class TrendingResult(BaseModel):
    symbol: str
    fractional_d: float
    hurst_exponent: float
    trend_strength: float
    adf_statistic: float
    is_stationary: bool
    wavelet_analysis: Optional[Dict[str, Any]] = None
    kalman_trend: Optional[List[float]] = None

class MicrostructureResult(BaseModel):
    symbol: str
    order_flow_imbalance: Optional[List[float]] = None
    amihud_illiquidity: Optional[float] = None
    volatility_metrics: Dict[str, float]

class AnalysisResponse(BaseModel):
    timestamp: datetime
    instruments: List[str]
    cointegration: Optional[List[CointegrationResult]] = None
    pca: Optional[PCAResult] = None
    volatility: Optional[List[VolatilityResult]] = None
    regimes: Optional[List[RegimeResult]] = None
    trending: Optional[List[TrendingResult]] = None
    microstructure: Optional[List[MicrostructureResult]] = None
    execution_recommendations: Optional[Dict[str, Any]] = None