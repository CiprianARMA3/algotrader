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
    adf_statistic: Optional[float] = None
    cointegrated: Optional[bool] = None
    hedge_ratio: Optional[float] = None
    half_life: Optional[float] = None
    kappa_reversion_speed: Optional[float] = None
    equilibrium_mu: Optional[float] = None
    kendall_tau: Optional[float] = None
    lower_tail_dependence: Optional[float] = None
    upper_tail_dependence: Optional[float] = None

class PCAResult(BaseModel):
    eigenvalues: Optional[List[float]] = None
    explained_variance: Optional[List[float]] = None
    eigenportfolios: Optional[List[List[float]]] = None
    principal_components: Optional[List[List[float]]] = None

class VolatilityResult(BaseModel):
    symbol: str
    garch_volatility: Optional[List[float]] = Field(default_factory=list)
    realized_volatility: Optional[List[float]] = Field(default_factory=list)
    short_term_vol: Optional[float] = None
    long_term_vol: Optional[float] = None
    term_structure: Optional[float] = None
    volatility_skew: Optional[float] = None
    jump_intensity: Optional[float] = None
    roughness_exponent: Optional[float] = None

class RegimeResult(BaseModel):
    symbol: str
    regime_statistics: Optional[Dict[str, Any]] = None
    volatility_regime_stats: Optional[Dict[str, Any]] = None

class TrendingResult(BaseModel):
    symbol: str
    fractional_d: Optional[float] = None
    hurst_exponent: Optional[float] = None
    trend_strength: Optional[float] = None
    is_stationary: Optional[bool] = None
    spectral_metrics: Optional[Dict[str, Any]] = None

class MicrostructureResult(BaseModel):
    symbol: str
    vpin_toxicity: Optional[float] = None
    kyles_lambda: Optional[float] = None
    amihud_illiquidity: Optional[float] = None
    pin_mle: Optional[float] = None
    ofi_proxy: Optional[float] = None
    volatility_metrics: Optional[Dict[str, Any]] = None

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