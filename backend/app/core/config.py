from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Algorithmic Trading Analytics"
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Data Configuration
    DEFAULT_LOOKBACK_DAYS: int = 365 * 2  # 2 years
    DEFAULT_INTERVAL: str = "1d"
    
    # Instruments
    FOREX_PAIRS: List[str] = [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
        "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X"
    ]
    
    STOCKS: List[str] = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "TSLA", "NVDA", "JPM", "JNJ", "V", "PG",
        "UNH", "HD", "MA", "DIS", "ADBE", "CRM",
        "NFLX", "PYPL", "INTC", "AMD", "CSCO"
    ]
    
    # Analysis Parameters
    COINTEGRATION_TEST_LEVEL: float = 0.05
    PCA_COMPONENTS: int = 5
    FRACTIONAL_D_OPTIMAL: float = 0.4
    HMM_STATES: int = 3
    GARCH_MODEL: str = "GJR-GARCH"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()