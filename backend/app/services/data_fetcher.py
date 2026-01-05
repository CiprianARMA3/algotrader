import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self):
        self.cache = {}
        
    async def fetch_data(
        self,
        symbols: List[str],
        timeframe: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_days: int = 365 * 2
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols
        """
        data_dict = {}
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
                
                # Clean symbol for yfinance
                clean_symbol = symbol.replace("=X", "")
                if "=X" in symbol:
                    ticker = yf.Ticker(f"{clean_symbol}=X")
                else:
                    ticker = yf.Ticker(symbol)
                
                # Fetch data
                hist = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=timeframe,
                    auto_adjust=True
                )
                
                if hist.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue

                # --- Professional Data Cleaning Layer ---
                # 1. Fill missing values using linear interpolation (standard institutional practice)
                hist = hist.interpolate(method='linear', limit_direction='both')
                
                # 2. Outlier Detection (Z-Score > 4) to prevent "Flash Crash" data errors from skewing models
                for col in ['Close', 'Open', 'High', 'Low']:
                    z_scores = (hist[col] - hist[col].mean()) / (hist[col].std() + 1e-9)
                    hist.loc[z_scores.abs() > 4, col] = np.nan
                    hist[col] = hist[col].fillna(method='ffill').fillna(method='bfill')

                # 3. Accurate return calculation
                hist['Returns'] = hist['Close'].pct_change().fillna(0)
                hist['Log_Returns'] = np.log(hist['Close'] / hist['Close'].shift(1)).fillna(0)
                hist['Volume'] = hist['Volume'].fillna(0)
                
                data_dict[symbol] = hist
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        return data_dict
    
    async def fetch_market_data(self) -> Dict[str, any]:
        """
        Fetch additional market data (VIX, treasury yields, etc.)
        """
        market_indicators = {
            "VIX": "^VIX",
            "SP500": "^GSPC",
            "DXY": "DX-Y.NYB",  # US Dollar Index
            "10Y_Yield": "^TNX",
            "2Y_Yield": "^IRX"
        }
        
        market_data = {}
        for name, symbol in market_indicators.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1mo", interval="1d")
                if not hist.empty:
                    market_data[name] = {
                        "current": hist['Close'].iloc[-1],
                        "change": hist['Close'].pct_change().iloc[-1] * 100
                    }
            except Exception as e:
                logger.error(f"Error fetching {name}: {str(e)}")
        
        return market_data
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various technical indicators
        """
        # Price-based indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        return df