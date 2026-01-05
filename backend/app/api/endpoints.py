from fastapi import APIRouter, HTTPException, Query, Body
from typing import List, Dict, Optional
import asyncio
from datetime import datetime
import logging
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

from app.models.schemas import (
    AnalysisRequest, AnalysisResponse, InstrumentRequest,
    CointegrationResult, PCAResult, VolatilityResult,
    RegimeResult, TrendingResult, MicrostructureResult
)
from app.services.data_fetcher import DataFetcher
from app.services.econometrics import EconometricAnalyzer
from app.services.signal_processing import SignalProcessor
from app.services.volatility import VolatilityAnalyzer
from app.services.machine_learning import MachineLearningAnalyzer
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
data_fetcher = DataFetcher()
econometrics = EconometricAnalyzer()
signal_processor = SignalProcessor()
volatility_analyzer = VolatilityAnalyzer()
ml_analyzer = MachineLearningAnalyzer()

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [convert_numpy(i) for i in obj.tolist()]
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_market(
    request: AnalysisRequest = Body(...)
):
    """
    Perform comprehensive market analysis on selected instruments
    """
    try:
        logger.info(f"Starting analysis for {len(request.instruments.symbols)} instruments")
        
        # Fetch data
        data = await data_fetcher.fetch_data(
            symbols=request.instruments.symbols,
            timeframe=request.instruments.timeframe.value,
            lookback_days=request.instruments.lookback_days
        )
        
        if not data:
            raise HTTPException(status_code=404, detail="No data fetched for the given symbols")
        
        # Initialize results
        results = {
            'timestamp': datetime.utcnow(),
            'instruments': list(data.keys()),
            'cointegration': [],
            'volatility': [],
            'regimes': [],
            'trending': [],
            'microstructure': []
        }
        
        # Extract returns for analysis
        returns_data = {}
        price_data = {}
        for symbol, df in data.items():
            if 'Returns' in df.columns:
                returns_data[symbol] = df['Returns'].dropna()
                price_data[symbol] = df['Close']
        
        # Perform requested analyses
        if 'cointegration' in request.analysis_types or 'full' in request.analysis_types:
            results['cointegration'] = await perform_cointegration_analysis(price_data)
        
        if 'pca' in request.analysis_types or 'full' in request.analysis_types:
            results['pca'] = await perform_pca_analysis(returns_data)
        
        if 'volatility' in request.analysis_types or 'full' in request.analysis_types:
            results['volatility'] = await perform_volatility_analysis(returns_data)
        
        if 'regime' in request.analysis_types or 'full' in request.analysis_types:
            results['regimes'] = await perform_regime_analysis(returns_data)
        
        if 'trending' in request.analysis_types or 'full' in request.analysis_types:
            results['trending'] = await perform_trending_analysis(price_data, returns_data)
        
        if 'microstructure' in request.analysis_types or 'full' in request.analysis_types:
            results['microstructure'] = await perform_microstructure_analysis(data)
        
        # Generate execution recommendations
        results['execution_recommendations'] = generate_execution_recommendations(results)
        
        # Convert all numpy types to native python types for serialization
        clean_results = convert_numpy(results)
        
        return AnalysisResponse(**clean_results)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def perform_cointegration_analysis(price_data: Dict) -> List[Dict]:
    """Perform cointegration analysis on all pairs"""
    cointegration_results = []
    symbols = list(price_data.keys())
    
    # Analyze all unique pairs
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            try:
                result = econometrics.calculate_cointegration(
                    price_data[symbols[i]],
                    price_data[symbols[j]]
                )
                
                if result['cointegrated']:
                    cointegration_results.append({
                        'pair': [symbols[i], symbols[j]],
                        **result
                    })
            except Exception as e:
                logger.error(f"Cointegration error for {symbols[i]}-{symbols[j]}: {str(e)}")
                continue
    
    return cointegration_results

async def perform_pca_analysis(returns_data: Dict) -> Optional[Dict]:
    """Perform PCA analysis on returns"""
    # Create returns matrix
    returns_df = pd.DataFrame(returns_data).dropna()
    
    if len(returns_df) < 10 or len(returns_df.columns) < 2:  # Reduced min columns for more results
        return None
    
    pca_result = econometrics.perform_pca_analysis(
        returns_df,
        n_components=min(5, len(returns_df.columns))
    )
    
    return pca_result

async def perform_volatility_analysis(returns_data: Dict) -> List[Dict]:
    """Perform volatility analysis for each instrument"""
    volatility_results = []
    
    for symbol, returns in returns_data.items():
        try:
            # Calculate realized volatility
            realized_vol = volatility_analyzer.calculate_realized_volatility(returns)
            
            # Fit GARCH model
            garch_result = volatility_analyzer.estimate_garch_models(
                returns,
                model_type=settings.GARCH_MODEL
            )
            
            # Calculate volatility surface metrics
            vol_metrics = volatility_analyzer.calculate_volatility_surface_metrics(returns)
            
            volatility_results.append({
                **vol_metrics,
                'symbol': symbol,
                'garch_volatility': garch_result.get('conditional_volatility', []),
                'realized_volatility': realized_vol.dropna().tolist(),
                'leverage_effect': garch_result.get('leverage_effect', 0),
                'vol_of_vol': garch_result.get('volatility_of_vol', 0),
            })
        except Exception as e:
            logger.error(f"Volatility analysis error for {symbol}: {str(e)}")
            continue
    
    return volatility_results

async def perform_regime_analysis(returns_data: Dict) -> List[Dict]:
    """Perform regime detection analysis"""
    regime_results = []
    
    for symbol, returns in returns_data.items():
        try:
            # HMM regime detection
            hmm_result = ml_analyzer.hidden_markov_model_regime(returns)
            
            # Volatility regime detection
            volatility = volatility_analyzer.calculate_realized_volatility(returns, annualize=False)
            vol_regime_result = volatility_analyzer.calculate_volatility_regime(volatility)
            
            regime_results.append({
                'symbol': symbol,
                **hmm_result,
                'volatility_regimes': vol_regime_result.get('regimes', []),
                'volatility_regime_stats': vol_regime_result.get('regime_stats', {})
            })
        except Exception as e:
            logger.error(f"Regime analysis error for {symbol}: {str(e)}")
            continue
    
    return regime_results

async def perform_trending_analysis(price_data: Dict, returns_data: Dict) -> List[Dict]:
    """Perform trending analysis"""
    trending_results = []
    
    for symbol in price_data.keys():
        try:
            prices = price_data[symbol]
            returns = returns_data.get(symbol, pd.Series())
            
            if prices.empty:
                continue
            
            # Calculate fractional differentiation
            frac_diff_result = signal_processor.find_optimal_fractional_d(prices)
            
            # Calculate Hurst exponent
            hurst = econometrics.calculate_hurst_exponent(prices.values)
            
            # Calculate trend strength
            trend_strength = signal_processor.calculate_trend_strength(prices)
            
            # Kalman filter trend
            kalman_result = signal_processor.kalman_filter_trend(prices)
            
            # Wavelet analysis
            wavelet_result = signal_processor.wavelet_transform(prices)
            
            # Stationarity test
            adf_result = adfuller(prices.dropna())
            
            trending_results.append({
                'symbol': symbol,
                'fractional_d': frac_diff_result.get('optimal_d', 0.5),
                'hurst_exponent': hurst,
                'trend_strength': float(trend_strength.mean()),
                'adf_statistic': adf_result[0],
                'is_stationary': adf_result[1] < 0.05,
                'wavelet_analysis': wavelet_result,
                'kalman_trend': kalman_result.get('filtered_trend', [])
            })
        except Exception as e:
            logger.error(f"Trending analysis error for {symbol}: {str(e)}")
            continue
    
    return trending_results

async def perform_microstructure_analysis(data: Dict) -> List[Dict]:
    """Perform microstructure analysis"""
    microstructure_results = []
    
    for symbol, df in data.items():
        try:
            if 'Volume' not in df.columns or 'Close' not in df.columns:
                continue
            
            # Calculate Amihud illiquidity ratio
            returns = df['Returns'].abs() if 'Returns' in df.columns else df['Close'].pct_change().abs()
            dollar_volume = df['Volume'] * df['Close']
            amihud_ratio = (returns / dollar_volume).mean()
            
            # Calculate volatility metrics
            volatility_metrics = volatility_analyzer.calculate_volatility_surface_metrics(
                df['Close'].pct_change() if 'Returns' not in df.columns else df['Returns']
            )
            
            microstructure_results.append({
                'symbol': symbol,
                'amihud_illiquidity': float(amihud_ratio) if not pd.isna(amihud_ratio) else 0,
                'volume_profile': {
                    'mean_volume': float(df['Volume'].mean()),
                    'volume_std': float(df['Volume'].std()),
                    'volume_skew': float(df['Volume'].skew())
                },
                'volatility_metrics': volatility_metrics
            })
        except Exception as e:
            logger.error(f"Microstructure analysis error for {symbol}: {str(e)}")
            continue
    
    return microstructure_results

def generate_execution_recommendations(results: Dict) -> Dict:
    """Generate execution recommendations based on analysis results"""
    recommendations = {
        'market_regime': 'neutral',
        'volatility_environment': 'medium',
        'recommended_strategies': [],
        'risk_warnings': [],
        'position_sizing': 'normal',
        'execution_timing': 'normal'
    }
    
    # Analyze regime information
    if results.get('regimes'):
        # Determine overall market regime
        regime_counts = {}
        for regime_result in results['regimes']:
            if 'hidden_states' in regime_result:
                # Find most common regime
                if regime_result['hidden_states']:
                    most_common = max(set(regime_result['hidden_states']), 
                                     key=regime_result['hidden_states'].count)
                    regime_counts[most_common] = regime_counts.get(most_common, 0) + 1
        
        if regime_counts:
            overall_regime = max(regime_counts, key=regime_counts.get)
            recommendations['market_regime'] = f'regime_{overall_regime}'
    
    # Analyze volatility
    if results.get('volatility'):
        avg_vol = np.mean([v.get('realized_volatility', 0) for v in results['volatility'] 
                          if isinstance(v.get('realized_volatility'), (int, float))])
        
        if avg_vol > 0.3:
            recommendations['volatility_environment'] = 'high'
            recommendations['position_sizing'] = 'reduced'
            recommendations['risk_warnings'].append('High volatility environment')
        elif avg_vol < 0.15:
            recommendations['volatility_environment'] = 'low'
            recommendations['execution_timing'] = 'aggressive'
    
    # Generate strategy recommendations
    if results.get('cointegration'):
        recommendations['recommended_strategies'].append('pairs_trading')
    
    if results.get('trending'):
        strong_trends = [t for t in results['trending'] 
                        if t.get('trend_strength', 0) > 0.7]
        if strong_trends:
            recommendations['recommended_strategies'].append('trend_following')
    
    if results.get('regimes'):
        recommendations['recommended_strategies'].append('regime_switching')
    
    return recommendations

@router.get("/instruments")
async def get_available_instruments():
    """Get list of available instruments"""
    return {
        "forex": settings.FOREX_PAIRS,
        "stocks": settings.STOCKS,
        "total_count": len(settings.FOREX_PAIRS) + len(settings.STOCKS)
    }

@router.get("/market-status")
async def get_market_status():
    """Get current market status and indicators"""
    try:
        market_data = await data_fetcher.fetch_market_data()
        
        # Calculate market breadth if we have stock data
        stock_data = await data_fetcher.fetch_data(
            symbols=settings.STOCKS[:20],  # Sample of stocks
            timeframe="1d",
            lookback_days=5
        )
        
        advancing = 0
        declining = 0
        for symbol, df in stock_data.items():
            if len(df) > 1:
                if df['Close'].iloc[-1] > df['Close'].iloc[-2]:
                    advancing += 1
                else:
                    declining += 1
        
        return {
            "timestamp": datetime.utcnow(),
            "market_indicators": market_data,
            "market_breadth": {
                "advancing": advancing,
                "declining": declining,
                "advance_decline_ratio": advancing / max(declining, 1)
            },
            "recommended_actions": {
                "risk_on": advancing > declining * 1.5,
                "caution_advised": advancing < declining
            }
        }
    except Exception as e:
        logger.error(f"Market status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quick-analysis/{symbol}")
async def quick_analysis(
    symbol: str,
    timeframe: str = "1d",
    lookback_days: int = 365
):
    """Quick analysis for a single symbol"""
    try:
        # Fetch data
        data = await data_fetcher.fetch_data(
            symbols=[symbol],
            timeframe=timeframe,
            lookback_days=lookback_days
        )
        
        if symbol not in data:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        df = data[symbol]
        
        # Calculate basic metrics
        returns = df['Returns'].dropna() if 'Returns' in df.columns else df['Close'].pct_change().dropna()
        
        basic_metrics = {
            "current_price": float(df['Close'].iloc[-1]),
            "daily_return": float(returns.iloc[-1] if len(returns) > 0 else 0),
            "volatility": float(returns.std() * np.sqrt(252)),
            "sharpe_ratio": float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            "max_drawdown": calculate_max_drawdown(df['Close']),
            "volume_trend": "increasing" if df['Volume'].iloc[-1] > df['Volume'].rolling(20).mean().iloc[-1] else "decreasing"
        }
        
        # Trend analysis
        trend_strength = signal_processor.calculate_trend_strength(df['Close'])
        basic_metrics["trend_strength"] = float(trend_strength.iloc[-1]) if len(trend_strength) > 0 else 0
        
        # Volatility regime
        realized_vol = volatility_analyzer.calculate_realized_volatility(returns, annualize=False)
        vol_regime = volatility_analyzer.calculate_volatility_regime(realized_vol)
        
        return {
            "symbol": symbol,
            "timestamp": datetime.utcnow(),
            "basic_metrics": basic_metrics,
            "volatility_regime": vol_regime.get('regime_stats', {}),
            "technical_indicators": {
                "rsi": float(df['RSI'].iloc[-1]) if 'RSI' in df.columns else None,
                "macd": float(df['MACD'].iloc[-1]) if 'MACD' in df.columns else None,
                "bollinger_position": calculate_bollinger_position(df)
            }
        }
    except Exception as e:
        logger.error(f"Quick analysis error for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown"""
    cumulative_returns = (prices / prices.iloc[0]) - 1
    running_max = cumulative_returns.expanding().max()
    drawdown = cumulative_returns - running_max
    max_drawdown = drawdown.min()
    return float(max_drawdown) if not pd.isna(max_drawdown) else 0

def calculate_bollinger_position(df: pd.DataFrame) -> str:
    """Calculate position within Bollinger Bands"""
    if 'Close' not in df.columns or 'BB_Upper' not in df.columns or 'BB_Lower' not in df.columns:
        return "unknown"
    
    current_price = df['Close'].iloc[-1]
    bb_upper = df['BB_Upper'].iloc[-1]
    bb_lower = df['BB_Lower'].iloc[-1]
    bb_middle = df['BB_Middle'].iloc[-1]
    
    bb_width = bb_upper - bb_lower
    
    if bb_width == 0:
        return "neutral"
    
    position = (current_price - bb_lower) / bb_width
    
    if position > 0.8:
        return "overbought"
    elif position < 0.2:
        return "oversold"
    else:
        return "neutral"