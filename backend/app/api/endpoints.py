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
from app.services.microstructure import MicrostructureAnalyzer
from app.services.information_theory import InformationAnalyzer
from app.services.execution import ExecutionOptimizer
from app.services.technical_analysis import TechnicalAnalyzer
from app.services.strategy import PositionAdvisor
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
data_fetcher = DataFetcher()
econometrics = EconometricAnalyzer()
signal_processor = SignalProcessor()
volatility_analyzer = VolatilityAnalyzer()
ml_analyzer = MachineLearningAnalyzer()
micro_analyzer = MicrostructureAnalyzer()
info_analyzer = InformationAnalyzer()
exec_optimizer = ExecutionOptimizer()
ta_analyzer = TechnicalAnalyzer()
position_advisor = PositionAdvisor()

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
            'microstructure': [],
            'position_advice': []
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

        # Generate Position Advice
        for symbol in data.keys():
            try:
                df = data[symbol]
                ta = ta_analyzer.calculate_all_indicators(df)
                micro = next((m for m in results['microstructure'] if m['symbol'] == symbol), {})
                trend = next((t for t in results['trending'] if t['symbol'] == symbol), {})
                coint = next((c for c in results['cointegration'] if symbol in c['pair']), {})
                
                advice = position_advisor.generate_advice(
                    symbol=symbol,
                    current_price=float(df['Close'].iloc[-1]),
                    atr=ta.get('atrr_14', 0.0) or 0.0,
                    trend_strength=trend.get('trend_strength', 0.5),
                    vpin=micro.get('vpin_toxicity', 0.2),
                    rsi=ta.get('rsi_14', 50.0),
                    half_life=coint.get('half_life'),
                    fib_levels=ta.get('fibonacci_levels')
                )
                results['position_advice'].append(advice)
            except Exception as e:
                logger.error(f"Advice error for {symbol}: {str(e)}")
        
        # Lead-Lag Analysis
        symbols = list(returns_data.keys())
        lead_lag_data = {}
        if len(symbols) >= 2:
            te = info_analyzer.calculate_transfer_entropy(returns_data[symbols[0]], returns_data[symbols[1]])
            lead_lag_data = {
                'lead_lag_score': float(te),
                'dominant_instrument': symbols[0] if te > 0 else symbols[1]
            }
        else:
            lead_lag_data = {
                'lead_lag_score': 0.0,
                'dominant_instrument': symbols[0] if symbols else 'NONE'
            }
        
        # Execution Optimization (Almgren-Chriss & Avellaneda-Stoikov)
        execution_data = perform_execution_optimization(price_data, returns_data)
        
        # Generate execution recommendations
        exec_recs = generate_execution_recommendations(results)
        results['execution_recommendations'] = {**lead_lag_data, **execution_data, **exec_recs}
        
        # Convert all numpy types to native python types for serialization
        clean_results = convert_numpy(results)
        
        return AnalysisResponse(**clean_results)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def perform_cointegration_analysis(price_data: Dict) -> List[Dict]:
    cointegration_results = []
    symbols = list(price_data.keys())
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            try:
                result = econometrics.calculate_cointegration(price_data[symbols[i]], price_data[symbols[j]])
                if result.get('cointegrated'):
                    spread = price_data[symbols[j]] - result['hedge_ratio'] * price_data[symbols[i]]
                    ou_params = econometrics.estimate_ou_process(spread)
                    copula_results = econometrics.calculate_copula_dependence(
                        price_data[symbols[i]].pct_change().dropna(),
                        price_data[symbols[j]].pct_change().dropna()
                    )
                    cointegration_results.append({'pair': [symbols[i], symbols[j]], **result, **ou_params, **copula_results})
            except Exception as e:
                logger.error(f"Cointegration error for {symbols[i]}-{symbols[j]}: {str(e)}")
    return cointegration_results

async def perform_pca_analysis(returns_data: Dict) -> Optional[Dict]:
    returns_df = pd.DataFrame(returns_data).dropna()
    if len(returns_df) < 10 or len(returns_df.columns) < 2: return None
    return econometrics.perform_pca_analysis(returns_df, n_components=min(5, len(returns_df.columns)))

async def perform_volatility_analysis(returns_data: Dict) -> List[Dict]:
    volatility_results = []
    for symbol, returns in returns_data.items():
        try:
            realized_vol = volatility_analyzer.calculate_realized_volatility(returns)
            garch_result = volatility_analyzer.estimate_garch_models(returns, model_type=settings.GARCH_MODEL)
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
    return volatility_results

async def perform_regime_analysis(returns_data: Dict) -> List[Dict]:
    regime_results = []
    for symbol, returns in returns_data.items():
        try:
            hmm_result = ml_analyzer.hidden_markov_model_regime(returns)
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
    return regime_results

async def perform_trending_analysis(price_data: Dict, returns_data: Dict) -> List[Dict]:
    trending_results = []
    for symbol in price_data.keys():
        try:
            prices = price_data[symbol]
            if prices.empty: continue
            frac_diff_result = signal_processor.find_optimal_fractional_d(prices)
            hurst = econometrics.calculate_hurst_exponent(prices.values)
            trend_strength = signal_processor.calculate_trend_strength(prices)
            spectral_metrics = signal_processor.calculate_instantaneous_metrics(prices)
            adf_result = adfuller(prices.dropna())
            trending_results.append({
                'symbol': symbol,
                'fractional_d': frac_diff_result.get('optimal_d', 0.5),
                'hurst_exponent': hurst,
                'trend_strength': float(trend_strength.mean()),
                'is_stationary': adf_result[1] < 0.05,
                'spectral_metrics': spectral_metrics
            })
        except Exception as e:
            logger.error(f"Trending analysis error for {symbol}: {str(e)}")
    return trending_results

async def perform_microstructure_analysis(data: Dict) -> List[Dict]:
    microstructure_results = []
    for symbol, df in data.items():
        try:
            if 'Volume' not in df.columns: continue
            returns = df['Returns'] if 'Returns' in df.columns else df['Close'].pct_change()
            vpin = micro_analyzer.calculate_vpin(df)
            k_lambda = micro_analyzer.calculate_kyles_lambda(returns, df['Volume'])
            amihud = micro_analyzer.calculate_amihud_illiquidity(returns, df['Volume'] * df['Close'])
            pin = micro_analyzer.estimate_pin_mle(df)
            ofi = micro_analyzer.calculate_ofi_proxy(df)
            volatility_metrics = volatility_analyzer.calculate_volatility_surface_metrics(returns)
            microstructure_results.append({
                'symbol': symbol,
                'vpin_toxicity': vpin,
                'kyles_lambda': k_lambda,
                'amihud_illiquidity': amihud,
                'pin_mle': pin,
                'ofi_proxy': ofi,
                'volatility_metrics': volatility_metrics
            })
        except Exception as e:
            logger.error(f"Microstructure analysis error for {symbol}: {str(e)}")
    return microstructure_results

def perform_execution_optimization(price_data: Dict, returns_data: Dict) -> Dict:
    results = {}
    for symbol in price_data.keys():
        try:
            curr_price = float(price_data[symbol].iloc[-1])
            vol = float(returns_data[symbol].std() * np.sqrt(252))
            # Almgren-Chriss Trajectory for 100,000 shares over 5 intervals
            trajectory = exec_optimizer.calculate_almgren_chriss_trajectory(100000, 5, vol)
            # Avellaneda-Stoikov Reservation Price
            res_price = exec_optimizer.calculate_avellaneda_stoikov_reservation(curr_price, 100, vol)
            results[f'{symbol}_execution'] = {
                'liquidation_trajectory': trajectory,
                'reservation_price': res_price
            }
        except: continue
    return results

def generate_execution_recommendations(results: Dict) -> Dict:
    recommendations = {
        'market_regime': 'neutral',
        'volatility_environment': 'medium',
        'recommended_strategies': [],
        'risk_warnings': [],
        'position_sizing': 'normal',
        'execution_timing': 'normal'
    }
    if results.get('volatility'):
        avg_vol = np.mean([v.get('current_realized_volatility', 0) for v in results['volatility'] if isinstance(v.get('current_realized_volatility'), (int, float))])
        if avg_vol > 0.3:
            recommendations['volatility_environment'] = 'high'
            recommendations['risk_warnings'].append('High volatility environment')
        elif avg_vol < 0.15:
            recommendations['volatility_environment'] = 'low'
    return recommendations

@router.get("/technical-suite/{symbol}")
async def get_technical_suite(symbol: str, timeframe: str = "1d", lookback_days: int = 365):
    """
    Returns 100+ technical indicators, momentum oscillators, and volume metrics.
    """
    try:
        data = await data_fetcher.fetch_data(symbols=[symbol], timeframe=timeframe, lookback_days=lookback_days)
        if symbol not in data:
            raise HTTPException(status_code=404, detail="Symbol not found")
        
        indicators = ta_analyzer.calculate_all_indicators(data[symbol])
        return {"symbol": symbol, "timestamp": datetime.utcnow(), "indicators": indicators}
    except Exception as e:
        logger.error(f"Suite error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quick-analysis/{symbol}")
async def quick_analysis(symbol: str, timeframe: str = "1d", lookback_days: int = 365):
    try:
        data = await data_fetcher.fetch_data(symbols=[symbol], timeframe=timeframe, lookback_days=lookback_days)
        if symbol not in data: raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        df = data[symbol]
        returns = df['Returns'].dropna() if 'Returns' in df.columns else df['Close'].pct_change().dropna()
        
        # Comprehensive TA for the quick view
        ta_data = ta_analyzer.calculate_all_indicators(df)
        
        basic_metrics = {
            "current_price": float(df['Close'].iloc[-1]), 
            "volatility": float(returns.std() * np.sqrt(252)), 
            "trend_strength": float(signal_processor.calculate_trend_strength(df['Close']).iloc[-1]),
            "rsi": ta_data.get('rsi_14'),
            "macd": ta_data.get('macdh_12_26_9')
        }
        return {
            "symbol": symbol, 
            "timestamp": datetime.utcnow(), 
            "basic_metrics": basic_metrics,
            "core_indicators": ta_data
        }
    except Exception as e:
        logger.error(f"Quick analysis error for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/instruments")
async def get_available_instruments():
    return {"forex": settings.FOREX_PAIRS, "stocks": settings.STOCKS, "total_count": len(settings.FOREX_PAIRS) + len(settings.STOCKS)}

@router.get("/market-status")
async def get_market_status():
    try:
        market_data = await data_fetcher.fetch_market_data()
        return {"timestamp": datetime.utcnow(), "market_indicators": market_data}
    except Exception as e:
        logger.error(f"Market status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))