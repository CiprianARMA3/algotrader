import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class PositionAdvisor:
    def __init__(self):
        pass

    def generate_advice(
        self, 
        symbol: str, 
        current_price: float, 
        atr: float, 
        trend_strength: float, 
        vpin: float, 
        rsi: float,
        half_life: float = None,
        fib_levels: Dict = None
    ) -> Dict[str, Any]:
        """
        Synthesizes multiple institutional indicators to provide specific position advice.
        """
        
        # 1. Calculate Accuracy / Confidence Score (0-100%)
        # Weights: Trend(40%), Toxicity(30%), Momentum(30%)
        confidence = 0.0
        
        # Trend component
        confidence += min(trend_strength * 40, 40)
        
        # Microstructure component (lower VPIN = higher confidence in stability)
        toxicity_penalty = min(vpin * 100, 30)
        confidence += (30 - toxicity_penalty)
        
        # Momentum component (RSI extremes provide reversal confidence)
        if rsi > 70 or rsi < 30:
            confidence += 30
        else:
            confidence += 15
            
        # 2. Determine Signal Direction
        direction = "NEUTRAL"
        if trend_strength > 0.6:
            direction = "LONG" if rsi < 60 else "NEUTRAL"
        elif trend_strength < 0.3 and rsi < 30:
            direction = "LONG_REVERSAL"
        elif rsi > 70:
            direction = "SHORT_REVERSAL"

        # 3. TP / SL Calculation (Volatility Based)
        # Institutional standard: SL = 2 * ATR, TP = 3-4 * ATR
        stop_loss = 0.0
        take_profit = 0.0
        
        if direction.startswith("LONG"):
            stop_loss = current_price - (2.0 * atr)
            take_profit = current_price + (3.5 * atr)
            # Refine TP with Fibonacci if available
            if fib_levels and fib_levels.get('ext_1618'):
                take_profit = (take_profit + fib_levels['ext_1618']) / 2
        elif direction.startswith("SHORT"):
            stop_loss = current_price + (2.0 * atr)
            take_profit = current_price - (3.5 * atr)

        # 4. Hold Duration (Based on OU Process Half-Life)
        # If no half-life provided, use trend-based estimate
        suggested_hold = half_life if half_life else (10 if trend_strength > 0.7 else 5)

        return {
            "symbol": symbol,
            "signal": direction,
            "confidence_pct": float(round(confidence, 2)),
            "entry_price": float(current_price),
            "take_profit": float(round(take_profit, 4)) if take_profit > 0 else None,
            "stop_loss": float(round(stop_loss, 4)) if stop_loss > 0 else None,
            "risk_reward_ratio": float(round((take_profit - current_price)/(current_price - stop_loss), 2)) if stop_loss and stop_loss != current_price else 0,
            "estimated_hold_days": float(round(suggested_hold, 1))
        }
