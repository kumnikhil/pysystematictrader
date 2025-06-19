from typing import List 
from src.utils.data_objs import Position
import numpy as np
from src.engine.oil import OilTradingEngine

import logging 

class TradingBot:
    """Automated trading strategies using the calibrated model"""
    
    def __init__(self, trading_engine: OilTradingEngine):
        self.engine = trading_engine
        self.logger = logging.getLogger('TradingBot')
        
    def volatility_arbitrage_strategy(self) -> List[Position]:
        """Identify volatility arbitrage opportunities"""
        opportunities = []
        
        try:
            # Get current forward curve
            forward_prices, maturities = self.engine.market_data.get_forward_curve()
            
            for i, maturity in enumerate(maturities[:6]):  # Focus on first 6 contracts
                if maturity < 0.02:  # Skip very short-term contracts
                    continue
                    
                # Calculate model implied volatility
                model_vol = self.engine.model.volatility_term_structure(maturity)
                
                # Estimate market implied volatility (simplified)
                # In production, would fetch from options market data
                F0 = forward_prices[i]
                historical_vol = self._estimate_historical_vol(maturity)
                
                # Compare model vs historical volatility
                vol_diff = model_vol - historical_vol
                
                if abs(vol_diff) > 0.05:  # 5% volatility difference threshold
                    # Create option position to capture vol difference
                    strike = F0  # ATM option
                    option_type = 'call' if vol_diff > 0 else 'put'
                    quantity = 100 if vol_diff > 0 else -100  # Long vol if model > historical
                    
                    option_price = self.engine.price_option(strike, maturity, option_type)
                    
                    position = Position(
                        instrument=f"WTI_{option_type}_{maturity:.2f}Y",
                        quantity=quantity,
                        entry_price=option_price,
                        maturity=maturity,
                        option_type=option_type,
                        strike=strike
                    )
                    
                    opportunities.append(position)
                    self.logger.info(f"Vol arb opportunity: {option_type} {maturity:.2f}Y, "
                                   f"Model vol: {model_vol:.1%}, Historical: {historical_vol:.1%}")
            
        except Exception as e:
            self.logger.error(f"Volatility arbitrage strategy failed: {e}")
            
        return opportunities
    
    def _estimate_historical_vol(self, maturity: float) -> float:
        """Estimate historical volatility for comparison"""
        # Simplified: use base volatility with some randomness
        # In production, would calculate from actual market data
        base_vol = 0.4
        maturity_adjustment = np.exp(-0.5 * maturity)  # Samuelson effect
        noise = np.random.normal(0, 0.05)  # Market noise
        
        return max(0.1, base_vol * maturity_adjustment + noise)
    
    def curve_trading_strategy(self) -> List[Position]:
        """Identify forward curve trading opportunities"""
        opportunities = []
        
        try:
            forward_prices, maturities = self.engine.market_data.get_forward_curve()
            
            if len(forward_prices) < 4:
                return opportunities
            
            # Look for curve anomalies
            for i in range(1, len(forward_prices) - 1):
                # Calculate expected price based on neighbors
                expected_price = (forward_prices[i-1] + forward_prices[i+1]) / 2
                actual_price = forward_prices[i]
                
                price_deviation = (actual_price - expected_price) / expected_price
                
                if abs(price_deviation) > 0.02:  # 2% deviation threshold
                    # Trade against the anomaly
                    quantity = -1000 if price_deviation > 0 else 1000  # Fade the move
                    
                    position = Position(
                        instrument=f"WTI_FUTURE_{maturities[i]:.2f}Y",
                        quantity=quantity,
                        entry_price=actual_price,
                        maturity=maturities[i]
                    )
                    
                    opportunities.append(position)
                    self.logger.info(f"Curve anomaly: {maturities[i]:.2f}Y contract "
                                   f"deviation: {price_deviation:.1%}")
            
        except Exception as e:
            self.logger.error(f"Curve trading strategy failed: {e}")
            
        return opportunities