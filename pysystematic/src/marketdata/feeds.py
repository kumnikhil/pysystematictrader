import datetime as dt
import numpy as np 
from typing import Optional, Tuple
import logging

from src.marketdata.yfinance import WTI

class MarketDataFeed:
    """Simulated real-time market data feed"""
    
    def __init__(self):
        self.wti = WTI()
        self.last_update = None
        
    def get_forward_curve(self, date: Optional[dt.datetime] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch current WTI forward curve"""
        try:
            if date is None:
                end_date = dt.datetime.now()
            else:
                end_date = date
                
            start_date = end_date - dt.timedelta(days=1)
            
            hist, ric_list = self.wti.forward_curve_history(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                forward_maturity=12
            )
            
            close_prices = self.wti.close_prices(hist, ric_list)
            
            if close_prices is not None and not close_prices.empty:
                # Use most recent data
                latest_prices = close_prices.iloc[-1].values
                maturities = np.arange(1, len(latest_prices) + 1) / 12  # Convert months to years
                
                # Clean data (remove NaN/invalid prices)
                valid_mask = ~np.isnan(latest_prices) & (latest_prices > 0)
                prices = latest_prices[valid_mask]
                mats = maturities[valid_mask]
                
                return prices, mats
            else:
                raise ValueError("No market data available")
                
        except Exception as e:
            logging.warning(f"Market data fetch failed: {e}. Using synthetic data.")
            # Fallback to synthetic curve
            maturities = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 12/12, 18/12, 24/12])
            # Create realistic oil forward curve (contango)
            base_price = 75.0
            curve_slope = 0.5  # $/year
            prices = base_price + curve_slope * maturities + np.random.normal(0, 0.2, len(maturities))
            return prices, maturities
    
    def get_current_price(self, maturity: float) -> float:
        """Get current price for specific maturity"""
        prices, maturities = self.get_forward_curve()
        return np.interp(maturity, maturities, prices)