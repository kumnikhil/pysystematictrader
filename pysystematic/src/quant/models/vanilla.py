from scipy import  optimize
from scipy.stats import norm
import numpy as np
from typing import Optional

class BlackScholesMerton:
    """
    Fixed Black-Scholes-Merton model for options pricing and implied volatility
    """
    @staticmethod
    def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes call option pricing"""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return call_price
    
    def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes put option pricing"""
        call_price =  BlackScholesMerton.black_scholes_call(S, K, T, r, sigma)
        put_price = call_price - S + K*np.exp(-r*T)
        return put_price
    
    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float, r: float, option_type: str = 'call') -> Optional[float]:
        """Calculate implied volatility using Brent's method"""
        
        def objective(sigma):
            if option_type.upper() in ['CALL','CE',"CALLS"]:
                bs_price =  BlackScholesMerton.black_scholes_call(S, K, T, r, sigma)
            else:
                bs_price =  BlackScholesMerton.black_scholes_put(S, K, T, r, sigma)
            return bs_price - market_price
        
        try:
            iv = optimize.brentq(objective, 0.001, 5.0, xtol=1e-6)
            return iv
        except (ValueError, RuntimeError):
            return None
                
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
        """Calculate option delta"""
        S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)
        
        if T <= 0 or sigma <= 0:
            if option_type.lower() == "call":
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        try:
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            
            if option_type.lower() == "call":
                return norm.cdf(d1)
            else:
                return norm.cdf(d1) - 1
        except:
            return 0.0
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option gamma"""
        S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)
        
        if T <= 0 or sigma <= 0:
            return 0.0
            
        try:
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            return norm.pdf(d1) / (S * sigma * np.sqrt(T))
        except:
            return 0.0
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
        """Calculate option theta"""
        S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)
        
        if T <= 0:
            return 0.0
            
        try:
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            theta_part1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            
            if option_type.lower() == "call":
                theta_part2 = -r * K * np.exp(-r*T) * norm.cdf(d2)
                return theta_part1 + theta_part2
            else:
                theta_part2 = r * K * np.exp(-r*T) * norm.cdf(-d2)
                return theta_part1 + theta_part2
        except:
            return 0.0
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega"""
        S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)
        
        if T <= 0:
            return 0.0
            
        try:
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            return S * norm.pdf(d1) * np.sqrt(T)
        except:
            return 0.0