import numpy as np
from typing import Optional
from scipy import stats

class BlacksModel:
    @staticmethod
    def d1(S:float, K:float, T:float, r:float, sigma:float)-> float:
        return (np.log(S/K)+0.5*sigma**2*T)/(sigma*np.sqrt(T))
    
    @staticmethod
    def d2(S:float, K:float, T:float, r:float, sigma:float)-> float:
        return BlacksModel.d1(S, K, T, r, sigma) - sigma*np.sqrt(T)
    
    @staticmethod
    def black76_call(S:float, K:float, T:float, r:float, sigma:float) -> float:
        """Black-76 call option pricing"""
        d1 = BlacksModel.d1(S, K, T, r, sigma)
        d2 = BlacksModel.d2(S, K, T, r, sigma)
        return np.exp(-r*T) * (S*stats.norm.cdf(d1) - K*stats.norm.cdf(d2))
    
    @staticmethod
    def black76_put(S:float, K:float, T:float, r:float, sigma:float) -> float:
        """Black-76 put option pricing"""
        d1 = BlacksModel.d1(S, K, T, r, sigma)
        d2 = BlacksModel.d2(S, K, T, r, sigma)
        return np.exp(-r*T) * (K*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1))
    
    @staticmethod
    def implied_volatility(option_price:float, S:float, K:float, T:float, r:float, option_type:str='call', tol:float=1e-6, max_iterations:int=100) -> Optional[float]:
        """Calculate implied volatility using Newton-Raphson method"""
        sigma = 0.2  # Initial guess
        def objective_function(sigma: float) -> float:
            if option_type == 'call':
                return BlacksModel.black76_call(S, K, T, r, sigma) - option_price
            else:
                return BlacksModel.black76_put(S, K, T, r, sigma) - option_price

        for _ in range(max_iterations):
            f_value = objective_function(sigma)
            if abs(f_value) < tol:
                return sigma
            # Numerical derivative
            dfdx = (objective_function(sigma + tol) - f_value) / tol
            sigma -= f_value / dfdx

        return None
    
    @staticmethod
    def delta(S:float, K:float, T:float, r:float, sigma:float, option_type:str='call') -> float:
        """Calculate the delta of a Black-76 option"""
        d1 = BlacksModel.d1(S, K, T, r, sigma)
        if option_type == 'call':
            return np.exp(-r*T) * stats.norm.cdf(d1)
        else:
            return np.exp(-r*T) * (stats.norm.cdf(d1) - 1)
        
    @staticmethod
    def gamma(S:float, K:float, T:float, r:float, sigma:float) -> float:
        """Calculate the gamma of a Black-76 option"""
        d1 = BlacksModel.d1(S, K, T, r, sigma)
        return (np.exp(-r*T) * stats.norm.pdf(d1)) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def vega(S:float, K:float, T:float, r:float, sigma:float) -> float:
        """Calculate the vega of a Black-76 option"""
        d1 = BlacksModel.d1(S, K, T, r, sigma)
        return S * np.exp(-r*T) * stats.norm.pdf(d1) * np.sqrt(T)
    
    @staticmethod
    def theta(S:float, K:float, T:float, r:float, sigma:float, option_type:str='call') -> float:
        """Calculate the theta of a Black-76 option"""
        d1 = BlacksModel.d1(S, K, T, r, sigma)
        d2 = BlacksModel.d2(S, K, T, r, sigma)
        term1 = - (S * stats.norm.pdf(d1) * sigma * np.exp(-r*T)) / (2 * np.sqrt(T))
        if option_type == 'call':
            term2 = r * np.exp(-r*T) *(S*stats.norm.cdf(d1) - K*stats.norm.cdf(d2)) 
            return term1 - term2
        else:
            term2 = r * np.exp(-r*T)*(K*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1))
        return term1 + term2
    
    @staticmethod
    def rho(S:float, K:float, T:float, r:float, sigma:float, option_type:str='call') -> float:
        """Calculate the rho of a Black-76 option"""
        if option_type == 'call':
            return -T * BlacksModel.black76_call(S, K, T, r, sigma)
        else:
            return -T * BlacksModel.black76_put(S, K, T, r, sigma)
        