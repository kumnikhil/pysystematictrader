import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Callable, Any

class SVI_parameteric:
    @staticmethod
    def svi_raw(k:np.ndarray, params:Tuple[float, float, float, float, float]) -> np.ndarray:
        """ calculate SVI total variance w =  IV**2*T
        params : (a, b, rho, m, sigma)
        """
        a, b, rho, m, sigma = params
        # enforce the bounds
        b =  max(b, 0.0)
        sigma = max(sigma, 1.0e-4)
        rho = max(min(rho, 1.0), -1.0)
        total_variance = a + b * ( rho *( k - m) + np.sqrt((k - m)**2 +sigma**2))
        return np.maximum(total_variance, 1.0e-6)
    
    @staticmethod
    def svi_objective(
        params:Tuple[float, float, float, float, float],
        k : np.ndarray, 
        market_w :np.ndarray, 
        weights: np.ndarray)->float:
        model_w = SVI_parameteric.svi_raw(k, params)
        error = np.sum(weights*(model_w-market_w)**2)
        return error
    
    @staticmethod
    def calibrate_SVI_params(
        k: np.ndarray,
        market_w: np.ndarray, 
        initial_guess: List[float] = None,
        bounds: List[Tuple[float, float]] = None,
        weights: np.ndarray = None) -> Tuple[np.ndarray, float]:
        """
        Args: 
        market_w : array of market total variances (IV**2*T)
        initial_guess: Optional initial guess for parameters
        bounds: Optional 
        """
        if isinstance(weights, np.ndarray):
            weights =  np.ones_like(market_w)
        weights =  weights/ np.sum(weights)
        
        if initial_guess is None:
            # Rough estimation based on data properties
            w_atm = market_w[np.argmin(np.abs(k))] # Approx variance at the money
            min_w = np.min(market_w)
            max_w = np.max(market_w)
            k_min_w = k[np.argmin(market_w)] # Log-moneyness at min variance

            a_guess = min_w # 'a' roughly anchors the minimum variance level
            m_guess = k_min_w # 'm' is roughly the log-moneyness of min variance
            rho_guess = np.sign(market_w[-1] - market_w[0]) * 0.5 # Skew direction * guess
            rho_guess = np.clip(rho_guess, -0.95, 0.95)
            # b and sigma control the wings/curvature - harder to guess simply
            # Estimate sigma from the width of the smile
            k_range = np.max(k) - np.min(k)
            sigma_guess = max(k_range / 4.0, 0.05) # Heuristic
            # Estimate b based on the variance range
            b_guess = max((max_w - min_w) / (sigma_guess * 2.0), 1e-3) # Heuristic, avoid zero

            initial_guess = [a_guess, b_guess, rho_guess, m_guess, sigma_guess]

        if bounds is None:
            max_market_w = np.max(market_w)
            k_min, k_max = np.min(k), np.max(k)
            bounds = [
                (1e-6, max_market_w * 1.5),  # a: level, must be positive, bound by observed max var
                (1e-6, None),                 # b: slope magnitude, must be non-negative
                (-0.999, 0.999),              # rho: correlation, strictly within (-1, 1)
                (k_min * 1.5 if k_min < 0 else k_min * 0.5, k_max * 1.5 if k_max > 0 else k_max * 0.5), # m: location of min, within/near observed k range
                (1e-4, None)                  # sigma: ATM curvature, must be positive
            ]

        result = minimize(SVI_parameteric.svi_objective,
        initial_guess,
        args=(k, market_w, weights),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-7}) # Increased maxiter and adjusted tolerances

        if not result.success:
            print(f"Warning: Optimization failed: {result.message}")

        return result.x, result