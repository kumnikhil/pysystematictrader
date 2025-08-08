from scipy.interpolate import UnivariateSpline
from sklearn.isotonic import IsotonicRegression
from scipy import interpolate, optimize
from typing import Callable
import numpy as np

def get_smoothened_vol_curve(strikes:np.ndarray,implied_vols:np.ndarray, return_spline:bool= False):
    # Interpolate missing data
    nans = np.isnan(implied_vols)
    implied_vols_interp = np.copy(implied_vols)
    implied_vols_interp[nans] = np.interp(strikes[nans], strikes[~nans], implied_vols[~nans])

    # Fit spline to the data with filled missing values
    spline = UnivariateSpline(strikes, implied_vols_interp, s=1)
    if return_spline:
        return spline(strikes), spline
    else:
        return spline(strikes), None
    
def fit_montonic_prices(strikes:float, prices:float, option_type:str, return_spline:bool=False):
    if option_type.upper() in  ['CALLS', 'CALL', 'CE']:
        increasing=False
    else:
        increasing=True
    iso_reg = IsotonicRegression(increasing=increasing)
    monotonic_prices = iso_reg.fit_transform(strikes,prices)
    spline = interpolate.UnivariateSpline(strikes, prices, s=0.01)
    if return_spline:
        return monotonic_prices, spline
    else:
        return monotonic_prices, None


def fit_convex_spline(strikes: np.ndarray, prices: np.ndarray) -> Callable:
    """
    Fit convex spline ensuring positive second derivative
    Args:
        strikes: Strike prices  
        prices: Option prices
        
    Returns:
        Interpolation function
    """
    
    def convexity_constraint(coeffs, strikes, prices):
        """Penalty function for non-convexity"""
        spline_temp = interpolate.UnivariateSpline(strikes, prices, s=0)
        spline_temp.set_smoothing_factor(coeffs[0])
        
        # Evaluate second derivative
        fine_strikes = np.linspace(strikes.min(), strikes.max(), 100)
        second_deriv = spline_temp.derivative(n=2)(fine_strikes)
        
        # Penalty for negative second derivatives
        penalty = np.sum(np.maximum(-second_deriv, 0)**2)
        
        # Add smoothness penalty
        smoothness = np.sum(np.diff(spline_temp(fine_strikes), 2)**2)
        
        return penalty + 0.01 * smoothness
    
    # Optimize smoothing parameter to maintain convexity
    result = optimize.minimize(
        convexity_constraint,
        x0=[0.01],
        args=(strikes, prices),
        bounds=[(1e-6, 1.0)]
    )
    
    optimal_smoothing = result.x[0]
    spline = interpolate.UnivariateSpline(strikes, prices, s=optimal_smoothing)
    
    return spline
